# SimpleStorage SSD offload implementation progress

Last updated: 2026-08-13 (Asia/Shanghai)

## Goal

Add local SSD/NVMe offload to the TransferQueue `SimpleStorage` backend. The
target RL workload may otherwise retain both model state and TransferQueue
payloads in host DRAM and exceed the machine memory limit.

The implementation must preserve the existing controller/data-plane boundary,
field-level update and selection semantics, sample capacity behavior,
checkpoint support, observability, and default in-memory behavior. SSD mode
must remain usable without installing an external storage service.

## Repository instructions and skill review

The repository `AGENTS.md` was read before making changes. It requires the
repository-local `.codex/skills/simplicity-first/SKILL.md`, which was also read
in full before editing code.

The skill's overall direction is appropriate for TransferQueue: explicit
control flow, small changes, traceability, preservation of controller/storage
boundaries, and protection of existing contracts are valuable for an async
distributed system.

Potential skill improvements were identified but **the skill has not been
modified**:

- Replace the subjective “could this diff be half the size?” check with a
  concrete requirement to remove unused configuration, extension points, and
  wrappers, and to justify abstractions by a named invariant or real reuse.
- Add performance evidence requirements for performance-sensitive changes:
  representative tensor/nested/non-tensor workloads, throughput, p50/p99,
  peak RSS, disk use, and scaling with storage-unit count.
- Add persistence invariants: atomic disk-full failure, metadata publication
  only after successful storage, checkpoint compatibility, reusable disk
  space, cleanup, and crash behavior.
- Clarify that “review findings only; fixes after approval” applies to a pure
  review request, not to implementation already explicitly authorized by the
  user.
- Require risk-based directed tests before the full suite and explicit notes
  for optional backend tests that cannot run.
- Clarify that reducing abstraction/line count must not weaken transaction
  boundaries, lifecycle handling, compatibility, or observability.

## Existing implementation analysis

The existing path is:

1. `transfer_queue/config.yaml` selects `backend.storage_backend` and configures
   `backend.SimpleStorage`.
2. `initialize_simple_storage()` creates Ray `SimpleStorageUnit` actors and
   passes their ZMQ endpoints to the config.
3. `AsyncSimpleStorageManager` hashes each global index to a storage unit and
   performs ZMQ PUT/GET/CLEAR requests.
4. Each storage unit has one data worker thread and an in-memory
   `StorageUnitData` dictionary indexed by field and global index.
5. The manager publishes production metadata only after all storage-unit PUTs
   complete successfully.

The old `total_storage_size` option limits sample count, not payload bytes. It
does not bound host memory. Existing checkpoints pickle the complete in-memory
dictionary in one operation.

## Mooncake SSD offload comparison

The local Mooncake bootstrap/client implementation and current official
SQLite/Python documentation were reviewed. Research references:

- SQLite WAL behavior: <https://sqlite.org/wal.html>
- SQLite PRAGMA behavior: <https://sqlite.org/pragma.html>
- Python `sqlite3`: <https://docs.python.org/3/library/sqlite3.html>
- Upstream TransferQueue: <https://github.com/Ascend/TransferQueue>

The relevant Mooncake design is:

- DRAM-first hierarchy with high-watermark eviction.
- `mooncake_master` controls eviction and an external `mooncake_client` owns
  the SSD pool.
- The current TransferQueue bootstrap starts one centralized offload client on
  the first `tq.init()` node, so offload from other nodes may add a network hop
  and the SSD pool is a topology bottleneck.
- It supports an optional io_uring path and heartbeat/client-TTL handling.
- Offload startup failure is a hard failure; hard pinning is disabled when
  offload is enabled.

Decisions taken from this comparison:

- Reuse the configuration shape `offload.enabled` and
  `offload.file_storage_path`, hard-fail startup behavior, and explicit
  lifecycle cleanup.
- Do not add a Mooncake dependency to SimpleStorage.
- Use a local SSD database per SimpleStorageUnit so aggregate bandwidth can
  scale with storage nodes and avoid a centralized SSD hop.
- Do not copy Mooncake's watermark settings. Accurate low-overhead byte
  accounting for arbitrary live Python objects is difficult, and background
  spill would add eviction races and transient duplicate memory.
- Do not claim Mooncake is inherently faster. Backend choice must follow
  deployed topology and benchmark results. SimpleStorage's node-local layout
  can target high throughput as well.

## Selected architecture

SSD mode is explicit and SSD-primary:

- Default mode remains the existing in-memory `StorageUnitData`.
- When `offload.enabled=true`, `SimpleStorageUnit` constructs
  `DiskStorageUnitData` in the storage process.
- Every storage unit creates an isolated SQLite metadata index plus immutable
  sequential batch-payload files under the configured absolute local path.
- SQLite uses the rollback journal (`journal_mode=DELETE`) and
  `synchronous=NORMAL`. The index remains transactionally consistent across
  process crashes without WAL checkpoint write amplification. Batch files are
  not fsynced per PUT; a machine power loss is outside the working-storage
  durability boundary, and training recovery relies on TransferQueue
  checkpoints.
- SQLite page cache is explicitly bounded per storage unit. The OS may retain
  file pages as reclaimable filesystem cache; active payloads no longer scale
  the storage process's Python heap.
- PUT writes new payload files before atomically publishing their SQLite
  references; replaced files are deleted only after commit. If serialization,
  capacity, permission, disk-full, or index commit fails, new files and index
  changes are rolled back and the manager does not publish controller
  production metadata.

### Batched on-disk layout

The first implementation stored one SQLite BLOB per `(global_index, field)`.
A real benchmark showed that Python serialization and thousands of SQLite BLOB
rows, rather than NVMe bandwidth, limited throughput.

The layout was then changed to batch-oriented storage. Its first version kept
the batch payload itself as a SQLite BLOB:

- Each incoming field batch is serialized and written once to `field_batches`.
- `field_values` stores the compact mapping
  `(global_index, field) -> (batch_id, position)`.
- `samples` remains the authoritative active-key index and enforces existing
  sample capacity behavior.
- GET groups references by batch, reads/decodes each required batch once,
  and returns values in the caller's original global-index order.
- Partial field updates create a new batch only for updated fields and update
  the affected references. Unreferenced old batches are deleted in the same
  transaction.
- CLEAR cascades field-reference deletion and removes batch payloads when their
  last reference disappears.

After the second PUT profiling round, the batch BLOB was replaced by an
immutable batch file. `encode()` still produces the same packed-frame format,
but the storage unit builds only its small header and sends header plus original
tensor/NumPy frames to the file with `writev`; it no longer creates a full-size
packed `bytearray` or copies the payload through SQLite B-tree pages. SQLite
stores `(batch_id, file_name, payload_size)` and the sample/field references.
The disk metric reports the physical database, SQLite auxiliary, temporary, and
batch-file bytes rather than logical live-payload bytes.

This preserves selected-field reads and partial-field updates while avoiding a
large BLOB insert and serialization call for every sample.

### Serialization and memory behavior

Disk payloads reuse TransferQueue's `encode`, `unpack_from`, and `decode`
functions. Tensor and NumPy data remain raw frames rather than generic tensor
pickle payloads. PUT builds only the packed-frame header and uses `writev` to
write the header and original frames without a full payload copy. GET reads
each required batch file directly into one writable `bytearray`; reconstructed
tensors are views over that request buffer.

The design bounds persistent payload memory but cannot eliminate the peak
memory of the PUT/GET request currently being received or returned.

## Configuration

New default configuration under `backend.SimpleStorage`:

```yaml
offload:
  enabled: false
  file_storage_path: /tmp/transfer_queue_simple_storage
  memory_cache_size_bytes: 67108864
```

- `enabled=false` preserves the original behavior.
- `file_storage_path` must be an absolute directory available at the same local
  path on every eligible Ray storage node. It should point to node-local NVMe,
  not NFS/shared storage.
- `memory_cache_size_bytes` is the SQLite page cache limit per storage unit.
  It must be non-negative.
- `total_storage_size` retains its old meaning: maximum active sample count,
  not bytes.

Bootstrap validates enabled configuration and forwards `offload_path` and the
cache size to every `SimpleStorageUnit`. Actor construction fails immediately
for invalid paths or SQLite initialization errors.

## Checkpoint decisions

In-memory checkpoints retain the old pickle representation for compatibility.
SSD checkpoints use a streamed, bounded-memory format containing:

- a versioned header and configured sample capacity;
- active sample-index chunks;
- packed batch-payload chunks;
- `(global_index, field, batch_id, position)` reference chunks.

Checkpoint restore writes uniquely named replacement batch files and updates
the index in one SQLite transaction. A required end record detects truncated
streamed checkpoints; corrupt or incomplete restores roll back the index,
delete newly created files, and leave previous SSD data and files intact.
Legacy in-memory checkpoints can be loaded into SSD mode, and streamed SSD
checkpoints can be loaded into memory mode.

## Lifecycle and failure handling

- `SimpleStorageUnit.close()` invokes the registered finalizer explicitly.
- `tq.close()` first asks every SimpleStorage actor to close gracefully, then
  kills the Ray actor even if graceful close reports an error.
- The bandwidth benchmark uses the same close path.
- Graceful close closes SQLite and removes only the generated storage-unit
  database, SQLite auxiliary files, batch files, and unit directory.
- An ungraceful process/node failure may leave an orphan unit directory. It can
  be removed when no TransferQueue job is using the configured root.

The new explicit close test exposed an existing ZMQ shutdown deadlock:
`Context.term()` waited for a worker thread's local socket. Shutdown now calls
`Context.destroy(linger=0)` to close all sockets before joining worker/proxy
threads. Poll errors caused by deliberate shutdown are no longer logged as
warnings.

## Observability

Storage-unit metrics now include `offload_disk_bytes`, the current total size
of the database, batch payloads, and SQLite auxiliary files. The controller
Prometheus exporter exposes it as:

```text
tq_storage_offload_disk_bytes{storage_unit_id="..."}
```

It is zero in memory mode. Existing active-key, capacity, request latency, and
RSS metrics remain unchanged.

## Files changed so far

- `transfer_queue/storage/simple_storage_disk.py`
  - Added `DiskStorageUnitData`.
  - Added packed SSD serialization helpers.
  - Added zero-copy-header/scatter-gather sequential batch-file storage with a
    transactional SQLite index, atomic PUT/GET/CLEAR, capacity checks,
    checkpoint streaming/rollback, disk usage, and cleanup.
- `transfer_queue/storage/simple_storage.py`
  - Added SSD constructor arguments to `SimpleStorageUnit`.
  - Kept memory/SSD checkpoint interoperability and selects the concrete data
    store explicitly during actor construction.
  - Routed checkpoint handlers through the selected data store.
  - Added disk metrics and explicit close.
  - Fixed ZMQ shutdown deadlock exposed by explicit close.
- `transfer_queue/storage/bootstrap/simple_storage_bootstrap.py`
  - Validates and forwards SimpleStorage offload configuration.
- `transfer_queue/config.yaml`
  - Added disabled-by-default SimpleStorage SSD configuration.
- `transfer_queue/interface.py`
  - Gracefully closes SimpleStorage actors before `ray.kill`.
- `transfer_queue/metrics.py`
  - Added the Prometheus SSD disk-usage gauge.
- `docs/storage_backends/simple_storage_ssd.md`
  - Added configuration, data-path, checkpoint, topology, cleanup, and
    benchmark documentation.
- `README.md`
  - Linked SimpleStorage to the SSD guide and mentioned optional local NVMe.
- `docs/checkpoint.md`
  - Documented that SimpleStorage checkpoints cover memory and SSD working
    data.
- `docs/metrics.md`
  - Documented the offload disk-usage metric.
- `tests/test_simple_storage_unit.py`
  - Added SSD round-trip/partial-update, capacity atomicity, cross-mode
    checkpoint, tensor/nested/non-tensor serialization, mocked disk-full
    rollback retaining old batch files, incomplete-checkpoint rollback, and
    Ray/ZMQ actor end-to-end tests.
- `tests/test_simple_storage_scheduling.py`
  - Added disabled/enabled bootstrap forwarding and missing-path validation.
- `tests/test_metrics.py`
  - Added metric registration and collection assertions.
- `scripts/put_benchmark.py`
  - Added SSD path/cache CLI options and actor forwarding.
  - Added graceful storage cleanup.
  - Fixed the script's stale manager registration name
    (`AsyncSimpleStorageManager` -> `SimpleStorage`) so the benchmark runs.
- `scripts/performance_test/perftest_config.yaml`
  - Added the disabled-by-default SimpleStorage offload configuration to the
    general performance-test template.
- `scripts/performance_test/README_PERFTEST.md`
  - Added SSD configuration and same-workload in-memory baseline guidance.
- `progress.md`
  - This design, decision, change, and verification log.

No repository skill file has been changed.

## Tests run and results

All commands used the existing Python 3.10 environment at
`/Users/jhz/miniforge3/envs/lynn/bin/python3.10`. No dependency installation was
required.

### Static validation

```bash
python -m compileall -q <changed Python files>
ruff check <changed Python files>
ruff format <changed Python files>
git diff --check
```

Status before the final documentation update: compile, changed-file Ruff, and
diff checks pass. The final static rerun is recorded below.

### Focused unit tests

```bash
pytest -q \
  tests/test_simple_storage_unit.py::test_disk_storage_unit_data_round_trip_and_partial_update \
  tests/test_simple_storage_unit.py::test_disk_storage_capacity_failure_is_atomic \
  tests/test_simple_storage_unit.py::test_disk_checkpoint_is_portable_between_storage_modes \
  tests/test_simple_storage_scheduling.py \
  tests/test_metrics.py
```

Result before the batch-layout optimization: `24 passed`. The three direct SSD
tests were rerun after the batch-layout change: `3 passed`.

### Ray/ZMQ SSD end-to-end test

```bash
pytest -q tests/test_simple_storage_unit.py::test_storage_unit_ssd_offload_e2e
```

The macOS sandbox initially blocked Ray's process enumeration, so Ray tests were
rerun with approved sandbox escalation. The first run exposed the ZMQ close
deadlock described above. After the fix, the test passed. It was rerun after
the batch-layout optimization and passed again: `1 passed`.

### SimpleStorage regression group

```bash
pytest -q \
  tests/test_simple_storage_unit.py \
  tests/test_simple_storage_scheduling.py \
  tests/test_async_simple_storage_manager.py \
  tests/test_metrics.py
```

Result after the batch-layout optimization: `65 passed in 58.79s`. The mocked
disk-full rollback test added immediately afterward passed separately:
`1 passed`.

### Full repository suite

```bash
pytest -q
```

This full-suite run preceded PUT optimization round 2. Result:
`584 passed, 10 skipped, 1 warning, 8 errors in 307s`.

All eight errors are setup errors from
`tests/test_yuanrong_storage_client_e2e.py`: the environment does not have the
optional openYuanrong client, and that fixture attempts to patch a missing
`yuanrong_client.datasystem` attribute without `create=True`. No
SimpleStorage, SSD-offload, Mooncake, or general regression test failed. This
unrelated optional-backend fixture was deliberately not changed under the
one-problem scope rule. The warning is the pre-existing immutable-buffer
warning in `serial_utils` tests.

## Performance measurements

The existing `scripts/put_benchmark.py` was run locally with two storage units,
five rounds, the `small` case, and about `0.1465 GB` per round. These macOS
results are smoke-test evidence, not a production NVMe performance claim.

### In-memory baseline

- PUT mean: `7.760 Gbps`
- GET mean: `5.138 Gbps`

### Initial per-sample/per-field SQLite BLOB layout

- PUT mean: `2.226 Gbps`
- GET mean: `2.572 Gbps`

### Batched field-BLOB layout

- PUT mean: `2.929 Gbps`
- GET mean: `4.112 Gbps`
- PUT improvement over the first SSD layout: about `31.6%`
- GET improvement over the first SSD layout: about `59.8%`
- GET reached about `80%` of the in-memory smoke-test throughput.

The first implementation intentionally kept SQLite WAL and atomic commits. It
was not changed to `journal_mode=OFF`, because silently trading away
disk-full/crash consistency for a peak benchmark number conflicts with the
availability requirement.

The benchmark's data-consistency reporter prints an existing false failure for
regular tensors reconstructed by `AsyncSimpleStorageManager._pack_field_values`
as `NestedTensor`. The exact same warning occurs in both in-memory and SSD
runs; direct SSD value tests and the Ray/ZMQ end-to-end test pass. This existing
manager/benchmark mismatch is not being fixed as part of SSD offload.

## PUT performance optimization round 2

After review feedback that PUT remained too slow, the same end-to-end small
benchmark was repeated before further changes. The current WAL baseline was:

- PUT mean: `3.073 Gbps` (five rounds, range `2.275–3.387 Gbps`)
- GET mean: `4.045 Gbps`

A direct per-storage-unit profile used the same post-routing shape: four
fields, 256 samples, and 75 MiB payload. Results:

- TransferQueue serialization plus packed-buffer copy: `0.0216s`
  (`3466 MiB/s`).
- SQLite WAL with its default automatic checkpoint: `0.2767s`
  (`271 MiB/s`), with about 151 MiB of DB/WAL files for a 75 MiB payload.
- WAL with foreground auto-checkpoint disabled: `0.1298s`
  (`578 MiB/s`).
- SQLite rollback journal modes: `0.121–0.124s`
  (`607–619 MiB/s`), with about 75 MiB of database files.

Conclusion: serialization was not the dominant direct-storage cost. The
default 1000-page WAL auto-checkpoint copied each large transaction from WAL
into the main database before PUT returned, nearly doubling physical payload
movement. Every SimpleStorageUnit has one data worker, so WAL's reader/writer
concurrency does not benefit this data path.

Decision: use `journal_mode=DELETE` with `synchronous=NORMAL`. This preserves
SQLite transaction rollback and disk-full/process-crash consistency while
avoiding WAL checkpoint write amplification. `journal_mode=OFF` remains
rejected. A separate payload-file protocol was considered but not implemented:
although scatter/gather file writes could remove another buffer copy, it would
introduce cross-file commit, orphan cleanup, compaction, and checkpoint
invariants before the smaller journal-mode correction had been measured
end-to-end.

The final end-to-end numbers for this change are recorded after the benchmark
rerun below.

### End-to-end results and final payload layout

The rollback-journal change alone raised SSD PUT mean from `3.073` to
`3.970 Gbps` (`+29.2%`). An intermediate 64 KiB SQLite page-size experiment
raised it to `4.627 Gbps` (`+50.6%` over WAL), confirming that B-tree page
processing remained material. That page-size setting was removed after payloads
left SQLite because it no longer served an invariant or measured hot path.

The in-memory benchmark was repeated in the same environment and measured:

- PUT mean: `7.526 Gbps`
- GET mean: `4.562 Gbps`

The smaller journal correction still left SSD PUT at only 61.5% of that memory
mean, so the previously deferred payload-file design was implemented with the
following ordering rules:

1. Serialize only the packed-frame header and write header plus original frames
   to a uniquely named temporary file using bounded `writev` batches.
2. Atomically rename the complete temporary file to its immutable batch name.
3. Insert the new batch path and sample/field references in the SQLite
   transaction.
4. On any write/index/commit error, roll back SQLite and delete every new batch
   file. Delete replaced or cleared files only after commit.

The final implementation was benchmarked twice. The first five-round run
measured PUT `5.724 Gbps`; after removing the now-unused 64 KiB page-size
setting, the final-code run measured:

- PUT mean: `5.582 Gbps`, max/p99 `6.577/6.575 Gbps`
- GET mean: `4.576 Gbps`
- final three PUT rounds: `6.30`, `6.55`, and `6.58 Gbps`
- improvement over the same-run WAL baseline: `81.6%`
- final SSD mean: `74.2%` of the same-environment memory PUT mean
- final three-round SSD mean: about `80.9%` of the memory run's final
  three-round mean

These remain local macOS page-cache smoke measurements, not target-NVMe
acceptance numbers. Payload files deliberately do not call `fsync` per PUT:
SimpleStorage working data is ephemeral and checkpoint recovery is required
after node/power loss. Adding a durability barrier to each field file would
serialize multiple device flushes and would present a misleading availability
contract because the controller and Ray actors are not independently durable.
Handled disk-full/write/commit failures remain atomic from the running job's
perspective and do not publish controller production metadata.

The file-backed checkpoint format was bumped to
`tq_simple_storage_batches_v2` and now includes a required end record. This
prevents a truncated stream from being mistaken for a successful shorter
checkpoint; restore rollback also deletes newly staged batch files and retains
the prior files.

## Final verification and open points

- After the final batch-file/checkpoint changes and the readability-only split
  into `simple_storage_disk.py`, the complete SimpleStorage regression group
  was rerun with the required Ray permission: `67 passed in 56.67s`.
- Final `ruff check` passed, `ruff format --check` reported all nine changed
  Python files already formatted, full `compileall -q transfer_queue tutorial
  tests scripts` passed, and `git diff --check` passed.
- Two attempts to start the final full suite with the macOS Ray/process/socket
  permission were blocked by the desktop automatic-approval timeout before a
  test result was returned. A normal-sandbox full run completed as expected but
  could not initialize Ray, enumerate processes, or bind sockets:
  `399 passed, 10 skipped, 3 failed, 192 errors in 17.84s`. The three failures
  are permission errors in a ZMQ client test, Ray-backed KV test, and real-socket
  serialization test; the setup errors are Ray/ZMQ permission errors plus the
  same eight unavailable-openYuanrong errors. This is recorded as an
  environment-limited run, not a code regression. The affected final
  SimpleStorage/Ray group did run with permission and passed all 67 tests.
- The final diff was reviewed against the simplicity checklist. The disk store
  remains a concrete storage implementation rather than a new framework or
  external service. It was moved out of the 1272-line actor module so the
  in-memory store, SSD store, and Ray/ZMQ actor are readable independently;
  this was a pure code move and the post-move 67-test group passed.
  Configuration is disabled by default and contains only the path and
  bounded-cache controls needed by the selected design.
- Production performance acceptance should be run on target Linux NVMe nodes
  with representative RL batches and should record PUT/GET throughput,
  p50/p99 latency, CPU, process RSS, disk bytes, physical NVMe bandwidth, and
  scaling across storage-unit counts.

## Mooncake Store integration and performance round

### Scope and measurement contract

This round evaluates Mooncake Store as a second SSD-offload implementation,
not as a replacement for SimpleStorage SSD mode. Both backends may be used for
high-throughput workloads; the comparison must be based on measured topology,
resource, latency, and availability trade-offs rather than assigning either
backend an artificial performance tier.

Mooncake's documented contract separates a synchronous memory write from
asynchronous SSD persistence. Therefore a normal `kv_batch_put` duration is a
front-door DRAM acknowledgement measurement, not SSD throughput. A valid SSD
test must report separately:

1. foreground PUT time;
2. time until the SSD offload queue has drained / the payload is observable on
   disk;
3. cold GET after the memory replica has actually been evicted; and
4. the exact DRAM segment, watermark, disk path, protocol, and filesystem used.

The existing generic performance script does not establish points 2 or 3, so
its Mooncake PUT/GET result cannot currently be labelled an SSD result.

### Repository and upstream findings

- TransferQueue already depends on `mooncake-transfer-engine>=0.3.10.post2`.
  That exact Mooncake release added SSD offload through the Python `setup()`
  interface, SSD metrics, and fixes for metadata-server/offload paths.
- The current TransferQueue bootstrap starts one additional standalone
  `mooncake_client` on the first node. This adds another configured global DRAM
  segment and makes one node's disk the cluster-wide SSD bottleneck. It is a
  valid Mooncake topology, but it works against the original goal of reducing
  host-memory pressure and leaves a multi-node-distributed-offload TODO.
- The official programmatic interface can instead enable SSD on each existing
  real client with `enable_ssd_offload=True` and `ssd_offload_path=...`. This
  avoids the extra storage process/DRAM segment and naturally contributes each
  node's local NVMe bandwidth. This is the selected integration direction.
- The current master uses `offload_on_evict=true`. This is appropriate when
  minimizing SSD write amplification, but it means neither a successful PUT
  nor a two-second sleep proves the object reached SSD. The performance test
  must create real memory pressure and verify disk/metrics before cold GET.
- `ReplicateConfig.with_soft_pin` defaults to false and TransferQueue disables
  hard pin when offload is enabled. The master flag
  `allow_evict_soft_pinned_objects=false` therefore does not block normal TQ
  values, although the unusually long soft-pin TTL remains unnecessary for
  values explicitly configured with soft pin in future callers.

Primary references used for these decisions are Mooncake's deployment/tuning
guide, design document, the `v0.3.10.post2` release notes, and the official
PyPI package pages. The current docs also show newer options and fixes; the
implementation is restricted to APIs present in TransferQueue's minimum
version.

### Local runtime preparation

The host is macOS/ARM and Mooncake publishes POSIX/Linux wheels, so it cannot
be imported natively. A real Linux test environment was prepared with the
existing Colima VM (Ubuntu 24.04, aarch64), resized from 2 CPU/2 GiB to
4 CPU/6 GiB. The shared repository is mounted read-only for runtime tests.
Docker image extraction repeatedly crashed Colima's containerd, so the test is
being run directly in the VM instead of treating that infrastructure failure
as a backend result. This VM filesystem is suitable for functional and
virtualized-disk throughput evidence only; it is not target NVMe acceptance.

### Scope correction after user review (2026-08-13)

The user clarified that the desired topology is **not** two independent TQ
backends. The intended path is:

```text
TransferQueue SimpleStorage data plane / 2-D sample index
    -> Mooncake Store payload tier
        -> node-local SSD/NVMe
```

Accordingly, the earlier statement that this round was evaluating Mooncake as
an independent alternative is superseded. The direct Mooncake measurements
remain useful only as a lower-layer component baseline. The implementation and
acceptance test must exercise the combined `SimpleStorage -> Mooncake -> SSD`
path through SimpleStorage's public ZMQ requests.

The revised design keeps SimpleStorage responsible for routing, capacity,
field/sample indexing, partial update, clear, checkpoint, and metrics. A small
SQLite index remains local and bounded; batch payloads are stored under unique
Mooncake keys. Each SimpleStorage actor embeds one real Mooncake client and
contributes its own node-local SSD path. Mooncake hard pin is disabled, its
DRAM segment is explicitly bounded per actor, and the master lease must be
short enough to let persisted replicas leave host memory.

The existing local-file payload implementation is retained as an explicit
fallback/reference provider while the Mooncake provider is implemented and
measured. It is not used as evidence for the requested combined path.

## Combined SimpleStorage -> Mooncake -> SSD implementation (final record)

This section is the authoritative record for the corrected scope and
supersedes the earlier independent-backend framing above.

### Selected architecture

```text
TransferQueueClient / AsyncSimpleStorageManager
    -> SimpleStorageUnit public ZMQ data plane
        -> SQLite sample/field index
        -> embedded MooncakeDistributedStore client
            -> bounded DRAM segment
            -> asynchronous node-local SSD/NVMe payload tier
```

SimpleStorage continues to own routing, sample capacity, `(sample, field)`
semantics, ordering, partial updates, clear, checkpoint, and storage-unit
metrics. Payloads never pass through the controller. Each Ray storage actor is
a separate process with one Mooncake client and one local SSD directory, so
storage nodes contribute their local device bandwidth rather than funneling
all SSD traffic through a standalone first-node client.

The exact compatibility target is TransferQueue's declared minimum
`mooncake-transfer-engine>=0.3.10.post2`. The implementation deliberately does
not pass newer master flags that are absent from post2.

### Key implementation decisions

1. `backend.SimpleStorage.offload.backend: mooncake` is the default provider
   when SimpleStorage offload is enabled. `local_file` remains an explicit
   dependency-free fallback and performance reference; it is not a separate
   performance tier and is not excluded from high-throughput use.
2. Each field batch is encoded with TransferQueue's existing packed-frame
   format. A compact SQLite index stores only the Mooncake payload reference,
   logical size, field, sample index, and position. SQLite transaction commit
   publishes a batch only after every Mooncake chunk has succeeded.
3. Mooncake PUT uses `batch_upsert_from` with one registered contiguous region
   per SimpleStorage request. Error `-200` (`NO_AVAILABLE_HANDLE`) is bounded
   backpressure and is retried until `put_timeout_seconds`; other errors fail
   immediately. Partial successes are removed if the logical PUT fails.
4. Hard pin is disabled. The SimpleStorage default lease is 500 ms rather than
   the old 999999 ms Mooncake bootstrap value, because an effectively permanent
   lease prevents the memory-pressure goal from being achieved. Foreground
   users may tune the lease, watermarks, and DRAM segment explicitly.
5. Post2's default bucket backend holds an incomplete bucket until it reaches
   256 MiB or 500 keys. This made small workloads appear to offload while their
   tail stayed in the ungrouped queue indefinitely. Since SimpleStorage already
   batches samples, the actor sets `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1` so
   each batch object becomes eligible for immediate persistence.
6. Post2 cold reads allocate Cachelib slices whose actual maximum is
   `16 MiB - 16 B`, plus 8 KiB of alignment/tail space per slice. The adapter
   transparently chunks payloads at a conservative 16 MiB minus 64 KiB,
   rejects unsafe overrides before entering native code, and windows cold GETs
   by their allocation footprint. This allows a logical model/batch larger
   than the Mooncake segment or SSD staging buffer.
7. Native PUT and remove calls are capped at 400 chunk keys, matching the
   existing Mooncake client integration. This prevents very large models from
   exceeding a native batch limit after transparent chunking.
8. Mooncake cold payloads use `batch_get_into`; post2's high-level `get()` path
   does not accept its local-disk descriptor variant. GET reconstructs the
   requested SimpleStorage batch only after all windows succeed.
9. Graceful close removes logical payload chunks, shuts down the native client,
   closes SQLite, removes the actor directory, then terminates only the
   Mooncake master process returned to and owned by this SimpleStorage
   bootstrap. Actor/bootstrap failures clean already-created actors and the
   owned master. `auto_init: false` attaches to an externally managed master
   and therefore does not claim ownership.
10. A successful normal PUT is a Mooncake DRAM acknowledgement, not a durable
    SSD acknowledgement. The running job remains consistent under handled
    backpressure/write/index errors, and Mooncake evicts memory only after an
    SSD replica exists. Node/power loss before asynchronous persistence, full
    master HA, and device loss remain checkpoint/deployment concerns rather
    than a stronger durability promise made by this feature.
11. The pre-existing Mooncake `auto_init: true` path terminates an existing
    local `mooncake_master` before starting its owned process. Production/shared
    clusters should use `auto_init: false` and supervise an external/HA master;
    the SSD guide now uses that safer production example. Changing the legacy
    auto-init contract itself is outside this coherent feature diff.

### Files changed for the combined path

- `transfer_queue/storage/simple_storage_mooncake.py`: new SimpleStorage
  payload provider using Mooncake batch pointer APIs, bounded backpressure,
  transparent chunking, windowed cold reads, cleanup, and disk metrics.
- `transfer_queue/storage/simple_storage_disk.py`: concrete shared
  SQLite-indexed SimpleStorage implementation with overridable batch payload
  operations; the direct local-file implementation remains here.
- `transfer_queue/storage/simple_storage.py`: selects memory, local-file, or
  Mooncake payload data at actor construction and provides explicit lifecycle,
  checkpoint, and metrics behavior.
- `transfer_queue/storage/bootstrap/simple_storage_bootstrap.py`: builds the
  post2-compatible combined configuration, starts/owns the master when asked,
  forwards settings to every actor, and cleans partial initialization.
- `transfer_queue/storage/bootstrap/mooncake_bootstrap.py`: removes the extra
  centralized standalone offload client, enables only post2-supported master
  flags, and makes the eviction lease configurable.
- `transfer_queue/storage/clients/mooncake_client.py`: enables SSD on the real
  embedded Mooncake client, normalizes its path/environment, and disables hard
  pin automatically when offload is active.
- `transfer_queue/config.yaml`: adds disabled-by-default SimpleStorage Mooncake
  offload and bounded per-actor defaults.
- `transfer_queue/interface.py`: gracefully closes SimpleStorage actors and
  terminates only a master owned by their bootstrap.
- `transfer_queue/metrics.py` and `tests/test_metrics.py`: export and verify
  per-actor offload directory bytes.
- `tests/test_simple_storage_mooncake.py`: fake-native semantic, setup,
  backpressure, large-field chunking, cold-window, unsafe-post2-boundary, and
  400-key native-batch tests.
- `tests/test_simple_storage_scheduling.py`: disabled/local-file/Mooncake
  configuration forwarding, master ownership, validation, and failed-bootstrap
  cleanup tests.
- `tests/test_simple_storage_unit.py`: environment-gated real
  SimpleStorage-ZMQ-to-Mooncake-SSD end-to-end test.
- `scripts/performance_test/simple_storage_mooncake_benchmark.py`: new public
  SimpleStorage data-path benchmark reporting foreground PUT, physical SSD
  visibility, pipeline throughput, eviction evidence, and verified cold GET.
- `scripts/performance_test/mooncake_offload_benchmark.py`: lower-layer
  component benchmark with post2-safe bucket, slice, and cold-buffer controls.
- `docs/storage_backends/simple_storage_ssd.md`, `README.md`,
  `docs/checkpoint.md`, `docs/metrics.md`, and the performance-test templates:
  configuration, topology, semantics, limitations, and reproducible commands.

The prior local-file work, PUT optimization, checkpoint format, metrics, and
tests remain in the diff because they form the SimpleStorage indexing/provider
base and an explicit fallback/reference. No repository skill file was changed.

### Mooncake failures found during implementation

- A first real 8 MiB-object E2E test filled the 64 MiB segment and timed out:
  the master fetched offload tasks but reported zero completions. A direct
  Mooncake control test failed the same way, proving the adapter was not the
  cause. Source inspection found the incomplete default 256 MiB/500-key bucket
  rule; one-key buckets fixed the issue.
- The old `default_kv_lease_ttl=999999` prevented DRAM eviction for roughly
  1000 seconds. The combined path now defaults to 500 ms.
- `store.get()` on a post2 local-disk replica failed with
  `Expected DiskDescriptor`; the supported `batch_get_into` cold path succeeds.
- Attempts with a 64 MiB logical chunk caused a native `SIGABRT` in
  `FileStorage::AllocateBatch`. Upstream source shows Cachelib's slab is 16
  MiB, not 64 MiB. The final 16 MiB-minus-64 KiB chunk boundary passed a 64
  MiB logical-batch cold read split into five chunks/two native GET windows.
- The aborted runs also produced noisy shutdown/native failure symptoms. With
  correct buckets and slice bounds, the real actor closes cleanly; no native
  crash occurred in the final E2E or performance runs.
- An initial command used the wrong Python environment, and one master launch
  collided with a stale port. These were test-environment errors, not backend
  results; the final commands use the isolated Linux environment and one owned
  master.

### Real functional and regression tests

The host is macOS/ARM, while Mooncake publishes Linux wheels. Real Mooncake
tests therefore ran in the existing Colima Ubuntu 24.04 ARM VM with 4 vCPU and
6 GiB RAM, exact `0.3.10.post2`, CPU PyTorch, Ray, TensorDict, and P2P/TCP.

- Fake Mooncake provider plus scheduling/bootstrap tests after the final large
  model and cleanup additions: `15 passed`.
- Final complete SimpleStorage/manager/metrics regression group:
  `74 passed, 1 skipped in 72.19s`. The skip is the opt-in real Mooncake case,
  which was run separately below.
- Final real public data-plane test with `TQ_RUN_MOONCAKE_E2E=1`:
  `1 passed in 15.98s`. It wrote twelve 8 MiB logical batches through ZMQ with
  only a 64 MiB Mooncake segment, observed backpressure/recovery and 72 MiB of
  master-confirmed eviction, then read verified values through the local-disk
  offload RPC.
- Repository-wide run before the final bootstrap-cleanup guard (covered by the
  later focused group): `592 passed, 11 skipped, 1 warning, 8 errors in
  394.53s`. All eight errors are the pre-existing optional openYuanrong fixture
  trying to patch a missing `yuanrong_client.datasystem` attribute. No
  SimpleStorage, Mooncake, SSD, controller, manager, or metrics test failed.
- `simplicity-first` itself passes the skill-creator `quick_validate.py`
  structural validator.

### Performance results

All Mooncake numbers below are single-actor TCP results on the 4-vCPU/6-GiB
Colima VM's virtual filesystem. They establish behavior and identify software
overhead; they are not production NVMe acceptance numbers.

#### Combined public SimpleStorage path

1. Pressure-heavy smoke test: 256 MiB total, 8 MiB logical batches, 64 MiB
   Mooncake segment, 32 MiB SSD staging buffer:
   - foreground PUT: `0.503 Gbps`;
   - SSD-visible pipeline: `0.447 Gbps`;
   - PUT p50/p99: `10.6/1037.0 ms`;
   - verified 8 MiB cold GET: `2.443 Gbps`;
   - physical directory growth: `268,450,816 B`;
   - 29 master-confirmed evicted keys.
   The tiny segment spends most of the run waiting for each one-second offload
   heartbeat and is a pressure/availability test, not a recommended setting.
2. Default-scale DRAM test: 768 MiB total, 8 MiB batches, 512 MiB segment,
   64 MiB staging:
   - foreground PUT: `2.414 Gbps`;
   - SSD-visible pipeline: `1.776 Gbps`;
   - PUT p50/p99: `13.3/118.9 ms`;
   - verified 8 MiB cold GET: `3.359 Gbps`;
   - directory growth: `805,389,312 B`;
   - 65 master-confirmed evicted keys.
3. Final post2-safe large-batch test: 768 MiB total, 64 MiB logical batches,
   512 MiB segment, 64 MiB staging, 16 MiB-minus-64 KiB chunks:
   - foreground PUT: `2.501 Gbps`;
   - SSD-visible pipeline: `1.769 Gbps`;
   - PUT p50/p99: `130.3/839.0 ms`;
   - verified 64 MiB cold GET: `2.393 Gbps`;
   - directory growth: `805,353,960 B`;
   - 41 master-confirmed evicted keys.

#### Comparable Mooncake component baseline

The lower-layer benchmark was rerun with the same one-key buckets, conservative
16 MiB-minus-64 KiB objects, 512 MiB segment, 64 MiB staging, TCP, and about
765 MiB total:

- foreground PUT: `3.408 Gbps`;
- SSD pipeline: `2.083 Gbps`;
- verified 63.75 MiB cold GET: `5.325 Gbps`;
- all 48 objects reached LOCAL_DISK, four selected objects were disk-only,
  final master memory usage was zero;
- logical/allocated disk growth: `802,163,788 / 802,553,856 B`.

Against this topology-matched component run, the combined public path measured
about 26.6% lower foreground PUT and 15.1% lower full SSD-pipeline bandwidth.
That gap includes ZMQ request/response serialization, TensorDict/PyTorch
packing, SQLite publication, chunk manifest work, and test-shape differences;
it must not be attributed to one adapter copy without a production profile.
Cold GET additionally reconstructs and returns SimpleStorage tensors over ZMQ.

The earlier 256 MiB direct component runs with Mooncake's default 256-object
bucket measured `15.56–24.49 Gbps` foreground PUT, `1.73–1.81 Gbps` SSD
pipeline, and `7.18–8.13 Gbps` cold GET. They remain recorded as diagnostic
history, but their bucket layout is not an apples-to-apples combined-path
baseline.

The direct local-file SimpleStorage fallback reached `6.30–6.58 Gbps` in its
stable final PUT rounds on the macOS page-cache benchmark. This confirms that
SimpleStorage SSD mode is not inherently excluded from high-throughput use.
For the combined Mooncake path, long steady-state PUT converges toward SSD
pipeline capacity once its bounded DRAM segment fills. Production throughput
should be raised and accepted by measuring real NVMe/io_uring and scaling
multiple SimpleStorage actors/devices; a single virtual disk result should not
be relabelled as a backend ceiling.

### `simplicity-first` skill review (proposal only)

The skill is structurally valid, concise, and directionally correct. It was
useful in keeping payload movement out of the controller, choosing concrete
providers instead of a framework, making retry/cleanup visible, and preserving
SimpleStorage's public semantics. The skill was not edited.

Suggested changes for user review:

1. Replace `One problem = one minimal diff` with `One coherent behavior = one
   reviewable diff`. Clarify that a distributed storage behavior normally
   requires its configuration, lifecycle, failure handling, observability,
   focused tests, and user documentation in the same coherent change. The
   current wording can incentivize an artificially incomplete feature.
2. Add an optional-dependency compatibility gate: treat the repository's
   minimum declared version as the implementation target, inspect the exact
   installed/upstream API, and do not copy flags from latest documentation
   without a version test.
3. Add a performance-evidence gate: define the acknowledgement boundary and
   report foreground latency/throughput separately from asynchronous drain,
   durable/persisted visibility, and cold-tier reads; always record topology,
   resource bounds, hardware/filesystem, and workload.
4. Add a distributed-lifecycle checklist item: name process/client ownership,
   partial-initialization rollback, graceful and forced cleanup, backpressure
   timeout, and node/service/device failure semantics.
5. Keep those additions short. If storage/backend work is frequent enough to
   need detailed matrices or benchmark recipes, create a separate repo-local
   `storage-backend-validation` skill/reference rather than bloating the
   general readability skill.
6. The repo-local skill is a symlink to `.agent/skills` and has no
   `agents/openai.yaml`. Skill Creator calls that metadata recommended rather
   than required. Add it only if this repo intends to surface the skill in UI
   lists; it is not needed for the current AGENTS.md-triggered workflow.

### Final static validation

- `ruff check` passed for all changed Python files.
- `ruff format --check` passed after formatting the new concrete disk provider.
- `python -m compileall -q transfer_queue tutorial tests scripts` passed.
- `git diff --check` passed.
- The final SimpleStorage focused group passed after the last code change, and
  no skill file was modified.

## Same-host Mooncake versus SQLite/local-file comparison

### Terminology and comparison boundary

The current implementation does not offer a production mode that stores
payload BLOBs in SQLite. Both selectable SSD providers use the same SQLite
sample/field index. The meaningful current A/B comparison is therefore:

1. `SQLite index -> immutable local-file payload` (`local_file`); and
2. `SQLite index -> Mooncake payload -> bounded DRAM + local SSD`
   (`mooncake`).

The older SQLite-BLOB prototypes remain useful diagnostic history but are not
current providers. Their macOS results cannot be compared numerically with a
Linux-VM Mooncake run. They were removed because copying large payloads through
SQLite B-tree pages and journals materially reduced PUT throughput; SQLite
remains well suited to the small transactional index.

### Benchmark change and measurement rules

`scripts/performance_test/simple_storage_mooncake_benchmark.py` now accepts
`--offload-backend local_file|mooncake`. The same public ZMQ PUT/GET path,
SQLite cache, actor count, serialization, 768 MiB total payload, twelve 64 MiB
logical batches, and 1 MiB samples are used for both providers. Only the
payload provider changes.

The output terminology was tightened:

- `storage_visible_*` means the actor's offload-directory byte metric accounts
  for the complete payload. It is not an `fsync` or power-loss durability
  claim.
- Mooncake GET is performed only after the master reports memory-replica
  eviction. The OS page cache is not dropped for either provider, so GET is a
  software-path comparison, not physical cold-NVMe bandwidth.
- Mooncake-only configuration fields are reported as null for `local_file`,
  rather than implying that its throughput depends on a DRAM segment or
  transport protocol.

The documentation now shows the explicit Mooncake selector and explains how
to repeat the same benchmark for `local_file` without mislabelling directory
visibility or page-cache reads.

### Environment and exact Mooncake settings

- same 4-vCPU/6-GiB Colima Ubuntu ARM VM and `/tmp` virtual Linux filesystem
  for every run;
- one SimpleStorage actor, public ZMQ data plane, TCP Mooncake transport;
- SQLite cache: 8 MiB per actor;
- Mooncake version: exact `0.3.10.post2`;
- Mooncake global segment: 512 MiB; transfer buffer: 32 MiB; SSD staging
  buffer: 64 MiB; maximum object: 16 MiB minus 64 KiB;
- master lease: 500 ms; high watermark: 0.5; eviction ratio: 0.5; one-second
  offload heartbeat;
- three independent runs per provider; table values are medians and ranges.

### Results

| Public SimpleStorage metric | SQLite index + local file | SQLite index + Mooncake | Mooncake / local file |
| --- | ---: | ---: | ---: |
| foreground PUT | 4.520 Gbps (3.630-6.110) | 2.662 Gbps (2.613-2.693) | 58.9% |
| storage-directory visible pipeline | 4.511 Gbps (3.625-6.096) | 1.915 Gbps (1.769-1.936) | 42.4% |
| 64 MiB GET, OS page cache retained | 7.198 Gbps (6.266-8.029) | 3.798 Gbps (3.257-3.981) | 52.8% |
| PUT p50 | 109.4 ms (77.6-147.3) | 106.6 ms (97.5-116.0) | 97.4% |
| PUT p99 | 203.7 ms (165.3-225.1) | 988.8 ms (917.2-994.4) | 4.86x |

Every Mooncake run grew the directory by 805,353,960 bytes, reported 39-41
new evicted keys, and logged zero memory keys plus five offload keys for the
verified 64 MiB read. Local-file runs grew it by 805,343,904 bytes.

### Interpretation and decision

The direct-file provider is currently faster for sustained single-actor
node-local traffic: Mooncake's median foreground PUT is 41.1% lower (equivalently,
local file is 69.8% higher), and local file's directory-visible pipeline is
2.36x Mooncake's.
The local-file run variance is high in this virtual/page-cache environment, but
even its slowest observed foreground result exceeds Mooncake's fastest.

Mooncake's median PUT p50 is essentially tied with direct file because its
foreground acknowledgement lands in DRAM. Once the 512 MiB segment fills, the
one-second offload cycle and bounded-space retries dominate the tail; that is
why p99 approaches one second and sustained PUT converges toward its roughly
1.9 Gbps background SSD pipeline. This is the main current PUT bottleneck, not
the shared SQLite index.

The providers offer different capabilities rather than separate eligibility
for high throughput:

- `local_file` is the simpler and currently faster node-local payload path. A
  PUT writes and renames the file before committing its SQLite reference; disk
  full/write errors are synchronous. It has no Mooncake master/native runtime,
  while filesystem pages remain reclaimable OS cache.
- `mooncake` adds an explicitly bounded application DRAM tier, automatic
  background SSD replicas/eviction, native network transports, and Mooncake
  backpressure. It also adds a master/native-library/port dependency and, in
  the tested embedded per-actor topology, does not yet turn those capabilities
  into higher single-node throughput.
- Neither provider promises power-loss durability per PUT; recovery still
  relies on TransferQueue checkpointing. An unsupervised Mooncake master is an
  additional availability dependency.

The user-requested SimpleStorage-through-Mooncake path remains the selected
combined implementation, with `local_file` retained as a fully valid
high-throughput fallback and reference. It would be inaccurate to claim that
Mooncake is already the performance winner. Before production acceptance, run
the same A/B on real NVMe, add multi-actor/device scaling, and profile/tune the
offload heartbeat, eviction watermarks, and segment-to-working-set ratio. The
historical SQLite-BLOB payload layout should not be restored: the current
SQLite-index plus pluggable payload boundary preserves SQLite's transactional
strength without putting bulk bytes back through its B-tree.

### Validation for this comparison update

- Both three-run A/B paths completed through the real public ZMQ actor API and
  verified returned values.
- All three Mooncake runs observed master-confirmed eviction; the exact
  `0.3.10.post2` native cold-tier API returned the verified batch.
- The updated benchmark passed Python bytecode compilation, `ruff check`, and
  `ruff format --check` in the Linux test environment. Ruff was installed only
  into the disposable `/tmp/tq-mooncake-venv` test environment; no repository
  dependency changed.
- Mooncake master processes started for the comparison were stopped after the
  runs.

## Local-file PUT optimization follow-up

### Scope and profiling result

The follow-up investigated whether the direct local-file payload provider
could be improved further without weakening its transaction/error behavior or
adding an unproven asynchronous I/O subsystem. All measurements used the same
4-vCPU/6-GiB Colima Linux VM and public SimpleStorage ZMQ path unless a result
is explicitly labelled as a direct-provider microbenchmark.

A direct `DiskStorageUnitData` profile wrote twelve 64 MiB batches (768 MiB
total) in 0.594 seconds, about 10.8 Gbps:

- the twelve `writev` calls used 0.463 seconds (78.0%);
- the twelve SQLite transaction exits/commits used 0.119 seconds (20.0%);
- packed-frame serialization used 0.002 seconds;
- all remaining Python/index/file-management work was negligible.

This rules out another serialization abstraction or routing the original ZMQ
frames directly into the storage format: it would add transport/storage
coupling to remove roughly 0.3% of the direct-provider time. The remaining
provider cost is real sequential filesystem writing plus the index commit.

### Selected change: bound threads once per actor process

`SimpleStorageUnit` already bounded `TQ_NUM_THREADS` to the physical CPU count
inside every PUT/GET/CLEAR context. On the four-core VM, however, the default
value of eight caused that same clamp and warning to run for every request.
The module now resolves the identical effective bound once when each actor
process starts and emits at most one warning per process. Public behavior,
thread cap, storage formats, acknowledgement boundary, and configuration all
remain unchanged.

A controlled interleaved 768 MiB comparison before the change measured:

| Effective setup | Foreground PUT runs | Median | PUT p99 runs | Median p99 |
| --- | --- | ---: | --- | ---: |
| default `TQ_NUM_THREADS=8`, clamped/warned per request | 9.005, 9.205 Gbps | 9.105 Gbps | 160.3, 149.2 ms | 154.8 ms |
| explicit `TQ_NUM_THREADS=4` | 9.398, 9.760 Gbps | 9.579 Gbps | 127.5, 127.9 ms | 127.7 ms |

The one-time bound therefore improved the controlled median by about 5.2% and
reduced median p99 by about 17.5% on this small-core environment. It does not
change hosts whose configured value already fits their physical CPU count.

With the final code and no manual `TQ_NUM_THREADS` override, three 768 MiB
public-path runs measured:

- foreground PUT: 7.807, 9.582, and 9.648 Gbps; median 9.582 Gbps;
- directory-visible pipeline: 7.791, 9.560, and 9.625 Gbps; median 9.560 Gbps;
- PUT p50 median: 45.7 ms; PUT p99 median: 135.1 ms;
- every run wrote and verified the complete 805,343,904-byte directory delta.

The first run remains sensitive to VM/filesystem cold state, so the 9.6 Gbps
steady runs are software-path evidence rather than a production NVMe claim.

### Explored and rejected changes

1. SQLite rollback-journal modes were tested without weakening
   `synchronous=NORMAL`. A fully interleaved direct-provider comparison gave
   medians of 0.379 seconds for `DELETE`, 0.375 seconds for `TRUNCATE`, 0.376
   seconds for `TRUNCATE + EXCLUSIVE`, and 0.377 seconds for `PERSIST +
   EXCLUSIVE`. The roughly 1% spread is too small and unstable to justify a
   storage-mode change. WAL and `synchronous=OFF` remain rejected for the
   previously recorded write-amplification and consistency reasons.
2. Linux `posix_fallocate` before `writev` improved the short direct-provider
   median from 0.389 to 0.354 seconds (about 9%). It did not survive the public
   path test: on six interleaved 3 GiB runs, no-preallocation median PUT was
   8.657 Gbps and preallocation median was 8.742 Gbps, with wide overlapping
   ranges and opposite ordering in individual pairs. The temporary diagnostic
   switch, implementation, and tests were removed; no unproven syscall path or
   configuration remains in the final diff.
3. Increasing logical request size does not monotonically raise throughput.
   One exploratory 3 GiB sweep measured 7.641/8.868/8.290/8.253 Gbps for
   32/64/128/256 MiB batches. Sixty-four MiB is the current VM sweet spot;
   larger requests raise latency and memory footprint without increasing PUT.
4. Parallel per-field writers, an actor-local asynchronous commit queue,
   `O_DIRECT`, and io_uring were not implemented. They change queue depth,
   acknowledgement/backpressure, alignment, cleanup, or crash semantics and
   need real NVMe profiling plus dedicated failure tests. The current profile
   does not justify that complexity in the portable Python fallback.

The practical next throughput step is horizontal: use multiple SimpleStorage
actors spread across storage nodes/devices, retain roughly 64 MiB logical
batches for this data shape, and run the same benchmark on the target NVMe.
Only if a real device remains underutilized should the local provider gain a
bounded asynchronous/direct-I/O path.

### Follow-up validation

- `tests/test_simple_storage_unit.py`: 22 passed, 1 environment-gated Mooncake
  case skipped.
- `ruff check` passed for the affected storage, test, and benchmark files.
- The affected files passed `ruff format --check` after the final formatting
  correction.
- `python -m compileall -q transfer_queue tutorial tests scripts` passed.
- `git diff --check` passed.
- The failed first journal experiment left one exact temporary directory; it
  was identified and deleted before the successful rerun. All benchmark actors
  cleaned their own ephemeral payload directories normally.
