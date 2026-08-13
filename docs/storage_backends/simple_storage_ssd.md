# SimpleStorage SSD offload through Mooncake Store

SimpleStorage can keep its sample/field indexing and public ZMQ data plane while
Mooncake Store moves batch payloads from bounded DRAM to each storage node's
local SSD/NVMe:

```text
TransferQueue client
  -> SimpleStorageUnit (routing, capacity, partial update, checkpoint)
     -> SQLite metadata index
     -> embedded Mooncake client -> bounded DRAM segment -> local SSD/NVMe
```

Payloads do not pass through the controller. Every SimpleStorage actor embeds
its own Mooncake client, so multiple storage nodes contribute both memory and
local-device bandwidth. This path requires
`mooncake-transfer-engine>=0.3.10.post2`, already declared by TransferQueue.

## Configuration

```yaml
backend:
  storage_backend: SimpleStorage
  SimpleStorage:
    num_data_storage_units: 2
    total_storage_size: null
    offload:
      enabled: true
      backend: mooncake
      file_storage_path: /local_nvme/transfer_queue
      memory_cache_size_bytes: 67108864
      mooncake:
        global_segment_size: 536870912
        local_buffer_size: 33554432
        offload_buffer_size_bytes: 67108864
        max_object_size_bytes: 16711680
        get_window_size_bytes: 67108864
        heartbeat_interval_seconds: 1
        lease_ttl_ms: 500
        eviction_high_watermark_ratio: 0.8
        eviction_ratio: 0.2
        use_uring: false

  MooncakeStore:
    auto_init: false
    metadata_server: P2PHANDSHAKE
    master_server_address: localhost:50051
    local_hostname: ""
    protocol: tcp
    device_name: ""
```

`file_storage_path` must be an absolute directory on a local SSD filesystem and
must be writable at the same path on every eligible Ray node. It need not be a
shared filesystem. The `MooncakeStore` block supplies cluster connection and
transport settings; SimpleStorage's nested `mooncake` block bounds resources
per actor and overrides the payload-tier tuning.

For a shared or production cluster, run Mooncake master under the platform's
service supervision and keep `auto_init: false`. The repository's existing
`auto_init: true` behavior is intended for an isolated single-job environment
and may replace an already-running local `mooncake_master`.

Host-memory capacity must include, per actor, the DRAM segment, registered
transfer buffer, SSD staging buffer, SQLite cache, and one in-flight request.
`total_storage_size` still limits the number of active samples rather than
bytes. Mooncake `NO_AVAILABLE_HANDLE` is treated as backpressure until
`put_timeout_seconds`; a failed PUT is not published as produced metadata.

## Data and failure semantics

An incoming field batch is serialized once and stored under unique Mooncake
keys. The local SQLite index maps `(global_index, field)` to its batch and
position, preserving selected-field reads, partial updates, clear, ordering,
capacity checks, and portable checkpoints.

Mooncake post2 has two constraints that the adapter handles explicitly:

- its default bucket backend retains a partial 256 MiB/500-key bucket instead
  of making the tail durable; SimpleStorage sets one already-batched object per
  bucket so every object is eligible for immediate asynchronous persistence;
- a cold-read slice is bounded by Cachelib's 16 MiB slab and consumes extra
  alignment space; large batches are transparently split below that boundary,
  and cold reads are windowed to the configured staging-buffer capacity.

Mooncake hard pin is disabled. The short lease protects an active PUT/GET long
enough to finish while allowing the master to reclaim memory promptly above
the watermark. Completed PUTs are persisted asynchronously; a successful PUT
is therefore a foreground acknowledgement, not proof that SSD drain is
complete. Checkpoints remain the recovery boundary for node or device loss.

Graceful `tq.close()` removes Mooncake objects, the SQLite index, per-actor SSD
directories, and an auto-started Mooncake master. An ungraceful node failure
can leave an orphan directory, which may be removed after confirming no job is
using the configured root. `tq_storage_offload_disk_bytes` reports files below
the actor's SSD directory.

Set `backend: local_file` to use the dependency-free, direct-file fallback. It
keeps the same SimpleStorage semantics and is useful for isolation and
comparison, but the supported integrated production path is `mooncake`.

## Performance testing

Start a post2-compatible master if `tq.init()` is not starting it for the
benchmark:

```bash
mooncake_master \
  --rpc_address=127.0.0.1 --rpc_port=50051 --metrics_port=9003 \
  --enable_offload=true --default_kv_lease_ttl=500 \
  --eviction_high_watermark_ratio=0.8 --eviction_ratio=0.2
```

Then exercise the public SimpleStorage path with a data volume larger than its
DRAM segment:

```bash
PYTHONPATH=. python scripts/performance_test/simple_storage_mooncake_benchmark.py \
  --offload-backend mooncake \
  --offload-path /local_nvme/transfer_queue_benchmark \
  --global-segment-size-bytes 536870912 \
  --total-bytes 805306368 \
  --batch-bytes 67108864 \
  --lease-ttl-ms 500
```

For an apples-to-apples SimpleStorage comparison, repeat the command on the
same host, filesystem, and data shape with `--offload-backend local_file` and a
different empty offload path. Both providers retain the same SQLite metadata
index; this switch changes only the payload provider.

The JSON result separates foreground PUT bandwidth, PUT p50/p99 latency, time
until the storage directory accounts for the payload, master-confirmed
Mooncake memory eviction, and a verified GET. Directory accounting is not an
`fsync` durability barrier. The benchmark does not drop the OS page cache, so
its GET result compares software paths rather than claiming physical NVMe cold
read bandwidth.
Record CPU count, filesystem, real NVMe model, transport, actor count, and all
buffer sizes when publishing numbers. A VM or page-cache run is a functional
smoke test, not NVMe acceptance.
