#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team

"""Measure Mooncake's foreground PUT, asynchronous SSD drain, and cold GET."""

import argparse
import ctypes
import json
import os
import re
import time
import urllib.request
from pathlib import Path

_STORAGE_VALUE = re.compile(r"(?P<value>[0-9.]+)\s*(?P<unit>[KMGTPE]?i?B)")


def _parse_storage_used(summary: str, tier: str) -> int:
    match = re.search(rf"{re.escape(tier)} Storage:\s*([^/|]+)", summary)
    if match is None:
        raise ValueError(f"{tier} Storage usage is missing from Mooncake metrics summary")
    value_match = _STORAGE_VALUE.search(match.group(1))
    if value_match is None:
        raise ValueError(f"Cannot parse {tier} Storage usage from: {match.group(1)!r}")

    value = float(value_match.group("value"))
    unit = value_match.group("unit")
    if unit == "B":
        return int(value)
    prefixes = "KMGTPE"
    exponent = prefixes.index(unit[0]) + 1
    base = 1024 if "iB" in unit else 1000
    return int(value * base**exponent)


def _metrics_summary(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _disk_usage(path: Path) -> tuple[int, int]:
    logical = 0
    allocated = 0
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        stat = entry.stat()
        logical += stat.st_size
        allocated += getattr(stat, "st_blocks", 0) * 512
    return logical, allocated


def _replica_state(store: object, keys: list[str]) -> tuple[set[str], list[str]]:
    descriptors = store.batch_get_replica_desc(keys)  # type: ignore[attr-defined]
    local_disk_keys = set()
    disk_only_keys = []
    for key in keys:
        replicas = descriptors.get(key, [])
        has_memory = any(replica.is_memory_replica() for replica in replicas)
        # v0.3.10.post2 does not bind is_local_disk_replica(); LOCAL_DISK is
        # the remaining descriptor variant after MEMORY and legacy DISK.
        has_local_disk = any(not replica.is_memory_replica() and not replica.is_disk_replica() for replica in replicas)
        if has_local_disk:
            local_disk_keys.add(key)
        if has_local_disk and not has_memory:
            disk_only_keys.append(key)
    return local_disk_keys, disk_only_keys


def _wait_for_ssd_replicas(store: object, keys: list[str], timeout: float) -> float:
    start = time.perf_counter()
    while True:
        local_disk_keys, _ = _replica_state(store, keys)
        if len(local_disk_keys) == len(keys):
            return time.perf_counter()
        if time.perf_counter() - start >= timeout:
            raise TimeoutError(
                f"Mooncake reached {len(local_disk_keys)}/{len(keys)} LOCAL_DISK replicas within {timeout}s"
            )
        time.sleep(0.05)


def _upsert_with_backpressure(
    store: object,
    keys: list[str],
    ptrs: list[int],
    sizes: list[int],
    replica_config: object,
    timeout: float,
    retry_interval: float,
) -> None:
    deadline = time.perf_counter() + timeout
    pending = list(range(len(keys)))
    while pending:
        results = store.batch_upsert_from(  # type: ignore[attr-defined]
            [keys[i] for i in pending],
            [ptrs[i] for i in pending],
            [sizes[i] for i in pending],
            config=replica_config,
        )
        next_pending = [pending[i] for i, result in enumerate(results) if result == -200]
        failures = [(keys[pending[i]], result) for i, result in enumerate(results) if result not in (0, -200)]
        if failures:
            raise RuntimeError(f"Mooncake PUT failures: {failures[:4]}")
        if not next_pending:
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"Mooncake did not reclaim DRAM for {len(next_pending)} PUT objects")
        pending = next_pending
        time.sleep(retry_interval)


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    os.environ["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = str(args.offload_buffer_size_bytes)
    os.environ["MOONCAKE_OFFLOAD_USE_URING"] = "1" if args.use_uring else "0"
    os.environ["MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS"] = str(args.heartbeat_interval_seconds)
    os.environ["MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"] = "bucket_storage_backend"
    os.environ["MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT"] = str(args.bucket_keys_limit)
    os.environ["MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES"] = str(args.bucket_size_limit_bytes)

    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    offload_path = Path(args.offload_path).expanduser().resolve()
    offload_path.mkdir(parents=True, exist_ok=True)
    store = MooncakeDistributedStore()
    setup_result = store.setup(
        args.local_hostname,
        args.metadata_server,
        args.global_segment_size_bytes,
        args.local_buffer_size_bytes,
        args.protocol,
        args.rdma_devices,
        args.master_server_address,
        None,
        True,
        str(offload_path),
    )
    if setup_result != 0:
        raise RuntimeError(f"Mooncake setup failed with error code {setup_result}")
    disk_logical_before, disk_allocated_before = _disk_usage(offload_path)

    batch_bytes = args.object_size_bytes * args.batch_size
    payload = ctypes.create_string_buffer(batch_bytes)
    ctypes.memset(ctypes.addressof(payload), 0x5A, batch_bytes)
    if store.register_buffer(ctypes.addressof(payload), batch_bytes) != 0:
        store.close()
        raise RuntimeError("Mooncake failed to register the benchmark buffer")

    keys = [f"tq-offload-bench-{os.getpid()}-{i}" for i in range(args.num_objects)]
    replica_config = ReplicateConfig()
    replica_config.with_hard_pin = False
    total_bytes = args.object_size_bytes * args.num_objects

    try:
        put_start = time.perf_counter()
        for offset in range(0, args.num_objects, args.batch_size):
            batch_keys = keys[offset : offset + args.batch_size]
            sizes = [args.object_size_bytes] * len(batch_keys)
            ptrs = [ctypes.addressof(payload) + i * args.object_size_bytes for i in range(len(batch_keys))]
            _upsert_with_backpressure(
                store,
                batch_keys,
                ptrs,
                sizes,
                replica_config,
                args.drain_timeout_seconds,
                args.backpressure_interval_seconds,
            )
        put_end = time.perf_counter()

        drain_end = _wait_for_ssd_replicas(store, keys, args.drain_timeout_seconds)
        disk_logical_after, disk_allocated_after = _disk_usage(offload_path)

        # Replica-descriptor queries grant a new Mooncake lease. Stop polling
        # after persistence is proven, wait for that lease to expire, then take
        # one snapshot to prove that cold keys no longer have a memory replica.
        eviction_wait_start = time.perf_counter()
        time.sleep(args.lease_ttl_ms / 1000 + args.eviction_grace_seconds)
        _, disk_only_keys = _replica_state(store, keys)
        cold_keys = disk_only_keys[: args.cold_get_objects]
        if len(cold_keys) < args.cold_get_objects:
            raise RuntimeError(
                f"Mooncake exposed only {len(cold_keys)}/{args.cold_get_objects} disk-only replicas "
                "after the configured lease and eviction grace period"
            )
        eviction_wait_end = time.perf_counter()

        ctypes.memset(ctypes.addressof(payload), 0, batch_bytes)
        get_start = time.perf_counter()
        for offset in range(0, len(cold_keys), args.batch_size):
            batch_keys = cold_keys[offset : offset + args.batch_size]
            ptrs = [ctypes.addressof(payload) + i * args.object_size_bytes for i in range(len(batch_keys))]
            sizes = [args.object_size_bytes] * len(batch_keys)
            results = store.batch_get_into(batch_keys, ptrs, sizes)
            failures = [
                (key, result)
                for key, result in zip(batch_keys, results, strict=True)
                if result != args.object_size_bytes
            ]
            if failures:
                raise RuntimeError(f"Mooncake cold batch_get_into failures: {failures[:4]}")
            if any(ctypes.string_at(ptr, 1) != b"Z" for ptr in ptrs):
                raise RuntimeError("Mooncake cold batch_get_into returned corrupt data")
        get_end = time.perf_counter()

        put_seconds = put_end - put_start
        pipeline_seconds = drain_end - put_start
        drain_seconds = drain_end - put_end
        get_seconds = get_end - get_start
        cold_get_bytes = args.object_size_bytes * len(cold_keys)
        return {
            "total_bytes": total_bytes,
            "put_seconds": put_seconds,
            "put_gbps": total_bytes * 8 / put_seconds / 1e9,
            "ssd_pipeline_seconds": pipeline_seconds,
            "ssd_pipeline_gbps": total_bytes * 8 / pipeline_seconds / 1e9,
            "post_put_drain_seconds": drain_seconds,
            "eviction_wait_seconds": eviction_wait_end - eviction_wait_start,
            "cold_get_bytes": cold_get_bytes,
            "cold_get_seconds": get_seconds,
            "cold_get_gbps": cold_get_bytes * 8 / get_seconds / 1e9,
            "final_mem_used_bytes": _parse_storage_used(_metrics_summary(args.metrics_url), "Mem"),
            "local_disk_replica_count": len(keys),
            "disk_only_replica_count": len(cold_keys),
            "disk_logical_delta_bytes": disk_logical_after - disk_logical_before,
            "disk_allocated_delta_bytes": disk_allocated_after - disk_allocated_before,
            "protocol": args.protocol,
            "offload_path": str(offload_path),
        }
    finally:
        for offset in range(0, len(keys), args.batch_size):
            store.batch_remove(keys[offset : offset + args.batch_size], force=True)
        store.unregister_buffer(ctypes.addressof(payload))
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload-path", required=True)
    parser.add_argument("--master-server-address", default="127.0.0.1:50051")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9003/metrics/summary")
    parser.add_argument("--local-hostname", default="127.0.0.1")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--global-segment-size-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--local-buffer-size-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--offload-buffer-size-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--object-size-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--bucket-keys-limit", type=int, default=1)
    parser.add_argument("--bucket-size-limit-bytes", type=int, default=16 * 1024 * 1024 - 16)
    parser.add_argument("--num-objects", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cold-get-objects", type=int, default=32)
    parser.add_argument("--drain-timeout-seconds", type=float, default=120)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=1)
    parser.add_argument("--lease-ttl-ms", type=int, default=5000)
    parser.add_argument("--eviction-grace-seconds", type=float, default=1)
    parser.add_argument("--backpressure-interval-seconds", type=float, default=0.1)
    parser.add_argument("--use-uring", action="store_true")
    args = parser.parse_args()

    if args.cold_get_objects > args.num_objects:
        parser.error("--cold-get-objects cannot exceed --num-objects")
    for name in (
        "global_segment_size_bytes",
        "local_buffer_size_bytes",
        "offload_buffer_size_bytes",
        "object_size_bytes",
        "bucket_keys_limit",
        "bucket_size_limit_bytes",
        "num_objects",
        "batch_size",
        "cold_get_objects",
        "heartbeat_interval_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lease_ttl_ms < 0:
        parser.error("--lease-ttl-ms must be non-negative")
    if args.object_size_bytes > args.bucket_size_limit_bytes:
        parser.error("--object-size-bytes cannot exceed --bucket-size-limit-bytes")
    if args.object_size_bytes > 16 * 1024 * 1024 - 16:
        parser.error("Mooncake post2 cold GET requires object size <= 16 MiB - 16 bytes")
    cold_batch_size = min(args.batch_size, args.cold_get_objects)
    cold_allocation_size = ((args.object_size_bytes + 4095) // 4096 * 4096 + 8192) * cold_batch_size
    if cold_allocation_size > args.offload_buffer_size_bytes:
        parser.error("cold GET batch plus Mooncake alignment overhead exceeds the offload buffer")
    if args.eviction_grace_seconds < 0 or args.backpressure_interval_seconds <= 0:
        parser.error("eviction grace must be non-negative and backpressure interval must be positive")

    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
