#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team

"""Compare SimpleStorage local-file and Mooncake SSD payload providers."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import ray
import torch
import zmq

from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, create_zmq_socket


class SimpleStorageClient:
    """Minimal public-ZMQ client used only by this storage-unit benchmark."""

    def __init__(self, server_info):
        self._context = zmq.Context()
        self._socket = create_zmq_socket(self._context, zmq.DEALER, server_info.ip)
        self._socket.setsockopt(zmq.RCVTIMEO, 120_000)
        self._socket.connect(server_info.to_addr("put_get_socket"))

    def request(self, request_type: ZMQRequestType, body: dict) -> ZMQMessage:
        message = ZMQMessage.create(request_type=request_type, sender_id="simple_ssd_benchmark", body=body)
        self._socket.send_multipart(message.serialize())
        return ZMQMessage.deserialize(self._socket.recv_multipart(copy=False))

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def _master_evicted_keys(metrics_url: str) -> int:
    with urllib.request.urlopen(metrics_url, timeout=5) as response:  # noqa: S310
        summary = response.read().decode("utf-8")
    match = re.search(r"Eviction: Success/Attempts=\d+/\d+, keys=(\d+)", summary)
    if match is None:
        raise ValueError("Mooncake master metrics do not expose the eviction key count")
    return int(match.group(1))


def _expect(response: ZMQMessage, expected: ZMQRequestType) -> None:
    if response.request_type != expected:
        raise RuntimeError(f"SimpleStorage request failed: {response.request_type}: {response.body}")


def _wait_for_disk(client: SimpleStorageClient, baseline: int, payload_bytes: int, timeout: float) -> tuple[int, float]:
    start = time.perf_counter()
    while True:
        response = client.request(ZMQRequestType.GET_METRICS, {})
        _expect(response, ZMQRequestType.METRICS_RESPONSE)
        disk_bytes = int(response.body["offload_disk_bytes"])
        if disk_bytes - baseline >= payload_bytes:
            return disk_bytes, time.perf_counter()
        if time.perf_counter() - start >= timeout:
            raise TimeoutError(
                f"SimpleStorage observed only {disk_bytes - baseline}/{payload_bytes} payload bytes on SSD"
            )
        time.sleep(0.05)


def run(args: argparse.Namespace) -> dict:
    offload_path = Path(args.offload_path).expanduser().resolve()
    offload_path.mkdir(parents=True, exist_ok=True)
    actor_kwargs = {
        "storage_unit_size": args.total_bytes // args.sample_bytes,
        "offload_path": str(offload_path),
        "offload_cache_size_bytes": args.sqlite_cache_size_bytes,
        "offload_backend": args.offload_backend,
    }
    if args.offload_backend == "mooncake":
        actor_kwargs["mooncake_config"] = {
            "local_hostname": args.local_hostname,
            "metadata_server": args.metadata_server,
            "master_server_address": args.master_server_address,
            "global_segment_size": args.global_segment_size_bytes,
            "local_buffer_size": args.local_buffer_size_bytes,
            "protocol": args.protocol,
            "device_name": args.rdma_devices,
            "put_timeout_seconds": args.put_timeout_seconds,
            "retry_interval_seconds": args.backpressure_interval_seconds,
            "offload": {
                "local_buffer_size_bytes": args.offload_buffer_size_bytes,
                "max_object_size_bytes": args.max_object_size_bytes,
                "get_window_size_bytes": args.get_window_size_bytes,
                "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
                "use_uring": args.use_uring,
            },
        }
    actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(
        **actor_kwargs,
    )
    server_info = ray.get(actor.get_zmq_server_info.remote())
    client = SimpleStorageClient(server_info)
    samples_per_batch = args.batch_bytes // args.sample_bytes
    num_batches = args.total_bytes // args.batch_bytes
    put_latencies = []

    try:
        metrics = client.request(ZMQRequestType.GET_METRICS, {})
        _expect(metrics, ZMQRequestType.METRICS_RESPONSE)
        disk_before = int(metrics.body["offload_disk_bytes"])
        evicted_before = _master_evicted_keys(args.metrics_url) if args.offload_backend == "mooncake" else None

        put_start = time.perf_counter()
        for batch_index in range(num_batches):
            indexes = list(range(batch_index * samples_per_batch, (batch_index + 1) * samples_per_batch))
            values = torch.full(
                (samples_per_batch, args.sample_bytes),
                batch_index % 251,
                dtype=torch.uint8,
            )
            batch_start = time.perf_counter()
            response = client.request(
                ZMQRequestType.PUT_DATA,
                {"global_indexes": indexes, "data": {"value": values}},
            )
            put_latencies.append(time.perf_counter() - batch_start)
            _expect(response, ZMQRequestType.PUT_DATA_RESPONSE)
        put_end = time.perf_counter()

        disk_after, drain_end = _wait_for_disk(
            client,
            disk_before,
            args.total_bytes,
            args.drain_timeout_seconds,
        )
        evicted_after = None
        if args.offload_backend == "mooncake":
            time.sleep(args.lease_ttl_ms / 1000 + args.eviction_grace_seconds)
            evicted_after = _master_evicted_keys(args.metrics_url)
            if evicted_after <= evicted_before:
                raise RuntimeError("Mooncake did not report memory-replica eviction; disk-tier GET was not exercised")

        read_indexes = list(range(samples_per_batch))
        get_start = time.perf_counter()
        response = client.request(
            ZMQRequestType.GET_DATA,
            {"global_indexes": read_indexes, "fields": ["value"]},
        )
        get_end = time.perf_counter()
        _expect(response, ZMQRequestType.GET_DATA_RESPONSE)
        read_values = response.body["data"]["value"]
        if len(read_values) != samples_per_batch or any(torch.count_nonzero(value) for value in read_values):
            raise RuntimeError("SimpleStorage GET returned corrupt data")

        put_seconds = put_end - put_start
        pipeline_seconds = drain_end - put_start
        get_seconds = get_end - get_start
        return {
            "path": f"SimpleStorage->SQLite-index->{args.offload_backend}-payload",
            "offload_backend": args.offload_backend,
            "total_bytes": args.total_bytes,
            "batch_bytes": args.batch_bytes,
            "num_batches": num_batches,
            "foreground_put_seconds": put_seconds,
            "foreground_put_gbps": args.total_bytes * 8 / put_seconds / 1e9,
            "put_latency_p50_ms": float(torch.tensor(put_latencies).quantile(0.5) * 1000),
            "put_latency_p99_ms": float(torch.tensor(put_latencies).quantile(0.99) * 1000),
            "storage_visible_seconds": pipeline_seconds,
            "storage_visible_gbps": args.total_bytes * 8 / pipeline_seconds / 1e9,
            "post_put_drain_seconds": drain_end - put_end,
            "get_bytes": args.batch_bytes,
            "get_seconds": get_seconds,
            "get_gbps": args.batch_bytes * 8 / get_seconds / 1e9,
            "read_cache_state": (
                "Mooncake DRAM replica evicted; OS page cache not dropped"
                if args.offload_backend == "mooncake"
                else "local file; OS page cache not dropped"
            ),
            "disk_delta_bytes": disk_after - disk_before,
            "master_evicted_key_delta": (
                evicted_after - evicted_before if evicted_after is not None and evicted_before is not None else None
            ),
            "global_segment_size_bytes": (
                args.global_segment_size_bytes if args.offload_backend == "mooncake" else None
            ),
            "offload_buffer_size_bytes": (
                args.offload_buffer_size_bytes if args.offload_backend == "mooncake" else None
            ),
            "protocol": args.protocol if args.offload_backend == "mooncake" else None,
            "offload_path": str(offload_path),
        }
    finally:
        client.close()
        try:
            ray.get(actor.close.remote(), timeout=30)
        finally:
            ray.kill(actor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload-path", required=True)
    parser.add_argument("--offload-backend", choices=("local_file", "mooncake"), default="mooncake")
    parser.add_argument("--master-server-address", default="127.0.0.1:50051")
    parser.add_argument("--metadata-server", default="P2PHANDSHAKE")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9003/metrics/summary")
    parser.add_argument("--local-hostname", default="127.0.0.1")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--total-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--batch-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--sample-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--global-segment-size-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--local-buffer-size-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--offload-buffer-size-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-object-size-bytes", type=int, default=16 * 1024 * 1024 - 65536)
    parser.add_argument("--get-window-size-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--sqlite-cache-size-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=1)
    parser.add_argument("--put-timeout-seconds", type=float, default=120)
    parser.add_argument("--drain-timeout-seconds", type=float, default=120)
    parser.add_argument("--backpressure-interval-seconds", type=float, default=0.05)
    parser.add_argument("--lease-ttl-ms", type=int, default=5000)
    parser.add_argument("--eviction-grace-seconds", type=float, default=1)
    parser.add_argument("--use-uring", action="store_true")
    args = parser.parse_args()

    positive = (
        "total_bytes",
        "batch_bytes",
        "sample_bytes",
        "global_segment_size_bytes",
        "local_buffer_size_bytes",
        "offload_buffer_size_bytes",
        "max_object_size_bytes",
        "get_window_size_bytes",
        "heartbeat_interval_seconds",
        "put_timeout_seconds",
        "drain_timeout_seconds",
        "backpressure_interval_seconds",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("all size, interval, and timeout arguments must be positive")
    if args.total_bytes % args.batch_bytes or args.batch_bytes % args.sample_bytes:
        parser.error("total bytes must divide into batches, and batches must divide into samples")
    if args.offload_backend == "mooncake" and args.total_bytes <= args.global_segment_size_bytes:
        parser.error("total bytes must exceed the DRAM segment so cold-path eviction is exercised")
    if args.lease_ttl_ms < 0 or args.eviction_grace_seconds < 0:
        parser.error("lease TTL and eviction grace must be non-negative")

    if not ray.is_initialized():
        ray.init()
    try:
        print(json.dumps(run(args), sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
