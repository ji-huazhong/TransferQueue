# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from transfer_queue.storage.simple_storage_disk import DiskStorageUnitData
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import get_node_ip_address

logger = get_logger(__name__)

_DIRECT_IO_ALIGNMENT = 4096
_GET_CHUNK_OVERHEAD = 2 * _DIRECT_IO_ALIGNMENT
# post2's FileStorage uses Cachelib's 16 MiB slab size for a cold-read slice.
_POST2_MAX_SLICE_SIZE = 16 * 1024 * 1024 - 16
_NATIVE_BATCH_KEYS = 400


def _cold_read_allocation_size(payload_size: int) -> int:
    aligned_size = (payload_size + _DIRECT_IO_ALIGNMENT - 1) // _DIRECT_IO_ALIGNMENT * _DIRECT_IO_ALIGNMENT
    return aligned_size + _GET_CHUNK_OVERHEAD


class MooncakeStorageUnitData(DiskStorageUnitData):
    """SimpleStorage index with batch payloads backed by Mooncake SSD offload."""

    def __init__(
        self,
        storage_size: int | None,
        offload_path: str,
        storage_unit_id: str,
        cache_size_bytes: int,
        mooncake_config: dict[str, Any],
    ):
        self.storage_unit_id = storage_unit_id
        self._store = None
        self._payload_dir: Path | None = None
        self._put_timeout = float(mooncake_config.get("put_timeout_seconds", 60))
        self._retry_interval = float(mooncake_config.get("retry_interval_seconds", 0.1))
        if self._put_timeout <= 0 or self._retry_interval <= 0:
            raise ValueError("SimpleStorage Mooncake timeout and retry interval must be positive")

        super().__init__(storage_size, offload_path, storage_unit_id, cache_size_bytes)
        self._payload_dir = self._unit_dir / "mooncake_payloads"
        self._payload_dir.mkdir()

        offload_config = mooncake_config.get("offload", {})
        offload_buffer_size = int(offload_config.get("local_buffer_size_bytes", 64 * 1024 * 1024))
        heartbeat_interval = int(offload_config.get("heartbeat_interval_seconds", 1))
        safe_default_object_size = min(
            _POST2_MAX_SLICE_SIZE,
            max(0, offload_buffer_size - _GET_CHUNK_OVERHEAD),
        )
        self._max_object_size = int(offload_config.get("max_object_size_bytes", safe_default_object_size))
        self._get_window_size = int(offload_config.get("get_window_size_bytes", offload_buffer_size))
        if (
            offload_buffer_size <= 0
            or heartbeat_interval <= 0
            or self._max_object_size <= 0
            or self._get_window_size <= 0
        ):
            self.close()
            raise ValueError("SimpleStorage Mooncake offload sizes and heartbeat interval must be positive")
        if self._get_window_size > offload_buffer_size:
            self.close()
            raise ValueError(
                "SimpleStorage Mooncake get_window_size_bytes must not exceed offload.local_buffer_size_bytes"
            )
        if self._max_object_size > _POST2_MAX_SLICE_SIZE:
            self.close()
            raise ValueError(
                f"SimpleStorage Mooncake max_object_size_bytes must be <= {_POST2_MAX_SLICE_SIZE} "
                "for mooncake-transfer-engine 0.3.10.post2"
            )
        if _cold_read_allocation_size(self._max_object_size) > self._get_window_size:
            self.close()
            raise ValueError(
                "SimpleStorage Mooncake max_object_size_bytes plus Mooncake's cold-read alignment "
                "overhead must fit offload.get_window_size_bytes"
            )

        # Ray gives each SimpleStorage actor its own process, so these
        # process-global Mooncake controls remain isolated per storage unit.
        os.environ["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = str(offload_buffer_size)
        os.environ["MOONCAKE_OFFLOAD_USE_URING"] = "1" if offload_config.get("use_uring", False) else "0"
        os.environ["MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS"] = str(heartbeat_interval)
        # Mooncake post2's default bucket backend holds an incomplete 256 MiB /
        # 500-key bucket in memory indefinitely. SimpleStorage already batches
        # samples into large objects, so one object per bucket is both immediate
        # and avoids the small-file-per-sample failure mode.
        os.environ["MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"] = "bucket_storage_backend"
        os.environ["MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT"] = "1"
        os.environ["MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES"] = str(self._max_object_size)

        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            self.close()
            raise ImportError("SimpleStorage Mooncake offload requires mooncake-transfer-engine>=0.3.10.post2") from e

        local_hostname = str(mooncake_config.get("local_hostname", "")).strip() or get_node_ip_address()
        metadata_server = self._normalize_metadata_server(mooncake_config.get("metadata_server"))
        master_server_address = mooncake_config.get("master_server_address")
        if not isinstance(master_server_address, str) or not master_server_address:
            self.close()
            raise ValueError("SimpleStorage Mooncake offload requires master_server_address")

        self._replica_config = ReplicateConfig()
        self._replica_config.with_hard_pin = False
        self._store = MooncakeDistributedStore()
        setup_result = self._store.setup(
            local_hostname,
            metadata_server,
            int(mooncake_config.get("global_segment_size", 512 * 1024 * 1024)),
            int(mooncake_config.get("local_buffer_size", 32 * 1024 * 1024)),
            str(mooncake_config.get("protocol", "tcp")),
            str(mooncake_config.get("device_name", "") or ""),
            master_server_address,
            None,
            True,
            str(self._payload_dir),
        )
        if setup_result != 0:
            self.close()
            raise RuntimeError(f"SimpleStorage Mooncake setup failed with error code: {setup_result}")

    @staticmethod
    def _normalize_metadata_server(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("SimpleStorage Mooncake offload requires metadata_server")
        value = value.strip()
        if value.upper() == "P2PHANDSHAKE":
            return "P2PHANDSHAKE"
        if not value.startswith(("http://", "etcd://")):
            value = f"http://{value}"
        if not value.startswith("etcd://") and not value.endswith("/metadata"):
            value += "/metadata"
        return value

    @property
    def disk_usage_bytes(self) -> int:
        """Physical bytes currently visible in this storage unit directory."""
        total = 0
        for path in self._unit_dir.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # Mooncake may rotate a bucket while metrics are sampled.
                continue
        return total

    @staticmethod
    def _pack_regions(
        packed_batches: list[list[bytes | bytearray | memoryview]],
    ) -> tuple[bytearray, list[int], list[int]]:
        sizes = [sum(memoryview(buffer).nbytes for buffer in buffers) for buffers in packed_batches]
        offsets = []
        region = bytearray(sum(sizes))
        batch_offset = 0
        for buffers, size in zip(packed_batches, sizes, strict=True):
            offsets.append(batch_offset)
            write_offset = batch_offset
            for buffer in buffers:
                view = memoryview(buffer).cast("B")
                region[write_offset : write_offset + view.nbytes] = view
                write_offset += view.nbytes
            assert write_offset == batch_offset + size
            batch_offset = write_offset
        return region, offsets, sizes

    @staticmethod
    def _region_address(region: bytearray) -> int:
        return ctypes.addressof(ctypes.c_ubyte.from_buffer(region))

    @staticmethod
    def _payload_ref(base_key: str, chunk_size: int, chunk_count: int) -> str:
        return f"mc1|{chunk_size}|{chunk_count}|{base_key}"

    @staticmethod
    def _parse_payload_ref(payload_ref: str) -> tuple[str, int, int]:
        try:
            version, chunk_size, chunk_count, base_key = payload_ref.split("|", 3)
            if version != "mc1":
                raise ValueError
            chunk_size_int = int(chunk_size)
            chunk_count_int = int(chunk_count)
            if chunk_size_int <= 0 or chunk_count_int <= 0 or not base_key:
                raise ValueError
        except ValueError as e:
            raise ValueError(f"Invalid SimpleStorage Mooncake payload reference: {payload_ref}") from e
        return base_key, chunk_size_int, chunk_count_int

    @staticmethod
    def _chunk_key(base_key: str, chunk_index: int) -> str:
        return f"{base_key}/{chunk_index:08x}"

    def _expand_payload_ref(
        self, payload_ref: str, payload_size: int | None = None
    ) -> list[tuple[str, int | None, int]]:
        base_key, chunk_size, chunk_count = self._parse_payload_ref(payload_ref)
        chunks = []
        for chunk_index in range(chunk_count):
            size = None
            if payload_size is not None:
                offset = chunk_index * chunk_size
                size = min(chunk_size, payload_size - offset)
                if size <= 0:
                    raise ValueError(f"Invalid size for SimpleStorage Mooncake payload reference: {payload_ref}")
            chunks.append((self._chunk_key(base_key, chunk_index), size, chunk_index * chunk_size))
        return chunks

    def _write_batches(self, packed_batches: list[list[bytes | bytearray | memoryview]]) -> list[tuple[str, int]]:
        if not packed_batches:
            return []
        assert self._store is not None
        region, offsets, sizes = self._pack_regions(packed_batches)
        base_pointer = self._region_address(region)
        payload_refs = []
        keys = []
        pointers = []
        chunk_sizes = []
        for offset, size in zip(offsets, sizes, strict=True):
            base_key = f"tq-simple/{self.storage_unit_id}/{uuid4().hex}"
            chunk_count = max(1, (size + self._max_object_size - 1) // self._max_object_size)
            payload_refs.append(self._payload_ref(base_key, self._max_object_size, chunk_count))
            for chunk_key, chunk_size, chunk_offset in self._expand_payload_ref(payload_refs[-1], size):
                assert chunk_size is not None
                keys.append(chunk_key)
                pointers.append(base_pointer + offset + chunk_offset)
                chunk_sizes.append(chunk_size)
        if self._store.register_buffer(base_pointer, len(region)) != 0:
            raise RuntimeError("SimpleStorage Mooncake failed to register PUT buffer")

        successful: set[int] = set()
        try:
            pending = list(range(len(keys)))
            deadline = time.perf_counter() + self._put_timeout
            while pending:
                batch = pending[:_NATIVE_BATCH_KEYS]
                results = self._store.batch_upsert_from(
                    [keys[index] for index in batch],
                    [pointers[index] for index in batch],
                    [chunk_sizes[index] for index in batch],
                    config=self._replica_config,
                )
                if len(results) != len(batch):
                    raise RuntimeError("SimpleStorage Mooncake returned an invalid PUT result count")
                failed = []
                for index, result in zip(batch, results, strict=True):
                    if result == 0:
                        successful.add(index)
                    elif result == -200:
                        failed.append(index)
                    else:
                        raise RuntimeError(f"SimpleStorage Mooncake PUT failed for {keys[index]}: {result}")
                if failed and time.perf_counter() >= deadline:
                    raise TimeoutError(f"SimpleStorage Mooncake PUT remained backpressured for {self._put_timeout}s")
                pending = failed + pending[len(batch) :]
                if failed:
                    time.sleep(self._retry_interval)
        except Exception:
            if successful:
                self._store.batch_remove([keys[index] for index in successful], force=True)
            raise
        finally:
            self._store.unregister_buffer(base_pointer)
        return list(zip(payload_refs, sizes, strict=True))

    def _write_batch_buffers(self, buffers: list[bytes | bytearray | memoryview]) -> tuple[str, int]:
        return self._write_batches([buffers])[0]

    def _read_batches(self, batch_refs: list[tuple[str, int]]) -> list[memoryview]:
        if not batch_refs:
            return []
        assert self._store is not None
        sizes = [size for _, size in batch_refs]
        payload_offsets = []
        current_offset = 0
        for size in sizes:
            payload_offsets.append(current_offset)
            current_offset += size
        region = bytearray(current_offset)
        base_pointer = self._region_address(region)
        if self._store.register_buffer(base_pointer, len(region)) != 0:
            raise RuntimeError("SimpleStorage Mooncake failed to register GET buffer")
        try:
            chunks = []
            for (payload_ref, payload_size), payload_offset in zip(batch_refs, payload_offsets, strict=True):
                for chunk_key, chunk_size, chunk_offset in self._expand_payload_ref(payload_ref, payload_size):
                    assert chunk_size is not None
                    chunks.append((chunk_key, base_pointer + payload_offset + chunk_offset, chunk_size))

            failures = []
            window = []
            window_size = 0
            for chunk in chunks:
                allocation_size = _cold_read_allocation_size(chunk[2])
                if window and window_size + allocation_size > self._get_window_size:
                    failures.extend(self._get_chunk_window(window))
                    window = []
                    window_size = 0
                window.append(chunk)
                window_size += allocation_size
            if window:
                failures.extend(self._get_chunk_window(window))
        finally:
            self._store.unregister_buffer(base_pointer)
        if failures:
            raise RuntimeError(f"SimpleStorage Mooncake GET failures: {failures[:4]}")
        return [memoryview(region)[offset : offset + size] for offset, size in zip(payload_offsets, sizes, strict=True)]

    def _get_chunk_window(self, chunks: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        assert self._store is not None
        results = self._store.batch_get_into(
            [key for key, _, _ in chunks],
            [pointer for _, pointer, _ in chunks],
            [size for _, _, size in chunks],
        )
        if len(results) != len(chunks):
            raise RuntimeError("SimpleStorage Mooncake returned an invalid GET result count")
        return [(key, result, size) for (key, _, size), result in zip(chunks, results, strict=True) if result != size]

    def _read_batch_file(self, file_name: str, payload_size: int) -> bytearray:
        return bytearray(self._read_batches([(file_name, payload_size)])[0])

    def _remove_batch_files(self, file_names: set[str]) -> None:
        if not file_names or self._store is None:
            return
        keys = [chunk_key for payload_ref in file_names for chunk_key, _, _ in self._expand_payload_ref(payload_ref)]
        for start in range(0, len(keys), _NATIVE_BATCH_KEYS):
            batch = keys[start : start + _NATIVE_BATCH_KEYS]
            results = self._store.batch_remove(batch, force=True)
            for key, result in zip(batch, results, strict=True):
                if result not in (0, -704):
                    logger.warning(f"Failed to remove SimpleStorage Mooncake payload {key}: {result}")

    def close(self) -> None:
        """Close Mooncake, SQLite, and this actor's ephemeral SSD directory."""
        if getattr(self, "_closed", False):
            return
        store = getattr(self, "_store", None)
        connection = getattr(self, "_connection", None)
        if store is not None:
            if connection is not None:
                try:
                    keys = {row[0] for row in connection.execute("SELECT file_name FROM field_batches")}
                    self._remove_batch_files(keys)
                except Exception as e:
                    logger.warning(f"Failed to clear SimpleStorage Mooncake payloads during close: {e}")
            try:
                store.close()
            except Exception as e:
                logger.warning(f"Failed to close SimpleStorage Mooncake client: {e}")
            self._store = None

        payload_dir = getattr(self, "_payload_dir", None)
        super().close()
        if payload_dir is not None:
            shutil.rmtree(payload_dir, ignore_errors=True)
        unit_dir = getattr(self, "_unit_dir", None)
        if unit_dir is not None:
            try:
                unit_dir.rmdir()
            except OSError:
                pass
