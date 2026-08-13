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

import os
import pickle
import sqlite3
import struct
from operator import index as integer_index
from pathlib import Path
from typing import Any
from uuid import uuid4

from tensordict import NonTensorStack

from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import decode, encode, unpack_from

logger = get_logger(__name__)

_OFFLOAD_CHECKPOINT_FORMAT = "tq_simple_storage_batches_v2"
_SQL_BATCH_SIZE = 128
_SQL_QUERY_KEYS = 500
_WRITEV_MAX_BUFFERS = 1024


def _packed_storage_buffers(value: Any) -> list[memoryview]:
    """Build the existing packed-frame header without copying frame payloads."""
    frames = encode(value)
    frame_views = [memoryview(frame).cast("B") for frame in frames]
    header_size = 4 + len(frame_views) * 8
    header = bytearray(header_size)
    struct.pack_into("<I", header, 0, len(frame_views))
    payload_offset = header_size
    for index, frame in enumerate(frame_views):
        struct.pack_into("<II", header, 4 + index * 8, payload_offset, frame.nbytes)
        payload_offset += frame.nbytes
    return [memoryview(header), *frame_views]


def _deserialize_storage_value(payload: bytes | bytearray | memoryview) -> Any:
    writable_payload = payload if isinstance(payload, bytearray) else bytearray(payload)
    return decode(unpack_from(writable_payload))


def _storage_batch_items(payload: bytes | bytearray | memoryview) -> Any:
    values = _deserialize_storage_value(payload)
    if isinstance(values, NonTensorStack):
        return values.tolist()
    return values.unbind() if getattr(values, "is_nested", False) else values


class DiskStorageUnitData:
    """SQLite-indexed 2D storage for bounded-host-memory SimpleStorage.

    Each incoming field batch is stored once. Compact per-sample references keep
    selected-field reads and partial updates from touching unrelated payloads.
    SQLite atomically publishes new file references before old files are removed.
    """

    def __init__(
        self,
        storage_size: int | None,
        offload_path: str,
        storage_unit_id: str,
        cache_size_bytes: int,
    ):
        root = Path(offload_path).expanduser()
        if not root.is_absolute():
            raise ValueError(f"SimpleStorage offload path must be absolute, got: {offload_path}")
        if cache_size_bytes < 0:
            raise ValueError(f"SimpleStorage offload cache_size_bytes must be >= 0, got: {cache_size_bytes}")

        root.mkdir(parents=True, exist_ok=True)
        self._unit_dir = root / storage_unit_id
        self._unit_dir.mkdir()
        self._database_path = self._unit_dir / "data.sqlite3"
        self.storage_size = storage_size
        self._active_key_count = 0
        self._closed = False

        try:
            self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
            # One worker serializes all requests, so WAL adds checkpoint write
            # amplification without providing useful read/write concurrency.
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            cache_size_kib = 0 if cache_size_bytes == 0 else max(1, (cache_size_bytes + 1023) // 1024)
            self._connection.execute(f"PRAGMA cache_size={-cache_size_kib}")
            self._connection.execute("CREATE TABLE samples (global_index INTEGER PRIMARY KEY)")
            self._connection.execute(
                """
                CREATE TABLE field_batches (
                    batch_id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL UNIQUE,
                    payload_size INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE field_values (
                    global_index INTEGER NOT NULL REFERENCES samples(global_index) ON DELETE CASCADE,
                    field TEXT NOT NULL,
                    batch_id INTEGER NOT NULL REFERENCES field_batches(batch_id),
                    position INTEGER NOT NULL,
                    PRIMARY KEY (global_index, field)
                ) WITHOUT ROWID
                """
            )
            self._connection.execute("CREATE INDEX field_values_batch_id ON field_values(batch_id)")
            self._connection.commit()
        except Exception:
            self.close()
            raise

    @property
    def active_key_count(self) -> int:
        """Number of active sample keys in the SQLite index."""
        return self._active_key_count

    @property
    def disk_usage_bytes(self) -> int:
        """Current database, auxiliary-file, and batch-payload size."""
        return sum(path.stat().st_size for path in self._unit_dir.iterdir() if path.is_file())

    @staticmethod
    def _write_all(fd: int, buffers: list[bytes | bytearray | memoryview]) -> int:
        pending = []
        for buffer in buffers:
            view = memoryview(buffer).cast("B")
            if view.nbytes:
                pending.append(view)
        total_size = sum(buffer.nbytes for buffer in pending)
        while pending:
            written = os.writev(fd, pending[:_WRITEV_MAX_BUFFERS])
            if written <= 0:
                raise OSError("Failed to write SimpleStorage SSD batch payload")
            consumed = 0
            while consumed < len(pending) and written >= pending[consumed].nbytes:
                written -= pending[consumed].nbytes
                consumed += 1
            if consumed:
                del pending[:consumed]
            if written:
                pending[0] = pending[0][written:]
        return total_size

    def _write_batch_buffers(self, buffers: list[bytes | bytearray | memoryview]) -> tuple[str, int]:
        token = uuid4().hex
        file_name = f"{token}.batch"
        temporary_path = self._unit_dir / f".{token}.tmp"
        final_path = self._unit_dir / file_name
        try:
            fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                payload_size = self._write_all(fd, buffers)
            finally:
                os.close(fd)
            os.replace(temporary_path, final_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise
        return file_name, payload_size

    def _write_batches(self, packed_batches: list[list[bytes | bytearray | memoryview]]) -> list[tuple[str, int]]:
        """Write field payloads and return their storage keys and sizes."""
        written = []
        try:
            for buffers in packed_batches:
                written.append(self._write_batch_buffers(buffers))
        except Exception:
            self._remove_batch_files({file_name for file_name, _ in written})
            raise
        return written

    def _read_batch_file(self, file_name: str, payload_size: int) -> bytearray:
        payload = bytearray(payload_size)
        view = memoryview(payload)
        with open(self._unit_dir / file_name, "rb", buffering=0) as batch_file:
            while view:
                bytes_read = batch_file.readinto(view)
                if not bytes_read:
                    raise EOFError(f"SimpleStorage SSD batch file is truncated: {file_name}")
                view = view[bytes_read:]
        return payload

    def _read_batches(self, batch_refs: list[tuple[str, int]]) -> list[bytes | bytearray | memoryview]:
        """Read field payloads in the requested order."""
        return [self._read_batch_file(file_name, payload_size) for file_name, payload_size in batch_refs]

    def _remove_batch_files(self, file_names: set[str]) -> None:
        for file_name in file_names:
            try:
                (self._unit_dir / file_name).unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Failed to remove unused SimpleStorage SSD batch file {file_name}: {e}")

    @staticmethod
    def _normalize_indexes(global_indexes: list) -> list[int]:
        indexes = []
        for global_index in global_indexes:
            try:
                indexes.append(integer_index(global_index))
            except TypeError as e:
                raise TypeError(f"SimpleStorage global index must be an integer, got: {global_index!r}") from e
        return indexes

    @staticmethod
    def _chunks(values: list, size: int):
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def _existing_keys(self, indexes: list[int]) -> set[int]:
        existing = set()
        for chunk in self._chunks(list(dict.fromkeys(indexes)), _SQL_QUERY_KEYS):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT global_index FROM samples WHERE global_index IN ({placeholders})", chunk
            )
            existing.update(row[0] for row in rows)
        return existing

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Read only the requested fields, preserving the caller's key order."""
        indexes = self._normalize_indexes(global_indexes)
        positions: dict[int, list[int]] = {}
        for position, global_index in enumerate(indexes):
            positions.setdefault(global_index, []).append(position)

        missing = object()
        result = {field: [missing] * len(indexes) for field in fields}
        unique_indexes = list(positions)
        references: dict[int, list[tuple[str, int, int]]] = {}
        for field in fields:
            for chunk in self._chunks(unique_indexes, _SQL_QUERY_KEYS):
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT global_index, batch_id, position FROM field_values "
                    f"WHERE field = ? AND global_index IN ({placeholders})",
                    [field, *chunk],
                )
                for global_index, batch_id, batch_position in rows:
                    references.setdefault(batch_id, []).append((field, global_index, batch_position))

        batch_rows = []
        for chunk in self._chunks(list(references), _SQL_QUERY_KEYS):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT batch_id, file_name, payload_size FROM field_batches WHERE batch_id IN ({placeholders})",
                chunk,
            )
            batch_rows.extend(rows)

        payloads = self._read_batches([(file_name, payload_size) for _, file_name, payload_size in batch_rows])
        for (batch_id, _, _), payload in zip(batch_rows, payloads, strict=True):
            batch_values = _storage_batch_items(payload)
            for field, global_index, batch_position in references[batch_id]:
                value = batch_values[batch_position]
                for result_position in positions[global_index]:
                    result[field][result_position] = value

        for field in fields:
            for position, value in enumerate(result[field]):
                if value is missing:
                    raise KeyError(f"StorageUnitData get_data: key {indexes[position]} not found in field '{field}'")
        return result

    def put_data(self, field_data: dict[str, Any], global_indexes: list) -> None:
        """Atomically write batch files and their SQLite references."""
        indexes = self._normalize_indexes(global_indexes)
        for field, values in field_data.items():
            if len(values) != len(indexes):
                raise ValueError(
                    f"StorageUnitData put_data: field '{field}' values length {len(values)} "
                    f"!= global_indexes length {len(indexes)}, length mismatch"
                )

        unique_indexes = list(dict.fromkeys(indexes))
        existing = self._existing_keys(unique_indexes)
        new_key_count = len(unique_indexes) - len(existing)
        if self.storage_size is not None and self._active_key_count + new_key_count > self.storage_size:
            raise ValueError(
                f"Storage capacity exceeded: {self._active_key_count} existing + "
                f"{new_key_count} new > {self.storage_size}"
            )

        created_files: set[str] = set()
        obsolete_files: set[str] = set()
        try:
            with self._connection:
                self._connection.executemany(
                    "INSERT OR IGNORE INTO samples(global_index) VALUES (?)",
                    ((global_index,) for global_index in unique_indexes),
                )
                if not indexes:
                    return
                field_batches = [
                    (
                        field,
                        self._batch_ids_for_field(field, unique_indexes),
                        _packed_storage_buffers(values.tolist() if isinstance(values, NonTensorStack) else values),
                    )
                    for field, values in field_data.items()
                ]
                written_batches = self._write_batches([buffers for _, _, buffers in field_batches])
                for (field, old_batch_ids, _), (file_name, payload_size) in zip(
                    field_batches, written_batches, strict=True
                ):
                    created_files.add(file_name)
                    cursor = self._connection.execute(
                        "INSERT INTO field_batches(file_name, payload_size) VALUES (?, ?)",
                        (file_name, payload_size),
                    )
                    batch_id = cursor.lastrowid
                    records = [
                        (global_index, field, batch_id, position) for position, global_index in enumerate(indexes)
                    ]
                    self._upsert_values(records)
                    obsolete_files.update(self._delete_unreferenced_batches(old_batch_ids))
        except Exception:
            self._remove_batch_files(created_files)
            raise
        self._remove_batch_files(obsolete_files)
        self._active_key_count += new_key_count

    def _batch_ids_for_field(self, field: str, indexes: list[int]) -> set[int]:
        batch_ids = set()
        for chunk in self._chunks(indexes, _SQL_QUERY_KEYS):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT DISTINCT batch_id FROM field_values WHERE field = ? AND global_index IN ({placeholders})",
                [field, *chunk],
            )
            batch_ids.update(row[0] for row in rows)
        return batch_ids

    def _upsert_values(self, records: list[tuple[int, str, int, int]]) -> None:
        self._connection.executemany(
            """
            INSERT INTO field_values(global_index, field, batch_id, position) VALUES (?, ?, ?, ?)
            ON CONFLICT(global_index, field) DO UPDATE SET
                batch_id=excluded.batch_id,
                position=excluded.position
            """,
            records,
        )

    def _delete_unreferenced_batches(self, batch_ids: set[int]) -> set[str]:
        file_names = set()
        for chunk in self._chunks(list(batch_ids), _SQL_QUERY_KEYS):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT file_name FROM field_batches WHERE batch_id IN ({placeholders}) "
                "AND NOT EXISTS (SELECT 1 FROM field_values WHERE field_values.batch_id = field_batches.batch_id)",
                chunk,
            )
            file_names.update(row[0] for row in rows)
            self._connection.execute(
                f"DELETE FROM field_batches WHERE batch_id IN ({placeholders}) "
                "AND NOT EXISTS (SELECT 1 FROM field_values WHERE field_values.batch_id = field_batches.batch_id)",
                chunk,
            )
        return file_names

    def clear(self, keys: list[int]) -> None:
        """Atomically delete keys and all their field values."""
        indexes = list(dict.fromkeys(self._normalize_indexes(keys)))
        existing_count = len(self._existing_keys(indexes))
        obsolete_files: set[str] = set()
        with self._connection:
            old_batch_ids = set()
            for chunk in self._chunks(indexes, _SQL_QUERY_KEYS):
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT DISTINCT batch_id FROM field_values WHERE global_index IN ({placeholders})", chunk
                )
                old_batch_ids.update(row[0] for row in rows)
                self._connection.execute(f"DELETE FROM samples WHERE global_index IN ({placeholders})", chunk)
            obsolete_files.update(self._delete_unreferenced_batches(old_batch_ids))
        self._remove_batch_files(obsolete_files)
        self._active_key_count -= existing_count

    def save_checkpoint(self, path: str, storage_unit_id: str) -> None:
        """Stream the SQLite rows to a portable, bounded-memory checkpoint."""
        header = {
            "format": _OFFLOAD_CHECKPOINT_FORMAT,
            "storage_unit_id": storage_unit_id,
            "storage_unit_size": self.storage_size,
        }
        with open(path, "wb") as f:
            pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)

            cursor = self._connection.execute("SELECT global_index FROM samples ORDER BY global_index")
            while rows := cursor.fetchmany(_SQL_BATCH_SIZE):
                pickle.dump(("samples", [row[0] for row in rows]), f, protocol=pickle.HIGHEST_PROTOCOL)

            cursor = self._connection.execute(
                "SELECT batch_id, file_name, payload_size FROM field_batches ORDER BY batch_id"
            )
            while row := cursor.fetchone():
                batch_id, file_name, payload_size = row
                payload = self._read_batch_file(file_name, payload_size)
                pickle.dump(("batches", [(batch_id, payload)]), f, protocol=pickle.HIGHEST_PROTOCOL)

            cursor = self._connection.execute(
                "SELECT global_index, field, batch_id, position FROM field_values ORDER BY global_index, field"
            )
            while rows := cursor.fetchmany(_SQL_BATCH_SIZE):
                pickle.dump(("values", rows), f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(("end", None), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path: str) -> tuple[int | None, int, int]:
        """Atomically replace SQLite data from SSD or legacy memory checkpoints."""
        old_files = {row[0] for row in self._connection.execute("SELECT file_name FROM field_batches")}
        created_files: set[str] = set()
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
                with self._connection:
                    self._connection.execute("DELETE FROM field_values")
                    self._connection.execute("DELETE FROM field_batches")
                    self._connection.execute("DELETE FROM samples")
                    if state.get("format") == _OFFLOAD_CHECKPOINT_FORMAT:
                        self._load_streamed_checkpoint(f, created_files)
                    else:
                        self._load_legacy_checkpoint(state, created_files)
        except Exception:
            self._remove_batch_files(created_files)
            raise
        self._remove_batch_files(old_files)

        self._active_key_count = self._connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        field_count = self._connection.execute("SELECT COUNT(DISTINCT field) FROM field_values").fetchone()[0]
        return state["storage_unit_size"], self._active_key_count, field_count

    def _load_streamed_checkpoint(self, checkpoint_file, created_files: set[str]) -> None:
        checkpoint_complete = False
        while True:
            try:
                record_type, records = pickle.load(checkpoint_file)
            except EOFError:
                break
            if record_type == "samples":
                self._connection.executemany(
                    "INSERT INTO samples(global_index) VALUES (?)", ((global_index,) for global_index in records)
                )
            elif record_type == "batches":
                for batch_id, payload in records:
                    file_name, payload_size = self._write_batch_buffers([payload])
                    created_files.add(file_name)
                    self._connection.execute(
                        "INSERT INTO field_batches(batch_id, file_name, payload_size) VALUES (?, ?, ?)",
                        (batch_id, file_name, payload_size),
                    )
            elif record_type == "values":
                self._connection.executemany(
                    "INSERT INTO field_values(global_index, field, batch_id, position) VALUES (?, ?, ?, ?)", records
                )
            elif record_type == "end":
                checkpoint_complete = True
                break
            else:
                raise ValueError(f"Unknown checkpoint record type: {record_type}")
        if not checkpoint_complete:
            raise ValueError("Incomplete SimpleStorage SSD checkpoint")

    def _load_legacy_checkpoint(self, state: dict[str, Any], created_files: set[str]) -> None:
        self._connection.executemany(
            "INSERT INTO samples(global_index) VALUES (?)",
            ((int(global_index),) for global_index in state["active_keys"]),
        )
        for field, values in state["field_data"].items():
            if not values:
                continue
            indexes = [int(global_index) for global_index in values]
            file_name, payload_size = self._write_batch_buffers(_packed_storage_buffers(list(values.values())))
            created_files.add(file_name)
            cursor = self._connection.execute(
                "INSERT INTO field_batches(file_name, payload_size) VALUES (?, ?)",
                (file_name, payload_size),
            )
            self._upsert_values(
                [(global_index, field, cursor.lastrowid, position) for position, global_index in enumerate(indexes)]
            )

    def close(self) -> None:
        """Close SQLite and remove this storage unit's ephemeral working files."""
        if self._closed:
            return
        self._closed = True
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

        unit_dir = getattr(self, "_unit_dir", None)
        if unit_dir is not None:
            for path in unit_dir.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
            try:
                unit_dir.rmdir()
            except OSError:
                pass
