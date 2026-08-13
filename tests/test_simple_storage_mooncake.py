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
import sys
from types import ModuleType

import pytest

from transfer_queue.storage.simple_storage_mooncake import MooncakeStorageUnitData


class _FakeReplicateConfig:
    with_hard_pin = True


class _FakeMooncakeStore:
    instances = []
    backpressure_once = False

    def __init__(self):
        self.data = {}
        self.setup_args = None
        self.put_calls = 0
        self.closed = False
        self.max_put_batch = 0
        self.max_remove_batch = 0
        self.__class__.instances.append(self)

    def setup(self, *args):
        self.setup_args = args
        return 0

    def register_buffer(self, _pointer, _size):
        return 0

    def unregister_buffer(self, _pointer):
        return 0

    def batch_upsert_from(self, keys, pointers, sizes, config):
        assert config.with_hard_pin is False
        self.put_calls += 1
        self.max_put_batch = max(self.max_put_batch, len(keys))
        if self.backpressure_once and self.put_calls == 1:
            return [-200] * len(keys)
        for key, pointer, size in zip(keys, pointers, sizes, strict=True):
            self.data[key] = ctypes.string_at(pointer, size)
        return [0] * len(keys)

    def batch_get_into(self, keys, pointers, sizes):
        results = []
        for key, pointer, size in zip(keys, pointers, sizes, strict=True):
            payload = self.data[key]
            assert len(payload) == size
            ctypes.memmove(pointer, payload, size)
            results.append(size)
        return results

    def batch_remove(self, keys, force):
        assert force is True
        self.max_remove_batch = max(self.max_remove_batch, len(keys))
        results = []
        for key in keys:
            results.append(0 if self.data.pop(key, None) is not None else -704)
        return results

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mooncake(monkeypatch):
    _FakeMooncakeStore.instances = []
    _FakeMooncakeStore.backpressure_once = False
    package = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = _FakeMooncakeStore
    store_module.ReplicateConfig = _FakeReplicateConfig
    package.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    return _FakeMooncakeStore


def _config() -> dict:
    return {
        "local_hostname": "127.0.0.1",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": "127.0.0.1:50051",
        "global_segment_size": 4096,
        "local_buffer_size": 2048,
        "protocol": "tcp",
        "device_name": "",
        "put_timeout_seconds": 1,
        "retry_interval_seconds": 0.001,
        "offload": {
            "local_buffer_size_bytes": 64 * 1024,
            "max_object_size_bytes": 1024,
            "get_window_size_bytes": 64 * 1024,
            "heartbeat_interval_seconds": 1,
            "use_uring": False,
        },
    }


def test_mooncake_payload_tier_preserves_simple_storage_semantics(fake_mooncake, tmp_path):
    storage = MooncakeStorageUnitData(10, str(tmp_path), "unit", 1024, _config())
    store = fake_mooncake.instances[0]
    try:
        storage.put_data(
            {"value": [{"v": 1}, {"v": 2}], "score": [0.1, 0.2]},
            [1, 2],
        )
        assert store.put_calls == 1
        assert storage.get_data(["value", "score"], [2, 1]) == {
            "value": [{"v": 2}, {"v": 1}],
            "score": [0.2, 0.1],
        }

        storage.put_data({"value": [{"v": 20}]}, [2])
        assert storage.get_data(["value", "score"], [2]) == {
            "value": [{"v": 20}],
            "score": [0.2],
        }
        storage.clear([1])
        assert storage.active_key_count == 1
        with pytest.raises(KeyError):
            storage.get_data(["value"], [1])

        assert store.setup_args[-2] is True
        assert store.setup_args[-1].endswith("unit/mooncake_payloads")
        assert os.environ["MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT"] == "1"
    finally:
        storage.close()

    assert store.closed is True
    assert list(tmp_path.iterdir()) == []


def test_mooncake_payload_tier_retries_memory_backpressure(fake_mooncake, tmp_path):
    fake_mooncake.backpressure_once = True
    storage = MooncakeStorageUnitData(10, str(tmp_path), "retry", 1024, _config())
    try:
        storage.put_data({"value": [1, 2]}, [1, 2])
        assert fake_mooncake.instances[0].put_calls == 2
        assert storage.get_data(["value"], [1, 2])["value"] == [1, 2]
    finally:
        storage.close()


def test_mooncake_payload_tier_chunks_large_fields_and_windows_get(fake_mooncake, tmp_path):
    config = _config()
    config["offload"]["max_object_size_bytes"] = 32
    config["offload"]["get_window_size_bytes"] = 64 * 1024
    storage = MooncakeStorageUnitData(10, str(tmp_path), "chunks", 1024, config)
    store = fake_mooncake.instances[0]
    try:
        storage.put_data({"value": ["x" * 80, "y" * 80]}, [1, 2])
        assert len(store.data) > 1
        assert max(map(len, store.data.values())) <= 32
        assert storage.get_data(["value"], [2, 1])["value"] == ["y" * 80, "x" * 80]
        storage.clear([1, 2])
        assert store.data == {}
    finally:
        storage.close()


def test_mooncake_payload_tier_rejects_post2_oversized_cold_read_slice(fake_mooncake, tmp_path):
    config = _config()
    config["offload"]["local_buffer_size_bytes"] = 128 * 1024 * 1024
    config["offload"]["get_window_size_bytes"] = 128 * 1024 * 1024
    config["offload"]["max_object_size_bytes"] = 16 * 1024 * 1024

    with pytest.raises(ValueError, match="0.3.10.post2"):
        MooncakeStorageUnitData(10, str(tmp_path), "oversized", 1024, config)

    assert list(tmp_path.iterdir()) == []


def test_mooncake_payload_tier_bounds_native_batches_for_large_models(fake_mooncake, tmp_path):
    config = _config()
    config["offload"]["max_object_size_bytes"] = 1
    storage = MooncakeStorageUnitData(10, str(tmp_path), "native-batches", 1024, config)
    store = fake_mooncake.instances[0]
    try:
        storage.put_data({"value": ["x" * 500]}, [1])
        assert len(store.data) > 400
        assert store.max_put_batch == 400
        storage.clear([1])
        assert store.max_remove_batch == 400
        assert store.data == {}
    finally:
        storage.close()
