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
import time

import pytest
import ray
import tensordict
import torch
import zmq

from transfer_queue.storage.simple_storage import DiskStorageUnitData, SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, create_zmq_socket


class MockStorageClient:
    """Mock client for testing storage unit operations."""

    def __init__(self, storage_put_get_address, storage_ip):
        self.context = zmq.Context()
        self.socket = create_zmq_socket(self.context, zmq.DEALER, storage_ip)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.connect(storage_put_get_address)

    def send_put(self, client_id, global_indexes, field_data, data_parser=None):
        body = {"global_indexes": global_indexes, "data": field_data}
        if data_parser is not None:
            body["data_parser"] = data_parser
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=f"mock_client_{client_id}",
            body=body,
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def send_get(self, client_id, global_indexes, fields):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes, "fields": fields},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def send_clear(self, client_id, global_indexes):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.CLEAR_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def send_get_metrics(self, client_id):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_METRICS,
            sender_id=f"mock_client_{client_id}",
            body={},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def send_save_checkpoint(self, client_id, path):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT,
            sender_id=f"mock_client_{client_id}",
            body={"path": path},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def send_load_checkpoint(self, client_id, path):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT,
            sender_id=f"mock_client_{client_id}",
            body={"path": path},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart(copy=False))

    def close(self):
        self.socket.close()
        self.context.term()


@pytest.fixture(scope="session")
def ray_setup():
    """Initialize Ray for testing."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def storage_setup(ray_setup):
    """Set up storage unit for testing."""
    storage_size = 10000
    tensordict.set_list_to_stack(True).set()

    # Start Ray actor for SimpleStorageUnit
    storage_actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(storage_unit_size=storage_size)

    # Get ZMQ server info from storage unit
    zmq_info = ray.get(storage_actor.get_zmq_server_info.remote())
    put_get_address = zmq_info.to_addr("put_get_socket")
    time.sleep(1)  # Wait for socket to be ready

    yield storage_actor, put_get_address, zmq_info.ip

    # Cleanup
    ray.kill(storage_actor)


def test_put_get_single_client(storage_setup):
    """Test basic put and get operations with a single client."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT data
    global_indexes = [0, 1, 2]
    field_data = {
        "log_probs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0]), torch.tensor([30.0])],
    }

    response = client.send_put(0, global_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1], ["log_probs", "rewards"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "log_probs" in retrieved_data
    assert "rewards" in retrieved_data
    assert len(retrieved_data["log_probs"]) == 2
    assert len(retrieved_data["rewards"]) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([10.0]))
    torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([20.0]))

    client.close()


def test_put_get_multiple_clients(storage_setup):
    """Test put and get operations with multiple clients."""
    _, put_get_address, storage_ip = storage_setup

    num_clients = 3
    clients = [MockStorageClient(put_get_address, storage_ip) for _ in range(num_clients)]

    # Each client puts unique data using different global_indexes
    for i, client in enumerate(clients):
        global_indexes = [i * 10 + 0, i * 10 + 1, i * 10 + 2]
        field_data = {
            "log_probs": [
                torch.tensor([i, i + 1, i + 2]),
                torch.tensor([i + 3, i + 4, i + 5]),
                torch.tensor([i + 6, i + 7, i + 8]),
            ],
            "rewards": [torch.tensor([i * 10]), torch.tensor([i * 10 + 10]), torch.tensor([i * 10 + 20])],
        }

        response = client.send_put(i, global_indexes, field_data)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Test overlapping global indexes
    overlapping_client = MockStorageClient(put_get_address, storage_ip)
    overlap_global_indexes = [0]  # Overlaps with first client's index 0
    overlap_field_data = {"log_probs": [torch.tensor([999, 999, 999])], "rewards": [torch.tensor([999])]}
    response = overlapping_client.send_put(99, overlap_global_indexes, overlap_field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Each original client gets its own data (except for index 0 which was overwritten)
    for i, client in enumerate(clients):
        response = client.send_get(i, [i * 10 + 0, i * 10 + 1], ["log_probs", "rewards"])
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

        retrieved_data = response.body["data"]
        assert len(retrieved_data["log_probs"]) == 2
        assert len(retrieved_data["rewards"]) == 2

        # For index 0, expect data from overlapping_client; others from original client
        if i == 0:
            # Index 0 was overwritten
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([999, 999, 999]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([999]))
            # Index 1 remains original
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([3, 4, 5]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([10]))
        else:
            # All data remains original
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([i, i + 1, i + 2]))
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([i + 3, i + 4, i + 5]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([i * 10]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([i * 10 + 10]))

    # Cleanup
    for client in clients:
        client.close()
    overlapping_client.close()


def test_performance_basic(storage_setup):
    """Basic performance test with larger data volume."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT performance test
    put_latencies = []
    num_puts = 10  # Reduced for faster testing
    batch_size = 16  # Reduced for faster testing

    for i in range(num_puts):
        start = time.time()

        # Use batch size and index mapping
        global_indexes = list(range(i * batch_size, (i + 1) * batch_size))

        # Create tensor data
        log_probs_data = []
        rewards_data = []

        for _ in range(batch_size):
            # Smaller tensors for faster testing
            log_probs_tensor = torch.randn(100)
            rewards_tensor = torch.randn(100)
            log_probs_data.append(log_probs_tensor)
            rewards_data.append(rewards_tensor)

        field_data = {"log_probs": log_probs_data, "rewards": rewards_data}

        response = client.send_put(0, global_indexes, field_data)
        latency = time.time() - start
        put_latencies.append(latency)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET performance test
    get_latencies = []
    num_gets = 10

    for i in range(num_gets):
        start = time.time()
        # Retrieve batch of data
        global_indexes = list(range(i * batch_size, (i + 1) * batch_size))
        response = client.send_get(0, global_indexes, ["log_probs", "rewards"])
        latency = time.time() - start
        get_latencies.append(latency)
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    avg_put_latency = sum(put_latencies) / len(put_latencies) * 1000  # ms
    avg_get_latency = sum(get_latencies) / len(get_latencies) * 1000  # ms

    # More lenient performance thresholds for testing environment
    assert avg_put_latency < 1500, f"Avg PUT latency {avg_put_latency}ms exceeds threshold"
    assert avg_get_latency < 1500, f"Avg GET latency {avg_get_latency}ms exceeds threshold"

    client.close()


def test_put_get_nested_tensor(storage_setup):
    """Test put and get operations with nested tensors."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT data with nested tensors
    global_indexes = [0, 1, 2]
    field_data = {
        "variable_length_sequences": [
            torch.tensor([-0.5, -1.2, -0.8]),
            torch.tensor([-0.3, -1.5, -2.1, -0.9]),
            torch.tensor([-1.1, -0.7]),
        ],
        "attention_mask": [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1])],
    }

    response = client.send_put(0, global_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 2], ["variable_length_sequences", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "variable_length_sequences" in retrieved_data
    assert "attention_mask" in retrieved_data
    assert len(retrieved_data["variable_length_sequences"]) == 2
    assert len(retrieved_data["attention_mask"]) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][0], torch.tensor([-0.5, -1.2, -0.8]))
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][1], torch.tensor([-1.1, -0.7]))
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))
    torch.testing.assert_close(retrieved_data["attention_mask"][1], torch.tensor([1, 1]))

    client.close()


def test_put_get_non_tensor_data(storage_setup):
    """Test put and get operations with non-tensor data (strings)."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT data with non-tensor data
    global_indexes = [0, 1, 2]
    field_data = {
        "prompt_text": ["Hello world!", "This is a longer sentence for testing", "Test case"],
        "response_text": ["Hi there!", "This is the response to the longer sentence", "Test response"],
    }

    response = client.send_put(0, global_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1, 2], ["prompt_text", "response_text"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "prompt_text" in retrieved_data
    assert "response_text" in retrieved_data

    # Verify data correctness
    assert isinstance(retrieved_data["prompt_text"][0], str)
    assert isinstance(retrieved_data["response_text"][0], str)

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert retrieved_data["prompt_text"][1] == "This is a longer sentence for testing"
    assert retrieved_data["prompt_text"][2] == "Test case"
    assert retrieved_data["response_text"][0] == "Hi there!"
    assert retrieved_data["response_text"][1] == "This is the response to the longer sentence"
    assert retrieved_data["response_text"][2] == "Test response"

    client.close()


def test_put_get_single_item(storage_setup):
    """Test put and get operations for a single item."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT single item data
    field_data = {
        "prompt_text": ["Hello world!"],
        "attention_mask": [torch.tensor([1, 1, 1])],
    }
    response = client.send_put(0, [0], field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0], ["prompt_text", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]

    assert "prompt_text" in retrieved_data
    assert "attention_mask" in retrieved_data

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert len(retrieved_data["attention_mask"]) == 1
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))

    client.close()


def test_clear_data(storage_setup):
    """Test clear operations."""
    _, put_get_address, storage_ip = storage_setup

    client = MockStorageClient(put_get_address, storage_ip)

    # PUT data first
    global_indexes = [0, 1, 2]
    field_data = {
        "log_probs": [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0]), torch.tensor([30.0])],
    }

    response = client.send_put(0, global_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Verify data exists
    response = client.send_get(0, [0, 1, 2], ["log_probs"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert len(response.body["data"]["log_probs"]) == 3

    # Clear data
    response = client.send_clear(0, [0, 2])  # Clear only indexes 0 and 2
    assert response.request_type == ZMQRequestType.CLEAR_DATA_RESPONSE

    # Verify some data is cleared (but index 1 should still exist)
    response = client.send_get(0, [1], ["log_probs"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert len(response.body["data"]["log_probs"]) == 1
    torch.testing.assert_close(response.body["data"]["log_probs"][0], torch.tensor([2.0]))

    client.close()


def test_storage_unit_data_direct():
    """Test StorageUnitData class directly without ZMQ."""
    from transfer_queue.storage import StorageUnitData

    storage_data = StorageUnitData(storage_size=10)

    field_data = {
        "log_probs": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0])],
    }
    # global_indexes = global_index values (e.g., 0 and 1)
    storage_data.put_data(field_data, [0, 1])

    result = storage_data.get_data(["log_probs", "rewards"], [0, 1])
    assert "log_probs" in result
    assert "rewards" in result
    assert len(result["log_probs"]) == 2
    assert len(result["rewards"]) == 2

    result_single = storage_data.get_data(["log_probs"], [0])
    torch.testing.assert_close(result_single["log_probs"][0], torch.tensor([1.0, 2.0]))

    # clear: key is removed (not set to None)
    storage_data.clear([0])
    assert 0 not in storage_data.field_data["log_probs"]  # key gone
    assert 1 in storage_data.field_data["log_probs"]  # other key intact


def test_storage_unit_data_capacity_uses_active_keys():
    """Capacity check must use _active_keys, not scan field_data."""
    from transfer_queue.storage.simple_storage import StorageUnitData

    storage = StorageUnitData(storage_size=3)

    # Fill to capacity
    storage.put_data({"f": [1, 2, 3]}, global_indexes=[0, 1, 2])
    assert len(storage._active_keys) == 3

    # Exceeding capacity must raise
    with pytest.raises(ValueError, match="Storage capacity exceeded"):
        storage.put_data({"f": [4]}, global_indexes=[3])

    # After clearing one key, adding one more should succeed
    storage.clear(keys=[2])
    assert len(storage._active_keys) == 2
    storage.put_data({"f": [4]}, global_indexes=[3])
    assert storage._active_keys == {0, 1, 3}


def test_disk_storage_unit_data_round_trip_and_partial_update(tmp_path):
    """SSD storage preserves field-level SimpleStorage semantics."""
    storage = DiskStorageUnitData(3, str(tmp_path), "test_unit", 1024 * 1024)
    try:
        assert storage._connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        storage.put_data(
            {
                "tensor": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
                "text": ["first", "second"],
            },
            [10, 11],
        )
        storage.put_data({"text": ["updated"]}, [10])

        result = storage.get_data(["tensor", "text"], [11, 10])
        torch.testing.assert_close(result["tensor"][0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result["tensor"][1], torch.tensor([1.0, 2.0]))
        assert result["text"] == ["second", "updated"]
        assert storage.active_key_count == 2
        assert storage.disk_usage_bytes > 0

        storage.clear([11])
        assert storage.active_key_count == 1
        with pytest.raises(KeyError, match="key 11 not found"):
            storage.get_data(["tensor"], [11])
        storage.clear([10])
        assert not list((tmp_path / "test_unit").glob("*.batch"))
    finally:
        storage.close()

    assert not (tmp_path / "test_unit").exists()


def test_disk_storage_capacity_failure_is_atomic(tmp_path):
    storage = DiskStorageUnitData(1, str(tmp_path), "capacity_unit", 0)
    try:
        storage.put_data({"value": [torch.tensor([1])]}, [1])
        with pytest.raises(ValueError, match="Storage capacity exceeded"):
            storage.put_data({"value": [torch.tensor([2])]}, [2])

        assert storage.active_key_count == 1
        torch.testing.assert_close(storage.get_data(["value"], [1])["value"][0], torch.tensor([1]))
    finally:
        storage.close()


def test_disk_storage_write_failure_rolls_back(monkeypatch, tmp_path):
    storage = DiskStorageUnitData(10, str(tmp_path), "failure_unit", 1024 * 1024)

    def fail_write(_, __):
        raise sqlite3.OperationalError("disk full")

    try:
        storage.put_data({"value": [torch.tensor([1.0])]}, [1])
        old_batch_files = set((tmp_path / "failure_unit").glob("*.batch"))
        monkeypatch.setattr(storage, "_write_all", fail_write)
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            storage.put_data({"value": [torch.tensor([2.0])]}, [1])

        assert storage.active_key_count == 1
        assert storage._connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert storage._connection.execute("SELECT COUNT(*) FROM field_batches").fetchone()[0] == 1
        assert set((tmp_path / "failure_unit").glob("*.batch")) == old_batch_files
        torch.testing.assert_close(storage.get_data(["value"], [1])["value"][0], torch.tensor([1.0]))
    finally:
        storage.close()


def test_disk_storage_nested_and_non_tensor_batches(tmp_path):
    storage = DiskStorageUnitData(10, str(tmp_path), "complex_unit", 1024 * 1024)
    nested = torch.nested.as_nested_tensor(
        [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
        layout=torch.jagged,
    )
    metadata = tensordict.NonTensorStack("first", "second")
    try:
        storage.put_data({"nested": nested, "metadata": metadata}, [1, 2])
        result = storage.get_data(["nested", "metadata"], [2, 1])

        torch.testing.assert_close(result["nested"][0], torch.tensor([3, 4, 5]))
        torch.testing.assert_close(result["nested"][1], torch.tensor([1, 2]))
        assert result["metadata"] == ["second", "first"]
    finally:
        storage.close()


def test_disk_checkpoint_is_portable_between_storage_modes(tmp_path):
    disk_checkpoint = str(tmp_path / "disk_checkpoint.pkl")
    memory_checkpoint = str(tmp_path / "memory_checkpoint.pkl")
    disk = DiskStorageUnitData(10, str(tmp_path), "checkpoint_source", 1024 * 1024)
    memory = StorageUnitData(10)
    restored_disk = DiskStorageUnitData(10, str(tmp_path), "checkpoint_target", 1024 * 1024)
    try:
        disk.put_data({"value": [torch.tensor([1.0]), torch.tensor([2.0])]}, [1, 2])
        disk.save_checkpoint(disk_checkpoint, "source")
        memory.load_checkpoint(disk_checkpoint)
        torch.testing.assert_close(memory.get_data(["value"], [2])["value"][0], torch.tensor([2.0]))

        memory.save_checkpoint(memory_checkpoint, "memory")
        restored_disk.load_checkpoint(memory_checkpoint)
        restored = restored_disk.get_data(["value"], [1, 2])["value"]
        torch.testing.assert_close(restored[0], torch.tensor([1.0]))
        torch.testing.assert_close(restored[1], torch.tensor([2.0]))
    finally:
        disk.close()
        restored_disk.close()


def test_incomplete_disk_checkpoint_rolls_back(tmp_path):
    checkpoint = tmp_path / "incomplete_checkpoint.pkl"
    storage = DiskStorageUnitData(10, str(tmp_path), "rollback_target", 1024 * 1024)
    try:
        storage.put_data({"value": [torch.tensor([1.0])]}, [1])
        old_batch_files = set((tmp_path / "rollback_target").glob("*.batch"))
        storage.save_checkpoint(str(checkpoint), "source")
        checkpoint.write_bytes(checkpoint.read_bytes()[:-10])

        with pytest.raises((ValueError, EOFError, pickle.UnpicklingError)):
            storage.load_checkpoint(str(checkpoint))

        assert set((tmp_path / "rollback_target").glob("*.batch")) == old_batch_files
        torch.testing.assert_close(storage.get_data(["value"], [1])["value"][0], torch.tensor([1.0]))
    finally:
        storage.close()


def test_storage_unit_ssd_offload_e2e(ray_setup, tmp_path):
    """The configured actor path uses SSD storage through the public ZMQ data plane."""
    actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(
        storage_unit_size=10,
        offload_path=str(tmp_path),
        offload_cache_size_bytes=1024 * 1024,
    )
    info = ray.get(actor.get_zmq_server_info.remote())
    client = MockStorageClient(info.to_addr("put_get_socket"), info.ip)
    time.sleep(1)
    try:
        response = client.send_put(0, [1, 2], {"value": [torch.tensor([1.0]), torch.tensor([2.0])]})
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

        response = client.send_get(0, [2, 1], ["value"])
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
        torch.testing.assert_close(response.body["data"]["value"][0], torch.tensor([2.0]))

        metrics = client.send_get_metrics(0).body
        assert metrics["active_keys"] == 2
        assert metrics["offload_disk_bytes"] > 0
    finally:
        client.close()
        ray.get(actor.close.remote(), timeout=10)
        ray.kill(actor)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    os.environ.get("TQ_RUN_MOONCAKE_E2E") != "1",
    reason="requires a local mooncake_master with SSD offload enabled",
)
def test_storage_unit_mooncake_offload_e2e(ray_setup, tmp_path):
    """Exercise SimpleStorage's public ZMQ path through a real Mooncake SSD client."""
    mooncake_config = {
        "local_hostname": "127.0.0.1",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": "127.0.0.1:50051",
        "global_segment_size": 64 * 1024 * 1024,
        "local_buffer_size": 16 * 1024 * 1024,
        "protocol": "tcp",
        "device_name": "",
        "put_timeout_seconds": 30,
        "retry_interval_seconds": 0.1,
        "offload": {
            "local_buffer_size_bytes": 32 * 1024 * 1024,
            "heartbeat_interval_seconds": 1,
            "use_uring": False,
        },
    }
    actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(
        storage_unit_size=128,
        offload_path=str(tmp_path),
        offload_cache_size_bytes=1024 * 1024,
        offload_backend="mooncake",
        mooncake_config=mooncake_config,
    )
    info = ray.get(actor.get_zmq_server_info.remote())
    client = MockStorageClient(info.to_addr("put_get_socket"), info.ip)
    time.sleep(1)
    try:
        batch_size = 8
        sample_size = 1024 * 1024
        for batch_index in range(12):
            indexes = list(range(batch_index * batch_size, (batch_index + 1) * batch_size))
            values = torch.full((batch_size, sample_size), batch_index, dtype=torch.uint8)
            response = client.send_put(0, indexes, {"value": values})
            assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE, response.body

        # The first values exceed the 64 MiB Mooncake segment and must be read
        # through its local-disk path after the 500 ms master lease expires.
        time.sleep(2)
        response = client.send_get(0, [0, 7, 88, 95], ["value"])
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE, response.body
        values = response.body["data"]["value"]
        torch.testing.assert_close(values[0], torch.zeros(sample_size, dtype=torch.uint8))
        torch.testing.assert_close(values[-1], torch.full((sample_size,), 11, dtype=torch.uint8))

        metrics = client.send_get_metrics(0).body
        assert metrics["active_keys"] == 96
        assert metrics["offload_disk_bytes"] > 64 * 1024 * 1024
    finally:
        client.close()
        ray.get(actor.close.remote(), timeout=20)
        ray.kill(actor)

    assert list(tmp_path.iterdir()) == []


def test_storage_unit_data_parser(storage_setup):
    """Test data_parser functionality in SimpleStorageUnit.

    Writes two columns:
    - normal_data: regular tensors, should remain unchanged
    - data_to_be_parsed: list of shape descriptors (list of ints)

    data_parser converts shape descriptors into random tensors of those shapes.
    """
    _, put_get_address, storage_ip = storage_setup
    client = MockStorageClient(put_get_address, storage_ip)

    def create_data_by_shape_parser(field_data):
        if "data_to_be_parsed" in field_data:
            shapes = field_data["data_to_be_parsed"]
            field_data["data_to_be_parsed"] = [torch.randn(shape) for shape in shapes]
        return field_data

    # Prepare data: normal_data is a batch tensor, data_to_be_parsed is a list of shape lists
    field_data = {
        "normal_data": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "data_to_be_parsed": [[2, 3], [1, 4], [3, 2]],
    }
    global_indexes = [0, 1, 2]

    # Put with data_parser
    response = client.send_put(0, global_indexes, field_data, data_parser=create_data_by_shape_parser)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE, f"Put failed: {response.body}"

    # Get back
    response = client.send_get(0, global_indexes, ["normal_data", "data_to_be_parsed"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    result = response.body["data"]

    # Verify normal_data is unchanged
    torch.testing.assert_close(result["normal_data"][0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result["normal_data"][1], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(result["normal_data"][2], torch.tensor([5.0, 6.0]))

    # Verify data_to_be_parsed shapes match the input shape descriptors
    expected_shapes = [(2, 3), (1, 4), (3, 2)]
    for i, expected_shape in enumerate(expected_shapes):
        actual_shape = tuple(result["data_to_be_parsed"][i].shape)
        assert actual_shape == expected_shape, (
            f"Shape mismatch at index {i}: expected {expected_shape}, got {actual_shape}"
        )

    client.close()


def test_storage_unit_data_parser_callable_types(storage_setup):
    """Test that various callable types (partial, callable class) work as data_parser."""
    _, put_get_address, storage_ip = storage_setup
    client = MockStorageClient(put_get_address, storage_ip)

    from functools import partial

    # 1. Test functools.partial
    def _partial_parser(field_data, prefix):
        if "text" in field_data:
            field_data["text"] = [f"{prefix}{t}" for t in field_data["text"]]
        return field_data

    partial_parser = partial(_partial_parser, prefix="parsed_")

    response = client.send_put(
        0,
        [0, 1],
        {"text": ["a", "b"]},
        data_parser=partial_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE, f"partial parser failed: {response.body}"

    response = client.send_get(0, [0, 1], ["text"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert response.body["data"]["text"] == ["parsed_a", "parsed_b"]

    # 2. Test callable class instance
    class CallableParser:
        def __call__(self, field_data):
            if "value" in field_data:
                field_data["value"] = [v * 2 for v in field_data["value"]]
            return field_data

    callable_parser = CallableParser()
    response = client.send_put(
        0,
        [2, 3],
        {"value": [1, 2]},
        data_parser=callable_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE, f"callable class parser failed: {response.body}"

    response = client.send_get(0, [2, 3], ["value"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert response.body["data"]["value"] == [2, 4]

    client.close()


def test_storage_unit_data_parser_validation(storage_setup):
    """Test that invalid data_parser inputs produce clear error messages."""
    _, put_get_address, storage_ip = storage_setup
    client = MockStorageClient(put_get_address, storage_ip)

    # 1. Non-callable data_parser should return a clear TypeError
    response = client.send_put(
        0,
        [0],
        {"data": [1]},
        data_parser="not_callable",
    )
    assert response.request_type == ZMQRequestType.PUT_ERROR
    assert "data_parser must be callable" in response.body["message"]

    # 2. data_parser returning non-dict should return a clear TypeError
    def bad_parser(field_data):
        return "not_a_dict"

    response = client.send_put(
        0,
        [1],
        {"data": [1]},
        data_parser=bad_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_ERROR
    assert "data_parser must return a dict" in response.body["message"]

    # 3. data_parser deleting a key should return a clear ValueError
    def delete_key_parser(field_data):
        del field_data["data"]
        return field_data

    response = client.send_put(
        0,
        [2],
        {"data": [1], "extra": [2]},
        data_parser=delete_key_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_ERROR
    assert "data_parser must not change dict keys" in response.body["message"]

    # 4. data_parser adding a key should return a clear ValueError
    def add_key_parser(field_data):
        field_data["new_key"] = [999]
        return field_data

    response = client.send_put(
        0,
        [3],
        {"data": [1]},
        data_parser=add_key_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_ERROR
    assert "data_parser must not change dict keys" in response.body["message"]

    # 5. data_parser changing element count should return a clear ValueError
    def wrong_len_parser(field_data):
        field_data["data"] = field_data["data"][:-1]
        return field_data

    response = client.send_put(
        0,
        [4, 5],
        {"data": [1, 2]},
        data_parser=wrong_len_parser,
    )
    assert response.request_type == ZMQRequestType.PUT_ERROR
    assert "data_parser changed the number of elements" in response.body["message"]

    client.close()


def test_storage_unit_checkpoint_round_trip(storage_setup, tmp_path):
    """Save storage state to a file, load it into a fresh unit, verify data."""
    _, put_get_address, storage_ip = storage_setup
    ckpt_path = str(tmp_path / "storage_unit.pkl")
    client = MockStorageClient(put_get_address, storage_ip)

    # 1. Put some data
    global_indexes = [10, 11, 12]
    field_data = {
        "log_probs": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])],
        "rewards": [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])],
    }
    response = client.send_put(0, global_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # 2. Save checkpoint
    response = client.send_save_checkpoint(0, ckpt_path)
    assert response.request_type == ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE
    assert response.body["success"] is True
    assert (tmp_path / "storage_unit.pkl").exists()

    # 3. Create a fresh storage unit and load the checkpoint into it
    fresh_actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(storage_unit_size=10000)
    fresh_zmq_info = ray.get(fresh_actor.get_zmq_server_info.remote())
    import time as _time

    _time.sleep(1)
    fresh_address = fresh_zmq_info.to_addr("put_get_socket")
    fresh_client = MockStorageClient(fresh_address, fresh_zmq_info.ip)

    response = fresh_client.send_load_checkpoint(0, ckpt_path)
    assert response.request_type == ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE
    assert response.body["success"] is True

    # 4. Verify data is accessible in the fresh unit
    response = fresh_client.send_get(0, global_indexes, ["log_probs", "rewards"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    retrieved = response.body["data"]
    torch.testing.assert_close(retrieved["log_probs"][0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(retrieved["log_probs"][2], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(retrieved["rewards"][1], torch.tensor([0.2]))

    fresh_client.close()
    client.close()
    ray.kill(fresh_actor)


def test_storage_unit_checkpoint_overwrites_existing_data(storage_setup, tmp_path):
    """Loading a checkpoint into a unit that already has data replaces it entirely."""
    _, put_get_address, storage_ip = storage_setup
    ckpt_path = str(tmp_path / "storage_unit_overwrite.pkl")
    client = MockStorageClient(put_get_address, storage_ip)

    # 1. Put original data and save checkpoint
    response = client.send_put(0, [20, 21], {"val": [torch.tensor([1.0]), torch.tensor([2.0])]})
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE
    response = client.send_save_checkpoint(0, ckpt_path)
    assert response.body["success"] is True

    # 2. Create a second unit, pre-populate it with different data, then load the checkpoint
    second_actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(storage_unit_size=10000)
    second_zmq_info = ray.get(second_actor.get_zmq_server_info.remote())
    import time as _time

    _time.sleep(1)
    second_address = second_zmq_info.to_addr("put_get_socket")
    second_client = MockStorageClient(second_address, second_zmq_info.ip)

    # 3. Write different data into the second unit before loading
    response = second_client.send_put(0, [99], {"val": [torch.tensor([999.0])]})
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # 4. Load checkpoint — should overwrite
    response = second_client.send_load_checkpoint(0, ckpt_path)
    assert response.body["success"] is True

    # 5. Old data (index 99) should be gone; checkpoint data (indexes 20, 21) should be present
    response = second_client.send_get(0, [20, 21], ["val"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    retrieved = response.body["data"]
    torch.testing.assert_close(retrieved["val"][0], torch.tensor([1.0]))
    torch.testing.assert_close(retrieved["val"][1], torch.tensor([2.0]))

    second_client.close()
    client.close()
    ray.kill(second_actor)


def test_storage_unit_checkpoint_load_missing_file(storage_setup, tmp_path):
    """Loading from a non-existent file returns success=False."""
    _, put_get_address, storage_ip = storage_setup
    client = MockStorageClient(put_get_address, storage_ip)
    missing_path = str(tmp_path / "does_not_exist.pkl")

    response = client.send_load_checkpoint(0, missing_path)
    assert response.request_type == ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE
    assert response.body["success"] is False
    assert "message" in response.body

    client.close()
