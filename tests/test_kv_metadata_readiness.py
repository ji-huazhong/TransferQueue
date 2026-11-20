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

import pytest

from transfer_queue.controller import (
    DataPartitionStatus,
    PartitionIndexManager,
    TransferQueueController,
)


@pytest.fixture
def controller():
    cls = TransferQueueController.__ray_metadata__.modified_class
    controller = cls.__new__(cls)
    controller.controller_id = "readiness-test"
    controller.index_manager = PartitionIndexManager()
    controller.index_manager.allocate_indexes("p", 3)
    partition = DataPartitionStatus("p")
    partition.keys_mapping = {"a": 0, "b": 1, "c": 2}
    partition.revert_keys_mapping = {index: key for key, index in partition.keys_mapping.items()}
    partition.global_indexes.update([0, 1, 2])
    partition.ensure_samples_capacity(3)
    schema = {"shared": {"dtype": "torch.float32", "shape": (2,), "is_nested": False, "is_non_tensor": False}}
    assert partition.update_production_status([0, 1], ["shared"], schema)
    assert partition.update_production_status([0], ["left"], {"left": schema["shared"]})
    partition.custom_meta[0] = {"tag": "a"}
    partition.field_custom_backend_meta[0] = {"shared": "backend-a"}
    controller.partitions = {"p": partition}
    return controller


@pytest.mark.parametrize(
    "keys,indexes,fields,ready",
    [
        (["a"], [0], {"left", "shared"}, [1]),
        (["a", "b"], [0, 1], {"shared"}, [1, 1]),
        (["b", "a", "b"], [1, 0, 1], {"shared"}, [1, 1, 1]),
        (["a", "c"], [0, 2], set(), [0, 0]),
        (["c"], [2], set(), [0]),
        (["a", "absent"], [0], {"left", "shared"}, [1]),
    ],
)
def test_kv_metadata_returns_only_common_ready_fields(controller, keys, indexes, fields, ready):
    metadata = controller.kv_retrieve_meta(keys, "p")
    assert metadata.global_indexes == indexes
    assert set(metadata.field_names) == fields
    assert metadata.production_status.tolist() == ready
    assert metadata.custom_meta == [{"tag": "a"} if index == 0 else {} for index in indexes]


def test_kv_metadata_stays_unready_after_mark_clearing(controller):
    controller.mark_clearing([0], ["p"])
    metadata = controller.kv_retrieve_meta(["a", "b"], "p")
    assert metadata.global_indexes == [0, 1]
    assert metadata.field_names == []
    assert metadata.production_status.tolist() == [0, 0]
    assert controller.kv_retrieve_meta(["b"], "p").production_status.tolist() == [1]


def test_kv_metadata_new_key_is_unready(controller):
    metadata = controller.kv_retrieve_meta(["new"], "p", create=True)
    assert metadata.size == 1
    assert metadata.field_names == []
    assert metadata.production_status.tolist() == [0]


def test_kv_metadata_missing_keys_and_partition_are_empty(controller):
    assert controller.kv_retrieve_meta(["absent"], "p").size == 0
    assert controller.kv_retrieve_meta(["a"], "absent").size == 0
