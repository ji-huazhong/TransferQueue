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

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from transfer_queue.storage.bootstrap import simple_storage_bootstrap
from transfer_queue.utils import common

_NODE_A = "01" * 28
_NODE_B = "02" * 28
_NODE_C = "03" * 28
_NODE_D = "04" * 28


def _node(node_id: str, *, alive: bool = True, resources: dict[str, float] | None = None) -> dict:
    return {"NodeID": node_id, "Alive": alive, "Resources": resources or {}}


def _node_ids(strategies) -> list[str]:
    return [strategy.node_id for strategy in strategies]


def test_round_robin_uses_all_alive_nodes_by_default(monkeypatch):
    nodes = [_node(_NODE_B), _node(_NODE_C, alive=False), _node(_NODE_A)]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(5)

    assert _node_ids(strategies) == [_NODE_A, _NODE_B, _NODE_A, _NODE_B, _NODE_A]


def test_required_node_resource_filters_nodes_and_zero_capacity(monkeypatch):
    nodes = [
        _node(_NODE_C, resources={"storage_pool": 2}),
        _node(_NODE_B, resources={"storage_pool": 0}),
        _node(_NODE_A, resources={"storage_pool": 1}),
        _node(_NODE_D, resources={"compute_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(4, required_node_resource="storage_pool")

    assert _node_ids(strategies) == [_NODE_A, _NODE_C, _NODE_A, _NODE_C]


def test_required_node_resource_excludes_dead_nodes(monkeypatch):
    nodes = [
        _node(_NODE_A, alive=False, resources={"storage_pool": 1}),
        _node(_NODE_B, resources={"storage_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(2, required_node_resource="storage_pool")

    assert _node_ids(strategies) == [_NODE_B, _NODE_B]


def test_required_node_resource_raises_when_no_alive_node_matches(monkeypatch):
    nodes = [
        _node(_NODE_A, resources={"storage_pool": 0}),
        _node(_NODE_B, alive=False, resources={"storage_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'storage_pool'"):
        common.get_node_round_robin_scheduling_strategies(1, required_node_resource="storage_pool")


def test_default_no_alive_node_error_is_unchanged(monkeypatch):
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(RuntimeError, match="No alive Ray nodes found. Is Ray initialized?"):
        common.get_node_round_robin_scheduling_strategies(1)


def test_simple_storage_initialization_forwards_required_node_resource(monkeypatch):
    strategy = MagicMock(node_id=_NODE_A)
    get_strategies = MagicMock(return_value=[strategy])
    storage_unit = MagicMock()
    storage_handle = MagicMock()
    storage_unit.options.return_value.remote.return_value = storage_handle

    monkeypatch.setattr(simple_storage_bootstrap, "get_node_round_robin_scheduling_strategies", get_strategies)
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", lambda _: {})

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "total_storage_size": None,
                    "required_node_resource": "storage_pool",
                },
            }
        }
    )

    handles = simple_storage_bootstrap.initialize_simple_storage(conf)

    get_strategies.assert_called_once_with(1, required_node_resource="storage_pool")
    storage_unit.options.return_value.remote.assert_called_once_with(
        storage_unit_size=None,
        offload_path=None,
        offload_cache_size_bytes=64 * 1024 * 1024,
        offload_backend="local_file",
        mooncake_config=None,
    )
    assert handles == {"TransferQueueStorageUnit#0": storage_handle}


def test_simple_storage_initialization_forwards_ssd_offload(monkeypatch, tmp_path):
    strategy = MagicMock(node_id=_NODE_A)
    storage_unit = MagicMock()
    storage_handle = MagicMock()
    storage_unit.options.return_value.remote.return_value = storage_handle

    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        MagicMock(return_value=[strategy]),
    )
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", lambda _: {})

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "offload": {
                        "enabled": True,
                        "backend": "local_file",
                        "file_storage_path": str(tmp_path),
                        "memory_cache_size_bytes": 8 * 1024 * 1024,
                    },
                },
            }
        }
    )

    simple_storage_bootstrap.initialize_simple_storage(conf)

    storage_unit.options.return_value.remote.assert_called_once_with(
        storage_unit_size=None,
        offload_path=str(tmp_path),
        offload_cache_size_bytes=8 * 1024 * 1024,
        offload_backend="local_file",
        mooncake_config=None,
    )


def test_simple_storage_initialization_embeds_mooncake_offload(monkeypatch, tmp_path):
    strategy = MagicMock(node_id=_NODE_A)
    storage_unit = MagicMock()
    storage_handle = MagicMock()
    storage_unit.options.return_value.remote.return_value = storage_handle
    master = MagicMock()
    initialize_mooncake = MagicMock(return_value=master)

    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        MagicMock(return_value=[strategy]),
    )
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", lambda _: {})
    monkeypatch.setattr(simple_storage_bootstrap, "initialize_mooncake_storage", initialize_mooncake)

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "offload": {
                        "enabled": True,
                        "backend": "mooncake",
                        "file_storage_path": str(tmp_path),
                        "memory_cache_size_bytes": 1024,
                        "mooncake": {
                            "global_segment_size": 4096,
                            "local_buffer_size": 2048,
                            "offload_buffer_size_bytes": 1024,
                            "lease_ttl_ms": 250,
                        },
                    },
                },
                "MooncakeStore": {
                    "auto_init": True,
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": "127.0.0.1:50051",
                    "local_hostname": "127.0.0.1",
                    "protocol": "tcp",
                    "device_name": "",
                    "offload": {"enabled": False},
                },
            }
        }
    )

    resources = simple_storage_bootstrap.initialize_simple_storage(conf)

    master_conf = initialize_mooncake.call_args.args[0]
    assert master_conf.backend.MooncakeStore.offload.enabled is True
    assert master_conf.backend.MooncakeStore.offload.lease_ttl_ms == 250
    actor_config = storage_unit.options.return_value.remote.call_args.kwargs["mooncake_config"]
    assert actor_config["global_segment_size"] == 4096
    assert actor_config["local_buffer_size"] == 2048
    assert actor_config["offload"]["file_storage_path"] == str(tmp_path)
    assert actor_config["hard_pin"] is False
    assert resources.mooncake_master is master


def test_simple_storage_offload_requires_path(monkeypatch):
    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "offload": {"enabled": True, "file_storage_path": ""},
                },
            }
        }
    )

    with pytest.raises(ValueError, match="offload.file_storage_path must be set"):
        simple_storage_bootstrap.initialize_simple_storage(conf)


def test_simple_storage_mooncake_bootstrap_failure_cleans_owned_master(monkeypatch, tmp_path):
    strategy = MagicMock(node_id=_NODE_A)
    storage_unit = MagicMock()
    storage_unit.options.return_value.remote.side_effect = RuntimeError("actor setup failed")
    master = MagicMock()
    master.poll.return_value = None

    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        MagicMock(return_value=[strategy]),
    )
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "initialize_mooncake_storage", MagicMock(return_value=master))

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "offload": {
                        "enabled": True,
                        "backend": "mooncake",
                        "file_storage_path": str(tmp_path),
                    },
                },
                "MooncakeStore": {
                    "auto_init": True,
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": "127.0.0.1:50051",
                    "offload": {"enabled": False},
                },
            }
        }
    )

    with pytest.raises(RuntimeError, match="actor setup failed"):
        simple_storage_bootstrap.initialize_simple_storage(conf)

    master.terminate.assert_called_once_with()
    master.wait.assert_called_once_with(timeout=5)
