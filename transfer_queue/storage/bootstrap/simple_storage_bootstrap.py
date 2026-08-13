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

import math
import subprocess
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from transfer_queue.storage.bootstrap.mooncake_bootstrap import initialize_mooncake_storage
from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.common import get_node_round_robin_scheduling_strategies
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)


class SimpleStorageResources(dict[str, Any]):
    """Actor mapping plus an optional Mooncake master owned by this bootstrap."""

    def __init__(self, *args, mooncake_master: subprocess.Popen | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mooncake_master = mooncake_master


def _build_mooncake_config(conf: DictConfig, offload_conf: DictConfig) -> dict[str, Any]:
    if "MooncakeStore" not in conf.backend:
        raise ValueError("SimpleStorage Mooncake offload requires backend.MooncakeStore configuration")
    config = OmegaConf.to_container(conf.backend.MooncakeStore, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("backend.MooncakeStore must be a mapping")

    tuning_node = offload_conf.get("mooncake", None)
    tuning = {} if tuning_node is None else OmegaConf.to_container(tuning_node, resolve=True)
    if not isinstance(tuning, dict):
        raise TypeError("SimpleStorage offload.mooncake must be a mapping")
    config["global_segment_size"] = int(tuning.get("global_segment_size", 512 * 1024 * 1024))
    config["local_buffer_size"] = int(tuning.get("local_buffer_size", 32 * 1024 * 1024))
    config["hard_pin"] = False
    config["put_timeout_seconds"] = float(tuning.get("put_timeout_seconds", 60))
    config["retry_interval_seconds"] = float(tuning.get("retry_interval_seconds", 0.1))

    base_offload = config.get("offload", {})
    if not isinstance(base_offload, dict):
        raise TypeError("backend.MooncakeStore.offload must be a mapping")
    offload_buffer_size = int(tuning.get("offload_buffer_size_bytes", 64 * 1024 * 1024))
    config["offload"] = {
        **base_offload,
        "enabled": True,
        "file_storage_path": str(offload_conf.file_storage_path),
        "local_buffer_size_bytes": offload_buffer_size,
        "max_object_size_bytes": int(
            tuning.get("max_object_size_bytes", min(16 * 1024 * 1024, offload_buffer_size) - 65536)
        ),
        "get_window_size_bytes": int(tuning.get("get_window_size_bytes", offload_buffer_size)),
        "heartbeat_interval_seconds": int(tuning.get("heartbeat_interval_seconds", 1)),
        "use_uring": bool(tuning.get("use_uring", False)),
        "lease_ttl_ms": int(tuning.get("lease_ttl_ms", 500)),
        "eviction_high_watermark_ratio": float(tuning.get("eviction_high_watermark_ratio", 0.8)),
        "eviction_ratio": float(tuning.get("eviction_ratio", 0.2)),
    }
    return config


@StorageBootstrapProvider.register_provider("SimpleStorage")
def initialize_simple_storage(conf: DictConfig) -> dict[str, Any]:
    """Initialize Simple storage with metastore mode."""

    simple_storage_handles = {}
    num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
    total_storage_size = conf.backend.SimpleStorage.get("total_storage_size", None)
    required_node_resource = conf.backend.SimpleStorage.get("required_node_resource", None)
    offload_conf = conf.backend.SimpleStorage.get("offload", {})
    offload_path = None
    offload_backend = "local_file"
    offload_cache_size_bytes = 64 * 1024 * 1024
    mooncake_config = None
    mooncake_master = None
    if offload_conf.get("enabled", False):
        offload_path = offload_conf.get("file_storage_path", None)
        if not offload_path:
            raise ValueError("SimpleStorage offload.file_storage_path must be set when offload is enabled")
        offload_cache_size_bytes = int(offload_conf.get("memory_cache_size_bytes", offload_cache_size_bytes))
        if offload_cache_size_bytes < 0:
            raise ValueError(
                f"SimpleStorage offload.memory_cache_size_bytes must be >= 0, got {offload_cache_size_bytes}"
            )
        offload_backend = str(offload_conf.get("backend", "mooncake")).strip().lower()
        if offload_backend not in ("local_file", "mooncake"):
            raise ValueError(f"Unsupported SimpleStorage offload backend: {offload_backend}")
        if offload_backend == "mooncake":
            mooncake_config = _build_mooncake_config(conf, offload_conf)
            master_conf = OmegaConf.create({"backend": {"MooncakeStore": mooncake_config}})
            mooncake_master = initialize_mooncake_storage(master_conf)
    # Compute per-unit capacity: None means unlimited
    storage_unit_size = (
        math.ceil(total_storage_size / num_data_storage_units) if total_storage_size is not None else None
    )
    try:
        scheduling_strategies = get_node_round_robin_scheduling_strategies(
            num_data_storage_units, required_node_resource=required_node_resource
        )
        for storage_unit_rank in range(num_data_storage_units):
            storage_node = SimpleStorageUnit.options(  # type: ignore[attr-defined]
                scheduling_strategy=scheduling_strategies[storage_unit_rank],
                name=f"TransferQueueStorageUnit#{storage_unit_rank}",
            ).remote(
                storage_unit_size=storage_unit_size,
                offload_path=offload_path,
                offload_cache_size_bytes=offload_cache_size_bytes,
                offload_backend=offload_backend,
                mooncake_config=mooncake_config,
            )
            simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
            logger.info(
                f"TransferQueueStorageUnit#{storage_unit_rank} has been created "
                f"on node {scheduling_strategies[storage_unit_rank].node_id}."
            )

        storage_zmq_info = process_zmq_server_info(simple_storage_handles)
    except Exception:
        for storage_node in simple_storage_handles.values():
            ray.kill(storage_node)
        if mooncake_master is not None and mooncake_master.poll() is None:
            mooncake_master.terminate()
            try:
                mooncake_master.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mooncake_master.kill()
                mooncake_master.wait(timeout=5)
        raise

    backend_name = conf.backend.storage_backend
    conf.backend[backend_name].zmq_info = storage_zmq_info

    return SimpleStorageResources(simple_storage_handles, mooncake_master=mooncake_master)
