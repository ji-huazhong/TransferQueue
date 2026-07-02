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

__all__ = (
    [
        # High-Level KV Interface
        "init",
        "close",
        "get_metrics_endpoint",
        "kv_put",
        "kv_batch_put",
        "kv_batch_get",
        "kv_batch_get_by_meta",
        "kv_list",
        "kv_clear",
        "async_kv_put",
        "async_kv_batch_put",
        "async_kv_batch_get",
        "async_kv_batch_get_by_meta",
        "async_kv_list",
        "async_kv_clear",
        "KVBatchMeta",
    ]
    + [
        # High-Level StreamingDataLoader Interface
        "StreamingDataset",
        "StreamingDataLoader",
    ]
    + [
        # Low-Level Native Interface
        "get_client",
        "BatchMeta",
        "TransferQueueClient",
    ]
    + [
        # Sampler
        "BaseSampler",
        "GRPOGroupNSampler",
        "SequentialSampler",
        "RankAwareSampler",
        "SeqlenBalancedSampler",
    ]
)

_LAZY_EXPORTS = {
    # High-Level KV Interface
    "init": (".interface", "init"),
    "close": (".interface", "close"),
    "get_metrics_endpoint": (".interface", "get_metrics_endpoint"),
    "kv_put": (".interface", "kv_put"),
    "kv_batch_put": (".interface", "kv_batch_put"),
    "kv_batch_get": (".interface", "kv_batch_get"),
    "kv_batch_get_by_meta": (".interface", "kv_batch_get_by_meta"),
    "kv_list": (".interface", "kv_list"),
    "kv_clear": (".interface", "kv_clear"),
    "async_kv_put": (".interface", "async_kv_put"),
    "async_kv_batch_put": (".interface", "async_kv_batch_put"),
    "async_kv_batch_get": (".interface", "async_kv_batch_get"),
    "async_kv_batch_get_by_meta": (".interface", "async_kv_batch_get_by_meta"),
    "async_kv_list": (".interface", "async_kv_list"),
    "async_kv_clear": (".interface", "async_kv_clear"),
    "get_client": (".interface", "get_client"),
    # High-Level StreamingDataLoader Interface
    "StreamingDataset": (".dataloader", "StreamingDataset"),
    "StreamingDataLoader": (".dataloader", "StreamingDataLoader"),
    # Low-Level Native Interface
    "TransferQueueClient": (".client", "TransferQueueClient"),
    "BatchMeta": (".metadata", "BatchMeta"),
    "KVBatchMeta": (".metadata", "KVBatchMeta"),
    # Sampler
    "BaseSampler": (".sampler", "BaseSampler"),
    "GRPOGroupNSampler": (".sampler.grpo_group_n_sampler", "GRPOGroupNSampler"),
    "SequentialSampler": (".sampler.sequential_sampler", "SequentialSampler"),
    "RankAwareSampler": (".sampler.rank_aware_sampler", "RankAwareSampler"),
    "SeqlenBalancedSampler": (".sampler.seqlen_balanced_sampler", "SeqlenBalancedSampler"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()
