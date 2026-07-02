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

__all__ = [
    "StorageKVClient",
    "StorageClientFactory",
    "RayStorageClient",
    "MooncakeStoreClient",
    "YuanrongStorageClient",
]

_LAZY_EXPORTS = {
    "StorageKVClient": (".base", "StorageKVClient"),
    "StorageClientFactory": (".base", "StorageClientFactory"),
    "RayStorageClient": (".ray_storage_client", "RayStorageClient"),
    "MooncakeStoreClient": (".mooncake_client", "MooncakeStoreClient"),
    "YuanrongStorageClient": (".yuanrong_client", "YuanrongStorageClient"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
