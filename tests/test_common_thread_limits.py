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

from unittest.mock import Mock

import pytest

from transfer_queue.utils import common


@pytest.fixture
def thread_limits(monkeypatch):
    common._physical_cpu_count.cache_clear()
    query = Mock(return_value=24)
    set_threads = Mock()
    monkeypatch.setattr(common.psutil, "cpu_count", query)
    monkeypatch.setattr(common.torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(common.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(common, "DEFAULT_TORCH_NUM_THREADS", 6)
    yield query, set_threads
    common._physical_cpu_count.cache_clear()


@pytest.mark.parametrize("physical,target,expected", [(24, None, 16), (8, None, 8), (24, 48, 24), (24, 4, 4)])
def test_repeated_thread_limits_reuse_cpu_count(thread_limits, physical, target, expected):
    query, set_threads = thread_limits
    query.return_value = physical
    query.assert_not_called()
    for _ in range(2):
        with common.limit_pytorch_auto_parallel_threads(target):
            set_threads.assert_called_with(expected)
        set_threads.assert_called_with(6)
    query.assert_called_once_with(logical=False)
    assert [call.args[0] for call in set_threads.call_args_list] == [expected, 6, expected, 6]


def test_thread_limit_restores_default_after_exception(thread_limits):
    query, set_threads = thread_limits
    with pytest.raises(RuntimeError, match="operation failed"):
        with common.limit_pytorch_auto_parallel_threads(4):
            raise RuntimeError("operation failed")
    set_threads.assert_called_with(6)
    query.assert_called_once_with(logical=False)
