# Copyright 2025 The TransferQueue Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from transfer_queue.controller import TransferQueueController
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@pytest.fixture
def controller():
    cls = TransferQueueController.__ray_metadata__.modified_class
    return cls.__new__(cls)


def test_dispatch_resolves_the_current_handler_and_records_metrics(controller):
    request = ZMQMessage.create(ZMQRequestType.KV_LIST, "client", body={})
    monitor = Mock()
    monitor.measure.side_effect = lambda **kwargs: nullcontext()
    for result in ("first", "replacement"):
        controller._handle_kv_list_request = Mock(return_value=result)
        assert controller._handle_request(request, monitor) == result
        controller._handle_kv_list_request.assert_called_once_with(request)
    assert monitor.measure.call_count == 2
    monitor.measure.assert_called_with(op_type="KV_LIST")


def test_unknown_request_has_no_handler_or_metrics(controller):
    request = ZMQMessage.create(ZMQRequestType.GET_DATA, "client", body={})
    monitor = Mock()
    assert controller._handle_request(request, monitor) is None
    monitor.measure.assert_not_called()


def test_handler_failure_propagates_through_the_monitor(controller):
    request = ZMQMessage.create(ZMQRequestType.KV_LIST, "client", body={})
    controller._handle_kv_list_request = Mock(side_effect=ValueError("invalid partition"))
    monitor = Mock()
    monitor.measure.return_value = nullcontext()
    with pytest.raises(ValueError, match="invalid partition"):
        controller._handle_request(request, monitor)
    monitor.measure.assert_called_once_with(op_type="KV_LIST")


@pytest.mark.parametrize(
    "request_type,method",
    [
        (ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT, "_handle_save_controller_checkpoint_request"),
        (ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT, "_handle_load_controller_checkpoint_request"),
    ],
)
def test_checkpoint_dispatch_preserves_metrics_exemption(controller, request_type, method):
    request = ZMQMessage.create(request_type, "client", body={})
    handler = Mock(return_value="checkpoint")
    setattr(controller, method, handler)
    monitor = Mock()
    assert controller._handle_request(request, monitor) == "checkpoint"
    handler.assert_called_once_with(request)
    monitor.measure.assert_not_called()
