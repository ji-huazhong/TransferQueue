# Copyright 2025 The TransferQueue Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import AsyncMock

import pytest

from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.metadata import BatchMeta
from transfer_queue.utils.zmq_utils import ZMQRequestType


@pytest.mark.asyncio
async def test_tag_update_preserves_partition_order_overwrites_empty_values_and_snapshot():
    client = AsyncTransferQueueClient.__new__(AsyncTransferQueueClient)
    client.client_id = "tags-test"
    client._controller = object()
    client.close = lambda: None
    metadata = BatchMeta(
        global_indexes=[7, 7, 7, 8],
        partition_ids=["p", "q", "p", "p"],
        custom_meta=[{"version": 1}, {"nested": {"values": [2]}}, {"version": 3}, {}],
    )
    captured = {}

    async def request(**kwargs):
        captured.update(kwargs)
        metadata.custom_meta[1]["nested"]["values"].append(99)

    client._request_controller = request
    await client.async_set_custom_meta.__wrapped__(client, metadata, socket=object())
    assert captured["request_type"] == ZMQRequestType.SET_CUSTOM_META
    assert captured["body"] == {
        "partition_custom_meta": {"p": {7: {"version": 3}, 8: {}}, "q": {7: {"nested": {"values": [2]}}}}
    }


@pytest.mark.asyncio
async def test_empty_tag_batch_does_not_send_an_update():
    client = AsyncTransferQueueClient.__new__(AsyncTransferQueueClient)
    client.client_id = "tags-test"
    client._controller = object()
    client.close = lambda: None
    client._request_controller = AsyncMock()
    await client.async_set_custom_meta.__wrapped__(client, BatchMeta.empty(), socket=object())
    client._request_controller.assert_not_awaited()
