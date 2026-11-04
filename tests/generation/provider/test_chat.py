# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from usdcraft.pipeline.provider.client import (
    NvidiaHttpResponse,
    NvidiaProtocolError,
    NvidiaProviderConfig,
    NvidiaResponsesClient,
)


@pytest.mark.parametrize("effort", ["provider-default", "high"])
def test_chat_multitool_history_images_and_usage(effort):
    requests = []
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "retained reasoning",
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "inspect",
                    "arguments": json.dumps({"index": i}),
                },
            }
            for i in range(2)
        ],
    }

    def transport(url, *, headers, body, timeout_seconds):
        requests.append((url, json.loads(body)))
        return NvidiaHttpResponse(
            200,
            {},
            json.dumps(
                {
                    "id": "response",
                    "choices": [{"finish_reason": "tool_calls", "message": assistant}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                }
            ).encode(),
        )

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test", api_style="chat-completions", reasoning_effort=effort),
        transport=transport,
    )
    initial = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "inspect"},
            {"type": "input_image", "image_url": "data:image/png;base64,A", "detail": "high"},
        ],
    }
    tools = [{"type": "function", "name": "inspect", "parameters": {"type": "object"}}]
    a = client.create_response(input_items=[initial], tools=tools)
    assert a.usage["input_tokens"] == 10
    assert a.usage["cached_input_tokens"] == 3
    assert a.usage["reasoning_output_tokens"] == 2
    assert [x["call_id"] for x in a.output if x["type"] == "function_call"] == ["call_0", "call_1"]
    history = [
        initial,
        *a.output,
        *[
            {"type": "function_call_output", "call_id": f"call_{i}", "output": "ok"}
            for i in range(2)
        ],
        initial,
    ]
    client.create_response(input_items=history, tools=tools)
    url, payload = requests[-1]
    assert url.endswith("/chat/completions")
    assert ("reasoning_effort" in payload) == (effort == "high")
    assert "reasoning" not in payload and "include" not in payload
    assert payload["messages"][1] == assistant
    assert [x["role"] for x in payload["messages"]] == ["user", "assistant", "tool", "tool", "user"]
    assert payload["messages"][-1]["content"][1]["image_url"]["detail"] == "high"


def test_chat_length_is_incomplete_and_bad_payload_is_protocol_error():
    def client_returning(body: bytes) -> NvidiaResponsesClient:
        return NvidiaResponsesClient(
            NvidiaProviderConfig(api_key="test", api_style="chat-completions"),
            transport=lambda *_args, **_kwargs: NvidiaHttpResponse(200, {}, body),
        )

    d = {
        "choices": [
            {"finish_reason": "length", "message": {"role": "assistant", "content": "partial"}}
        ]
    }
    response = client_returning(json.dumps(d).encode()).create_response(input_items=[], tools=[])
    assert response.status == "incomplete"
    with pytest.raises(NvidiaProtocolError):
        client_returning(b'{"choices": []}').create_response(input_items=[], tools=[])


def test_responses_provider_default_omits_reasoning():
    def transport(url, *, headers, body, timeout_seconds):
        payload = json.loads(body)
        assert url.endswith("/responses")
        assert "reasoning" not in payload and "include" not in payload
        return NvidiaHttpResponse(200, {}, b'{"status":"completed","output":[]}')

    client = NvidiaResponsesClient(
        NvidiaProviderConfig(api_key="test", reasoning_effort="provider-default"),
        transport=transport,
    )
    client.create_response(input_items=[], tools=[])
