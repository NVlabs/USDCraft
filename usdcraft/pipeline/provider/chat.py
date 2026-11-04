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

"""Chat Completions wire mapping for the shared canonical tool history."""

from __future__ import annotations

from typing import Any


def chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if payload.get("instructions"):
        messages.append({"role": "system", "content": payload["instructions"]})
    native_assistant = False
    for item in payload["input"]:
        if item.get("type") == "chat_assistant_history":
            messages.append(item["message"])
            native_assistant = True
            continue
        kind = item.get("type")
        if native_assistant and kind in {"function_call", "message", "reasoning"}:
            continue
        if kind == "function_call_output":
            native_assistant = False
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
            )
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }
                    ],
                }
            )
        elif "role" in item:
            native_assistant = False
            content = item.get("content", "")
            if isinstance(content, list):
                converted = []
                for part in content:
                    if part["type"] in {"input_text", "output_text"}:
                        converted.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "input_image":
                        image = {"url": part["image_url"]}
                        if "detail" in part:
                            image["detail"] = part["detail"]
                        converted.append({"type": "image_url", "image_url": image})
                    else:
                        raise ValueError("Unsupported Chat Completions content type")
                content = converted
            messages.append({"role": item["role"], "content": content})
        else:
            raise ValueError("Unsupported Chat Completions history item")
    request = {
        "model": payload["model"],
        "messages": messages,
        "tools": [
            {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
            for tool in payload["tools"]
        ],
        "parallel_tool_calls": False,
    }
    effort = payload.get("reasoning", {}).get("effort")
    if effort and effort != "provider-default":
        request["reasoning_effort"] = effort
    return request


def chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected one Chat Completions choice")
    choice = choices[0]
    message = choice["message"]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected Chat Completions assistant message")
    # Preserve provider reasoning/signatures verbatim for replay; the harness consumes
    # only the normalized function_call/message items following this history item.
    output = [{"type": "chat_assistant_history", "message": message}]
    for call in message.get("tool_calls") or []:
        fn = call["function"]
        if not all(isinstance(v, str) and v for v in [call["id"], fn["name"], fn["arguments"]]):
            raise ValueError("Malformed Chat Completions function call")
        output.append(
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": fn["name"],
                "arguments": fn["arguments"],
            }
        )
    if message.get("content"):
        if not isinstance(message["content"], str):
            raise ValueError("Expected Chat Completions text content")
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message["content"]}],
            }
        )
    usage = payload.get("usage") or {}
    status = "completed" if choice.get("finish_reason") in {"stop", "tool_calls"} else "incomplete"
    return {
        "id": payload.get("id"),
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": usage.get("prompt_tokens_details", {}),
            "output_tokens_details": usage.get("completion_tokens_details", {}),
        },
    }
