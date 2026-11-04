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

"""Provider accounting metadata; never changes or retains request bodies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def decode_usage(raw: Any) -> dict[str, int]:
    """Keep reported counters, including explicit zero; absent is unknown."""
    if not isinstance(raw, dict):
        return {}
    paths = {
        "input_tokens": ("input_tokens",),
        "output_tokens": ("output_tokens",),
        "total_tokens": ("total_tokens",),
        "cached_input_tokens": ("input_tokens_details", "cached_tokens"),
        "cache_write_input_tokens": ("input_tokens_details", "cache_write_tokens"),
        "reasoning_output_tokens": ("output_tokens_details", "reasoning_tokens"),
    }
    usage: dict[str, int] = {}
    for name, path in paths.items():
        value: Any = raw
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if type(value) is int and value >= 0:
            usage[name] = value
    return usage


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _content_sizes(item: Mapping[str, Any]) -> dict[str, int]:
    sizes = dict(text_bytes=0, image_count=0, image_url_bytes=0, encrypted_reasoning_bytes=0)

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for entry in value:
                visit(entry)
        elif isinstance(value, dict):
            if value.get("type") == "input_image":
                sizes["image_count"] += 1
            for key, child in value.items():
                if isinstance(child, str):
                    if key in {"text", "arguments", "output", "content"}:
                        sizes["text_bytes"] += len(child.encode("utf-8"))
                    elif key == "image_url":
                        sizes["image_url_bytes"] += len(child.encode("utf-8"))
                    elif key == "encrypted_content":
                        sizes["encrypted_reasoning_bytes"] += len(child.encode("utf-8"))
                elif isinstance(child, (list, dict)):
                    visit(child)

    visit(item)
    return sizes


def request_composition(
    history: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    previous_item_count: int,
) -> dict[str, Any]:
    """Measure actual model-visible JSON, not the sometimes larger local tool trace.

    Replayed means sent in an earlier request, NOT a provider cache hit. New item
    descriptors are emitted once; their indexes bind subsequent aggregate counts.
    """
    allowed_names = {tool.get("name") for tool in tools}
    call_names = {
        item.get("call_id"): item.get("name")
        for item in history
        if item.get("type") == "function_call" and item.get("name") in allowed_names
    }
    groups: dict[str, dict[str, int]] = {}
    partitions: dict[str, dict[str, int]] = {"replayed": {}, "new": {}}
    new_items = []
    item_bytes = 0
    for index, item in enumerate(history):
        kind = item.get("type")
        if index == 0:
            group = "initial_task"
        elif kind == "function_call_output":
            group = "tool_result:" + str(call_names.get(item.get("call_id"), "unknown"))
        elif kind == "function_call":
            group = "tool_call:" + str(call_names.get(item.get("call_id"), "unknown"))
        elif kind == "reasoning":
            group = "reasoning"
        elif item.get("role") == "assistant":
            group = "assistant_message"
        elif item.get("role") == "user":
            group = "user_feedback"
        else:
            group = "other"
        encoded = _encoded(item)
        sizes = {"item_count": 1, "item_json_bytes": len(encoded), **_content_sizes(item)}
        item_bytes += len(encoded)
        partition = "replayed" if index < previous_item_count else "new"
        for bucket in (groups.setdefault(group, {}), partitions[partition]):
            for key, value in sizes.items():
                bucket[key] = bucket.get(key, 0) + value
        if partition == "new":
            new_items.append(
                {
                    "index": index,
                    "group": group,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    **sizes,
                }
            )
    schema = _encoded(list(tools))
    return {
        "version": 1,
        "input_item_count": len(history),
        "input_json_bytes": item_bytes + 2 + max(0, len(history) - 1),
        "tools_json_bytes": len(schema),
        "tools_sha256": hashlib.sha256(schema).hexdigest(),
        "tool_schema_count": len(tools),
        "groups": groups,
        "partitions": partitions,
        "new_items": new_items,
    }
