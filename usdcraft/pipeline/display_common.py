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

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

PROVIDER_SUCCESS_STATUSES = frozenset(
    {"complete", "generation_ready", "reconstruction_ready", "success", "succeeded"}
)
# Product labels accepted by the shared Provider/Codex batch display.
PROVIDER_PROGRESS_LABELS = (
    "Provider Generation",
    "Provider Reconstruction",
    "Codex Reconstruction",
    "Reconstruction Codex",
)
PROVIDER_SAFE_PROGRESS_FIELDS = frozenset(
    {
        "activity",
        "cycle",
        "detail",
        "duration",
        "effort",
        "elapsed",
        "error_type",
        "max_turns",
        "model",
        "record_id",
        "revision",
        "run_id",
        "stage",
        "status",
        "tokens",
        "tool",
        "tool_calls",
        "tool_duration",
        "tools",
        "turn",
        "turns",
    }
)


def single_line(value: object | None) -> str:
    """Return display text with terminal-control characters removed."""

    text = "" if value is None else str(value)
    return "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " " for character in text
    ).strip()


def quoted(value: object | None) -> str:
    return json.dumps(single_line(value), ensure_ascii=False)


def timestamp(value: datetime) -> str:
    aware = value
    if aware.tzinfo is None or aware.utcoffset() is None:
        # Progress is presentation-only. Repair a malformed observer timestamp instead of making
        # it execution authority.
        aware = aware.replace(tzinfo=timezone.utc)
    return aware.isoformat(timespec="seconds")


def environment_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def safe_provider_message(message: str | None, *, label: str | None = None) -> str:
    """Project progress onto safe fields, using the display's product label when supplied."""

    normalized = single_line(message)
    pieces = [piece.strip() for piece in normalized.split("|")]
    display_label = label if label in PROVIDER_PROGRESS_LABELS else None
    if not pieces or pieces[0] not in PROVIDER_PROGRESS_LABELS:
        return f'{display_label or "Provider Generation"} | activity="running"'

    safe_pieces = [display_label or pieces[0]]
    for piece in pieces[1:]:
        key, separator, _value = piece.partition("=")
        if separator and key.strip() in PROVIDER_SAFE_PROGRESS_FIELDS:
            safe_pieces.append(piece)
    if len(safe_pieces) == 1:
        safe_pieces.append('activity="running"')
    return " | ".join(safe_pieces)


def provider_message_fields(message: str | None) -> dict[str, str]:
    """Return raw values for the safe fields in one projected Provider progress message."""

    fields: dict[str, str] = {}
    for piece in safe_provider_message(message).split("|")[1:]:
        key, separator, value = piece.strip().partition("=")
        if separator and key in PROVIDER_SAFE_PROGRESS_FIELDS:
            fields[key] = value.strip()
    return fields


_TURN_RE = re.compile(r"^(?P<turn>[0-9]+)/(?P<maximum>[0-9]+)$")


@dataclass(slots=True, frozen=True)
class ProgressReading:
    """The display-relevant fields of one sanitized Provider progress message."""

    activity: str
    status: str
    stage: str
    turn: int | None
    max_turns: int | None
    model: str
    effort: str


def read_progress_message(message: str | None) -> ProgressReading:
    """Decode the activity, status, stage, turn and runtime fields of one progress message."""

    fields = provider_message_fields(safe_provider_message(message))
    turn: int | None = None
    max_turns: int | None = None
    turn_match = _TURN_RE.fullmatch(decoded_provider_field(fields.get("turn")))
    if turn_match is not None:
        turn = int(turn_match.group("turn"))
        maximum = int(turn_match.group("maximum"))
        max_turns = maximum if maximum > 0 else None
    else:
        parsed = decoded_provider_field(fields.get("max_turns"))
        if parsed.isdigit() and int(parsed) > 0:
            max_turns = int(parsed)
    return ProgressReading(
        activity=decoded_provider_field(fields.get("activity")).casefold(),
        status=decoded_provider_field(fields.get("status")).casefold(),
        stage=decoded_provider_field(fields.get("stage")),
        turn=turn,
        max_turns=max_turns,
        model=decoded_provider_field(fields.get("model")),
        effort=decoded_provider_field(fields.get("effort")),
    )


def decoded_provider_field(value: str | None) -> str:
    """Decode ordinary JSON/Python-quoted scalar fields without accepting structured payloads."""

    normalized = single_line(value)
    if not normalized:
        return ""
    if normalized[:1] in {"'", '"'} and normalized[-1:] == normalized[:1]:
        try:
            decoded = ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            decoded = normalized[1:-1]
        if isinstance(decoded, (str, int, float, bool)):
            return single_line(decoded)
        return ""
    return normalized


__all__ = [
    "PROVIDER_SAFE_PROGRESS_FIELDS",
    "PROVIDER_SUCCESS_STATUSES",
    "ProgressReading",
    "decoded_provider_field",
    "read_progress_message",
    "environment_flag",
    "provider_message_fields",
    "quoted",
    "safe_provider_message",
    "single_line",
    "timestamp",
]
