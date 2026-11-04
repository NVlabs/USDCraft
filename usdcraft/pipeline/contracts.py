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

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from usdcraft.values import normalize_reasoning_level


class CodexReasoningEffort(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ULTRA = "ultra"
    MAX = "max"


class CodexCompletionStatus(StrEnum):
    COMPLETE = "complete"
    BLOCKED = "blocked"


def _normalized_nonempty(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


@dataclass(slots=True, frozen=True)
class CodexRuntimeSelection:
    model_id: str
    reasoning_effort: str
    runtime_source: str
    selection_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _normalized_nonempty(self.model_id, label="Codex model"),
        )
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_reasoning_level(
                _normalized_nonempty(self.reasoning_effort, label="Codex reasoning effort")
            ),
        )
        object.__setattr__(
            self,
            "runtime_source",
            _normalized_nonempty(self.runtime_source, label="Codex runtime source"),
        )
        object.__setattr__(
            self,
            "selection_source",
            _normalized_nonempty(self.selection_source, label="Codex selection source"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "runtime_source": self.runtime_source,
            "selection_source": self.selection_source,
        }


@dataclass(slots=True, frozen=True)
class CodexFinalMessage:
    status: CodexCompletionStatus
    summary: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CodexFinalMessage:
        status_value = payload.get("status")
        if not isinstance(status_value, str):
            raise ValueError("Codex final response status must be 'complete' or 'blocked'.")
        raw_status = status_value.strip().lower()
        try:
            status = CodexCompletionStatus(raw_status)
        except ValueError as exc:
            raise ValueError(
                "Codex final response status must be 'complete' or 'blocked'."
            ) from exc
        summary_value = payload.get("summary")
        if not isinstance(summary_value, str):
            raise ValueError("Codex final response summary is required.")
        summary = summary_value.strip()
        if not summary:
            raise ValueError("Codex final response summary is required.")
        return cls(status=status, summary=summary)

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status.value, "summary": self.summary}


@dataclass(slots=True, frozen=True)
class CodexRunResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    final_message: CodexFinalMessage


@dataclass(slots=True, frozen=True)
class ProviderRuntimeSelection:
    model_id: str
    reasoning_effort: str
    runtime_source: str = "openai-responses-api"
    selection_source: str = "repository_env_or_provider_defaults"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_id", _normalized_nonempty(self.model_id, label="Provider model")
        )
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_reasoning_level(
                _normalized_nonempty(self.reasoning_effort, label="Provider reasoning effort")
            ),
        )
        object.__setattr__(
            self,
            "runtime_source",
            _normalized_nonempty(self.runtime_source, label="Provider runtime source"),
        )
        object.__setattr__(
            self,
            "selection_source",
            _normalized_nonempty(self.selection_source, label="Provider selection source"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "runtime_source": self.runtime_source,
            "selection_source": self.selection_source,
        }
