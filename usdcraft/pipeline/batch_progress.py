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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ProviderBatchProgressKind = Literal[
    "batch_started",
    "row_started",
    "row_progress",
    "row_finished",
    "batch_finished",
]


@dataclass(slots=True, frozen=True)
class ProviderBatchProgressEvent:
    """One presentation-only observation from Provider Batch execution.

    ``timestamp`` is supplied by the Batch coordinator and is expected to be timezone-aware.
    Every other field is optional because each event kind carries only the data applicable to it.
    The event deliberately has no prompt, image, tool arguments/results, or provider error-body
    field.
    """

    kind: ProviderBatchProgressKind
    timestamp: datetime
    total: int | None = None
    effective_concurrency: int | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    slot: int | None = None
    row_index: int | None = None
    row_id: str | None = None
    message: str | None = None
    status: str | None = None
    record_id: str | None = None
    error_type: str | None = None
    succeeded: int | None = None
    failed: int | None = None


ProviderBatchProgressCallback = Callable[[ProviderBatchProgressEvent], None]


def emit_provider_batch_progress(
    callback: ProviderBatchProgressCallback | None,
    event: ProviderBatchProgressEvent,
) -> None:
    """Report optional Batch presentation progress without affecting Generation."""

    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        # Batch progress is a presentation surface, never scheduling or promotion authority.
        pass


__all__ = [
    "ProviderBatchProgressCallback",
    "ProviderBatchProgressEvent",
    "ProviderBatchProgressKind",
    "emit_provider_batch_progress",
]
