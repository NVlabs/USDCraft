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

"""Read-only projections over recorded Reconstruction timing."""

from datetime import datetime
from typing import Any


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo is not None else None
    except ValueError:
        return None


def recorded_duration(runtime: dict[str, Any], created_at: object) -> float | None:
    """Approximate execution to record creation, excluding queue/final persistence tail."""
    end = _timestamp(created_at)
    stages = runtime.get("stage_timings")
    starts = [
        start
        for stage in (stages if isinstance(stages, list) else [])
        if isinstance(stage, dict)
        and (start := _timestamp(stage.get("started_at_utc"))) is not None
    ]
    if end is not None and starts:
        duration = (end - min(starts)).total_seconds()
        if duration >= 0:
            return duration
    return None
