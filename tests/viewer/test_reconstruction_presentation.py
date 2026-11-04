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

import pytest

from viewer.api.reconstruction_presentation import recorded_duration


def test_duration_includes_preprocessing_and_uses_timezone():
    runtime = {
        "stage_timings": [
            {"started_at_utc": "2026-01-10T00:02:00Z"},
            {"started_at_utc": "2026-01-10T00:00:00Z"},
        ],
    }
    assert recorded_duration(runtime, "2026-01-10T08:10:00+08:00") == 600
    assert recorded_duration(runtime, "bad") is None
    assert recorded_duration(runtime, "2026-01-09T00:00:00Z") is None


@pytest.mark.parametrize("stage_timings", [None, [], [{"started_at_utc": "bad"}], "x"])
def test_missing_or_invalid_time_is_not_zero(stage_timings):
    runtime = {"elapsed_seconds": 300, "stage_timings": stage_timings}
    assert recorded_duration(runtime, "2026-01-10T00:00:00Z") is None
