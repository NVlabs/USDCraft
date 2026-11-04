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

"""Fixed camera contract for Generation render diagnostics."""

from __future__ import annotations

SELF_CHECK_RENDER_SIZE = 512
STANDARD_RENDER_VIEWS: tuple[tuple[str, float, float], ...] = (
    ("level_front", 0.0, 8.0),
    ("level_back", 180.0, 8.0),
    ("level_right", 90.0, 8.0),
    ("level_left", -90.0, 8.0),
    ("elevated_3q", 45.0, 35.0),
    ("top_down", 15.0, 88.0),
    ("underside_3q", 225.0, -30.0),
)

__all__ = [
    "SELF_CHECK_RENDER_SIZE",
    "STANDARD_RENDER_VIEWS",
]
