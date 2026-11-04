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

from usdcraft.reconstruction.axial import support_reading
from usdcraft.reconstruction.axial_brief import render_axial_brief


def test_archived_support_statistics_keep_numbers_without_old_structure_instruction() -> None:
    statistics = (
        "support: contact_cover=3% body_full_width_from_z=+0.021 "
        "verdict=contact_patch_or_unscanned_underside"
    )
    archived = statistics + "; the body underside sits at body_full_width_from_z and feet fill gap"
    reading = support_reading(archived)
    assert reading.split(";", 1)[0] == statistics
    assert support_reading(reading) == reading
    # The incomplete-stack fallback also exposes only the current interpretation.
    brief = render_axial_brief("[BODY_WALLS_Z]\n" + archived + "\n")
    assert reading in brief
    assert archived not in brief
