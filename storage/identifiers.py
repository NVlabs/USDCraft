# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import re

RECORD_ID_RE = re.compile(r"^rec_[A-Za-z0-9_.-]+$")


def validate_record_id(value: str | None, *, label: str = "record_id") -> str:
    record_id = str(value or "").strip()
    if not record_id:
        raise ValueError(f"{label} is required.")
    if not RECORD_ID_RE.fullmatch(record_id):
        raise ValueError(
            f"{label} must start with rec_ and contain only letters, digits, "
            "underscores, periods, or hyphens."
        )
    return record_id
