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


def normalize_reasoning_level(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Codex reasoning effort must be a string.")
    normalized = value.strip()
    if normalized.lower() == "med":
        normalized = "medium"
    if not normalized:
        raise ValueError("Codex reasoning effort must not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("Codex reasoning effort must not contain control characters.")
    return normalized
