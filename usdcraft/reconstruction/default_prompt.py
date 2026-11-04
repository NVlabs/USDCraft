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

DEFAULT_RECONSTRUCTION_PROMPT = (
    "Reconstruct the primary subject object shown in the semantic image and the source mesh as a "
    "complete, functionally usable, simulator-ready 3D asset."
)


def resolve_reconstruction_prompt(
    *,
    prompt: str | None,
) -> str:
    """Return explicit request text or the deterministic default Reconstruction request."""

    explicit = str(prompt or "").strip()
    if explicit:
        return explicit
    return DEFAULT_RECONSTRUCTION_PROMPT
