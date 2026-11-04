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

from pathlib import Path

# Image-only Generation asks for the depicted subject; Reconstruction's default request names the
# same subject together with its source mesh.
DEFAULT_IMAGE_PROMPT = (
    "Create the primary subject object shown in the attached image as a complete, "
    "functionally usable, simulator-ready 3D asset."
)


def resolve_generation_prompt(
    *,
    prompt: str | None,
    image_path: Path | None,
) -> str:
    """Return explicit text or the deterministic image-only Generation prompt."""

    explicit = str(prompt or "").strip()
    if explicit:
        return explicit
    if image_path is None:
        raise ValueError("Generation requires an explicit text prompt or one inspiration image.")
    return DEFAULT_IMAGE_PROMPT


__all__ = [
    "DEFAULT_IMAGE_PROMPT",
    "resolve_generation_prompt",
]
