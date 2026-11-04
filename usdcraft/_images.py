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

from PIL import Image


def validate_decodable_image(path: Path, *, label: str) -> None:
    """Reject a file that Pillow cannot verify and fully decode as a raster image."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    try:
        # ``verify`` checks container integrity without decoding pixels. Reopen and ``load`` so the
        # public input contract also rejects failures that appear only during pixel decoding.
        with Image.open(resolved) as image:
            image.verify()
        with Image.open(resolved) as image:
            image.load()
    except Exception as exc:
        raise ValueError(f"{label} is invalid: {resolved}") from exc


__all__ = ["validate_decodable_image"]
