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

import pytest

from sdk import TextureCanvas, ValidationError


def test_renders_fill_at_expected_pixels() -> None:
    canvas = TextureCanvas(100, 100, background=(1.0, 0.0, 0.0, 1.0))
    canvas.rect((0.0, 0.0, 0.5, 1.0), fill=(0.0, 0.0, 1.0, 1.0))  # left half blue

    img = canvas.render()
    assert img.size == (100, 100)
    assert img.mode == "RGBA"
    assert img.getpixel((25, 50))[:3] == (0, 0, 255)  # left half blue
    assert img.getpixel((75, 50))[:3] == (255, 0, 0)  # right half red background


def test_default_background_is_transparent() -> None:
    canvas = TextureCanvas(40, 40)
    img = canvas.render()
    assert img.getpixel((5, 5))[3] == 0  # alpha 0


def test_text_renders_non_background_pixels() -> None:
    canvas = TextureCanvas(160, 80, background=(1.0, 1.0, 1.0, 1.0))
    canvas.text((0.5, 0.5), "AB", size=0.6, fill=(0.0, 0.0, 0.0, 1.0), anchor="mm", bold=True)

    img = canvas.render().convert("L")
    dark = sum(img.histogram()[:40])  # grayscale pixels with value < 40
    assert dark > 30  # the glyphs produced a meaningful number of dark pixels


def test_paste_composites_subcanvas() -> None:
    icon = TextureCanvas(20, 20, background=(0.0, 1.0, 0.0, 1.0))  # solid green
    canvas = TextureCanvas(100, 100, background=(0.0, 0.0, 0.0, 1.0))
    canvas.paste(icon, (0.4, 0.4, 0.2, 0.2))

    img = canvas.render()
    assert img.getpixel((50, 50))[:3] == (0, 255, 0)  # green icon center
    assert img.getpixel((5, 5))[:3] == (0, 0, 0)  # black background elsewhere


def test_content_digest_is_deterministic_and_sensitive() -> None:
    a = TextureCanvas(64, 64, background=(1, 0, 0, 1))
    a.rect((0.1, 0.1, 0.5, 0.5), fill=(0, 0, 1, 1))
    b = TextureCanvas(64, 64, background=(1, 0, 0, 1))
    b.rect((0.1, 0.1, 0.5, 0.5), fill=(0, 0, 1, 1))
    c = TextureCanvas(64, 64, background=(1, 0, 0, 1))
    c.rect((0.1, 0.1, 0.5, 0.5), fill=(0, 1, 0, 1))  # different color

    assert a.content_digest() == b.content_digest()
    assert a.content_digest() != c.content_digest()


def test_validates_inputs() -> None:
    with pytest.raises(ValidationError, match="width/height"):
        TextureCanvas(4, 4)
    with pytest.raises(ValidationError, match="0..1"):
        TextureCanvas(64, 64, background=(2.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="anchor"):
        TextureCanvas(64, 64).text((0.5, 0.5), "x", anchor="zz")
    with pytest.raises(ValidationError, match="xywh"):
        TextureCanvas(64, 64).rect((0.1, 0.1, 0.0, 0.5), fill=(0, 0, 0, 1))
