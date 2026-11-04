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

"""TextureCanvas: author surface graphics (labels, logos, printed warnings,
decorative patterns) as CODE inside model.py.

A TextureCanvas is the SOURCE OF TRUTH for a surface graphic — like geometry,
it is defined by code and re-rendered at compile time into a derived PNG. The
agent edits the label by editing these calls, never by hand-authoring an image.

All drawing coordinates and sizes are NORMALIZED (0..1) so the spec is
independent of the pixel resolution. Colors are display-referred sRGB rgba
floats in 0..1; generated images are exported as sRGB color textures.

Use a TextureCanvas with DecalGeometry to place the graphic on a surface; the
graphic is purely visual (it is not geometry and not a collision shape).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, cast

from .errors import ValidationError

Rgba = Tuple[float, float, float, float]
Vec2 = Tuple[float, float]

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_REGULAR = _FONT_DIR / "DejaVuSans.ttf"
_FONT_BOLD = _FONT_DIR / "DejaVuSans-Bold.ttf"

_ANCHORS = {"lt", "mt", "rt", "lm", "mm", "rm", "lb", "mb", "rb"}


def _norm_rgba(value: Optional[Sequence[float]], *, field: str) -> Optional[Rgba]:
    if value is None:
        return None
    nums = [float(v) for v in value]
    if len(nums) == 3:
        nums.append(1.0)
    if len(nums) != 4:
        raise ValidationError(f"{field} must have 3 or 4 values")
    if any(v < 0.0 or v > 1.0 for v in nums):
        raise ValidationError(f"{field} values must be in 0..1")
    return (nums[0], nums[1], nums[2], nums[3])


def _to_px_rgba(rgba: Rgba) -> Tuple[int, int, int, int]:
    return cast(Tuple[int, int, int, int], tuple(int(round(c * 255)) for c in rgba))


def _stroke_width(line_width: float, width: Optional[float]) -> float:
    """Resolve the stroke width, accepting ``width`` as a PIL-style alias."""
    return float(width if width is not None else line_width)


class TextureCanvas:
    """A normalized 2D drawing surface that renders to an RGBA raster."""

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        *,
        background: Optional[Sequence[float]] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        if self.width < 8 or self.height < 8:
            raise ValidationError("TextureCanvas width/height must be >= 8")
        if self.width > 4096 or self.height > 4096:
            raise ValidationError("TextureCanvas width/height must be <= 4096")
        # Default background is transparent so a decal shows only its graphics.
        self.background: Rgba = _norm_rgba(background, field="background") or (0.0, 0.0, 0.0, 0.0)
        self._ops: List[tuple] = []

    # --- drawing primitives (normalized coords) ---------------------------

    def rect(
        self,
        xywh: Sequence[float],
        *,
        fill: Optional[Sequence[float]] = None,
        outline: Optional[Sequence[float]] = None,
        line_width: float = 0.0,
        width: Optional[float] = None,
    ) -> "TextureCanvas":
        self._ops.append(
            (
                "rect",
                self._xywh(xywh),
                _norm_rgba(fill, field="fill"),
                _norm_rgba(outline, field="outline"),
                _stroke_width(line_width, width),
            )
        )
        return self

    def ellipse(
        self,
        xywh: Sequence[float],
        *,
        fill: Optional[Sequence[float]] = None,
        outline: Optional[Sequence[float]] = None,
        line_width: float = 0.0,
        width: Optional[float] = None,
    ) -> "TextureCanvas":
        self._ops.append(
            (
                "ellipse",
                self._xywh(xywh),
                _norm_rgba(fill, field="fill"),
                _norm_rgba(outline, field="outline"),
                _stroke_width(line_width, width),
            )
        )
        return self

    def line(
        self,
        points: Sequence[Sequence[float]],
        *,
        fill: Sequence[float],
        line_width: float = 0.01,
        width: Optional[float] = None,
    ) -> "TextureCanvas":
        pts = [self._xy(p) for p in points]
        if len(pts) < 2:
            raise ValidationError("line needs at least 2 points")
        self._ops.append(
            ("line", pts, _norm_rgba(fill, field="fill"), _stroke_width(line_width, width))
        )
        return self

    def polygon(
        self,
        points: Sequence[Sequence[float]],
        *,
        fill: Optional[Sequence[float]] = None,
        outline: Optional[Sequence[float]] = None,
        line_width: float = 0.0,
        width: Optional[float] = None,
    ) -> "TextureCanvas":
        pts = [self._xy(p) for p in points]
        if len(pts) < 3:
            raise ValidationError("polygon needs at least 3 points")
        self._ops.append(
            (
                "polygon",
                pts,
                _norm_rgba(fill, field="fill"),
                _norm_rgba(outline, field="outline"),
                _stroke_width(line_width, width),
            )
        )
        return self

    def text(
        self,
        xy: Sequence[float],
        text: str,
        *,
        size: float = 0.1,
        fill: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        anchor: str = "mm",
        bold: bool = False,
    ) -> "TextureCanvas":
        if anchor not in _ANCHORS:
            raise ValidationError(f"anchor must be one of {sorted(_ANCHORS)}")
        if not 0.0 < size <= 1.0:
            raise ValidationError("text size must be in (0, 1]")
        self._ops.append(
            (
                "text",
                self._xy(xy),
                str(text),
                float(size),
                _norm_rgba(fill, field="fill"),
                anchor,
                bool(bold),
            )
        )
        return self

    def paste(self, other: "TextureCanvas", xywh: Sequence[float]) -> "TextureCanvas":
        """Composite another canvas into a sub-rectangle (for repeated motifs)."""
        if not isinstance(other, TextureCanvas):
            raise ValidationError("paste expects a TextureCanvas")
        self._ops.append(("paste", other, self._xywh(xywh)))
        return self

    # --- helpers ----------------------------------------------------------

    def _xy(self, p: Sequence[float]) -> Vec2:
        if len(p) != 2:
            raise ValidationError("point must be [x, y]")
        return (float(p[0]), float(p[1]))

    def _xywh(self, v: Sequence[float]) -> Tuple[float, float, float, float]:
        if len(v) != 4:
            raise ValidationError("xywh must be [x, y, w, h] (normalized)")
        x, y, w, h = (float(c) for c in v)
        if w <= 0.0 or h <= 0.0:
            raise ValidationError("xywh width/height must be > 0")
        return (x, y, w, h)

    def content_digest(self) -> str:
        """Stable hash of the spec (drives content-addressed asset storage)."""
        payload = repr((self.width, self.height, self.background, self._ops))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # --- rendering --------------------------------------------------------

    def render(self):
        """Render to a PIL.Image (RGBA). Imported lazily to keep import light."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (self.width, self.height), _to_px_rgba(self.background))
        draw = ImageDraw.Draw(img)
        w, h = self.width, self.height

        def px(point: Vec2) -> Tuple[float, float]:
            return (point[0] * w, point[1] * h)

        def box(rect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
            x, y, bw, bh = rect
            return (x * w, y * h, (x + bw) * w, (y + bh) * h)

        for op in self._ops:
            kind = op[0]
            if kind in ("rect", "ellipse"):
                _, xywh, fill, outline, lw = op
                kwargs: dict[str, Any] = {}
                if fill is not None:
                    kwargs["fill"] = _to_px_rgba(fill)
                if outline is not None:
                    kwargs["outline"] = _to_px_rgba(outline)
                    kwargs["width"] = max(1, int(round(lw * min(w, h))))
                (draw.rectangle if kind == "rect" else draw.ellipse)(box(xywh), **kwargs)
            elif kind == "line":
                _, pts, fill, lw = op
                draw.line(
                    [px(p) for p in pts],
                    fill=_to_px_rgba(fill),
                    width=max(1, int(round(lw * min(w, h)))),
                )
            elif kind == "polygon":
                _, pts, fill, outline, lw = op
                kwargs = {}
                if fill is not None:
                    kwargs["fill"] = _to_px_rgba(fill)
                if outline is not None:
                    kwargs["outline"] = _to_px_rgba(outline)
                draw.polygon([px(p) for p in pts], **kwargs)
            elif kind == "text":
                _, xy, text, size, fill, anchor, bold = op
                font_path = _FONT_BOLD if bold else _FONT_REGULAR
                font = ImageFont.truetype(str(font_path), max(1, int(round(size * h))))
                draw.text(px(xy), text, font=font, fill=_to_px_rgba(fill), anchor=anchor)
            elif kind == "paste":
                _, other, xywh = op
                sub = other.render()
                x0, y0, x1, y1 = box(xywh)
                target = (max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
                sub = sub.resize(target)
                img.alpha_composite(sub, (int(round(x0)), int(round(y0))))

        return img
