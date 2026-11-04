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

"""DecalGeometry: a thin textured quad that carries a surface graphic.

A decal is a flat (or cylindrically curved) quad with per-vertex UVs and a
reference to a TextureCanvas. It is purely visual — place it slightly proud of
the host surface; do not use it as collision geometry.

Local frame:
- Flat decal: lies in the XY plane, centered at the origin, front face toward
  +Z. ``width`` spans X, ``height`` spans Y. UVs use glTF's bottom-left
  convention (``v=1`` at the visual top), so the canvas appears upright when
  viewed from +Z.
- Curved decal (``curvature_radius`` > 0): the width wraps around a cylinder of
  that radius whose axis is Z (matching the SDK's Z-axis ``Cylinder``); ``height``
  runs along Z and the front face points radially outward (-Y at the center).
  Place it on the host body's axis so it hugs the surface — an upright cylinder
  needs no tilt; rotate about Z (yaw) to aim the label around the body.

A small ``proud_offset`` is baked along the outward normal so a flush placement
does not z-fight the surface underneath.
"""

from __future__ import annotations

import io
from math import cos, isfinite, sin
from typing import List, Optional

from ..types import Mesh
from .primitives import MeshGeometry

_DEFAULT_PROUD = 0.0005  # 0.5 mm


class DecalGeometry(MeshGeometry):
    """Textured label quad of ``width`` x ``height`` facing +Z, ``proud_offset`` in front of z=0.

    With ``curvature_radius > 0`` the width wraps as arc length around a Z-axis cylinder of that
    radius, facing outward (-Y at the center).
    """

    def __init__(
        self,
        texture: object,
        width: float,
        height: float,
        *,
        curvature_radius: float = 0.0,
        segments: int = 16,
        proud_offset: float = _DEFAULT_PROUD,
    ):
        super().__init__()
        width = float(width)
        height = float(height)
        if not isfinite(width) or not isfinite(height) or width <= 0.0 or height <= 0.0:
            raise ValueError("DecalGeometry width/height must be finite and > 0")
        curvature_radius = float(curvature_radius)
        if not isfinite(curvature_radius) or curvature_radius < 0.0:
            raise ValueError("curvature_radius must be finite and >= 0")
        proud_offset = float(proud_offset)
        if not isfinite(proud_offset):
            raise ValueError("proud_offset must be finite")
        if curvature_radius > 0.0 and curvature_radius + proud_offset <= 0.0:
            raise ValueError("curvature_radius + proud_offset must be positive")

        half_h = height / 2.0
        vertices: List = []
        uvs: List = []

        if curvature_radius == 0.0:
            # Flat quad in XY, front face +Z, pushed slightly proud along +Z.
            half_w = width / 2.0
            z = proud_offset
            corners = [
                (-half_w, half_h, z, 0.0, 1.0),  # top-left
                (half_w, half_h, z, 1.0, 1.0),  # top-right
                (-half_w, -half_h, z, 0.0, 0.0),  # bottom-left
                (half_w, -half_h, z, 1.0, 0.0),  # bottom-right
            ]
            for x, y, zz, u, v in corners:
                vertices.append((x, y, zz))
                uvs.append((u, v))
            # CCW when viewed from +Z.
            faces = [(0, 2, 3), (0, 3, 1)]
        else:
            # Wrap width around a cylinder of `curvature_radius` about the Z axis, matching
            # the SDK's Z-axis Cylinder/CylinderGeometry. Height runs along Z; the front face
            # points radially outward (-Y at the center). Placed on the host body axis, an
            # upright cylinder needs no tilt — rotate about Z (yaw) to aim the label around
            # the body. (This is a rigid +90 deg rotation about X of the former Y-axis wrap.)
            segments = max(2, int(segments))
            radius = curvature_radius + proud_offset
            # `width` is the arc length along the circumference: angle = width / radius.
            half_angle = (width / curvature_radius) / 2.0
            cols = []
            for i in range(segments + 1):
                t = i / segments
                angle = -half_angle + t * (2.0 * half_angle)
                x = radius * sin(angle)
                y = -radius * cos(angle)
                cols.append((x, y, t))
            for x, y, u in cols:
                vertices.append((x, y, half_h))
                uvs.append((u, 1.0))
                vertices.append((x, y, -half_h))
                uvs.append((u, 0.0))
            faces = []
            for i in range(segments):
                tl = 2 * i
                bl = 2 * i + 1
                tr = 2 * (i + 1)
                br = 2 * (i + 1) + 1
                # Front face points radially outward (-Y at center); CCW seen from outside.
                faces.append((tl, bl, br))
                faces.append((tl, br, tr))

        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.texture = texture
        self.validate()


def build_decal_visual_mesh(
    canvas: object,
    *,
    width: float,
    height: float,
    curvature_radius: float = 0.0,
    proud_offset: float = _DEFAULT_PROUD,
    assets: object = None,
) -> Mesh:
    """Render a TextureCanvas, build a textured GLB for the decal quad, register
    it as a managed asset, and return a Mesh referencing the .glb.

    The texture is a derived artifact (regenerated from the canvas code at
    compile time), consistent with how geometry meshes are materialized.
    """
    import numpy as np

    from ..assets import get_active_asset_session
    from ..texture import TextureCanvas
    from ..viewer_assets import build_textured_glb_bytes

    if not isinstance(canvas, TextureCanvas):
        raise ValueError("decal expects a TextureCanvas")

    geom = DecalGeometry(
        canvas,
        width=width,
        height=height,
        curvature_radius=curvature_radius,
        proud_offset=proud_offset,
    )
    image = canvas.render()
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    alpha_mode = "MASK" if alpha is not None and alpha.getextrema()[0] < 255 else "OPAQUE"
    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")

    glb = build_textured_glb_bytes(
        np.asarray(geom.vertices, dtype=np.float32),
        np.asarray(geom.faces, dtype=np.uint32),
        np.asarray(geom.uvs, dtype=np.float32),
        png_buffer.getvalue(),
        alpha_mode=alpha_mode,
        alpha_cutoff=0.5,
    )

    session = get_active_asset_session(create_if_missing=True)
    if session is None:
        raise RuntimeError("Failed to create a managed asset session for decal")
    logical_name = f"decal_{canvas.content_digest()}"
    info = session.register_decal_glb(logical_name, glb)
    return Mesh(filename=info.ref, name=info.logical_name, materialized_path=info.path.as_posix())
