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

"""AXIAL BRIEF: the Author-facing digest of ``AXIAL`` for the Provider arm.

Five Provider runs showed what an Author actually takes from the full AXIAL text: the metric
envelope and ground plane, the primary body's extent per height, and the controls that need a
metric anchor. The raster stack and its glyph patterns are consumed only by ``check_axial``.
The brief is derived from the full text at workspace staging time (nothing new is measured), so
the full ``axial.txt`` stays the Host-side evidence of record and the Codex arm's input.
"""

from __future__ import annotations

import re

import numpy as np

from usdcraft.reconstruction.axial import support_reading
from usdcraft.reconstruction.diagnostics import parse_axial_stack
from usdcraft.reconstruction.provider_axial_audit import MicroIsland, parse_micro_slabs

_WALL_ROW = re.compile(
    r"^z\[(?P<z0>[-+\d.]+),(?P<z1>[-+\d.]+)\] "
    r"x_wall\[(?P<xw0>[-+\d.]+),(?P<xw1>[-+\d.]+)\] x_ext\[(?P<xe0>[-+\d.]+),(?P<xe1>[-+\d.]+)\] "
    r"y_wall\[(?P<yw0>[-+\d.]+),(?P<yw1>[-+\d.]+)\] y_ext\[(?P<ye0>[-+\d.]+),(?P<ye1>[-+\d.]+)\] "
    r"cover=(?P<cover>\d+)%(?P<round> section=round)?$"
)


def _wall_rows(full_text: str) -> dict[str, dict[str, str]]:
    """``BODY_WALLS_Z`` rows keyed by their ``z[a,b]`` text, plus the ``support:`` line."""

    rows: dict[str, dict[str, str]] = {}
    for line in _section(full_text, "[BODY_WALLS_Z"):
        match = _WALL_ROW.match(line.strip())
        if match:
            rows[f"z[{match.group('z0')},{match.group('z1')}]"] = match.groupdict()
        elif line.startswith("support:"):
            rows["support"] = {"line": support_reading(line)}
    return rows


BRIEF_HEADER_KEYS = (
    "profile",
    "pose",
    "frame",
    "semantic_front",
    "bounds_min_xyz",
    "bounds_max_xyz",
    "extent_xyz",
    "interior",
    "absence",
)


def _section(text: str, header_prefix: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith(header_prefix):
            inside = True
        elif inside and line.startswith("["):
            break
        if inside:
            out.append(line)
    return out


def render_axial_brief(full_text: str) -> str:
    """Render the brief: header, one body line per slab (wall, ext, centre, yaw, cover, others),
    support, FACE_FEATURES, FACE_OPENINGS, INTERIOR_ISLANDS, MEMBER_TRACKS."""

    lines = ["AXIAL BRIEF (derived from AXIAL; every value in recon_world q=0 metres)"]
    try:
        parsed = parse_axial_stack(full_text)
    except ValueError as exc:
        # A text without a decodable stack (fixtures, foreign inputs) still gets its header and
        # any FACE_FEATURES/MEMBER_TRACKS sections; the body-island digest needs the micro frame.
        for raw in full_text.splitlines():
            if raw.split(":", 1)[0] in BRIEF_HEADER_KEYS:
                lines.append(raw)
        lines.append("")
        lines.append(f"[BODY_Z unavailable] {exc}")
        for prefix in (
            "[BODY_WALLS_Z",
            "[FACE_FEATURES",
            "[FACE_OPENINGS",
            "[INTERIOR_ISLANDS",
            "[MEMBER_TRACKS192",
        ):
            section = _section(full_text, prefix)
            if section:
                lines.append("")
                lines.extend(
                    support_reading(line) if line.startswith("support:") else line
                    for line in section
                )
        return "\n".join(line.rstrip() for line in lines).rstrip("\n") + "\n"
    frame = parsed.micro_frame
    origin = np.asarray(frame.bounds_min, dtype=float)
    pitch = float(frame.pitch)
    for raw in full_text.splitlines():
        key = raw.split(":", 1)[0]
        if key in BRIEF_HEADER_KEYS:
            lines.append(raw)
    lines.append("")
    walls = _wall_rows(full_text)
    lines.append(
        "[BODY_Z metres per stack slab; body = the largest sampled-surface island of the slab; "
        "wall = median of the slab's column extremes on each side, held to the whole face's plane where "
        "an open door or lid spreads them; ext = bbox reach; other "
        "islands are sampled surfaces beside the largest island at that height]"
    )
    lines.append(
        "Use wall estimates as the starting dimensions for matching body faces; ext includes "
        "projecting parts. Check the image and source views before assigning a slab/island to a "
        "part: open flaps, rims and scan smear can contaminate walls. section=round gives a "
        "diameter, not straight walls. centre/yaw anchor placement and orientation; preserve "
        "measured asymmetry instead of making repeated parts identical. cover is sampled XY "
        "coverage, not solidity; islands are surfaces, not Parts. Remeasure disputed regions."
    )
    support = walls.get("support")
    if support:
        lines.append(support["line"])
    for slab in parse_micro_slabs(full_text):
        z0 = origin[2] + slab.micro_z0 * pitch
        z1 = origin[2] + slab.micro_z1 * pitch
        if not slab.islands:
            continue
        body = max(slab.islands, key=lambda i: (i.x1 - i.x0) * (i.y1 - i.y0))
        x0, x1 = origin[0] + body.x0 * pitch, origin[0] + body.x1 * pitch
        y0, y1 = origin[1] + body.y0 * pitch, origin[1] + body.y1 * pitch
        others = len(slab.islands) - 1
        body_yaw = _island_yaw_deg(body.contour)
        z_key = f"z[{z0:+.3f},{z1:+.3f}]"
        wall = walls.get(z_key)
        if wall is not None:
            extent_text = (
                f"wall x[{wall['xw0']},{wall['xw1']}] y[{wall['yw0']},{wall['yw1']}] "
                f"ext x[{wall['xe0']},{wall['xe1']}] y[{wall['ye0']},{wall['ye1']}]"
            )
            tail = f" cover={wall['cover']}%" + (" section=round" if wall["round"] else "")
        else:
            extent_text = f"body x[{x0:+.3f},{x1:+.3f}] y[{y0:+.3f},{y1:+.3f}]"
            tail = ""
        lines.append(
            f"{z_key} {extent_text} "
            f"centre({(x0 + x1) / 2:+.3f},{(y0 + y1) / 2:+.3f}) size={(x1 - x0) * 1000:.0f}x"
            f"{(y1 - y0) * 1000:.0f}mm"
            + (f" yaw={body_yaw:.0f}deg" if body_yaw is not None else "")
            + tail
            + f" others={others}"
        )
        described = _describe_other_islands(slab.islands, body, origin=origin, pitch=pitch)
        if described:
            lines.append("  others: " + "; ".join(described))
    features = _section(full_text, "[FACE_FEATURES")
    lines.append("")
    if features:
        lines.extend(features)
    else:
        lines.append(
            "[FACE_FEATURES none detected] no compact protrusion separated on any face; absence "
            "never proves a face flat — the semantic image still decides which controls exist."
        )
    openings = _section(full_text, "[FACE_OPENINGS")
    if openings:
        lines.append("")
        lines.append(
            "Opening bounds describe detected depth regions: verify the physical rim before using them as the full cavity/tray size; visible depth is a lower bound."
        )
        lines.extend(openings)
    interior = _section(full_text, "[INTERIOR_ISLANDS")
    if interior:
        lines.append("")
        lines.extend(interior)
    tracks = _section(full_text, "[MEMBER_TRACKS192")
    if tracks:
        lines.append("")
        lines.extend(tracks)
    return "\n".join(line.rstrip() for line in lines).rstrip("\n") + "\n"


OTHER_ISLAND_MIN_AREA_SHARE = 0.10
OTHER_ISLANDS_LISTED = 4


def _island_yaw_deg(contour: tuple[tuple[int, int], ...]) -> float | None:
    """Yaw of the outline's long axis (degrees from +X about +Z, in [0, 180)) or None."""

    if len(contour) < 3:
        return None
    points = np.asarray(contour, dtype=float)
    centred = points - points.mean(axis=0)
    cov = centred.T @ centred
    if not np.all(np.isfinite(cov)):
        return None
    values, vectors = np.linalg.eigh(cov)
    if values[-1] <= 0.0 or values[-1] < 1.5 * max(values[0], 1e-12):
        # No dominant axis (a square or round footprint): yaw is meaningless.
        return None
    axis = vectors[:, -1]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)


def _describe_other_islands(
    islands: tuple[MicroIsland, ...], body: MicroIsland, *, origin: np.ndarray, pitch: float
) -> list[str]:
    body_area = max((body.x1 - body.x0) * (body.y1 - body.y0), 1)
    others = [
        island
        for island in islands
        if island is not body
        and (island.x1 - island.x0) * (island.y1 - island.y0)
        >= OTHER_ISLAND_MIN_AREA_SHARE * body_area
    ]
    others.sort(key=lambda i: -(i.x1 - i.x0) * (i.y1 - i.y0))
    out: list[str] = []
    for island in others[:OTHER_ISLANDS_LISTED]:
        x0, x1 = origin[0] + island.x0 * pitch, origin[0] + island.x1 * pitch
        y0, y1 = origin[1] + island.y0 * pitch, origin[1] + island.y1 * pitch
        text = (
            f"x[{x0:+.3f},{x1:+.3f}] y[{y0:+.3f},{y1:+.3f}] "
            f"centre({(x0 + x1) / 2:+.3f},{(y0 + y1) / 2:+.3f}) "
            f"size={(x1 - x0) * 1000:.0f}x{(y1 - y0) * 1000:.0f}mm"
        )
        yaw = _island_yaw_deg(island.contour)
        if yaw is not None:
            text += f" yaw={yaw:.0f}deg"
        out.append(text)
    return out


def body_islands_in_band(full_text: str, z_band: tuple[float, float]) -> list[dict[str, object]]:
    """Body island extents (metres) of the stack slabs overlapping ``z_band``."""

    parsed = parse_axial_stack(full_text)
    origin = np.asarray(parsed.micro_frame.bounds_min, dtype=float)
    pitch = float(parsed.micro_frame.pitch)
    rows: list[dict[str, object]] = []
    for slab in parse_micro_slabs(full_text):
        z0 = origin[2] + slab.micro_z0 * pitch
        z1 = origin[2] + slab.micro_z1 * pitch
        if z1 < z_band[0] or z0 > z_band[1] or not slab.islands:
            continue
        body = max(slab.islands, key=lambda i: (i.x1 - i.x0) * (i.y1 - i.y0))
        rows.append(
            {
                "z_m": [round(z0, 4), round(z1, 4)],
                "body_x_m": [
                    round(origin[0] + body.x0 * pitch, 4),
                    round(origin[0] + body.x1 * pitch, 4),
                ],
                "body_y_m": [
                    round(origin[1] + body.y0 * pitch, 4),
                    round(origin[1] + body.y1 * pitch, 4),
                ],
                "other_islands": len(slab.islands) - 1,
            }
        )
    return rows


__all__ = ["BRIEF_HEADER_KEYS", "body_islands_in_band", "render_axial_brief"]
