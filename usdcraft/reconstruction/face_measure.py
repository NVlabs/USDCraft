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

"""Per-face evidence query shared by the Provider `measure_face` tool and `recon-measure-face`.

Both arms ask the same question of the same AXIAL text — every `FACE_FEATURES` protrusion and
`FACE_OPENINGS` recess of one axis face in metres and, for a height band, the body island extents
of the stack slabs inside it — and receive the same payload. Nothing here is a Host gate.
"""

from __future__ import annotations

from typing import Any

FACE_NAMES = ("-Y", "+Y", "-X", "+X", "+Z", "-Z")
MEASURE_NOTE = (
    "centres, sizes and heights are source measurements in metres; absence of a protrusion never "
    "proves the face flat — the semantic image decides existence. Bind centre, footprint and "
    "height separately: retained anchors are measurements; eroded/fused attributes may be lower "
    "bounds or unresolved hypotheses. Missing entries never cancel visible controls, and image "
    "pixels or ratios are not metric measurements."
)


def normalize_z_band(value: object) -> tuple[float, float] | None:
    """Validate an optional ``[lo, hi]`` z band; returns it ordered, or ``None`` when absent."""

    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError("z_band must be two numbers or null.")
    return (float(min(value)), float(max(value)))


def measure_face_payload(
    axial_text: str, *, face: str, z_band: tuple[float, float] | None = None
) -> dict[str, Any]:
    """The per-face evidence payload for ``face`` (one of :data:`FACE_NAMES`)."""

    from usdcraft.reconstruction.axial_brief import body_islands_in_band
    from usdcraft.reconstruction.face_features import parse_face_features
    from usdcraft.reconstruction.face_openings import parse_face_openings

    face = str(face).strip()
    if face not in FACE_NAMES:
        raise ValueError(f"face must be one of {', '.join(FACE_NAMES)}.")
    features = [f.to_dict() for f in parse_face_features(axial_text) if f.face == face]
    openings = [o.to_dict() for o in parse_face_openings(axial_text) if o.face == face]
    payload: dict[str, Any] = {
        "ok": True,
        "face": face,
        "protrusions": features,
        "openings": openings,
        "note": MEASURE_NOTE,
    }
    if z_band is not None:
        try:
            payload["body_islands"] = body_islands_in_band(axial_text, z_band)
        except ValueError as exc:
            payload["body_islands"] = []
            payload["body_islands_note"] = f"unavailable: {exc}"
    return payload


__all__ = ["FACE_NAMES", "MEASURE_NOTE", "measure_face_payload", "normalize_z_band"]
