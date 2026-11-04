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

"""Axis-face depth grid shared by the FACE_FEATURES and FACE_OPENINGS detectors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

FACE_FEATURE_FACES: tuple[tuple[str, int, int, int, int], ...] = (
    # name, depth axis, outward sign, u axis, v axis
    ("-Y", 1, -1, 0, 2),
    ("+Y", 1, +1, 0, 2),
    ("-X", 0, -1, 1, 2),
    ("+X", 0, +1, 1, 2),
    ("+Z", 2, +1, 0, 1),
    ("-Z", 2, -1, 0, 1),
)
FACE_AXIS_LABELS = {"-Y": "x,z", "+Y": "x,z", "-X": "y,z", "+X": "y,z", "+Z": "x,y", "-Z": "x,y"}

MIN_PITCH_M = 0.002
PITCH_DIVISOR = 400  # pitch = max(MIN_PITCH_M, longest extent / PITCH_DIVISOR)
MIN_FACE_CELLS = 50
SILHOUETTE_MARGIN_M = 0.012
SAMPLING_GAP_MAX_M2 = 1.5e-4
SAMPLES_PER_CELL = 4
SAMPLE_MIN = 400_000
SAMPLE_MAX = 3_000_000


def sample_surface_points(components: Sequence[Any], *, target: int, seed: int = 0) -> np.ndarray:
    """Area-weighted deterministic surface samples plus every vertex of the components."""

    rng = np.random.default_rng(seed)
    loose: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    for component in components:
        vertices = np.asarray(component.vertices, dtype=float)
        faces = np.asarray(component.faces, dtype=int)
        if len(faces) == 0:
            loose.append(vertices)
            continue
        triangles.append(vertices[faces])
    if not triangles:
        return np.vstack(loose) if loose else np.zeros((0, 3))
    tris = np.vstack(triangles)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return np.vstack([tris.reshape(-1, 3), *loose])
    counts = np.maximum(1, np.round(area / total * target)).astype(int)
    index = np.repeat(np.arange(len(tris)), counts)
    u = rng.random(len(index))
    v = rng.random(len(index))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    samples = tris[index, 0] + (tris[index, 1] - tris[index, 0]) * u[:, None]
    samples += (tris[index, 2] - tris[index, 0]) * v[:, None]
    return np.vstack([samples, tris.reshape(-1, 3), *loose])


def extent_and_pitch(components: Sequence[Any]) -> tuple[np.ndarray, float] | None:
    """Axis extent of all component vertices and the grid pitch; ``None`` without vertices."""

    vertices = np.vstack([np.asarray(c.vertices, dtype=float) for c in components])
    if len(vertices) == 0:
        return None
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    pitch = max(MIN_PITCH_M, float(extent.max()) / PITCH_DIVISOR)
    return extent, pitch


def rasterize_face(
    points: np.ndarray, axis: int, sign: int, u_axis: int, v_axis: int, pitch: float
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """First-surface depth map of one axis face as ``(depth, valid, u0, v0)``.

    ``depth`` holds the outward-most sample per cell (``-inf`` where no sample fell); ``None``
    when fewer than ``MIN_FACE_CELLS`` cells are covered.
    """

    u0 = float(points[:, u_axis].min())
    v0 = float(points[:, v_axis].min())
    nu = int((points[:, u_axis].max() - u0) / pitch) + 1
    nv = int((points[:, v_axis].max() - v0) / pitch) + 1
    iu = ((points[:, u_axis] - u0) / pitch).astype(int)
    iv = ((points[:, v_axis] - v0) / pitch).astype(int)
    outward = points[:, axis] * sign
    depth = np.full((nu, nv), -np.inf)
    np.maximum.at(depth, (iu, iv), outward)
    valid = np.isfinite(depth)
    if valid.sum() < MIN_FACE_CELLS:
        return None
    return depth, valid, u0, v0


def silhouette_margin_cells(pitch: float) -> int:
    """Erosion depth, in cells, of the silhouette margin kept clear of face detections."""

    return max(2, int(round(SILHOUETTE_MARGIN_M / pitch)))


def section_byte_count(lines: Sequence[str]) -> int:
    """UTF-8 size of a serialized section, one newline per line; 0 for an empty section."""

    return sum(len(line.encode("utf-8")) + 1 for line in lines) if lines else 0
