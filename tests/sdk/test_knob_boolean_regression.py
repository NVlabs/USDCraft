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

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow  # every test here builds real CadQuery geometry


def _run_bounded_geometry(source: str) -> None:
    # Native OCC cleanup does not yield to Python. Kill a regression in the child
    # before its repeated allocations can exhaust the machine running pytest.
    script = (
        """
import faulthandler
import cadquery as cq
import trimesh
from sdk import KnobGeometry, KnobGrip, KnobIndicator
from sdk._core._mesh.cadquery_helpers import _cut_with_pattern
faulthandler.dump_traceback_later(20, exit=True)
"""
        + source
        + "\nfaulthandler.cancel_dump_traceback_later()\n"
    )
    # A fresh HOME gives the child an empty parametric-mesh disk cache, so the native build runs.
    with tempfile.TemporaryDirectory() as home:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "HOME": home},
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("center", [True, False])
def test_small_filleted_scalloped_knob_is_closed_and_preserves_features(center: bool) -> None:
    _run_bounded_geometry(f"""
k = KnobGeometry(
    .027, .010, body_style='domed', crown_radius=.002, edge_radius=.001,
    grip=KnobGrip(style='scalloped', count=12, depth=.001),
    indicator=KnobIndicator(style='dot', mode='raised', depth=.0007),
    center={center!r},
)
m = trimesh.Trimesh(k.vertices, k.faces, process=True)
assert m.is_watertight and m.is_winding_consistent
assert len(m.split()) == 1
assert 5.4e-6 < m.volume < 5.6e-6
z0 = -.005 if {center!r} else 0.0
assert abs(m.bounds[0, 2] - z0) < 1e-7
assert abs(m.bounds[1, 2] - (z0 + .0108)) < 1e-7
# Scallops remain on the bottom edge; the raised dot remains above the cap.
bottom = m.vertices[abs(m.vertices[:, 2] - z0) < 1e-7]
radii = (bottom[:, 0] ** 2 + bottom[:, 1] ** 2) ** .5
assert radii.min() < .0126 and radii.max() > .0134
""")


def test_invalid_loft_cut_returns_repair_feedback_before_native_cleanup() -> None:
    _run_bounded_geometry("""
w = cq.Workplane('XY').workplane(offset=-.005).circle(.0135)
last = -.005
for z in [-.0032, -.0008, .0022, .005]:
    w = w.workplane(offset=z-last).circle(.0135)
    last = z
shape = w.loft().faces('>Z').edges().fillet(.0016)
r = .0135 * .18 * .55
cutter = (cq.Workplane('XY').circle(r).extrude(.020, both=True)
          .translate((.0135+r-.001, 0, 0)))
try:
    _cut_with_pattern(shape, [cutter])
except ValueError as exc:
    assert 'pattern cut 1' in str(exc) and 'before cleanup' in str(exc)
else:
    raise AssertionError('invalid native Boolean result was accepted')
""")
