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

from usdcraft.compile.feedback import (
    CADQUERY_NULL_SHAPE_HINT,
    CADQUERY_OCC_NOT_DONE_HINT,
    _compile_hint_lines,
    _location_lines_from_traceback_text,
)

_CADQUERY_TRACEBACK = """Traceback (most recent call last):
  File "/repo/data/local/generation_runs/gen_x/model.py", line 1049, in <module>
    object_model = build_object_model()
  File "/repo/data/local/generation_runs/gen_x/model.py", line 233, in build_object_model
    carcass_skin = carcass.visual(mesh_from_cadquery(_carcass_shape(), "carcass_soft_shell"))
  File "/repo/data/local/generation_runs/gen_x/model.py", line 118, in _rounded_box
    return shape.edges("|Z").fillet(radius)
  File "/x/.venv/lib/python3.12/site-packages/cadquery/cq.py", line 3021, in fillet
    s = solid.fillet(radius, edgeList)
  File "/x/.venv/lib/python3.12/site-packages/cadquery/occ_impl/shapes.py", line 399, in __init__
    raise ValueError("Null TopoDS_Shape object")
ValueError: Null TopoDS_Shape object
"""


def test_traceback_location_names_the_innermost_model_frame_before_the_library_frame() -> None:
    lines = _location_lines_from_traceback_text(_CADQUERY_TRACEBACK)
    assert lines == [
        "Model: .../generation_runs/gen_x/model.py:118",
        'Model code: return shape.edges("|Z").fillet(radius)',
        "Location: site-packages/cadquery/occ_impl/shapes.py:399",
        'Code: raise ValueError("Null TopoDS_Shape object")',
    ]


def test_traceback_location_omits_the_model_line_when_the_failure_is_already_in_model_py() -> None:
    text = """Traceback (most recent call last):
  File "/repo/data/local/generation_runs/gen_x/model.py", line 12, in <module>
    value = 1 / 0
ZeroDivisionError: division by zero
"""
    assert _location_lines_from_traceback_text(text) == [
        "Location: .../generation_runs/gen_x/model.py:12",
        "Code: value = 1 / 0",
    ]


def test_cadquery_failures_get_actionable_hints() -> None:
    assert _compile_hint_lines(["ValueError: Null TopoDS_Shape object"]) == [
        CADQUERY_NULL_SHAPE_HINT
    ]
    assert "half the shortest adjacent edge" in CADQUERY_NULL_SHAPE_HINT
    assert _compile_hint_lines(["StdFail_NotDone: BRep_API: command not done"]) == [
        CADQUERY_OCC_NOT_DONE_HINT
    ]
    assert _compile_hint_lines(["NameError: name 'x' is not defined"]) == []
