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

import numpy as np

from usdcraft.reconstruction import diagnostics


def _stack_text() -> str:
    return """AXIAL
profile: agent_core_surface
micro_grid192: dims_xyz=4,4,192 pitch=0.01 grid_min_xyz=0,0,0
[SLICE_STACK4 axis=Z dims_xyz=2,2,4 pitch=0.48 grid_min_xyz=0,0,0 slice_order=-Z->+Z rows=+Y->-Y cols=-X->+X]
slice_runs: half-open indices in slice_order
run a: slices=0:4

[PATTERN a]
~.
.~
"""


def test_stack_parser_and_reencode_keep_unknown_extra_as_soft_warning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    axial = tmp_path / "input.txt"
    rebuild = tmp_path / "candidate.usda"
    axial.write_text(_stack_text(), encoding="utf-8")
    rebuild.write_text("# test placeholder\n", encoding="utf-8")
    parsed = diagnostics.parse_axial_stack(_stack_text())
    candidate = parsed.volume.copy()
    positive = np.argwhere(candidate != 0)
    unknown = np.argwhere(candidate == 0)
    candidate[tuple(positive[0])] = 0
    candidate[tuple(unknown[0])] = 1

    monkeypatch.setattr(diagnostics, "load_normalized_source", lambda _source: object())
    monkeypatch.setattr(diagnostics, "_scene_components", lambda _scene: ())

    def fake_build(_components, frame):  # type: ignore[no-untyped-def]
        return np.zeros(tuple(int(value) for value in frame.dimensions), dtype=np.uint8), ()

    monkeypatch.setattr(
        diagnostics,
        "_build_micro_evidence",
        fake_build,
    )
    monkeypatch.setattr(
        diagnostics,
        "_quantize_volume",
        lambda _volume, _micro_frame, _target_frame: candidate,
    )

    result = diagnostics.check_axial_reencode(axial, rebuild)

    assert int(np.count_nonzero(parsed.volume)) == 8
    assert result["diagnostic_only"] is True
    assert result["host_gate"] is False
    assert result["evidence_mode"] == "surface_only"
    assert result["evidence_comparison"]["missing_cell_count"] == 1
    limitations = "\n".join(result["limitations"])
    assert "cannot classify or approve a localized source-artifact exception" in limitations
    assert "at most one AXIAL-driven repair pass" in limitations
    assert "does not treat nonzero recall as a completion or promotion blocker" in limitations
    assert "Do not reintroduce a documented bounded artifact" in limitations
    extra = result["evidence_comparison"]["candidate_only_evidence"]
    assert extra["cell_count"] == 1
    assert extra["severity"] == "soft_warning"
    profiles = result["per_slice_profiles"]["by_slice_axis"]
    assert set(profiles) == {"X", "Y", "Z"}
    assert profiles["Z"]["coordinate_summaries"]["Y"]["semantics"] == {
        "min": "-Y",
        "max": "+Y",
    }
    envelope = result["envelope_comparison"]
    assert envelope["pitch_m"] == 0.48
    assert set(envelope["axes"]) == {"X", "Y", "Z"}
    for axis_entry in envelope["axes"].values():
        assert axis_entry["input_span_m"] is not None
        assert axis_entry["candidate_short_low_m"] >= 0.0
        assert axis_entry["candidate_short_high_m"] >= 0.0
    hotspot_with_missing = next(
        hotspot
        for axis in "XYZ"
        for hotspot in profiles[axis]["hotspots"]
        if hotspot["missing_plane_bounds_m"] is not None
    )
    for span in hotspot_with_missing["missing_plane_bounds_m"].values():
        assert span[0] < span[1]


def test_parser_rejects_non_surface_glyphs() -> None:
    text = _stack_text().replace("~.", "#.", 1)

    try:
        diagnostics.parse_axial_stack(text)
    except ValueError as exc:
        assert "unsupported cell glyph '#'" in str(exc)
    else:
        raise AssertionError("AXIAL accepted a non-surface glyph")


def test_parser_requires_the_exact_axial_header() -> None:
    text = _stack_text().replace("AXIAL\n", "AXIAL 2\n", 1)

    try:
        diagnostics.parse_axial_stack(text)
    except ValueError as exc:
        assert "requires AXIAL text" in str(exc)
    else:
        raise AssertionError("AXIAL parser accepted a versioned header")
