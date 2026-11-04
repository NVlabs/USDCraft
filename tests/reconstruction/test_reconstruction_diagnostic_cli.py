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

import json
from pathlib import Path

import pytest

from usdcraft.reconstruction import diagnostic_cli


def _result(*, complete: bool) -> dict[str, object]:
    return {
        "schema": "axial_reencode_diagnostic",
        "evidence_comparison": {
            "complete": complete,
            "matched_cell_count": 9 if complete else 8,
            "input_cell_count": 9,
            "missing_cell_count": 0 if complete else 1,
            "recall": 1.0 if complete else 8 / 9,
            "candidate_only_evidence": {"cell_count": 3, "severity": "soft_warning"},
        },
        "grid": {
            "dimensions_xyz": [8, 8, 8],
            "pitch_m": 0.1,
            "bounds_min_xyz_m": [-0.4, -0.4, -0.4],
            "bounds_max_xyz_m": [0.4, 0.4, 0.4],
        },
        "envelope_comparison": {
            "pitch_m": 0.1,
            "axes": {
                "X": {
                    "input_span_m": [-0.4, 0.4],
                    "candidate_span_m": [-0.4, 0.4],
                    "candidate_short_low_m": 0.0,
                    "candidate_short_high_m": 0.0,
                },
                "Y": {
                    "input_span_m": [-0.4, 0.4],
                    "candidate_span_m": [-0.4, 0.4] if complete else [-0.25, 0.4],
                    "candidate_short_low_m": 0.0 if complete else 0.15,
                    "candidate_short_high_m": 0.0,
                },
                "Z": {
                    "input_span_m": [-0.4, 0.4],
                    "candidate_span_m": [-0.4, 0.4],
                    "candidate_short_low_m": 0.0,
                    "candidate_short_high_m": 0.0,
                },
            },
        },
        "per_slice_profiles": {
            "by_slice_axis": {
                axis: {
                    "missing_input_slices": [] if complete else [2, 3],
                    "candidate_only_slices": [4],
                    "hotspots": []
                    if complete
                    else [
                        {
                            "slice_index": 2,
                            "positive_recall": 0.5,
                            "extra_cell_count": 1,
                            "slice_range_m": [-0.2, -0.1],
                            "missing_plane_bounds_m": {"X": [-0.3, 0.1], "Z": [-0.2, 0.0]},
                        }
                    ],
                }
                for axis in "XYZ"
            }
        },
    }


def test_text_result_records_residual_without_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        diagnostic_cli, "check_axial_reencode", lambda *_args: _result(complete=False)
    )
    assert diagnostic_cli.main([str(tmp_path / "axial.txt"), str(tmp_path / "model.usdc")]) == 0
    output = capsys.readouterr().out
    # A residual is reported as data (the recorded missing slices), never as a failing exit.
    assert "missing_slices=" in output


@pytest.mark.parametrize("complete", [False, True])
def test_json_reports_completed_review_without_pass_fail_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete: bool,
) -> None:
    monkeypatch.setattr(
        diagnostic_cli, "check_axial_reencode", lambda *_args: _result(complete=complete)
    )
    assert (
        diagnostic_cli.main([str(tmp_path / "axial.txt"), str(tmp_path / "model.usdc"), "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["review_completed"] is True
    assert "passed" not in payload
    assert payload["evidence_comparison"]["complete"] is complete


def test_invalid_input_is_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object) -> None:
        raise ValueError("bad AXIAL")

    monkeypatch.setattr(diagnostic_cli, "check_axial_reencode", fail)
    assert diagnostic_cli.main([str(tmp_path / "axial.txt"), str(tmp_path / "model.usdc")]) == 2
    assert "bad AXIAL" in capsys.readouterr().err


def test_payload_dir_records_each_call_in_the_provider_tool_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic_cli,
        "check_axial_reencode",
        lambda *_args, **_kwargs: _result(complete=False),
    )
    payload_dir = tmp_path / "axial_checks"
    for _ in range(2):
        assert (
            diagnostic_cli.main(
                [
                    str(tmp_path / "axial.txt"),
                    str(tmp_path / "model.usdc"),
                    "--payload-dir",
                    str(payload_dir),
                ]
            )
            == 0
        )

    payloads = diagnostic_cli.read_axial_payloads(payload_dir)
    assert [payload["axial_call"] for payload in payloads] == [1, 2]
    assert all(payload["ok"] is True and payload["revision"] is None for payload in payloads)
    assert payloads[0]["output"]["result"]["evidence_comparison"]["recall"] == 8 / 9
    assert "recall=0.889" in payloads[0]["output"]["text"]
    assert len(list(payload_dir.glob("check_axial_*.json"))) == 2
    assert diagnostic_cli.read_axial_payloads(tmp_path / "missing") == []


def test_source_option_prints_the_mesh_to_mesh_audit_with_the_reference_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_audit(axial_path, candidate_usd_path, *, source_usd_path=None):  # type: ignore[no-untyped-def]
        observed["source"] = source_usd_path
        return {"defects": [], "face_depth": {"faces": []}}

    monkeypatch.setattr(
        diagnostic_cli,
        "check_axial_reencode",
        lambda *_args, **_kwargs: _result(complete=True),
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_axial_audit.audit_candidate_against_axial", fake_audit
    )
    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_axial_audit.format_provider_audit",
        lambda audit: "DEFECTS: none",
    )

    assert (
        diagnostic_cli.main(
            [
                str(tmp_path / "axial.txt"),
                str(tmp_path / "model.usdc"),
                "--source",
                str(tmp_path / "normalized.usdc"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.startswith(diagnostic_cli.AXIAL_REFERENCE_HEADER)
    assert "DEFECTS: none" in output
    assert observed["source"] == tmp_path / "normalized.usdc"


def test_measure_face_command_answers_the_provider_tool_question(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    axial = tmp_path / "axial.txt"
    axial.write_text(
        "AXIAL\nprofile: agent_core_surface\n"
        "micro_grid192: dims_xyz=1,1,1 pitch=1 grid_min_xyz=0,0,0\n"
        "[SLICE_STACK1 axis=Z dims_xyz=1,1,1 pitch=1 grid_min_xyz=0,0,0 "
        "slice_order=-Z->+Z rows=+Y->-Y cols=-X->+X]\n"
        "slice_runs: half-open indices in slice_order\nrun a: slices=0:1\n\n[PATTERN a]\n~\n",
        encoding="utf-8",
    )

    assert diagnostic_cli.measure_main([str(axial), "--face=-Y", "--z-band", "0.2", "0.0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["face"] == "-Y"
    assert payload["protrusions"] == [] and payload["openings"] == []
    assert "body_islands" in payload
    assert "semantic image decides existence" in payload["note"]

    with pytest.raises(SystemExit) as excinfo:
        diagnostic_cli.measure_main([str(axial), "--face=front"])
    assert excinfo.value.code == 2
    # The CLI entry accepts the same equals form for a negative axis.
    from usdcraft.cli import main as cli_main

    assert cli_main(["recon-measure-face", str(axial), "--face=-X"]) == 0
    assert json.loads(capsys.readouterr().out)["face"] == "-X"
