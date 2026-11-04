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

from usdcraft.reconstruction.axial_summary import (
    AXIAL_DIAGNOSTIC_SCHEMA,
    summarize_axial_payloads,
    summarize_axial_result,
)


def _payload(recall: float, *, ok: bool = True, revision: int = 1) -> dict:
    return {
        "ok": ok,
        "revision": revision,
        "output": {
            "text": "",
            "result": {
                "evidence_comparison": {
                    "recall": recall,
                    "input_cell_count": 200,
                    "matched_cell_count": 150,
                    "missing_cell_count": 50,
                    "candidate_only_evidence": {"cell_count": 9},
                },
                "provider_audit": {
                    "exterior_excess": {"excess_cells": 3, "fraction_of_candidate": 0.05},
                    "surface_fit": {
                        "candidate_to_source_p50_m": 0.01,
                        "candidate_to_source_p90_m": 0.02,
                        "faces": [{"face": "-Z bottom", "reading": "face plane or scale off"}],
                    },
                    "opening_placement": {
                        "candidate_openings": [],
                        "unmodelled_source_openings": [{}, {}, {}],
                    },
                },
            },
        },
    }


def test_result_summary_copies_recall_excess_fit_and_opening_counts() -> None:
    summary = summarize_axial_result(_payload(0.75)["output"]["result"])
    assert summary["recall"] == 0.75
    assert summary["input_cells"] == 200 and summary["candidate_only_cells"] == 9
    assert summary["exterior_excess_fraction"] == 0.05
    assert summary["surface_fit_p50_mm"] == 10.0 and summary["surface_faces_off_plane"] == 1
    assert summary["unmodelled_source_openings"] == 3 and summary["candidate_openings"] == 0
    assert not any(key.startswith("controls_") for key in summary)


def test_run_summary_keeps_first_recall_last_readings_and_skips_failed_calls() -> None:
    summary = summarize_axial_payloads(
        [
            _payload(0.40, revision=1),
            {"ok": False, "revision": 2, "error": "needs compile"},
            _payload(0.70, revision=3),
        ]
    )
    assert summary is not None
    assert summary["schema"] == AXIAL_DIAGNOSTIC_SCHEMA
    assert summary["calls"] == 2 and summary["revision"] == 3
    assert summary["first_recall"] == 0.40 and summary["recall"] == 0.70
    assert summarize_axial_payloads([]) is None
    assert summarize_axial_payloads([{"ok": False}]) is None
