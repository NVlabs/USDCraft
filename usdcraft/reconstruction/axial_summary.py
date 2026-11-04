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

"""Compact record-level summary of the Provider `check_axial` diagnostic.

The Author's `check_axial` calls return the full AXIAL re-encode diagnostic (recall, exterior
excess, surface fit and opening placement). This module reduces the successful `check_axial`
payloads of one run to a flat, mechanical summary that the Host writes into
`runtime_report.axial_diagnostic` at promotion time, where the Viewer reads it. Every value is
copied or counted from the diagnostic's own output; nothing here judges the reconstruction.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

AXIAL_DIAGNOSTIC_SCHEMA = "axial_diagnostic"
CHECK_AXIAL_TOOL_NAME = "check_axial"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _payload_result(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The diagnostic `result` dict inside one successful `check_axial` tool payload."""

    if payload.get("ok") is not True:
        return None
    output = payload.get("output")
    if not isinstance(output, Mapping):
        return None
    result = output.get("result")
    return result if isinstance(result, Mapping) else None


def summarize_axial_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one diagnostic result to the fields the record and the Viewer carry."""

    if isinstance(result.get("summary"), Mapping):
        return dict(result["summary"])
    comparison = result.get("evidence_comparison")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    candidate_only = comparison.get("candidate_only_evidence")
    candidate_only = candidate_only if isinstance(candidate_only, Mapping) else {}
    audit = result.get("provider_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    depth = audit.get("face_depth")
    depth = depth if isinstance(depth, Mapping) else {}
    # The depth-map audit when present, otherwise the stack-based excess / KD-tree surface fit.
    excess = depth.get("excess") if depth else audit.get("exterior_excess")
    excess = excess if isinstance(excess, Mapping) else {}
    fit = audit.get("surface_fit")
    fit = fit if isinstance(fit, Mapping) else {}
    openings = audit.get("opening_placement")
    openings = openings if isinstance(openings, Mapping) else {}

    if depth:
        faces = [row for row in depth.get("faces", ()) if isinstance(row, Mapping)]
        faces_off = sum(1 for row in faces if row.get("defects"))
        p50 = _number(depth.get("abs_delta_p50_m"))
        p90 = _number(depth.get("abs_delta_p90_m"))
    else:
        faces = [row for row in fit.get("faces", ()) if isinstance(row, Mapping)]
        faces_off = sum(1 for row in faces if "off" in str(row.get("reading", "")))
        p50 = _number(fit.get("candidate_to_source_p50_m"))
        p90 = _number(fit.get("candidate_to_source_p90_m"))
    candidate_openings = openings.get("candidate_openings")
    unmodelled_openings = openings.get("unmodelled_source_openings")
    protrusions = audit.get("feature_placement")
    protrusions = protrusions if isinstance(protrusions, Mapping) else {}
    matched_protrusions = protrusions.get("matched")
    unbuilt_protrusions = protrusions.get("unbuilt_source")
    return {
        "recall": _number(comparison.get("recall")),
        "input_cells": _integer(comparison.get("input_cell_count")),
        "matched_cells": _integer(comparison.get("matched_cell_count")),
        "missing_cells": _integer(comparison.get("missing_cell_count")),
        "candidate_only_cells": _integer(candidate_only.get("cell_count")),
        "exterior_excess_cells": _integer(excess.get("columns", excess.get("excess_cells"))),
        "exterior_excess_fraction": _number(
            excess.get("share_of_candidate_columns", excess.get("fraction_of_candidate"))
        ),
        "surface_fit_p50_mm": round(p50 * 1000, 1) if p50 is not None else None,
        "surface_fit_p90_mm": round(p90 * 1000, 1) if p90 is not None else None,
        "surface_faces_off_plane": faces_off if faces else None,
        "candidate_openings": (
            len(candidate_openings) if isinstance(candidate_openings, list) else None
        ),
        "unmodelled_source_openings": (
            len(unmodelled_openings) if isinstance(unmodelled_openings, list) else None
        ),
        "protrusions_matched": (
            len(matched_protrusions) if isinstance(matched_protrusions, list) else None
        ),
        "unbuilt_source_protrusions": (
            len(unbuilt_protrusions) if isinstance(unbuilt_protrusions, list) else None
        ),
    }


def summarize_axial_payloads(
    payloads: Iterable[Mapping[str, Any]], *, final_binding: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Summarize a run's successful `check_axial` payloads in call order; None when there were none."""

    ordered: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for payload in payloads:
        result = _payload_result(payload)
        if result is not None:
            ordered.append((payload, result))
    if not ordered:
        return None
    first_payload, first_result = ordered[0]
    last_payload, last_result = ordered[-1]
    binding = last_payload.get("binding") or {}
    status = "unverified"
    if (
        final_binding is not None
        and binding.get("candidate_sha256")
        and binding.get("source_sha256")
    ):
        status = "current" if binding == final_binding else "stale"
    summary = summarize_axial_result(last_result)
    first_recall = summarize_axial_result(first_result).get("recall")
    return {
        "schema": AXIAL_DIAGNOSTIC_SCHEMA,
        "binding": binding or None,
        "binding_status": status,
        "calls": len(ordered),
        "revision": _integer(last_payload.get("revision")),
        "first_recall": first_recall,
        **summary,
    }


__all__ = [
    "AXIAL_DIAGNOSTIC_SCHEMA",
    "CHECK_AXIAL_TOOL_NAME",
    "summarize_axial_payloads",
    "summarize_axial_result",
]
