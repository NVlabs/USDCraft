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

"""Authoring-only AXIAL diagnostics for the Codex Reconstruction Author (shell commands).

``recon-check-axial`` re-encodes one compiled candidate against the AXIAL text. With ``--source``
(the Host's normalized source USD) it prints the same mesh-to-mesh audit — numbered discrepancies first,
then the envelope, face-depth and opening readings, recall last — that the Provider Author's
``check_axial`` tool returns, and with ``--payload-dir`` it also writes the structured payload of
each call so the Host can reduce the run's calls to ``runtime_report.axial_diagnostic`` exactly as
it does for the Provider arm. ``recon-measure-face`` answers the Provider ``measure_face`` question
from the shell. Neither command is a Host gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from usdcraft.reconstruction.diagnostics import check_axial_reencode

# One-line framing prepended to every audit output: the source mesh is reference evidence; the
# numbered discrepancies are the repair list and recall is only a departure reading.
AXIAL_REFERENCE_HEADER = (
    "reference mesh — resolve each numbered discrepancy: repair supported geometry errors; "
    "verify region/part correspondence before changing shape. Explain retained differences. "
    "Recall is a reference reading, never a target.\n"
)
AXIAL_PAYLOAD_PREFIX = "check_axial_"


def _number(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.3f}"


def format_axial_diagnostic(result: dict[str, Any]) -> str:
    """Return bounded repair feedback without exposing another evidence channel."""

    evidence = result["evidence_comparison"]
    candidate_only = evidence["candidate_only_evidence"]
    coverage = "complete" if evidence["complete"] else "residuals recorded"
    lines = [
        "AXIAL re-encode (authoring-only; not a Host gate)",
        "status: review complete",
        f"sampled-source coverage: {coverage}",
        (
            f"sampled source surfaces: matched={evidence['matched_cell_count']}/"
            f"{evidence['input_cell_count']} recall={_number(evidence['recall'])} "
            f"missing={evidence['missing_cell_count']}"
        ),
        (
            "candidate-only surfaces in UNKNOWN cells: "
            f"{candidate_only['cell_count']} severity={candidate_only['severity']}"
        ),
    ]

    envelope = result.get("envelope_comparison")
    if envelope:
        pitch = envelope.get("pitch_m")
        lines.append("envelope coverage vs AXIAL (meters):")
        for axis in "XYZ":
            entry = envelope["axes"][axis]
            input_span = entry.get("input_span_m")
            if input_span is None:
                continue
            candidate_span = entry.get("candidate_span_m")
            piece = f"  {axis}: input=[{input_span[0]:.3f}..{input_span[1]:.3f}]"
            if candidate_span is None:
                piece += " candidate=absent"
            else:
                piece += f" candidate=[{candidate_span[0]:.3f}..{candidate_span[1]:.3f}]"
                for key, label in (
                    ("candidate_short_low_m", "short_low"),
                    ("candidate_short_high_m", "short_high"),
                ):
                    shortfall = float(entry.get(key) or 0.0)
                    if shortfall > 1e-9:
                        piece += f" {label}={shortfall:.3f}"
                        if pitch and shortfall > float(pitch):
                            piece += " (exceeds one stack pitch)"
            lines.append(piece)

    grid = result.get("grid") or {}
    grid_min = grid.get("bounds_min_xyz_m")
    grid_pitch = grid.get("pitch_m")

    def _indices_with_spans(indices: list[int], axis_index: int) -> str:
        if not indices or grid_min is None or grid_pitch is None:
            return str(indices)
        groups: list[tuple[int, int]] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            groups.append((start, previous))
            start = previous = index
        groups.append((start, previous))
        rendered = []
        for low, high in groups:
            low_m = grid_min[axis_index] + low * grid_pitch
            high_m = grid_min[axis_index] + (high + 1) * grid_pitch
            label = str(low) if low == high else f"{low}-{high}"
            rendered.append(f"{label} ({'xyz'[axis_index]}={low_m:.3f}..{high_m:.3f}m)")
        return "[" + ", ".join(rendered) + "]"

    lines.append("worst slice profiles:")
    for axis in "XYZ":
        axis_index = "XYZ".index(axis)
        profile = result["per_slice_profiles"]["by_slice_axis"][axis]
        lines.append(
            f"  {axis}: missing_slices={_indices_with_spans(profile['missing_input_slices'], axis_index)} "
            f"candidate_only_slices={profile['candidate_only_slices']}"
        )
        for hotspot in profile["hotspots"][:2]:
            piece = (
                f"    slice={hotspot['slice_index']} "
                f"recall={_number(hotspot['positive_recall'])} "
                f"extra={hotspot['extra_cell_count']}"
            )
            slice_range = hotspot.get("slice_range_m")
            if slice_range:
                piece += f" at {'xyz'[axis_index]}={slice_range[0]:.3f}..{slice_range[1]:.3f}m"
            missing_bounds = hotspot.get("missing_plane_bounds_m")
            if missing_bounds:
                piece += " missing_at " + " ".join(
                    f"{plane_axis}[{span[0]:.3f}..{span[1]:.3f}]"
                    for plane_axis, span in missing_bounds.items()
                )
            lines.append(piece)
    lines.extend(
        (
            "repair policy:",
            "  investigate consequential differences, repair supported errors and recheck the final geometry within the run budget;",
            "  record retained differences and the evidence supporting them;",
            "  AXIAL residuals alone never block Author completion or Host promotion;",
            "  candidate-only surfaces remain soft because every input nonsurface cell is UNKNOWN;",
            "  this command cannot classify artifacts; verify correspondence against the source and semantic image;",
            "  recall is a reference reading, never a target or completion threshold.",
        )
    )
    return "\n".join(lines) + "\n"


def axial_audit_report(
    axial_path: Path,
    candidate_usd_path: Path,
    *,
    source_usd_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """The mesh-to-mesh audit text (header, discrepancies, readings, recall) and its structured result.

    The stack re-encode runs first; the audit is added on top of it, and when the audit itself
    cannot run the stack report is returned with a note so a diagnostic edge case never hides all
    feedback. The result is JSON-clean.
    """

    from usdcraft.reconstruction.provider_axial_audit import (
        audit_candidate_against_axial,
        format_provider_audit,
    )

    result: dict[str, Any] = dict(check_axial_reencode(axial_path, candidate_usd_path))
    comparison = result.get("evidence_comparison") or {}
    recall_line = (
        f"recall {float(comparison.get('recall') or 0.0):.3f} "
        f"({comparison.get('matched_cell_count')}/{comparison.get('input_cell_count')} sampled "
        "source cells covered; a reference reading against the reference mesh, never a target)"
    )
    try:
        audit = audit_candidate_against_axial(
            axial_path,
            candidate_usd_path,
            source_usd_path=source_usd_path,
        )
    except Exception as exc:
        text = format_axial_diagnostic(result).rstrip("\n")
        text += f"\nprovider audit unavailable: {type(exc).__name__}: {exc}"
    else:
        result["provider_audit"] = audit
        text = format_provider_audit(audit) + "\n" + recall_line
    return (
        AXIAL_REFERENCE_HEADER + text + "\n",
        json.loads(json.dumps(result, ensure_ascii=False, default=str)),
    )


def write_axial_payload(
    payload_dir: Path, *, text: str, result: dict[str, Any], binding: dict[str, Any] | None = None
) -> Path:
    """Append one ``check_axial``-shaped payload to ``payload_dir`` (Codex arm sidecar).

    The payload has the Provider tool's shape (``ok``, ``axial_call``, ``output.text``,
    ``output.result``; ``revision`` is ``None`` because the shell arm has no staged revision
    counter) so ``summarize_axial_payloads`` reduces both arms identically.
    """

    payload_dir = Path(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    call = len(sorted(payload_dir.rglob(f"{AXIAL_PAYLOAD_PREFIX}*.json"))) + 1
    payload = {
        "call_id": uuid.uuid4().hex,
        "recorded_at_ns": time.time_ns(),
        "binding": binding,
        "ok": True,
        "revision": None,
        "axial_call": call,
        "output": {"text": text, "result": result},
    }
    path = (
        payload_dir / f"{AXIAL_PAYLOAD_PREFIX}{payload['recorded_at_ns']}_{payload['call_id']}.json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Register even when the Author chooses a nested or alternative feedback directory.
    workspace = os.environ.get("USDCRAFT_RECONSTRUCTION_WORKSPACE")
    if workspace:
        registry = Path(workspace) / "axial_checks"
        registry.mkdir(exist_ok=True)
        archive = Path(workspace) / "author_feedback" / "axial_checks"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f"{payload['call_id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        )
        if registry.resolve() != payload_dir.resolve():
            registered = registry / f"{AXIAL_PAYLOAD_PREFIX}{payload['call_id']}.json"
            registered.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_axial_payloads(payload_dir: Path) -> list[dict[str, Any]]:
    """The payloads a Codex run's ``recon-check-axial --payload-dir`` calls left, in call order."""

    payload_dir = Path(payload_dir)
    if not payload_dir.is_dir():
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(payload_dir.rglob(f"{AXIAL_PAYLOAD_PREFIX}*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    payloads.sort(key=lambda p: (p.get("recorded_at_ns", 0), p.get("axial_call", 0)))
    unique: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        call_id = payload.get("call_id")
        if isinstance(call_id, str) and call_id:
            unique[call_id] = payload
    return list(unique.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("axial", type=Path)
    parser.add_argument("candidate_usd", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Host-side normalized source USD. With it the command prints the mesh-to-mesh audit "
            "(numbered discrepancies, envelope, face depth, openings, recall) the Provider check_axial "
            "tool returns; without it, the stack re-encode report."
        ),
    )
    parser.add_argument(
        "--payload-dir",
        type=Path,
        default=None,
        help="Directory receiving one structured payload per call (Host reduces them at promotion).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.source is not None:
            text, result = axial_audit_report(
                args.axial, args.candidate_usd, source_usd_path=args.source
            )
        else:
            result = check_axial_reencode(args.axial, args.candidate_usd)
            text = format_axial_diagnostic(result)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    workspace = os.environ.get("USDCRAFT_RECONSTRUCTION_WORKSPACE")
    payload_dir = args.payload_dir or (Path(workspace) / "axial_checks" if workspace else None)
    if payload_dir is not None:
        from usdcraft.reconstruction.diagnostic_binding import diagnostic_binding

        write_axial_payload(
            payload_dir,
            text=text,
            binding=diagnostic_binding(args.source, args.candidate_usd),
            result=json.loads(json.dumps(result, ensure_ascii=False, default=str)),
        )
    if args.json:
        payload = {**result, "review_completed": True}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(text, end="")
    return 0


def measure_main(argv: list[str] | None = None) -> int:
    """``recon-measure-face``: the Provider ``measure_face`` question from the shell."""

    from usdcraft.reconstruction.face_measure import FACE_NAMES, measure_face_payload

    parser = argparse.ArgumentParser(
        description="Per-face FACE_FEATURES / FACE_OPENINGS readings (and body islands of a z band)."
    )
    parser.add_argument("axial", type=Path)
    parser.add_argument(
        "--face",
        required=True,
        choices=FACE_NAMES,
        help="Axis face; write the equals form (`--face=-Y`) so a negative axis is not an option.",
    )
    parser.add_argument(
        "--z-band",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="Also list the body island extents of the stack slabs inside this z band (metres).",
    )
    args = parser.parse_args(argv)
    try:
        text = Path(args.axial).read_text(encoding="utf-8")
        band = (min(args.z_band), max(args.z_band)) if args.z_band is not None else None
        payload = measure_face_payload(text, face=args.face, z_band=band)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AXIAL_PAYLOAD_PREFIX",
    "AXIAL_REFERENCE_HEADER",
    "axial_audit_report",
    "format_axial_diagnostic",
    "main",
    "measure_main",
    "read_axial_payloads",
    "write_axial_payload",
]
