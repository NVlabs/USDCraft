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

"""Reconstruction tools for one Author: bounded SDK access, direct mesh measurement,
compiled candidate images, AXIAL diagnostics and completion. Feedback stays in the Author
conversation; these tools launch no semantic agents and add no Host quality gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from usdcraft.pipeline.tools import (
    ProviderCheckRunner,
    ProviderCompileRunner,
    ProviderToolExecution,
    ProviderToolExecutor,
    ProviderToolResult,
    build_provider_tool_schemas,
)
from usdcraft.reconstruction.diagnostic_binding import diagnostic_binding
from usdcraft.reconstruction.diagnostic_cli import AXIAL_REFERENCE_HEADER, axial_audit_report
from usdcraft.reconstruction.face_measure import (
    FACE_NAMES,
    measure_face_payload,
    normalize_z_band,
)
from usdcraft.reconstruction.workspace import ReconstructionAuthorWorkspace

AXIAL_TOOL_NAME = "check_axial"
FINISH_TOOL_NAME = "finish_reconstruction"
AXIAL_TEXT_LOGICAL_PATH = "inputs/axial.txt"
AXIAL_BRIEF_LOGICAL_PATH = "inputs/axial_brief.txt"
EVIDENCE_MANIFEST_LOGICAL_PATH = "evidence_manifest.json"
MEASURE_TOOL_NAME = "measure_face"


class ProviderAxialRunner(Protocol):
    def __call__(
        self,
        axial_path: Path,
        candidate_usd_path: Path,
        *,
        source_usd_path: Path | None = None,
    ) -> ProviderToolResult: ...


def default_axial_runner(
    axial_path: Path,
    candidate_usd_path: Path,
    *,
    source_usd_path: Path | None = None,
) -> ProviderToolResult:
    """Run the mesh-to-mesh AXIAL audit and return its text and structured result.

    The report is shared with the Codex arm's ``recon-check-axial --source`` command.
    """

    try:
        text, result = axial_audit_report(
            axial_path, candidate_usd_path, source_usd_path=source_usd_path
        )
    except Exception as exc:
        return ProviderToolResult.error(f"{type(exc).__name__}: {exc}")
    return ProviderToolResult.ok({"text": text, "result": result})


def build_reconstruction_tool_schemas() -> tuple[dict[str, Any], ...]:
    """The Reconstruction Author's complete function-tool surface."""

    schemas: list[dict[str, Any]] = []
    for schema in build_provider_tool_schemas():
        if schema["name"] == "read_file":
            schema = dict(schema)
            schema["description"] = (
                "Read the current staged model.py, the staged evidence "
                f"(`{AXIAL_BRIEF_LOGICAL_PATH}`, `{EVIDENCE_MANIFEST_LOGICAL_PATH}`, returned "
                "verbatim without line numbers so grids stay column-aligned), or an exact shared "
                "SDK documentation/example file, or a tool-returned detail_path. Page a long file with offset and limit. "
                "Repository source outside sdk/_docs and sdk/_examples is unavailable."
            )
        schemas.append(schema)
    schemas.append(
        {
            "type": "function",
            "name": AXIAL_TOOL_NAME,
            "description": (
                "Re-encode the current compiled candidate in the AXIAL grid and report coverage, "
                "per-axis envelope comparison, face planes, exterior excess and opening placement "
                "as authoring repair feedback. Requires a successful compile_model for the "
                "current model revision. AXIAL recall is never a completion threshold."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            "strict": True,
        }
    )
    schemas.append(
        {
            "type": "function",
            "name": MEASURE_TOOL_NAME,
            "description": (
                "Query the source mesh evidence for one axis face: every FACE_FEATURES protrusion "
                "(centre, size, height) and every FACE_OPENINGS recess or opening (centre, plan "
                "size, visible depth as a lower bound) on that face in metres and, when a z band is "
                "given, the "
                "body island extents of the stack slabs inside that band. Host-computed from the "
                "same AXIAL evidence; use measure_mesh to independently inspect uncertain detections."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "face": {"type": "string", "enum": list(FACE_NAMES)},
                    "z_band": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["face", "z_band"],
            },
            "strict": True,
        }
    )
    schemas.append(
        {
            "type": "function",
            "name": "compare_part",
            "description": "Compare a named compiled part (optional visual element) at q=0 to an Author-selected source region. Requires compile_model or validate_model to succeed after the latest model.py edit. Returns local surface/profile residuals and an attached projection image; correspondence is your hypothesis. Use for small controls or thin panels whose shape global recall hides.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "part": {"type": "string"},
                    "element": {"type": ["string", "null"]},
                    "face": {"type": "string", "enum": ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]},
                    "bounds": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 6,
                        "maxItems": 6,
                    },
                },
                "required": ["part", "element", "face", "bounds"],
            },
            "strict": True,
        }
    )
    schemas.extend(
        (
            {
                "type": "function",
                "name": "render_source",
                "description": "View the frozen source mesh at a chosen camera when shape or occlusion is unclear. Returns an image attached to your next turn. No candidate or compile is required. Angles are degrees in the normalized metre/Z-up frame; source views are qualitative evidence, not calibrated pixel measurements.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"azimuth": {"type": "number"}, "elevation": {"type": "number"}},
                    "required": ["azimuth", "elevation"],
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "measure_mesh",
                "description": "Read the frozen source mesh directly, independent of AXIAL detections. "
                "Return selected-vertex bounds, quantiles, bounded XYZ points and orthographic images. "
                "Use bounds to inspect a region or thin slice in normalized Z-up metres. "
                "An empty selection does not prove a missing part.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bounds": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 6,
                            "maxItems": 6,
                        },
                        "max_points": {"type": "integer", "minimum": 1, "maximum": 4096},
                    },
                    "required": ["bounds", "max_points"],
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "render_candidate",
                "description": "View the current compiled candidate yourself. Returns an image at "
                "azimuth/elevation degrees; optional joint poses are radians/metres, not physics. "
                "Requires compile_model or validate_model to succeed after the latest model.py edit. "
                "The image is attached automatically to your next turn. "
                "Use this to inspect appearance, geometry and articulated clearances, then repair.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "azimuth": {"type": "number"},
                        "elevation": {"type": "number"},
                        "joints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "position": {"type": "number"},
                                },
                                "required": ["name", "position"],
                            },
                        },
                    },
                    "required": ["azimuth", "elevation", "joints"],
                },
                "strict": True,
            },
        )
    )
    schemas.append(
        {
            "type": "function",
            "name": FINISH_TOOL_NAME,
            "description": (
                "Finish only after authoring is done. `complete` requires successful check_model "
                "and compile_model calls for the current model revision; a remaining AXIAL "
                "residual never blocks completion. `blocked` is reserved for a concrete evidence "
                "or SDK limitation, never a repairable model/check/compile failure. "
                "Report missing or stale inspections from validation_state in your summary."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["complete", "blocked"]},
                    "summary": {"type": "string", "minLength": 1},
                },
                "required": ["status", "summary"],
            },
            "strict": True,
        }
    )
    schemas.append(
        {
            "type": "function",
            "name": "validate_model",
            "description": "Default complete candidate checkpoint: check, compile, run AXIAL and render q=0 in one call. Stops on code/export failure. ok reports mechanical success; diagnostics_complete reports diagnostic execution success, not shape quality. Inspect the attached image and axial results before repairing. Use individual tools for targeted checks or additional views. validation_state reports author revision coverage, not visual acceptance.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            "strict": True,
        }
    )
    return tuple(json.loads(json.dumps(schema)) for schema in schemas)


class ReconstructionToolExecutor(ProviderToolExecutor):
    """Shared Provider tools plus the Reconstruction evidence reads and AXIAL diagnostic.

    Every ``check_axial`` report opens with ``AXIAL_REFERENCE_HEADER``.
    """

    workspace: ReconstructionAuthorWorkspace

    def __init__(
        self,
        workspace: ReconstructionAuthorWorkspace,
        *,
        check_runner: ProviderCheckRunner,
        compile_runner: ProviderCompileRunner,
        axial_runner: ProviderAxialRunner = default_axial_runner,
        source_usd_path: Path | None = None,
    ) -> None:
        super().__init__(
            workspace,
            finish_tool_name=FINISH_TOOL_NAME,
            check_runner=check_runner,
            compile_runner=compile_runner,
            extra_readable_files={
                **(
                    {AXIAL_BRIEF_LOGICAL_PATH: workspace.axial_brief_path}
                    if workspace.axial_brief_path is not None
                    else {AXIAL_TEXT_LOGICAL_PATH: workspace.axial_path}
                ),
                EVIDENCE_MANIFEST_LOGICAL_PATH: workspace.context_manifest_path,
            },
        )
        self.axial_runner = axial_runner
        self.axial_call_count = 0
        self.feedback_call_count = 0
        feedback_root = workspace.root / "author_feedback"
        # A restarted Host retains query identities and read access to saved detail files.
        for directory in feedback_root.glob("call_*"):
            suffix = directory.name.removeprefix("call_")
            if directory.is_dir() and suffix.isdigit():
                self.feedback_call_count = max(self.feedback_call_count, int(suffix))
        for file in (*feedback_root.glob("call_*/*.json"), *feedback_root.glob("axial_*.json")):
            if file.is_file() and file.resolve().is_relative_to(feedback_root.resolve()):
                self.extra_readable_files[str(file.relative_to(workspace.root))] = file.resolve()
        # Host reference used by the audit; direct queries read the staged Author copy.
        self.source_usd_path = source_usd_path
        # Every successful `check_axial` payload in call order; the Host reduces them to the
        # record's `runtime_report.axial_diagnostic` at promotion time.
        self.axial_payloads: list[dict[str, Any]] = []

    @property
    def candidate_usd_path(self) -> Path:
        return self.workspace.build_dir / "isaac" / "model.usdc"

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        return self.track_validation(
            name,
            self._execute(name, arguments),
            {
                "check_model": "check",
                "compile_model": "compile",
                "check_axial": "axial",
                "render_candidate": "render",
            },
        )

    def validation_state(self, stages: tuple[str, ...] = ()) -> dict[str, Any]:
        """Report this Author's fixed stage set; ``stages`` exists only for signature parity."""
        del stages
        return super().validation_state(("check", "compile", "axial", "render"))

    def _execute(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        if name == "validate_model":
            if arguments:
                return ProviderToolExecution(self._error("validate_model accepts no arguments."))
            return self.validate_candidate(
                [
                    ("axial", lambda: ProviderToolExecution(self._check_axial({}))),
                    (
                        "render",
                        lambda: self._author_feedback(
                            "render_candidate", {"azimuth": 45, "elevation": 30, "joints": []}
                        ),
                    ),
                ]
            )
        if name in {"measure_mesh", "render_source", "render_candidate", "compare_part"}:
            return self._author_feedback(name, arguments)
        if name == AXIAL_TOOL_NAME:
            payload = self._check_axial(arguments)
            return ProviderToolExecution(
                payload,
                trace_payload=self.axial_payloads[-1]
                if payload.get("ok") and self.axial_payloads
                else None,
            )
        if name == MEASURE_TOOL_NAME:
            return ProviderToolExecution(self._measure_face(arguments))
        return super().execute(name, arguments)

    def _author_feedback(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        from usdcraft.reconstruction.author_tools import (
            measure_mesh,
            render_candidate,
            render_source,
        )

        self.feedback_call_count += 1
        out = self.workspace.root / "author_feedback" / f"call_{self.feedback_call_count:04d}"
        try:
            out.mkdir(parents=True, exist_ok=False)
            if name == "render_source":
                payload = render_source(
                    self.workspace.root / "inputs" / "source_render",
                    out_dir=out,
                    azimuth=float(arguments["azimuth"]),
                    elevation=float(arguments["elevation"]),
                )
            elif name == "measure_mesh":
                payload = measure_mesh(
                    self.workspace.source_mesh_path,
                    out_dir=out,
                    bounds=arguments.get("bounds"),
                    max_points=arguments.get("max_points", 256),
                )
            elif name == "compare_part":
                if self.compile_success_revision != self.revision:
                    raise ValueError(
                        "compare_part needs compile_model for the current revision first"
                    )
                from usdcraft.reconstruction.part_comparison import compare_part

                payload = compare_part(
                    self.workspace.source_mesh_path,
                    self.workspace.build_dir,
                    out_dir=out,
                    part=str(arguments["part"]),
                    element=arguments.get("element"),
                    face=str(arguments["face"]),
                    bounds=arguments["bounds"],
                )
            else:
                if self.compile_success_revision != self.revision:
                    return ProviderToolExecution(
                        self._error(
                            "render_candidate needs compile_model for the current model revision first."
                        )
                    )
                joints = arguments.get("joints", [])
                positions = {j["name"]: float(j["position"]) for j in joints}
                if len(positions) != len(joints):
                    raise ValueError("Joint names must be unique")
                payload = render_candidate(
                    self.workspace.build_dir,
                    out_dir=out,
                    azimuth=float(arguments.get("azimuth", 45)),
                    elevation=float(arguments.get("elevation", 30)),
                    joint_positions=positions,
                )
            for file in out.glob("*.json"):
                self.extra_readable_files[str(file.relative_to(self.workspace.root))] = (
                    file.resolve()
                )
            payload["revision"] = self.revision
            return ProviderToolExecution(
                payload, image_paths=tuple(Path(p) for p in payload["images"])
            )
        except Exception as exc:
            return ProviderToolExecution(self._error(f"{name}: {type(exc).__name__}: {exc}"))

    def _measure_face(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        face = str(arguments.get("face", "")).strip()
        if face not in FACE_NAMES:
            return self._error(f"{MEASURE_TOOL_NAME} face must be one of {', '.join(FACE_NAMES)}.")
        try:
            band = normalize_z_band(arguments.get("z_band"))
        except ValueError as exc:
            return self._error(f"{MEASURE_TOOL_NAME} {exc}")
        text = self.workspace.axial_path.read_text(encoding="utf-8")
        return measure_face_payload(text, face=face, z_band=band)

    def _check_axial(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if arguments:
            return self._error(f"{AXIAL_TOOL_NAME} takes no arguments.")
        if self.compile_success_revision != self.revision:
            return self._error(
                f"{AXIAL_TOOL_NAME} needs a successful compile_model for the current model "
                "revision first."
            )
        candidate = self.candidate_usd_path
        if not candidate.is_file():
            return self._error(f"Compiled candidate is missing: {candidate.name}")
        self.axial_call_count += 1
        try:
            result = self.axial_runner(
                self.workspace.axial_path,
                candidate,
                source_usd_path=self.source_usd_path,
            )
        except Exception as exc:
            result = ProviderToolResult.error(f"{type(exc).__name__}: {exc}")
        if not isinstance(result, ProviderToolResult):
            return self._error("axial_runner returned an invalid ProviderToolResult.")
        if result.success and not isinstance(result.output, Mapping):
            return self._error("axial_runner returned no structured diagnostic output.")
        payload: dict[str, Any] = {
            "ok": result.success,
            "revision": self.revision,
            "axial_call": self.axial_call_count,
        }
        output = result.output
        if result.success:
            payload["binding"] = diagnostic_binding(self.source_usd_path, candidate)
        if result.success and isinstance(output, Mapping):
            text = str(output.get("text", ""))
            if not text.startswith(AXIAL_REFERENCE_HEADER):
                text = AXIAL_REFERENCE_HEADER + text
            output = {**output, "text": text}
        payload["output" if result.success else "error"] = output
        if result.success:
            self.axial_payloads.append(payload)
            detail = (
                self.workspace.root / "author_feedback" / f"axial_{self.axial_call_count:04d}.json"
            )
            detail.parent.mkdir(exist_ok=True)
            detail.write_text(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self.extra_readable_files[str(detail.relative_to(self.workspace.root))] = (
                detail.resolve()
            )
            from usdcraft.reconstruction.axial_summary import summarize_axial_result

            compact = (
                summarize_axial_result(output.get("result", {}))
                if isinstance(output, Mapping)
                else {}
            )
            return {
                **payload,
                "output": {
                    "text": output.get("text", ""),
                    "result": {"summary": compact},
                    "detail_path": str(detail),
                },
            }
        return payload


__all__ = [
    "AXIAL_BRIEF_LOGICAL_PATH",
    "AXIAL_REFERENCE_HEADER",
    "AXIAL_TEXT_LOGICAL_PATH",
    "AXIAL_TOOL_NAME",
    "FACE_NAMES",
    "MEASURE_TOOL_NAME",
    "EVIDENCE_MANIFEST_LOGICAL_PATH",
    "FINISH_TOOL_NAME",
    "ProviderAxialRunner",
    "ReconstructionToolExecutor",
    "build_reconstruction_tool_schemas",
    "default_axial_runner",
]
