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

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal

ASSET_STRUCTURE_VERSION = "3.0"
ASSET_STRUCTURE_PROFILE = "minimal_shared_physics"
COMPILE_REPORT_SCHEMA_VERSION = 2
GENERATION_MANIFEST_SCHEMA_VERSION = 3
GENERATION_VALIDATION_PROFILE = "generation"
RECONSTRUCTION_MANIFEST_SCHEMA_VERSION = 1
RECONSTRUCTION_VALIDATION_PROFILE = "reconstruction"
RECONSTRUCTION_INPUT_MANIFEST_SCHEMA = "reconstruction_input_manifest"
RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION = 2
RECONSTRUCTION_APPROVED_INPUTS_SCHEMA = "reconstruction_approved_inputs"
# The semantic image is always the reconstruction target's semantic record.
RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE = "reconstruction_target"
PROVENANCE_SCHEMA_VERSION = 3

PHYSICS_BACKENDS = ("physx", "newton")
PHYSX_USDC_ARTIFACT = "isaac/model.usdc"
NEWTON_USDC_ARTIFACT = "isaac/model_newton.usdc"
ASSET_INTERFACE_ARTIFACT = "isaac/asset.usda"
SHARED_PHYSICS_ARTIFACT = "isaac/physics.usdc"
PHYSX_LAYER_ARTIFACT = "isaac/physx.usda"
MUJOCO_LAYER_ARTIFACT = "isaac/mujoco.usda"
USDZ_ARTIFACT = "model.usdz"
GENERATION_ROOT_FLOATING_CHECK = "generation_root_floating"
FINITE_AUTHORED_VALUES_CHECK = "finite_authored_values"
ARTICULATION_ROOT_BODY_BINDING_CHECK = "articulation_root_body_binding"

ASSET_STRUCTURE_LAYER_ARTIFACTS = MappingProxyType(
    {
        "interface": ASSET_INTERFACE_ARTIFACT,
        "physics": SHARED_PHYSICS_ARTIFACT,
        "physx": PHYSX_LAYER_ARTIFACT,
        "newton_mjc": MUJOCO_LAYER_ARTIFACT,
    }
)
PHYSICS_ENTRY_ARTIFACTS = MappingProxyType(
    {
        "physx": PHYSX_USDC_ARTIFACT,
        "newton": NEWTON_USDC_ARTIFACT,
    }
)
REQUIRED_ASSET_STRUCTURE_ARTIFACTS = (
    ASSET_INTERFACE_ARTIFACT,
    SHARED_PHYSICS_ARTIFACT,
    PHYSX_LAYER_ARTIFACT,
    MUJOCO_LAYER_ARTIFACT,
    PHYSX_USDC_ARTIFACT,
    NEWTON_USDC_ARTIFACT,
)
USD_STAGE_ARTIFACTS = REQUIRED_ASSET_STRUCTURE_ARTIFACTS
REQUIRED_COMPILE_CHECKS = frozenset(
    {
        "urdf_export",
        "usd_strict",
        "physx_usd_strict",
        "newton_usd_strict",
        "usd_dependency_closure",
        GENERATION_ROOT_FLOATING_CHECK,
    }
)
_COMMON_BACKEND_CHECKS = frozenset(
    {
        "stage_open",
        "default_prim",
        "z_up",
        "meters",
        "kilograms",
        "reusable_asset_no_scene",
        FINITE_AUTHORED_VALUES_CHECK,
        "dependencies_resolved",
        "relative_asset_paths",
        "vendor_namespace_isolation",
        "physics_graph_acyclic",
        "joint_parent_child_order",
        "articulation_root_coverage",
        ARTICULATION_ROOT_BODY_BINDING_CHECK,
    }
)
REQUIRED_BACKEND_CHECKS = MappingProxyType(
    {
        "physx": _COMMON_BACKEND_CHECKS | {"physx_schema_consistency"},
        "newton": _COMMON_BACKEND_CHECKS
        | {
            "newton_schema_consistency",
            "mass_inertia_or_collision_derivation",
            "collider_dimensions",
            "collider_scale",
        },
    }
)
STANDARD_RENDER_ARTIFACTS = (
    "renders/level_front.png",
    "renders/level_back.png",
    "renders/level_right.png",
    "renders/level_left.png",
    "renders/elevated_3q.png",
    "renders/top_down.png",
    "renders/underside_3q.png",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(slots=True, frozen=True)
class ArtifactContractIssue:
    path: str
    message: str


@dataclass(slots=True, frozen=True)
class MaterializationContractResult:
    issues: tuple[ArtifactContractIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        return "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)


MaterializationDisplayStatus = Literal[
    "available",
    "contract_invalid",
    "missing",
    "compile_failed",
]


@dataclass(slots=True, frozen=True)
class MaterializationDisplayResult:
    """Viewer displayability without weakening the canonical artifact contract.

    ``available`` remains reserved for a materialization that passes the complete
    canonical contract. A contract-invalid materialization is still displayable when
    its required Viewer entry file exists and the compile report does not explicitly
    record a failed compile. Missing Viewer input and an explicit compile failure are
    the only blocking classifications here.
    """

    status: MaterializationDisplayStatus
    issues: tuple[ArtifactContractIssue, ...]

    @property
    def displayable(self) -> bool:
        return self.status in {"available", "contract_invalid"}


def physics_output_contract() -> dict[str, Any]:
    """Return the stable, JSON-ready Generation physics artifact contract."""

    return {
        "asset_structure_version": ASSET_STRUCTURE_VERSION,
        "profile": ASSET_STRUCTURE_PROFILE,
        "interface_usd": ASSET_INTERFACE_ARTIFACT,
        "entrypoints_flattened": True,
        "physics_backends": list(PHYSICS_BACKENDS),
        "layers": dict(ASSET_STRUCTURE_LAYER_ARTIFACTS),
        "entries": dict(PHYSICS_ENTRY_ARTIFACTS),
        "shared_usdz": USDZ_ARTIFACT,
        "usdz_backend": "physx",
    }


def exporter_asset_structure_contract() -> dict[str, Any]:
    """Return the exact SDK exporter evidence required by the Host."""

    return {
        "version": ASSET_STRUCTURE_VERSION,
        "profile": ASSET_STRUCTURE_PROFILE,
        "entrypoints_flattened": True,
        "interface_usd": ASSET_INTERFACE_ARTIFACT,
        "layers": {
            "physics": SHARED_PHYSICS_ARTIFACT,
            "physx": PHYSX_LAYER_ARTIFACT,
            "newton_mjc": MUJOCO_LAYER_ARTIFACT,
        },
    }


def validate_materialization_contract(
    materialization: Path,
    generation_manifest: Mapping[str, Any] | Path | None,
    *,
    phase: Literal["compile_output", "canonical"] = "canonical",
    revision_dir: Path | None = None,
) -> MaterializationContractResult:
    """Validate the current Generation artifact contract.

    Validation is fail-closed for exported artifact structure, hashes and USD dependency
    closures. ``compile_output`` is used before promotion, while ``canonical`` additionally binds
    the report and manifest to their record/revision. Renders are optional presentation artifacts
    and never part of the canonical integrity gate.
    """

    if phase not in {"compile_output", "canonical"}:
        raise ValueError(f"Unsupported artifact contract phase: {phase}")
    root = Path(materialization).resolve()
    issues: list[ArtifactContractIssue] = []
    manifest = _manifest_object(generation_manifest, issues)
    report = _json_object(root / "compile_report.json", "compile_report.json", issues)
    if manifest is None or report is None:
        return MaterializationContractResult(tuple(issues))

    for key, expected in (
        ("schema_version", GENERATION_MANIFEST_SCHEMA_VERSION),
        ("pipeline", "generation"),
        ("status", "generation_ready"),
        ("validation_profile", GENERATION_VALIDATION_PROFILE),
    ):
        if manifest.get(key) != expected:
            _add_issue(issues, "generation.json", f"{key} must be {expected!r}")
    if manifest.get("outputs") != physics_output_contract():
        _add_issue(
            issues,
            "generation.json",
            "outputs must match the current dual-backend physics contract",
        )

    if report.get("status") != "success":
        _add_issue(issues, "compile_report.json", "status must be 'success'")
    if _integer_schema(report.get("schema_version")) != COMPILE_REPORT_SCHEMA_VERSION:
        _add_issue(
            issues,
            "compile_report.json",
            f"schema_version must be {COMPILE_REPORT_SCHEMA_VERSION}",
        )
    if report.get("validation_profile") != GENERATION_VALIDATION_PROFILE:
        _add_issue(
            issues,
            "compile_report.json",
            f"validation_profile must be {GENERATION_VALIDATION_PROFILE!r}",
        )
    _validate_current_materialization(
        root,
        manifest,
        report,
        issues,
        phase=phase,
        revision_dir=revision_dir,
    )
    return MaterializationContractResult(tuple(issues))


def validate_reconstruction_materialization_contract(
    materialization: Path,
    reconstruction_manifest: Mapping[str, Any] | Path | None,
    *,
    phase: Literal["compile_output", "canonical"] = "canonical",
    revision_dir: Path | None = None,
) -> MaterializationContractResult:
    """Validate Reconstruction's binding around the shared compiled-asset contract.

    Reconstruction intentionally reuses the same SDK exporter, compile-report schema, backend
    evidence and dependency-closure checks as current Generation. Its product manifest and input
    binding are distinct: semantic images, source surfaces, AXIAL and source-overlay carriers are
    Reconstruction evidence and must never be represented as Generation inspiration input.
    """

    if phase not in {"compile_output", "canonical"}:
        raise ValueError(f"Unsupported artifact contract phase: {phase}")
    root = Path(materialization).resolve()
    issues: list[ArtifactContractIssue] = []
    manifest = _pipeline_manifest_object(
        reconstruction_manifest,
        label="reconstruction.json",
        issues=issues,
    )
    report = _json_object(root / "compile_report.json", "compile_report.json", issues)
    if manifest is None or report is None:
        return MaterializationContractResult(tuple(issues))

    for key, expected in (
        ("schema_version", RECONSTRUCTION_MANIFEST_SCHEMA_VERSION),
        ("pipeline", "reconstruction"),
        ("status", "reconstruction_ready"),
        ("validation_profile", RECONSTRUCTION_VALIDATION_PROFILE),
        ("compile_validation_profile", GENERATION_VALIDATION_PROFILE),
    ):
        if manifest.get(key) != expected:
            _add_issue(issues, "reconstruction.json", f"{key} must be {expected!r}")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        _add_issue(issues, "reconstruction.json", "run_id must be a non-empty string")
    if manifest.get("outputs") != physics_output_contract():
        _add_issue(
            issues,
            "reconstruction.json",
            "outputs must match the current dual-backend physics contract",
        )

    if report.get("status") != "success":
        _add_issue(issues, "compile_report.json", "status must be 'success'")
    if _integer_schema(report.get("schema_version")) != COMPILE_REPORT_SCHEMA_VERSION:
        _add_issue(
            issues,
            "compile_report.json",
            f"schema_version must be {COMPILE_REPORT_SCHEMA_VERSION}",
        )
    if report.get("validation_profile") != GENERATION_VALIDATION_PROFILE:
        _add_issue(
            issues,
            "compile_report.json",
            f"validation_profile must be {GENERATION_VALIDATION_PROFILE!r}",
        )

    # This is the one shared exporter/Host gate. Keep product-specific canonical bindings outside
    # it so Reconstruction cannot silently fork compilation semantics.
    _validate_current_materialization(
        root,
        manifest,
        report,
        issues,
        phase="compile_output",
        revision_dir=None,
    )
    if phase == "canonical":
        _validate_reconstruction_canonical_binding(
            root,
            manifest,
            report,
            issues,
            revision_dir=revision_dir,
        )
    return MaterializationContractResult(tuple(issues))


def classify_materialization_displayability(
    materialization: Path,
    generation_manifest: Mapping[str, Any] | Path | None,
    *,
    revision_dir: Path | None = None,
) -> MaterializationDisplayResult:
    """Classify whether an existing materialization can be shown by Viewer.

    This is deliberately narrower than canonical validation. It does not reinterpret
    contract failures as success: ``available`` still means the complete canonical
    contract passed. It only separates non-blocking contract evidence problems from
    the two objective conditions that prevent Viewer display.
    """

    root = Path(materialization).resolve()
    contract = validate_materialization_contract(
        root,
        generation_manifest,
        phase="canonical",
        revision_dir=revision_dir,
    )
    return _classify_materialization_displayability(root, contract)


def classify_reconstruction_materialization_displayability(
    materialization: Path,
    reconstruction_manifest: Mapping[str, Any] | Path | None,
    *,
    revision_dir: Path | None = None,
) -> MaterializationDisplayResult:
    """Project Reconstruction's complete canonical contract into Viewer display state."""

    root = Path(materialization).resolve()
    contract = validate_reconstruction_materialization_contract(
        root,
        reconstruction_manifest,
        phase="canonical",
        revision_dir=revision_dir,
    )
    return _classify_materialization_displayability(root, contract)


def cheap_display_hint(root: Path) -> tuple[MaterializationDisplayStatus | None, bool]:
    """The two objective, hash-free Viewer display conditions, without the full contract.

    Returns ``(status, displayable)`` when a compile failure or a missing required Viewer
    entry file alone is already conclusive. Returns ``(None, True)`` when distinguishing
    ``available`` from ``contract_invalid`` still requires the full canonical contract (the
    hash/dependency-closure work this function deliberately does not do). A caller using only
    this hint must never report the ``None`` case as ``available``; it is displayable but
    unvalidated.
    """

    report_issues: list[ArtifactContractIssue] = []
    report = _json_object(root / "compile_report.json", "compile_report.json", report_issues)
    if report is not None and report.get("status") == "failed":
        return "compile_failed", False
    if not (root / "model.urdf").is_file():
        return "missing", False
    return None, True


def _classify_materialization_displayability(
    root: Path,
    contract: MaterializationContractResult,
) -> MaterializationDisplayResult:
    """Apply the shared Viewer projection after the applicable pipeline contract runs."""

    issues = list(contract.issues)

    hint_status, _ = cheap_display_hint(root)
    if hint_status == "compile_failed" and not any(
        issue.path == "compile_report.json" and "status" in issue.message for issue in issues
    ):
        _add_issue(issues, "compile_report.json", "compile status is explicitly 'failed'")
    if hint_status == "missing" and not any(issue.path == "model.urdf" for issue in issues):
        _add_issue(issues, "model.urdf", "missing required Viewer model file")

    if hint_status is not None:
        status: MaterializationDisplayStatus = hint_status
    elif contract.ok:
        status = "available"
    else:
        status = "contract_invalid"
    return MaterializationDisplayResult(status=status, issues=tuple(issues))


def _validate_current_materialization(
    root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
    *,
    phase: Literal["compile_output", "canonical"],
    revision_dir: Path | None,
) -> None:
    if report.get("asset_structure") != physics_output_contract():
        _add_issue(
            issues,
            "compile_report.json",
            "asset_structure must match the current dual-backend physics contract",
        )

    checks = report.get("checks_run")
    check_names = (
        {str(item) for item in checks if isinstance(item, str)}
        if isinstance(checks, list)
        else set()
    )
    missing_checks = sorted(REQUIRED_COMPILE_CHECKS - check_names)
    if missing_checks:
        _add_issue(
            issues,
            "compile_report.json",
            "checks_run is missing: " + ", ".join(missing_checks),
        )
    metrics = report.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("asset_structure") != exporter_asset_structure_contract()
    ):
        _add_issue(
            issues,
            "compile_report.json",
            "metrics.asset_structure must match the SDK exporter contract",
        )
    backend_validation = metrics
    backend_validation = (
        backend_validation.get("backend_validation")
        if isinstance(backend_validation, Mapping)
        else None
    )
    if not isinstance(backend_validation, Mapping):
        _add_issue(
            issues,
            "compile_report.json",
            "metrics.backend_validation must be an object",
        )
    else:
        for backend, required_checks in REQUIRED_BACKEND_CHECKS.items():
            evidence = backend_validation.get(backend)
            if not isinstance(evidence, Mapping):
                _add_issue(
                    issues,
                    "compile_report.json",
                    f"metrics.backend_validation.{backend} must be an object",
                )
                continue
            if evidence.get("status") != "success":
                _add_issue(
                    issues,
                    "compile_report.json",
                    f"metrics.backend_validation.{backend}.status must be 'success'",
                )
            backend_checks = evidence.get("checks")
            backend_check_names = (
                {str(item) for item in backend_checks if isinstance(item, str)}
                if isinstance(backend_checks, list)
                else set()
            )
            missing_backend_checks = sorted(required_checks - backend_check_names)
            if missing_backend_checks:
                _add_issue(
                    issues,
                    "compile_report.json",
                    f"metrics.backend_validation.{backend}.checks is missing: "
                    + ", ".join(missing_backend_checks),
                )
            if evidence.get("runtime_status") not in {"not_run", "success"}:
                _add_issue(
                    issues,
                    "compile_report.json",
                    f"metrics.backend_validation.{backend}.runtime_status must be "
                    "'not_run' or 'success'",
                )

    required_files = [
        "model.urdf",
        USDZ_ARTIFACT,
        *REQUIRED_ASSET_STRUCTURE_ARTIFACTS,
    ]
    asset_root = root / "assets"
    if asset_root.is_dir():
        try:
            required_files.extend(
                path.relative_to(root).as_posix()
                for path in sorted(asset_root.rglob("*"))
                if path.is_file()
            )
        except OSError as exc:
            _add_issue(issues, "assets", f"could not enumerate assets: {exc}")

    artifact_hashes = report.get("artifacts")
    if not isinstance(artifact_hashes, Mapping):
        _add_issue(issues, "compile_report.json", "artifacts must be an object")
        artifact_hashes = {}
    for relative_path in dict.fromkeys(required_files):
        path = _require_file(root, relative_path, issues)
        if path is not None:
            _validate_hash(path, relative_path, artifact_hashes, issues)

    issues.extend(validate_urdf_dependencies(root, artifact_hashes))
    _validate_dependency_closures(root, report, artifact_hashes, issues)

    if phase == "canonical":
        _validate_canonical_binding(
            root,
            manifest,
            report,
            issues,
            revision_dir=revision_dir,
        )


def _validate_canonical_binding(
    root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
    *,
    revision_dir: Path | None,
) -> None:
    record_id = manifest.get("record_id")
    revision_id = manifest.get("revision_id")
    if not isinstance(record_id, str) or not record_id:
        _add_issue(issues, "generation.json", "record_id must be a non-empty string")
    if not isinstance(revision_id, str) or not revision_id:
        _add_issue(issues, "generation.json", "revision_id must be a non-empty string")
    if not isinstance(record_id, str) or not isinstance(revision_id, str):
        return

    expected_report = f"../../../../cache/record_materialization/{record_id}/compile_report.json"
    if manifest.get("compile_report") != expected_report:
        _add_issue(
            issues,
            "generation.json",
            f"compile_report must be {expected_report!r}",
        )
    if report.get("record_id") != record_id:
        _add_issue(
            issues,
            "compile_report.json",
            f"record_id must be {record_id!r}",
        )
    expected_model_path = f"data/records/{record_id}/revisions/{revision_id}/model.py"
    if report.get("model_path") != expected_model_path:
        _add_issue(
            issues,
            "compile_report.json",
            f"model_path must be {expected_model_path!r}",
        )
    if revision_dir is None:
        _add_issue(
            issues,
            "generation.json",
            "canonical validation requires the active revision directory",
        )
        return
    revision_root = Path(revision_dir).resolve()
    model_path = revision_root / "model.py"
    if model_path.is_file():
        if manifest.get("model_sha256") != _sha256(model_path):
            _add_issue(
                issues,
                "generation.json",
                "model_sha256 does not match model.py",
            )
    else:
        _add_issue(issues, "model.py", "missing canonical model")

    prompt_path = revision_root / "prompt.txt"
    input_payload = manifest.get("input")
    if not isinstance(input_payload, Mapping):
        _add_issue(issues, "generation.json", "input must be an object")
        return
    input_kind = input_payload.get("kind")
    if input_kind not in {"text", "text_image"}:
        _add_issue(
            issues,
            "generation.json",
            "input.kind must be 'text' or 'text_image'",
        )
    expected_image_role = "inspiration" if input_kind == "text_image" else None
    if input_payload.get("image_role") != expected_image_role:
        _add_issue(
            issues,
            "generation.json",
            f"input.image_role must be {expected_image_role!r}",
        )
    if prompt_path.is_file():
        try:
            prompt = prompt_path.read_text(encoding="utf-8").rstrip("\n")
        except OSError as exc:
            _add_issue(issues, "prompt.txt", f"could not read canonical prompt: {exc}")
        else:
            if input_payload.get("prompt_sha256") != _sha256_bytes(prompt.encode("utf-8")):
                _add_issue(
                    issues,
                    "generation.json",
                    "input.prompt_sha256 does not match prompt.txt",
                )
    else:
        _add_issue(issues, "prompt.txt", "missing canonical prompt")

    if input_kind != "text_image":
        if (
            input_payload.get("image_path") is not None
            or input_payload.get("image_sha256") is not None
        ):
            _add_issue(
                issues,
                "generation.json",
                "text input must not declare an inspiration image",
            )
        return
    image_value = input_payload.get("image_path")
    if not isinstance(image_value, str) or not _safe_relative_path(image_value):
        _add_issue(
            issues,
            "generation.json",
            "input.image_path must be a safe revision-relative path",
        )
        return
    image_path = _contained_path(revision_root, image_value, issues, label="generation.json")
    if image_path is None or not image_path.is_file():
        _add_issue(issues, image_value, "missing inspiration image")
    elif input_payload.get("image_sha256") != _sha256(image_path):
        _add_issue(
            issues,
            "generation.json",
            "input.image_sha256 does not match the inspiration image",
        )


def _validate_reconstruction_canonical_binding(
    root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
    *,
    revision_dir: Path | None,
) -> None:
    record_id = manifest.get("record_id")
    revision_id = manifest.get("revision_id")
    if not isinstance(record_id, str) or not record_id:
        _add_issue(issues, "reconstruction.json", "record_id must be a non-empty string")
    if not isinstance(revision_id, str) or not revision_id:
        _add_issue(issues, "reconstruction.json", "revision_id must be a non-empty string")
    if not isinstance(record_id, str) or not isinstance(revision_id, str):
        return

    expected_report = f"../../../../cache/record_materialization/{record_id}/compile_report.json"
    if manifest.get("compile_report") != expected_report:
        _add_issue(
            issues,
            "reconstruction.json",
            f"compile_report must be {expected_report!r}",
        )
    if report.get("record_id") != record_id:
        _add_issue(
            issues,
            "compile_report.json",
            f"record_id must be {record_id!r}",
        )
    expected_model_path = f"data/records/{record_id}/revisions/{revision_id}/model.py"
    if report.get("model_path") != expected_model_path:
        _add_issue(
            issues,
            "compile_report.json",
            f"model_path must be {expected_model_path!r}",
        )
    if revision_dir is None:
        _add_issue(
            issues,
            "reconstruction.json",
            "canonical validation requires the active revision directory",
        )
        return

    revision_root = Path(revision_dir).resolve()
    model_path = revision_root / "model.py"
    if not model_path.is_file():
        _add_issue(issues, "model.py", "missing canonical model")
    elif manifest.get("model_sha256") != _sha256(model_path):
        _add_issue(
            issues,
            "reconstruction.json",
            "model_sha256 does not match model.py",
        )
    input_payload = manifest.get("input")
    if not isinstance(input_payload, Mapping):
        _add_issue(issues, "reconstruction.json", "input must be an object")
        return
    if input_payload.get("kind") not in {"mesh_image", "mesh_image_text"}:
        _add_issue(
            issues,
            "reconstruction.json",
            "input.kind must be 'mesh_image' or 'mesh_image_text'",
        )
    # Optional user guidance, recorded inline: null for a run without guidance, otherwise the
    # normalized non-empty text. Records promoted before the field existed simply omit it.
    guidance = input_payload.get("guidance")
    if guidance is not None and (
        not isinstance(guidance, str) or not guidance.strip() or guidance != guidance.strip()
    ):
        _add_issue(
            issues,
            "reconstruction.json",
            "input.guidance must be null or a non-empty string without surrounding whitespace",
        )

    request_value = input_payload.get("request_path")
    if not isinstance(request_value, str) or not _safe_relative_path(request_value):
        _add_issue(
            issues,
            "reconstruction.json",
            "input.request_path must be a safe revision-relative path",
        )
    else:
        request_path = _contained_path(
            revision_root,
            request_value,
            issues,
            label="reconstruction.json",
        )
        if request_path is None or not request_path.is_file():
            _add_issue(issues, request_value, "missing Reconstruction request text")
        else:
            try:
                request_text = request_path.read_text(encoding="utf-8").rstrip("\n")
            except OSError as exc:
                _add_issue(issues, request_value, f"could not read request text: {exc}")
            else:
                expected = _sha256_bytes(request_text.encode("utf-8"))
                if input_payload.get("request_sha256") != expected:
                    _add_issue(
                        issues,
                        "reconstruction.json",
                        "input.request_sha256 does not match request.txt",
                    )

    input_manifest_value = input_payload.get("manifest_path")
    if not isinstance(input_manifest_value, str) or not _safe_relative_path(input_manifest_value):
        _add_issue(
            issues,
            "reconstruction.json",
            "input.manifest_path must be a safe revision-relative path",
        )
        return
    input_manifest_path = _contained_path(
        revision_root,
        input_manifest_value,
        issues,
        label="reconstruction.json",
    )
    if input_manifest_path is None or not input_manifest_path.is_file():
        _add_issue(issues, input_manifest_value, "missing Reconstruction input manifest")
        return
    if input_payload.get("manifest_sha256") != _sha256(input_manifest_path):
        _add_issue(
            issues,
            "reconstruction.json",
            "input.manifest_sha256 does not match the Reconstruction input manifest",
        )
        return
    input_manifest = _json_object(
        input_manifest_path,
        input_manifest_value,
        issues,
    )
    if input_manifest is not None:
        _validate_reconstruction_input_manifest(
            revision_root,
            input_manifest,
            issues,
            label=input_manifest_value,
        )
        _validate_reconstruction_input_projection(
            input_payload,
            input_manifest,
            issues,
            label="reconstruction.json",
        )


def _validate_reconstruction_input_projection(
    input_payload: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
    *,
    label: str,
) -> None:
    raw_files = input_manifest.get("files")
    if not isinstance(raw_files, list):
        return
    files = {
        item.get("key"): item
        for item in raw_files
        if isinstance(item, Mapping) and isinstance(item.get("key"), str)
    }
    for group_key in ("author_inputs", "host_preprocessing_inputs"):
        raw_keys = input_manifest.get(group_key)
        if not isinstance(raw_keys, list):
            continue
        expected = []
        for key in raw_keys:
            descriptor = files.get(key)
            if descriptor is None:
                continue
            expected.append(
                {key: value for key, value in descriptor.items() if key != "author_input"}
            )
        if input_payload.get(group_key) != expected:
            _add_issue(
                issues,
                label,
                f"input.{group_key} must match the generated input manifest",
            )


def _validate_reconstruction_approved_snapshot(
    revision_root: Path,
    files: Mapping[str, Mapping[str, Any]],
    issues: list[ArtifactContractIssue],
    *,
    label: str,
) -> None:
    """Cross-check current frozen-input provenance against the copied canonical bytes."""

    manifest_descriptor = files.get("original_source_manifest")
    if manifest_descriptor is None:
        # Reported by the required-key check.
        return
    manifest_relative = manifest_descriptor.get("path")
    if not isinstance(manifest_relative, str) or not _safe_relative_path(manifest_relative):
        return
    manifest_path = revision_root / manifest_relative
    if not manifest_path.is_file():
        return
    approved_label = f"{label}.approved_inputs"
    try:
        approved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add_issue(issues, approved_label, f"invalid approved-input manifest: {exc}")
        return
    if not isinstance(approved, Mapping) or approved.get("schema") != (
        RECONSTRUCTION_APPROVED_INPUTS_SCHEMA
    ):
        _add_issue(
            issues,
            approved_label,
            f"schema must be {RECONSTRUCTION_APPROVED_INPUTS_SCHEMA!r}",
        )
        return
    if approved.get("schema_version") != 1:
        _add_issue(issues, approved_label, "schema_version must be 1")
    if approved.get("pipeline") != "reconstruction":
        _add_issue(issues, approved_label, "pipeline must be 'reconstruction'")
    fingerprint = approved.get("approved_fingerprint")
    if not (
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
    ):
        _add_issue(issues, approved_label, "approved_fingerprint must be a lowercase sha256")

    source = approved.get("source")
    raw_source_files = source.get("files") if isinstance(source, Mapping) else None
    if not isinstance(raw_source_files, list) or not raw_source_files:
        _add_issue(issues, approved_label, "source.files must be a non-empty array")
        return
    approved_files: dict[str, Mapping[str, Any]] = {}
    approved_entries: list[str] = []
    for index, raw_file in enumerate(raw_source_files):
        if not isinstance(raw_file, Mapping):
            _add_issue(issues, approved_label, f"source.files[{index}] must be an object")
            continue
        relative_path = raw_file.get("tree_relative_path")
        frozen_path = raw_file.get("frozen_path")
        if not isinstance(relative_path, str) or not _safe_relative_path(relative_path):
            _add_issue(
                issues,
                approved_label,
                f"source.files[{index}].tree_relative_path must be safe",
            )
            continue
        if frozen_path != f"source/tree/{relative_path}":
            _add_issue(
                issues,
                approved_label,
                f"source.files[{index}].frozen_path must preserve the source tree",
            )
        if relative_path in approved_files:
            _add_issue(issues, approved_label, f"duplicate source file {relative_path!r}")
            continue
        approved_files[relative_path] = raw_file
        if raw_file.get("entry") is True:
            approved_entries.append(relative_path)
        elif raw_file.get("entry") is not False:
            _add_issue(
                issues,
                approved_label,
                f"source.files[{index}].entry must be a boolean",
            )

    canonical_source_descriptors = {
        descriptor.get("source_relative_path"): descriptor
        for key, descriptor in files.items()
        if key == "original_source" or key.startswith("source_dependency_")
    }
    if set(canonical_source_descriptors) != set(approved_files):
        _add_issue(
            issues,
            approved_label,
            "source.files must bind the complete canonical original-source closure",
        )
    declared_entry = source.get("entry_tree_relative_path") if isinstance(source, Mapping) else None
    original_source = files.get("original_source")
    canonical_entry = (
        original_source.get("source_relative_path")
        if isinstance(original_source, Mapping)
        else None
    )
    if approved_entries != [canonical_entry] or declared_entry != canonical_entry:
        _add_issue(issues, approved_label, "source entry binding is inconsistent")

    for relative_path, raw_file in approved_files.items():
        descriptor = canonical_source_descriptors.get(relative_path)
        if not isinstance(descriptor, Mapping):
            continue
        for field in ("sha256", "byte_count"):
            if descriptor.get(field) != raw_file.get(field):
                _add_issue(
                    issues,
                    approved_label,
                    f"source file {relative_path!r} {field} differs from approval",
                )
        if descriptor.get("path") != f"inputs/source/original/{relative_path}":
            _add_issue(
                issues,
                approved_label,
                f"source file {relative_path!r} does not preserve canonical topology",
            )
        if descriptor.get("source_resolved_path") != raw_file.get("original_path"):
            _add_issue(
                issues,
                approved_label,
                f"source file {relative_path!r} original path differs from approval",
            )

    for approved_key, descriptor_key in (
        ("semantic_image", "semantic_image"),
        ("scaffold", "author_scaffold"),
    ):
        raw_file = approved.get(approved_key)
        descriptor = files.get(descriptor_key)
        if not isinstance(raw_file, Mapping) or not isinstance(descriptor, Mapping):
            _add_issue(issues, approved_label, f"{approved_key} binding is missing")
            continue
        for field in ("sha256", "byte_count"):
            if descriptor.get(field) != raw_file.get(field):
                _add_issue(
                    issues,
                    approved_label,
                    f"{approved_key} {field} differs from approval",
                )


def _validate_reconstruction_input_manifest(
    revision_root: Path,
    manifest: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
    *,
    label: str,
) -> None:
    if manifest.get("schema") != RECONSTRUCTION_INPUT_MANIFEST_SCHEMA:
        _add_issue(issues, label, f"schema must be {RECONSTRUCTION_INPUT_MANIFEST_SCHEMA!r}")
    if manifest.get("schema_version") != RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION:
        _add_issue(
            issues,
            label,
            f"schema_version must be {RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION}",
        )
    if manifest.get("pipeline") != "reconstruction":
        _add_issue(issues, label, "pipeline must be 'reconstruction'")
    if manifest.get("path_base") != "revision":
        _add_issue(issues, label, "path_base must be 'revision'")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        _add_issue(issues, label, "files must be an array")
        return
    files: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        file_label = f"{label}.files[{index}]"
        if not isinstance(raw_file, Mapping):
            _add_issue(issues, label, f"files[{index}] must be an object")
            continue
        key = raw_file.get("key")
        relative_path = raw_file.get("path")
        if not isinstance(key, str) or not key:
            _add_issue(issues, label, f"files[{index}].key must be a non-empty string")
            continue
        if key in files:
            _add_issue(issues, label, f"duplicate file key {key!r}")
            continue
        if (
            not isinstance(relative_path, str)
            or not _safe_relative_path(relative_path)
            or not relative_path.startswith("inputs/")
        ):
            _add_issue(
                issues,
                label,
                f"files[{index}].path must be a safe inputs/ revision-relative path",
            )
            continue
        if relative_path in seen_paths:
            _add_issue(issues, label, f"duplicate file path {relative_path!r}")
            continue
        seen_paths.add(relative_path)
        files[key] = raw_file
        path = _contained_path(revision_root, relative_path, issues, label=file_label)
        if path is None or not path.is_file():
            _add_issue(issues, relative_path, "missing Reconstruction input evidence")
            continue
        if raw_file.get("sha256") != _sha256(path):
            _add_issue(issues, file_label, "sha256 does not match the input file")
        byte_count = raw_file.get("byte_count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or byte_count != path.stat().st_size
        ):
            _add_issue(issues, file_label, "byte_count does not match the input file")
        if not isinstance(raw_file.get("role"), str) or not raw_file.get("role"):
            _add_issue(issues, file_label, "role must be a non-empty string")
        if not isinstance(raw_file.get("authority"), str) or not raw_file.get("authority"):
            _add_issue(issues, file_label, "authority must be a non-empty string")
        if not isinstance(raw_file.get("author_input"), bool):
            _add_issue(issues, file_label, "author_input must be a boolean")

    _validate_reconstruction_approved_snapshot(
        revision_root,
        files,
        issues,
        label=label,
    )

    required_keys = {
        "original_source",
        "original_source_manifest",
        "normalized_source",
        "source_manifest",
        "subject_isolation_binding",
        "subject_isolation_manifest",
        "mesh_preprocess_manifest",
        "semantic_image",
        "author_scaffold",
        "axial",
        "author_evidence_manifest",
        "source_overlay_manifest",
        "source_overlay_urdf",
        "source_overlay_glb",
    }
    missing = sorted(required_keys - files.keys())
    if missing:
        _add_issue(issues, label, "files is missing required keys: " + ", ".join(missing))

    # Raw source closures and preprocessing internals remain Host-only on both arms.
    required_author_inputs = {
        "normalized_source",
        "semantic_image",
        "author_scaffold",
        "axial",
        "author_evidence_manifest",
    }
    raw_author_inputs = manifest.get("author_inputs")
    raw_host_inputs = manifest.get("host_preprocessing_inputs")
    if not isinstance(raw_author_inputs, list) or not all(
        isinstance(item, str) for item in raw_author_inputs
    ):
        _add_issue(issues, label, "author_inputs must be an array of file keys")
        author_inputs: set[str] = set()
    else:
        author_inputs = set(raw_author_inputs)
        if len(author_inputs) != len(raw_author_inputs):
            _add_issue(issues, label, "author_inputs must not contain duplicates")
        if author_inputs != required_author_inputs:
            _add_issue(
                issues,
                label,
                "author_inputs must contain exactly the declared Reconstruction Author allowlist",
            )
    if not isinstance(raw_host_inputs, list) or not all(
        isinstance(item, str) for item in raw_host_inputs
    ):
        _add_issue(issues, label, "host_preprocessing_inputs must be an array of file keys")
        host_inputs: set[str] = set()
    else:
        host_inputs = set(raw_host_inputs)
        if len(host_inputs) != len(raw_host_inputs):
            _add_issue(issues, label, "host_preprocessing_inputs must not contain duplicates")
    if author_inputs & host_inputs:
        _add_issue(issues, label, "Author and Host input sets must be disjoint")
    if author_inputs | host_inputs != set(files):
        _add_issue(issues, label, "Author and Host input sets must partition files")
    for key, descriptor in files.items():
        expected_author_input = key in author_inputs
        if descriptor.get("author_input") is not expected_author_input:
            _add_issue(
                issues,
                label,
                f"files[{key!r}].author_input does not match the input partition",
            )

    channels = manifest.get("channels")
    if not isinstance(channels, Mapping):
        _add_issue(issues, label, "channels must be an object")
        return
    original_source = channels.get("original_source")
    expected_source_dependencies = sorted(
        key for key in files if key.startswith("source_dependency_")
    )
    if not isinstance(original_source, Mapping):
        _add_issue(issues, label, "channels.original_source must be an object")
    else:
        if original_source.get("entry_key") != "original_source":
            _add_issue(
                issues,
                label,
                "channels.original_source.entry_key must be 'original_source'",
            )
        if original_source.get("dependency_keys") != expected_source_dependencies:
            _add_issue(
                issues,
                label,
                "channels.original_source.dependency_keys must bind the complete copied closure",
            )
        if original_source.get("manifest_key") != "original_source_manifest":
            _add_issue(
                issues,
                label,
                "channels.original_source.manifest_key does not match persisted provenance",
            )

    subject_isolation = channels.get("subject_isolation")
    if not isinstance(subject_isolation, Mapping) or subject_isolation != {
        "binding_key": "subject_isolation_binding",
        "manifest_key": "subject_isolation_manifest",
    }:
        _add_issue(
            issues,
            label,
            "channels.subject_isolation must bind its decision and manifest",
        )
    mesh_preprocess = channels.get("mesh_preprocess")
    if not isinstance(mesh_preprocess, Mapping) or mesh_preprocess.get("manifest_key") != (
        "mesh_preprocess_manifest"
    ):
        _add_issue(
            issues,
            label,
            "channels.mesh_preprocess must bind mesh_preprocess_manifest",
        )

    semantic = channels.get("semantic_image")
    if not isinstance(semantic, Mapping):
        _add_issue(issues, label, "channels.semantic_image must be an object")
    else:
        if semantic.get("file_key") != "semantic_image":
            _add_issue(
                issues,
                label,
                "channels.semantic_image.file_key must be 'semantic_image'",
            )
        if semantic.get("provenance_role") != RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE:
            _add_issue(
                issues,
                label,
                "channels.semantic_image.provenance_role must be "
                f"{RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE!r}",
            )

    normalized = channels.get("normalized_source")
    if not isinstance(normalized, Mapping):
        _add_issue(issues, label, "channels.normalized_source must be an object")
    elif (
        normalized.get("file_key") != "normalized_source"
        or normalized.get("same_frame_as_candidate") is not True
    ):
        _add_issue(
            issues,
            label,
            "channels.normalized_source must bind normalized_source in the candidate frame",
        )

    overlay = channels.get("source_overlay")
    if not isinstance(overlay, Mapping):
        _add_issue(issues, label, "channels.source_overlay must be an object")
    else:
        if overlay.get("manifest_key") != "source_overlay_manifest":
            _add_issue(
                issues,
                label,
                "channels.source_overlay.manifest_key must be 'source_overlay_manifest'",
            )
        if overlay.get("base_path") != "inputs/source/overlay/carrier":
            _add_issue(
                issues,
                label,
                "channels.source_overlay.base_path must name the persisted carrier",
            )
        if overlay.get("urdf_key") != "source_overlay_urdf":
            _add_issue(
                issues,
                label,
                "channels.source_overlay.urdf_key must be 'source_overlay_urdf'",
            )
        if overlay.get("glb_key") != "source_overlay_glb":
            _add_issue(
                issues,
                label,
                "channels.source_overlay.glb_key must be 'source_overlay_glb'",
            )
        if overlay.get("same_frame_as_candidate") is not True:
            _add_issue(
                issues,
                label,
                "channels.source_overlay.same_frame_as_candidate must be true",
            )
        if overlay.get("viewer_only") is not True:
            _add_issue(issues, label, "channels.source_overlay.viewer_only must be true")
        expected_overlay_dependencies = sorted(
            key for key in files if key.startswith("source_overlay_dependency_")
        )
        if overlay.get("dependency_keys") != expected_overlay_dependencies:
            _add_issue(
                issues,
                label,
                "channels.source_overlay.dependency_keys must bind the copied carrier closure",
            )

    geometry = channels.get("geometry")
    if not isinstance(geometry, Mapping):
        _add_issue(issues, label, "channels.geometry must be an object")
    else:
        if geometry.get("axial_key") != "axial":
            _add_issue(issues, label, "channels.geometry.axial_key must be 'axial'")
        if geometry.get("evidence_manifest_key") != "author_evidence_manifest":
            _add_issue(
                issues,
                label,
                "channels.geometry.evidence_manifest_key must be 'author_evidence_manifest'",
            )

    _validate_reconstruction_audit_closures(
        channels.get("audit_closures"),
        files,
        issues,
        label=label,
    )


def _validate_reconstruction_audit_closures(
    raw_closures: Any,
    files: Mapping[str, Mapping[str, Any]],
    issues: list[ArtifactContractIssue],
    *,
    label: str,
) -> None:
    """Validate complete relocated audit trees and byte-identical primary bindings."""

    if not isinstance(raw_closures, Mapping):
        _add_issue(issues, label, "channels.audit_closures must be an object")
        return
    expected_channels = {
        "subject_isolation": "inputs/preprocessing/audit/subject_isolation",
        "mesh_preprocess": "inputs/preprocessing/audit/mesh_preprocess",
        "source_overlay": "inputs/preprocessing/audit/source_overlay",
        "author_context": "inputs/author_context",
    }
    expected_manifest_keys = {
        "subject_isolation": "subject_isolation_manifest",
        "mesh_preprocess": "mesh_preprocess_manifest",
        "source_overlay": "source_overlay_manifest",
        "author_context": "author_evidence_manifest",
    }
    described_audit_keys: set[str] = set()
    for channel, base_path in expected_channels.items():
        raw_channel = raw_closures.get(channel)
        if not isinstance(raw_channel, Mapping):
            _add_issue(issues, label, f"channels.audit_closures.{channel} must be an object")
            continue
        if raw_channel.get("base_path") != base_path:
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel}.base_path must be {base_path!r}",
            )
        raw_file_keys = raw_channel.get("file_keys")
        if not isinstance(raw_file_keys, list) or not all(
            isinstance(item, str) for item in raw_file_keys
        ):
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel}.file_keys must be an array of keys",
            )
            file_keys: list[str] = []
        else:
            file_keys = list(raw_file_keys)
            if len(set(file_keys)) != len(file_keys):
                _add_issue(
                    issues,
                    label,
                    f"channels.audit_closures.{channel}.file_keys must not contain duplicates",
                )
        expected_channel_keys = {
            key for key, descriptor in files.items() if descriptor.get("audit_channel") == channel
        }
        if set(file_keys) != expected_channel_keys:
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel}.file_keys must bind every audit file",
            )
        described_audit_keys.update(file_keys)
        for key in file_keys:
            descriptor = files.get(key)
            if descriptor is None:
                continue
            path = descriptor.get("path")
            relative = descriptor.get("audit_relative_path")
            expected_path = (
                f"{base_path}/{relative}" if isinstance(relative, str) and relative else None
            )
            if descriptor.get("audit_channel") != channel or path != expected_path:
                _add_issue(
                    issues,
                    label,
                    f"audit descriptor {key!r} does not resolve below the declared base path",
                )

        manifest_primary_key = expected_manifest_keys[channel]
        if raw_channel.get("manifest_primary_key") != manifest_primary_key:
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel}.manifest_primary_key must be "
                f"{manifest_primary_key!r}",
            )
        manifest_audit_key = raw_channel.get("manifest_audit_key")
        primary = files.get(manifest_primary_key)
        audit = files.get(manifest_audit_key) if isinstance(manifest_audit_key, str) else None
        if primary is None or audit is None or audit.get("audit_channel") != channel:
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel}.manifest_audit_key is invalid",
            )
        elif primary.get("sha256") != audit.get("sha256"):
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.{channel} manifest copies are not byte-identical",
            )

    all_audit_keys = {
        key for key, descriptor in files.items() if isinstance(descriptor.get("audit_channel"), str)
    }
    if described_audit_keys != all_audit_keys:
        _add_issue(issues, label, "channels.audit_closures must partition every audit descriptor")
    if raw_closures.get("all_copies_byte_identical") is not True:
        _add_issue(issues, label, "channels.audit_closures.all_copies_byte_identical must be true")
    if raw_closures.get("internal_relative_paths_resolve_against_each_base_path") is not True:
        _add_issue(
            issues,
            label,
            "channels.audit_closures.internal_relative_paths_resolve_against_each_base_path "
            "must be true",
        )

    bindings = raw_closures.get("primary_bindings")
    if not isinstance(bindings, Mapping):
        _add_issue(issues, label, "channels.audit_closures.primary_bindings must be an object")
        return
    required_primary_bindings = {
        "subject_isolation_manifest",
        "normalized_source",
        "source_manifest",
        "subject_isolation_binding",
        "mesh_preprocess_manifest",
        "source_overlay_manifest",
        "source_overlay_urdf",
        "source_overlay_glb",
        "author_evidence_manifest",
        "author_scaffold",
        "semantic_image",
        "axial",
    }
    if "voxel_sidecar" in files:
        required_primary_bindings.add("voxel_sidecar")
    if set(bindings) != required_primary_bindings:
        _add_issue(
            issues,
            label,
            "channels.audit_closures.primary_bindings must bind every relocated primary input",
        )
    for primary_key, audit_key in bindings.items():
        primary = files.get(primary_key)
        audit = files.get(audit_key) if isinstance(audit_key, str) else None
        if primary is None or audit is None:
            _add_issue(
                issues,
                label,
                f"channels.audit_closures.primary_bindings[{primary_key!r}] is invalid",
            )
            continue
        if audit_key not in all_audit_keys or primary.get("sha256") != audit.get("sha256"):
            _add_issue(
                issues,
                label,
                f"primary input {primary_key!r} is not byte-identical to its audit copy",
            )


def validate_urdf_dependencies(
    root: Path,
    artifact_hashes: Mapping[str, Any],
) -> tuple[ArtifactContractIssue, ...]:
    """Validate local URDF references and hashes during compile and canonical validation."""
    root = Path(root).resolve()
    issues: list[ArtifactContractIssue] = []
    urdf_path = root / "model.urdf"
    if not urdf_path.is_file():
        return ()
    try:
        document = ET.parse(urdf_path)
    except (OSError, ET.ParseError) as exc:
        _add_issue(issues, "model.urdf", f"invalid URDF XML: {exc}")
        return tuple(issues)
    for element in document.iter():
        for attribute_name in ("filename", "uri", "url"):
            value = element.attrib.get(attribute_name)
            if value is None:
                continue
            if not _safe_relative_path(value):
                _add_issue(
                    issues,
                    "model.urdf",
                    f"{attribute_name} contains an unsafe path: {value}",
                )
                continue
            dependency = _contained_path(root, value, issues, label="model.urdf")
            if dependency is None or not dependency.is_file():
                _add_issue(issues, value, "missing URDF dependency")
                continue
            _validate_hash(dependency, Path(value).as_posix(), artifact_hashes, issues)
    return tuple(issues)


def _validate_dependency_closures(
    root: Path,
    report: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
) -> None:
    closures = report.get("usd_dependency_closure")
    if not isinstance(closures, Mapping):
        _add_issue(
            issues,
            "compile_report.json",
            "usd_dependency_closure must be an object",
        )
        return
    try:
        from pxr import Sdf, UsdUtils
    except ImportError as exc:
        _add_issue(
            issues,
            "compile_report.json",
            f"USD dependency validation is unavailable: {exc}",
        )
        return

    for entry_path in USD_STAGE_ARTIFACTS:
        reported = closures.get(entry_path)
        if (
            not isinstance(reported, list)
            or not reported
            or not all(isinstance(value, str) and value for value in reported)
        ):
            _add_issue(
                issues,
                "compile_report.json",
                f"usd_dependency_closure.{entry_path} must be a non-empty string list",
            )
            continue
        normalized_reported = [str(value) for value in reported]
        if normalized_reported != sorted(set(normalized_reported)):
            _add_issue(
                issues,
                "compile_report.json",
                f"usd_dependency_closure.{entry_path} must be sorted and unique",
            )
        if entry_path not in normalized_reported:
            _add_issue(
                issues,
                "compile_report.json",
                f"usd_dependency_closure.{entry_path} must contain its entry path",
            )

        entry = _contained_path(root, entry_path, issues, label="compile_report.json")
        if entry is None or not entry.is_file():
            continue
        try:
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(entry)))
        except Exception as exc:
            _add_issue(
                issues,
                entry_path,
                f"could not compute USD dependencies: {exc}",
            )
            continue
        if unresolved:
            _add_issue(
                issues,
                entry_path,
                "USD dependencies are unresolved: "
                + ", ".join(sorted(str(value) for value in unresolved)),
            )

        actual: set[str] = set()
        for layer in layers:
            layer_path = Path(str(layer.realPath or layer.identifier))
            relative = _relative_dependency(root, layer_path, entry_path, issues)
            if relative is not None:
                actual.add(relative)
            try:
                dependency_groups = UsdUtils.ExtractExternalReferences(layer.identifier)
            except Exception as exc:
                _add_issue(
                    issues,
                    entry_path,
                    f"could not inspect USD layer {layer.identifier}: {exc}",
                )
                continue
            for dependency_group in dependency_groups:
                for authored_path in dependency_group:
                    if not _safe_authored_dependency_path(str(authored_path)):
                        _add_issue(
                            issues,
                            entry_path,
                            f"USD layer authors an unsafe dependency: {authored_path}",
                        )
        for asset in assets:
            relative = _relative_dependency(root, Path(str(asset)), entry_path, issues)
            if relative is not None:
                actual.add(relative)

        expected = sorted(actual)
        if normalized_reported != expected:
            _add_issue(
                issues,
                "compile_report.json",
                f"usd_dependency_closure.{entry_path} must equal {expected!r}",
            )
        for relative_path in expected:
            dependency = _contained_path(
                root,
                relative_path,
                issues,
                label="compile_report.json",
            )
            if dependency is None or not dependency.is_file():
                _add_issue(issues, relative_path, "missing reported USD dependency")
                continue
            _validate_hash(dependency, relative_path, artifact_hashes, issues)


def _relative_dependency(
    root: Path,
    dependency: Path,
    entry_path: str,
    issues: list[ArtifactContractIssue],
) -> str | None:
    try:
        return dependency.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        _add_issue(
            issues,
            entry_path,
            f"USD dependency escapes the materialization: {dependency}",
        )
        return None


def _manifest_object(
    source: Mapping[str, Any] | Path | None,
    issues: list[ArtifactContractIssue],
) -> Mapping[str, Any] | None:
    if isinstance(source, Path):
        return _json_object(source, "generation.json", issues)
    if isinstance(source, Mapping):
        return source
    _add_issue(issues, "generation.json", "missing Generation manifest")
    return None


def _pipeline_manifest_object(
    source: Mapping[str, Any] | Path | None,
    *,
    label: str,
    issues: list[ArtifactContractIssue],
) -> Mapping[str, Any] | None:
    if isinstance(source, Path):
        return _json_object(source, label, issues)
    if isinstance(source, Mapping):
        return source
    _add_issue(issues, label, f"missing {label}")
    return None


def _json_object(
    path: Path,
    label: str,
    issues: list[ArtifactContractIssue],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        _add_issue(issues, label, "missing JSON object")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add_issue(issues, label, f"invalid JSON: {exc}")
        return None
    if not isinstance(value, Mapping):
        _add_issue(issues, label, "expected a JSON object")
        return None
    return value


def _integer_schema(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _require_file(
    root: Path,
    relative_path: str,
    issues: list[ArtifactContractIssue],
) -> Path | None:
    path = _contained_path(root, relative_path, issues, label=relative_path)
    if path is None:
        return None
    if not path.is_file():
        _add_issue(issues, relative_path, "missing compiled artifact")
        return None
    return path


def _contained_path(
    root: Path,
    relative_path: str,
    issues: list[ArtifactContractIssue],
    *,
    label: str,
) -> Path | None:
    if not _safe_relative_path(relative_path):
        _add_issue(issues, label, f"unsafe relative path: {relative_path}")
        return None
    candidate = root / Path(relative_path)
    try:
        candidate.resolve().relative_to(root)
    except (OSError, ValueError):
        _add_issue(issues, label, f"path escapes the materialization: {relative_path}")
        return None
    return candidate


def _safe_relative_path(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    path = Path(normalized)
    windows_path = PureWindowsPath(normalized)
    return not (
        path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
    )


def _safe_authored_dependency_path(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return True
    package_path = normalized.split("[", 1)[0]
    return not (
        Path(package_path).is_absolute()
        or PureWindowsPath(package_path).is_absolute()
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", package_path)
    )


def _validate_hash(
    path: Path,
    relative_path: str,
    artifact_hashes: Mapping[str, Any],
    issues: list[ArtifactContractIssue],
) -> None:
    expected = artifact_hashes.get(relative_path)
    if not isinstance(expected, str) or not _HASH_RE.fullmatch(expected):
        _add_issue(
            issues,
            "compile_report.json",
            f"artifacts.{relative_path} must be a lowercase SHA-256",
        )
        return
    try:
        actual = _sha256(path)
    except OSError as exc:
        _add_issue(issues, relative_path, f"could not hash artifact: {exc}")
        return
    if actual != expected:
        _add_issue(
            issues,
            "compile_report.json",
            f"artifacts.{relative_path} does not match the file",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _add_issue(
    issues: list[ArtifactContractIssue],
    path: str,
    message: str,
) -> None:
    issue = ArtifactContractIssue(path=path, message=message)
    if issue not in issues:
        issues.append(issue)


__all__ = [
    "ASSET_INTERFACE_ARTIFACT",
    "ASSET_STRUCTURE_LAYER_ARTIFACTS",
    "ASSET_STRUCTURE_PROFILE",
    "ASSET_STRUCTURE_VERSION",
    "ARTICULATION_ROOT_BODY_BINDING_CHECK",
    "ArtifactContractIssue",
    "COMPILE_REPORT_SCHEMA_VERSION",
    "FINITE_AUTHORED_VALUES_CHECK",
    "GENERATION_MANIFEST_SCHEMA_VERSION",
    "GENERATION_ROOT_FLOATING_CHECK",
    "GENERATION_VALIDATION_PROFILE",
    "MaterializationDisplayResult",
    "MaterializationDisplayStatus",
    "MUJOCO_LAYER_ARTIFACT",
    "NEWTON_USDC_ARTIFACT",
    "PHYSICS_BACKENDS",
    "PHYSICS_ENTRY_ARTIFACTS",
    "PHYSX_LAYER_ARTIFACT",
    "PHYSX_USDC_ARTIFACT",
    "PROVENANCE_SCHEMA_VERSION",
    "RECONSTRUCTION_APPROVED_INPUTS_SCHEMA",
    "RECONSTRUCTION_INPUT_MANIFEST_SCHEMA",
    "RECONSTRUCTION_INPUT_MANIFEST_SCHEMA_VERSION",
    "RECONSTRUCTION_MANIFEST_SCHEMA_VERSION",
    "RECONSTRUCTION_SEMANTIC_PROVENANCE_ROLE",
    "RECONSTRUCTION_VALIDATION_PROFILE",
    "REQUIRED_BACKEND_CHECKS",
    "REQUIRED_ASSET_STRUCTURE_ARTIFACTS",
    "REQUIRED_COMPILE_CHECKS",
    "SHARED_PHYSICS_ARTIFACT",
    "STANDARD_RENDER_ARTIFACTS",
    "USDZ_ARTIFACT",
    "USD_STAGE_ARTIFACTS",
    "MaterializationContractResult",
    "classify_materialization_displayability",
    "classify_reconstruction_materialization_displayability",
    "cheap_display_hint",
    "exporter_asset_structure_contract",
    "physics_output_contract",
    "validate_materialization_contract",
    "validate_reconstruction_materialization_contract",
    "validate_urdf_dependencies",
]
