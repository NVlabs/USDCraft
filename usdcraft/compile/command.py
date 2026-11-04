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
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from storage.artifact_contract import (
    ASSET_INTERFACE_ARTIFACT,
    COMPILE_REPORT_SCHEMA_VERSION,
    GENERATION_ROOT_FLOATING_CHECK,
    GENERATION_VALIDATION_PROFILE,
    MUJOCO_LAYER_ARTIFACT,
    NEWTON_USDC_ARTIFACT,
    PHYSICS_BACKENDS,
    PHYSX_LAYER_ARTIFACT,
    PHYSX_USDC_ARTIFACT,
    REQUIRED_BACKEND_CHECKS,
    SHARED_PHYSICS_ARTIFACT,
    USDZ_ARTIFACT,
    exporter_asset_structure_contract,
    physics_output_contract,
    validate_urdf_dependencies,
)
from usdcraft.compile.compiler import compile_urdf_report_maybe_timeout
from usdcraft.compile.models import CompileSignalBundle
from usdcraft.render.contract import SELF_CHECK_RENDER_SIZE, STANDARD_RENDER_VIEWS
from usdcraft.render.offscreen import render_urdf_views


@dataclass(slots=True, frozen=True)
class AuthoringCheckResult:
    """Non-canonical diagnostics returned by the authoring-only check command."""

    model_path: Path
    warnings: tuple[str, ...]
    signal_bundle: CompileSignalBundle


@dataclass(slots=True, frozen=True)
class CompileArtifacts:
    output_dir: Path
    urdf_path: Path
    usdc_path: Path
    newton_usdc_path: Path
    asset_usda_path: Path
    physics_usdc_path: Path
    physx_usda_path: Path
    mujoco_usda_path: Path
    usdz_path: Path
    report_path: Path
    render_paths: tuple[Path, ...]
    warnings: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_urdf_assets(
    urdf_xml: str,
    *,
    source_root: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Materialize every referenced local dependency beside the URDF."""

    try:
        document = ET.fromstring(urdf_xml)
    except ET.ParseError as exc:
        raise RuntimeError(f"Compiled URDF is invalid XML: {exc}") from exc

    relative_paths: set[Path] = set()
    for element in document.iter():
        for attribute_name in ("filename", "uri", "url"):
            value = element.attrib.get(attribute_name)
            if value is None:
                continue
            normalized = value.strip()
            relative_path = Path(normalized)
            if (
                not normalized
                or relative_path.is_absolute()
                or PureWindowsPath(normalized).is_absolute()
                or ".." in relative_path.parts
                or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", normalized)
            ):
                raise RuntimeError(
                    f"Compiled URDF {attribute_name} contains an unsafe path: {value}"
                )
            relative_paths.add(relative_path)

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    target_assets = output_dir / "assets"
    if target_assets.exists():
        shutil.rmtree(target_assets)

    copied: list[Path] = []
    for relative_path in sorted(relative_paths, key=lambda path: path.as_posix()):
        source = (source_root / relative_path).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(f"URDF asset escapes the model directory: {relative_path}") from exc
        if not source.is_file():
            raise RuntimeError(f"URDF asset is missing: {source}")
        destination = output_dir / relative_path
        if not destination.resolve().is_relative_to(output_dir):
            raise RuntimeError(f"URDF asset escapes the compile output: {relative_path}")
        if relative_path.as_posix() in {"model.urdf", "model.usdz", "compile_report.json"}:
            raise RuntimeError(f"URDF asset conflicts with compiled output: {relative_path}")
        if destination.exists() and relative_path.parts[0] == "isaac":
            if not destination.is_file() or _sha256(destination) != _sha256(source):
                raise RuntimeError(f"URDF asset conflicts with compiled output: {relative_path}")
            copied.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def _relative_dependency_path(path: Path, *, output_dir: Path, entry_path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"USD dependency for {entry_path} escapes the compile output: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"USD dependency for {entry_path} is missing: {resolved}")
    return relative.as_posix()


def _validate_authored_dependency_path(
    value: object,
    *,
    layer_identifier: str,
    entry_path: Path,
) -> None:
    raw = str(value or "").strip()
    if not raw:
        return
    package_path = raw.split("[", 1)[0]
    if (
        Path(package_path).is_absolute()
        or PureWindowsPath(package_path).is_absolute()
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", package_path)
    ):
        raise RuntimeError(
            f"USD layer {layer_identifier} authors an absolute dependency for {entry_path}: {raw}"
        )


def _validate_usd_dependency_closure(
    entry_path: Path,
    *,
    output_dir: Path,
) -> tuple[str, ...]:
    from pxr import Usd, UsdUtils

    stage = Usd.Stage.Open(entry_path.as_posix())
    if stage is None:
        raise RuntimeError(f"USD entry could not be opened: {entry_path}")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(entry_path.as_posix())
    if unresolved:
        raise RuntimeError(
            f"USD entry has unresolved dependencies: {entry_path}: "
            + ", ".join(sorted(str(item) for item in unresolved))
        )

    closure: set[str] = set()
    for layer in layers:
        identifier = str(layer.identifier)
        layer_path = Path(str(layer.realPath or identifier))
        closure.add(
            _relative_dependency_path(layer_path, output_dir=output_dir, entry_path=entry_path)
        )
        dependency_groups = UsdUtils.ExtractExternalReferences(identifier)
        for dependency_group in dependency_groups:
            for authored_path in dependency_group:
                _validate_authored_dependency_path(
                    authored_path,
                    layer_identifier=identifier,
                    entry_path=entry_path,
                )
    for asset in assets:
        closure.add(
            _relative_dependency_path(
                Path(str(asset)), output_dir=output_dir, entry_path=entry_path
            )
        )
    return tuple(sorted(closure))


def _require_backend_validation(metrics: dict[str, object]) -> None:
    backend_validation = metrics.get("backend_validation")
    if not isinstance(backend_validation, dict):
        raise RuntimeError("Strict USD export did not report backend_validation evidence.")
    failures: list[str] = []
    for backend in PHYSICS_BACKENDS:
        evidence = backend_validation.get(backend)
        required_checks = REQUIRED_BACKEND_CHECKS[backend]
        checks = evidence.get("checks") if isinstance(evidence, dict) else None
        check_names = (
            {str(item) for item in checks if isinstance(item, str)}
            if isinstance(checks, list)
            else set()
        )
        if (
            not isinstance(evidence, dict)
            or evidence.get("status") != "success"
            or evidence.get("runtime_status") not in {"not_run", "success"}
            or not required_checks.issubset(check_names)
        ):
            failures.append(backend)
    if failures:
        raise RuntimeError(
            "Strict USD export did not pass backend validation for: " + ", ".join(failures)
        )


def _require_asset_structure_validation(metrics: dict[str, object]) -> None:
    actual = metrics.get("asset_structure")
    expected = exporter_asset_structure_contract()
    if actual != expected:
        raise RuntimeError(
            "Strict USD export did not report the required Asset Structure "
            f"{expected['version']} {expected['profile']!r} evidence."
        )


def _require_generation_root_floating(asset_usda_path: Path) -> None:
    """Require the effective exported Generation asset root to remain floating.

    This is deliberately narrower than the Generation authoring policy. It does not inspect
    source shape, require an explicit constructor argument, or judge installation semantics; it
    only verifies the objective root mode carried by the exported asset interface.
    """

    from pxr import Usd

    stage = Usd.Stage.Open(asset_usda_path.as_posix())
    if stage is None:
        raise RuntimeError(
            f"Generation root-mode check could not open the asset interface: {asset_usda_path}"
        )
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(
            f"Generation root-mode check found no default prim in: {asset_usda_path}"
        )
    actual = root.GetCustomDataByKey("usdcraft:baseMode")
    if not isinstance(actual, str) or actual != "floating":
        rendered = "missing" if actual is None else repr(actual)
        raise RuntimeError(
            "Generation assets must export a floating root; "
            f"{asset_usda_path} declares usdcraft:baseMode={rendered}. "
            "Set ArticulatedObject(base_mode=BaseMode.FLOATING)."
        )


def check_model(model_path: Path) -> AuthoringCheckResult:
    """Run model-authored tests and semantic diagnostics without producing canonical artifacts.

    This command exists for the Generation Agent's repair loop. Its failures are intentionally
    not replayed by the final Host gate and are never written into the canonical compile report.
    """

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model.py not found: {model_path}")
    report = compile_urdf_report_maybe_timeout(model_path, run_checks=True)
    return AuthoringCheckResult(
        model_path=model_path,
        warnings=tuple(str(item) for item in report.warnings),
        signal_bundle=report.signal_bundle,
    )


def compile_model(
    model_path: Path,
    output_dir: Path,
    *,
    clean: bool = True,
    render: bool = True,
    require_generation_root_floating: bool = False,
) -> CompileArtifacts:
    """Compile and validate the current dual-backend Generation artifact profile.

    Generated ``model.py`` is executable Python and must come from a trusted Codex run.
    The Host gate is deliberately limited to executable/exportable source, valid URDF/USD,
    complete output files, and resolvable dependencies. Model-authored tests and semantic
    geometry/physics quality checks belong to the authoring loop, not final promotion.
    """

    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model.py not found: {model_path}")
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = compile_urdf_report_maybe_timeout(
        model_path,
        run_checks=False,
        usd_output_dir=output_dir,
        usd_strict=True,
    )

    urdf_path = output_dir / "model.urdf"
    urdf_path.write_text(report.urdf_xml, encoding="utf-8")

    urdf_dependencies = _copy_urdf_assets(
        report.urdf_xml,
        source_root=model_path.parent,
        output_dir=output_dir,
    )

    usdc_path = output_dir / PHYSX_USDC_ARTIFACT
    newton_usdc_path = output_dir / NEWTON_USDC_ARTIFACT
    asset_usda_path = output_dir / ASSET_INTERFACE_ARTIFACT
    physics_usdc_path = output_dir / SHARED_PHYSICS_ARTIFACT
    physx_usda_path = output_dir / PHYSX_LAYER_ARTIFACT
    mujoco_usda_path = output_dir / MUJOCO_LAYER_ARTIFACT
    usdz_path = output_dir / USDZ_ARTIFACT
    required_usd_paths = (
        usdc_path,
        newton_usdc_path,
        asset_usda_path,
        physics_usdc_path,
        physx_usda_path,
        mujoco_usda_path,
    )
    missing = [path for path in (*required_usd_paths, usdz_path) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Strict USD export did not produce: " + ", ".join(str(path) for path in missing)
        )
    if require_generation_root_floating:
        _require_generation_root_floating(asset_usda_path)
    metrics = dict(report.artifact_metrics)
    _require_backend_validation(metrics)
    _require_asset_structure_validation(metrics)
    dependency_closure = {
        path.relative_to(output_dir).as_posix(): list(
            _validate_usd_dependency_closure(path, output_dir=output_dir)
        )
        for path in required_usd_paths
    }

    warnings = list(report.warnings)
    render_paths: tuple[Path, ...] = ()
    if render:
        try:
            render_paths = tuple(
                render_urdf_views(
                    output_dir,
                    output_dir / "renders",
                    views=STANDARD_RENDER_VIEWS,
                    size=SELF_CHECK_RENDER_SIZE,
                )
            )
            if len(render_paths) != len(STANDARD_RENDER_VIEWS):
                warnings.append(
                    "Generation render diagnostics produced an incomplete view set "
                    f"({len(render_paths)}/{len(STANDARD_RENDER_VIEWS)}); export remains valid."
                )
        except Exception as exc:
            warnings.append(
                f"Generation render diagnostics failed (non-blocking): {type(exc).__name__}: {exc}"
            )

    artifacts = {
        "model.urdf": _sha256(urdf_path),
        USDZ_ARTIFACT: _sha256(usdz_path),
    }
    dependency_artifacts = {
        relative_path for paths in dependency_closure.values() for relative_path in paths
    }
    dependency_artifacts.update(
        path.relative_to(output_dir).as_posix() for path in required_usd_paths
    )
    dependency_artifacts.update(
        path.relative_to(output_dir).as_posix() for path in urdf_dependencies
    )
    artifacts.update(
        {
            relative_path: _sha256(output_dir / relative_path)
            for relative_path in sorted(dependency_artifacts)
        }
    )
    artifacts.update({str(path.relative_to(output_dir)): _sha256(path) for path in render_paths})
    dependency_issues = validate_urdf_dependencies(output_dir, artifacts)
    if dependency_issues:
        raise RuntimeError(
            "URDF dependency validation failed: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in dependency_issues)
        )
    payload = {
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "status": "success",
        "validation_profile": GENERATION_VALIDATION_PROFILE,
        "asset_structure": physics_output_contract(),
        "model_path": str(model_path),
        "checks_run": [
            "urdf_export",
            "usd_strict",
            "physx_usd_strict",
            "newton_usd_strict",
            "usd_dependency_closure",
            *([GENERATION_ROOT_FLOATING_CHECK] if require_generation_root_floating else []),
            *(["generation_render_diagnostics"] if render else []),
        ],
        "warnings": warnings,
        "notes": list(report.notes),
        "artifacts": artifacts,
        "usd_dependency_closure": dependency_closure,
        "metrics": metrics,
        "signal_bundle": report.signal_bundle.to_dict(),
    }
    report_path = output_dir / "compile_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CompileArtifacts(
        output_dir=output_dir,
        urdf_path=urdf_path,
        usdc_path=usdc_path,
        newton_usdc_path=newton_usdc_path,
        asset_usda_path=asset_usda_path,
        physics_usdc_path=physics_usdc_path,
        physx_usda_path=physx_usda_path,
        mujoco_usda_path=mujoco_usda_path,
        usdz_path=usdz_path,
        report_path=report_path,
        render_paths=render_paths,
        warnings=tuple(warnings),
    )
