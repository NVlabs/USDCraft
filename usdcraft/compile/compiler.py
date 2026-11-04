# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import ast
import logging
import math
import os
import runpy
import sys
import threading
import traceback
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sdk
from sdk._core.assets import activate_asset_session, asset_session_for_script
from storage.artifact_contract import ASSET_STRUCTURE_LAYER_ARTIFACTS
from usdcraft.compile.feedback import build_compile_signal_bundle
from usdcraft.compile.models import CompileReport, CompileSignalBundle
from usdcraft.compile.mp_utils import get_mp_context
from usdcraft.compile.physics_checks import (
    _check_generation_gravity_hold_coverage,
    _check_generation_physics_completeness,
    _check_generation_physx_armature_stability,
)
from usdcraft.compile.test_report import (
    _build_test_report,
    _merge_test_reports,
    _raise_for_failed_test_report,
    _run_required_tests,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)

_COMPILER_SAMPLED_POSE_OVERLAP_CHECK = "compiler_sampled_pose_overlap"
_COMPILER_SAMPLED_POSE_MAX_SAMPLES = 128
_MODEL_EXECUTION_LOCK = threading.Lock()


COMPILE_TIMEOUT_ENV = "USDCRAFT_COMPILE_TIMEOUT_SECONDS"
DEFAULT_COMPILE_TIMEOUT_SECONDS = 300.0


def compile_timeout_seconds(environ: Mapping[str, str] | None = None) -> float:
    """Return the compile-worker timeout from ``USDCRAFT_COMPILE_TIMEOUT_SECONDS``.

    Unset or empty uses the 300 s default. Any other value must be a finite positive number of
    seconds; invalid, zero or negative values are configuration errors, never silently replaced.
    """

    env = os.environ if environ is None else environ
    raw = env.get(COMPILE_TIMEOUT_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_COMPILE_TIMEOUT_SECONDS
    try:
        value = float(str(raw).strip())
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Invalid {COMPILE_TIMEOUT_ENV}={raw!r}: compile requires a positive finite "
            f"{COMPILE_TIMEOUT_ENV} in seconds (leave it unset or empty for the "
            f"{DEFAULT_COMPILE_TIMEOUT_SECONDS:.0f} s default)."
        )
    return value


def _extract_urdf_xml(
    globals_dict: dict,
    *,
    validate_export: bool = True,
    suppress_exceptions: bool = False,
) -> str | None:
    object_model = globals_dict.get("object_model")
    urdf_xml: str | None = None
    export_exc: Exception | None = None
    try:
        if object_model is not None:
            from sdk._core._urdf_export import compile_object_to_urdf_xml

            script_dir = globals_dict.get("__file__")
            asset_root = Path(script_dir).resolve().parent if isinstance(script_dir, str) else None
            urdf_xml = compile_object_to_urdf_xml(
                object_model,
                asset_root=asset_root,
                include_physical_collisions=True,
                validate=validate_export,
            )
            globals_dict["urdf_xml"] = urdf_xml
    except Exception as exc:
        export_exc = exc
        urdf_xml = None

    if not isinstance(urdf_xml, str):
        maybe_urdf_xml = globals_dict.get("urdf_xml")
        if isinstance(maybe_urdf_xml, str):
            urdf_xml = maybe_urdf_xml
    if not isinstance(urdf_xml, str) and export_exc is not None and not suppress_exceptions:
        raise export_exc
    return urdf_xml


def _attach_compiled_urdf_on_failure(
    exc: BaseException,
    *,
    urdf_xml: str | None,
    warnings: list[str],
    signal_bundle: CompileSignalBundle,
) -> BaseException:
    wrapped = RuntimeError(f"{type(exc).__name__}: {exc}")
    if isinstance(urdf_xml, str) and urdf_xml.strip():
        setattr(wrapped, "compiled_urdf_xml", urdf_xml)
    setattr(wrapped, "warnings", list(warnings))
    test_report = getattr(exc, "test_report", None)
    if test_report is not None:
        setattr(wrapped, "test_report", test_report)
    setattr(wrapped, "compile_signal_bundle", signal_bundle)
    return wrapped


def compile_urdf_report(
    script_path: Path,
    *,
    run_checks: bool = True,
    usd_output_dir: Path | None = None,
    usd_strict: bool = False,
) -> CompileReport:
    session = asset_session_for_script(script_path)
    with activate_asset_session(session):
        return _compile_urdf_report_impl(
            script_path,
            run_checks=run_checks,
            usd_output_dir=usd_output_dir,
            usd_strict=usd_strict,
        )


def _compile_urdf_report_impl(
    script_path: Path,
    *,
    run_checks: bool = True,
    usd_output_dir: Path | None = None,
    usd_strict: bool = False,
) -> CompileReport:
    """Execute a generated script and return export XML plus non-blocking warnings."""
    globals_dict = load_model_globals(script_path)
    warnings: list[str] = []
    notes: list[str] = []
    test_report = None
    script_path = script_path.resolve()
    if run_checks:
        try:
            authored_report = _run_required_tests(globals_dict)
            baseline_report = _run_compiler_owned_baseline_tests(
                globals_dict,
                script_path=script_path,
                authored_report=authored_report,
            )
            test_report = _merge_test_reports(authored_report, baseline_report)
            _raise_for_failed_test_report(test_report)
        except Exception as exc:
            urdf_xml = _extract_urdf_xml(globals_dict, suppress_exceptions=True)
            signal_bundle = build_compile_signal_bundle(
                status="failure",
                warnings=warnings,
                test_report=getattr(exc, "test_report", None),
                exc=exc,
            )
            wrapped = _attach_compiled_urdf_on_failure(
                exc,
                urdf_xml=urdf_xml,
                warnings=warnings,
                signal_bundle=signal_bundle,
            )
            raise wrapped from exc

        warnings.extend(str(item) for item in getattr(test_report, "warnings", ()) or ())

    urdf_xml = _extract_urdf_xml(globals_dict, validate_export=not run_checks)
    if not isinstance(urdf_xml, str):
        raise ValueError("object_model must compile into an exportable XML payload")
    artifacts: dict[str, str] = {}
    artifact_metrics: dict[str, object] = {}
    if usd_output_dir is not None:
        object_model = globals_dict.get("object_model")
        if object_model is None:
            message = "USD export skipped: model.py did not define object_model"
            if usd_strict:
                raise ValueError(message)
            warnings.append(message)
        else:
            try:
                from sdk._core._usd_export import export_object_to_usd

                usd_result = export_object_to_usd(
                    object_model,
                    usd_output_dir,
                    asset_root=script_path.parent,
                    include_physical_collisions=True,
                    validate=usd_strict or not run_checks,
                )
                asset_structure_paths = dict(usd_result.asset_structure_paths)
                if set(asset_structure_paths) != set(ASSET_STRUCTURE_LAYER_ARTIFACTS):
                    raise ValueError(
                        "USD exporter returned invalid Asset Structure roles: "
                        + ", ".join(sorted(str(role) for role in asset_structure_paths))
                    )
                expected_structure_paths = {
                    role: (Path(usd_output_dir) / relative_path).resolve()
                    for role, relative_path in ASSET_STRUCTURE_LAYER_ARTIFACTS.items()
                }
                for role, expected_path in expected_structure_paths.items():
                    if Path(asset_structure_paths[role]).resolve() != expected_path:
                        raise ValueError(
                            f"USD exporter returned unexpected {role!r} layer path: "
                            f"{asset_structure_paths[role]}"
                        )
                artifacts = {
                    "usdc": str(usd_result.usdc_path),
                    "newton_usdc": str(usd_result.newton_usdc_path),
                    "usdz": str(usd_result.usdz_path),
                    **{
                        f"asset_structure_{role}": str(path)
                        for role, path in asset_structure_paths.items()
                    },
                }
                artifact_metrics = dict(usd_result.metrics)
                warnings.extend(str(item) for item in usd_result.warnings)
                notes.extend(str(item) for item in getattr(usd_result, "notes", ()))
            except Exception as exc:
                if usd_strict:
                    raise
                warnings.append(f"USD export warning (URDF remains available): {exc}")

    signal_bundle = build_compile_signal_bundle(
        status="success",
        warnings=warnings,
        test_report=test_report,
    )
    return CompileReport(
        urdf_xml=urdf_xml,
        warnings=warnings,
        signal_bundle=signal_bundle,
        artifacts=artifacts,
        artifact_metrics=artifact_metrics,
        notes=notes,
    )


def load_model_globals(script_path: Path) -> dict[str, Any]:
    """Execute a model script and return its globals."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = script_path.resolve()
    with _MODEL_EXECUTION_LOCK:
        prev_cwd = Path.cwd()
        os.chdir(script_path.parent)
        sys.path.insert(0, str(repo_root))
        try:
            globals_dict = runpy.run_path(script_path.name)
        finally:
            os.chdir(prev_cwd)
            if sys.path and sys.path[0] == str(repo_root):
                sys.path.pop(0)
    return globals_dict


def _compile_worker(
    script_path_str: str,
    run_checks: bool,
    usd_output_dir_str: str | None,
    usd_strict: bool,
    timeout_seconds: float,
    conn: Connection,
) -> None:
    try:
        report = compile_urdf_report(
            Path(script_path_str),
            run_checks=run_checks,
            usd_output_dir=Path(usd_output_dir_str) if usd_output_dir_str else None,
            usd_strict=usd_strict,
        )
        payload = {
            "ok": True,
            "urdf_xml": report.urdf_xml,
            "warnings": report.warnings,
            "notes": report.notes,
            "signal_bundle": report.signal_bundle.to_dict(),
            "artifacts": report.artifacts,
            "artifact_metrics": report.artifact_metrics,
        }
        conn.send(payload)
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        compiled_urdf_xml = getattr(exc, "compiled_urdf_xml", None)
        if isinstance(compiled_urdf_xml, str) and compiled_urdf_xml.strip():
            payload["compiled_urdf_xml"] = compiled_urdf_xml
        warnings = getattr(exc, "warnings", None)
        if isinstance(warnings, list):
            payload["warnings"] = [str(w) for w in warnings]
        signal_bundle = getattr(exc, "compile_signal_bundle", None)
        if isinstance(signal_bundle, CompileSignalBundle):
            payload["signal_bundle"] = signal_bundle.to_dict()
        conn.send(payload)
    finally:
        with suppress(Exception):
            conn.close()


def compile_urdf_report_maybe_timeout(
    script_path: Path,
    *,
    run_checks: bool = True,
    usd_output_dir: Path | None = None,
    usd_strict: bool = False,
) -> CompileReport:
    """
    Run `compile_urdf_report` with a hard timeout to prevent indefinite hangs.

    The compile always runs in a worker process bounded by `compile_timeout_seconds()`
    (`USDCRAFT_COMPILE_TIMEOUT_SECONDS`, default 300) so accidental infinite geometry loops
    cannot hang the CLI.
    """
    timeout_seconds = compile_timeout_seconds()

    ctx = get_mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_compile_worker,
        args=(
            str(script_path),
            run_checks,
            str(usd_output_dir) if usd_output_dir is not None else None,
            usd_strict,
            timeout_seconds,
            child_conn,
        ),
        daemon=True,
    )
    proc.start()
    try:
        with suppress(Exception):
            child_conn.close()

        if parent_conn.poll(timeout_seconds):
            try:
                msg = parent_conn.recv()
            except EOFError:
                proc.join(timeout=2.0)
                exitcode = proc.exitcode
                if exitcode is not None and exitcode < 0:
                    detail = f"terminated by signal {-exitcode}"
                elif exitcode is not None:
                    detail = f"exit code {exitcode}"
                else:
                    detail = "unknown exit status"
                raise RuntimeError(
                    f"URDF compile worker died before returning a result ({detail}). "
                    "This usually indicates a crash in native geometry/USD code while "
                    "processing the generated model."
                ) from None
        else:
            with suppress(Exception):
                proc.terminate()
            proc.join(timeout=2.0)
            raise TimeoutError(
                f"URDF compile timed out after {timeout_seconds:.0f}s. "
                "This can happen if the generated script contains a long-running loop, "
                "expensive mesh processing, or a stalled SDK/export operation. "
                "To adjust the safety limit, set USDCRAFT_COMPILE_TIMEOUT_SECONDS."
            )
    finally:
        with suppress(Exception):
            parent_conn.close()

    proc.join(timeout=2.0)
    if proc.is_alive():
        with suppress(Exception):
            proc.terminate()
        proc.join(timeout=2.0)

    if not isinstance(msg, dict):
        raise RuntimeError(
            f"URDF compile failed: worker returned non-dict payload ({type(msg).__name__})"
        )
    if msg.get("ok") is True:
        urdf_xml = msg.get("urdf_xml")
        warnings = msg.get("warnings")
        notes = msg.get("notes")
        signal_bundle_payload = msg.get("signal_bundle")
        artifacts = msg.get("artifacts")
        artifact_metrics = msg.get("artifact_metrics")
        if not isinstance(urdf_xml, str):
            raise RuntimeError("URDF compile failed: missing urdf_xml from worker")
        if not isinstance(warnings, list):
            warnings = []
        if not isinstance(notes, list):
            notes = []
        if isinstance(signal_bundle_payload, dict):
            signal_bundle = CompileSignalBundle.from_dict(signal_bundle_payload)
        else:
            signal_bundle = build_compile_signal_bundle(status="success", warnings=warnings)
        return CompileReport(
            urdf_xml=urdf_xml,
            warnings=[str(w) for w in warnings],
            signal_bundle=signal_bundle,
            artifacts={str(k): str(v) for k, v in artifacts.items()}
            if isinstance(artifacts, dict)
            else {},
            artifact_metrics=dict(artifact_metrics) if isinstance(artifact_metrics, dict) else {},
            notes=[str(n) for n in notes],
        )

    error_text = str(msg.get("error", "Unknown compile worker error")).strip()
    error_type = str(msg.get("error_type", "")).strip()
    tb_text = str(msg.get("traceback", "")).strip()
    if tb_text:
        logger.debug("Compile worker traceback:\n%s", tb_text)
    exc = RuntimeError(error_text or "Unknown compile worker error")
    if error_type:
        setattr(exc, "remote_error_type", error_type)
    if tb_text:
        setattr(exc, "remote_traceback", tb_text)
    compiled_urdf_xml = msg.get("compiled_urdf_xml")
    if isinstance(compiled_urdf_xml, str) and compiled_urdf_xml.strip():
        setattr(exc, "compiled_urdf_xml", compiled_urdf_xml)
    warnings = msg.get("warnings")
    if isinstance(warnings, list):
        setattr(exc, "warnings", [str(w) for w in warnings])
    signal_bundle_payload = msg.get("signal_bundle")
    if isinstance(signal_bundle_payload, dict):
        setattr(
            exc,
            "compile_signal_bundle",
            CompileSignalBundle.from_dict(signal_bundle_payload),
        )
    raise exc


def _check_model_has_single_root_part(ctx: Any) -> bool:
    root_parts = getattr(ctx.model, "root_parts", None)
    if not callable(root_parts):
        return bool(ctx.check("check_single_root_part", False, "model has no .root_parts()"))

    try:
        roots = root_parts()
    except Exception as exc:
        return bool(ctx.check("check_single_root_part", False, f"{type(exc).__name__}: {exc}"))

    root_names = tuple(
        str(getattr(part, "name", "")).strip()
        for part in roots
        if str(getattr(part, "name", "")).strip()
    )
    if len(root_names) != 1:
        return bool(
            ctx.check(
                "check_single_root_part",
                False,
                f"Expected exactly one root part, found {len(root_names)}: {list(root_names)!r}",
            )
        )
    return bool(ctx.check("check_single_root_part", True))


def _check_no_shadowed_model_bindings(ctx: Any, script_path: Path) -> bool:
    """Reject duplicate top-level model factories or exported model bindings.

    Python keeps the last definition, so duplicates can compile while hiding dead or unintended
    asset code. A Generation model must expose one unambiguous factory and one exported object.
    """
    try:
        source = Path(script_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))
    except (OSError, ValueError, SyntaxError):
        # Read/syntax problems are surfaced by the compiler's own execution path.
        return bool(ctx.check("check_no_shadowed_model_bindings", True))

    build_defs = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_object_model"
    ]
    model_bindings: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "object_model" for target in targets):
            model_bindings.append(node.lineno)

    problems: list[str] = []
    if len(build_defs) > 1:
        problems.append(
            "multiple top-level `def build_object_model` definitions at lines "
            f"{', '.join(str(line) for line in build_defs)} (only the last is used)"
        )
    if len(model_bindings) > 1:
        problems.append(
            "multiple top-level `object_model = ...` bindings at lines "
            f"{', '.join(str(line) for line in model_bindings)} (only the last is exported)"
        )
    if problems:
        return bool(
            ctx.check(
                "check_no_shadowed_model_bindings",
                False,
                "model.py contains shadowed top-level model code; keep exactly one "
                "build_object_model definition and one object_model binding. Offending "
                "structure: " + "; ".join(problems) + ".",
            )
        )
    return bool(ctx.check("check_no_shadowed_model_bindings", True))


def _apply_authored_allowances_to_baseline_context(ctx: Any, authored_report: object) -> None:
    for part_name in getattr(authored_report, "allowed_isolated_parts", ()):
        ctx.allow_isolated_part(
            str(part_name),
            reason="carried over from authored run_tests() allowance",
        )

    for overlap in getattr(authored_report, "allowed_overlaps", ()):
        ctx.allow_overlap(
            str(getattr(overlap, "link_a", "")),
            str(getattr(overlap, "link_b", "")),
            reason=str(getattr(overlap, "reason", "")),
            elem_a=(
                None
                if getattr(overlap, "elem_a", None) is None
                else str(getattr(overlap, "elem_a"))
            ),
            elem_b=(
                None
                if getattr(overlap, "elem_b", None) is None
                else str(getattr(overlap, "elem_b"))
            ),
        )

    for item in getattr(authored_report, "disconnected_geometry_classifications", ()):
        ctx.classify_disconnected_geometry(
            str(getattr(item, "part", "")),
            classification=str(getattr(item, "classification", "")),
            elements=tuple(str(value) for value in getattr(item, "elements", ())),
            reason=str(getattr(item, "reason", "")),
        )


def _run_compiler_owned_baseline_tests(
    globals_dict: dict,
    *,
    script_path: Path,
    authored_report: object,
) -> object:
    object_model = globals_dict.get("object_model")
    if object_model is None:
        raise ValueError("Generated script must define top-level `object_model`")

    ctx = sdk.TestContext(object_model, asset_root=script_path.parent)
    _apply_authored_allowances_to_baseline_context(ctx, authored_report)
    ctx.check_model_valid()
    _check_model_has_single_root_part(ctx)
    _check_no_shadowed_model_bindings(ctx, script_path)
    preliminary_report = ctx.report()
    if not bool(getattr(preliminary_report, "passed", False)):
        return _build_test_report(
            type(preliminary_report),
            checks=tuple(str(name) for name in getattr(preliminary_report, "checks", ())),
            failures=tuple(getattr(preliminary_report, "failures", ())),
            warnings=tuple(str(item) for item in getattr(preliminary_report, "warnings", ())),
            allowances=(),
            allowed_isolated_parts=(),
            allowed_overlaps=(),
        )
    _check_generation_physics_completeness(ctx, object_model)
    physics_report = ctx.report()
    if not bool(getattr(physics_report, "passed", False)):
        return _build_test_report(
            type(physics_report),
            checks=tuple(str(name) for name in getattr(physics_report, "checks", ())),
            failures=tuple(getattr(physics_report, "failures", ())),
            warnings=tuple(str(item) for item in getattr(physics_report, "warnings", ())),
            allowances=(),
            allowed_isolated_parts=(),
            allowed_overlaps=(),
        )
    _check_generation_physx_armature_stability(ctx, object_model)
    _check_generation_gravity_hold_coverage(
        ctx,
        object_model,
        asset_root=script_path.parent,
    )
    gravity_report = ctx.report()
    if not bool(getattr(gravity_report, "passed", False)):
        return _build_test_report(
            type(gravity_report),
            checks=tuple(str(name) for name in getattr(gravity_report, "checks", ())),
            failures=tuple(getattr(gravity_report, "failures", ())),
            warnings=tuple(str(item) for item in getattr(gravity_report, "warnings", ())),
            allowances=(),
            allowed_isolated_parts=(),
            allowed_overlaps=(),
        )
    ctx.check_mesh_assets_ready()
    ctx.fail_if_isolated_parts()
    ctx.warn_if_part_contains_disconnected_geometry_islands()
    ctx.fail_if_parts_overlap_in_current_pose()
    articulation_type = sdk.ArticulationType
    articulations = getattr(object_model, "articulations", ())
    if any(
        getattr(articulation, "articulation_type", None) != articulation_type.FIXED
        for articulation in articulations
    ):
        ctx.fail_if_parts_overlap_in_sampled_poses(
            max_pose_samples=_COMPILER_SAMPLED_POSE_MAX_SAMPLES,
            ignore_adjacent=False,
            ignore_fixed=True,
            name=_COMPILER_SAMPLED_POSE_OVERLAP_CHECK,
        )
    baseline_report = ctx.report()
    return _build_test_report(
        type(baseline_report),
        checks=tuple(str(name) for name in getattr(baseline_report, "checks", ())),
        failures=tuple(getattr(baseline_report, "failures", ())),
        warnings=tuple(str(item) for item in getattr(baseline_report, "warnings", ())),
        allowances=(),
        allowed_isolated_parts=(),
        allowed_overlaps=(),
    )
