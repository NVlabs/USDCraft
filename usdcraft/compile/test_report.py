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

"""Run the Author's ``run_tests()`` and merge it with the compiler-owned baseline report."""

from __future__ import annotations

import sdk

_AUTOMATED_BASELINE_WARNING_CHECK_NAME = (
    "warn_if_part_contains_disconnected_geometry_islands(tol=1e-06)"
)
_AUTOMATED_BASELINE_DEFAULT_CHECK_NAMES = frozenset(
    {
        "check_model_valid",
        "check_single_root_part",
        "check_mesh_assets_ready",
        "fail_if_isolated_parts()",
        _AUTOMATED_BASELINE_WARNING_CHECK_NAME,
        "fail_if_parts_overlap_in_current_pose()",
    }
)


def _run_required_tests(globals_dict: dict) -> object:
    run_tests = globals_dict.get("run_tests")
    if not callable(run_tests):
        raise ValueError(
            "Missing required `run_tests()` in generated script. "
            "Add a top-level `def run_tests() -> TestReport:` and return `ctx.report()`."
        )

    report = run_tests()
    if not isinstance(report, sdk.TestReport):
        raise ValueError(f"run_tests() must return sdk.TestReport (got {type(report).__name__})")

    return report


def _report_warning_check_name(text: object) -> str | None:
    warning_text = str(text).strip()
    if not warning_text:
        return None
    check_name, has_separator, _detail = warning_text.partition(":")
    if not has_separator:
        return None
    return check_name.strip() or None


def _build_test_report(
    report_type: type,
    *,
    checks: tuple[str, ...],
    failures: tuple[object, ...],
    warnings: tuple[str, ...],
    allowances: tuple[str, ...],
    allowed_isolated_parts: tuple[str, ...],
    allowed_overlaps: tuple[object, ...],
    disconnected_geometry_classifications: tuple[object, ...] = (),
) -> object:
    return report_type(
        passed=not failures,
        checks_run=len(checks),
        checks=checks,
        failures=failures,
        warnings=warnings,
        allowances=allowances,
        allowed_isolated_parts=allowed_isolated_parts,
        allowed_overlaps=allowed_overlaps,
        disconnected_geometry_classifications=disconnected_geometry_classifications,
    )


def _filter_duplicate_automated_baseline_results(
    authored_report: object, baseline_report: object
) -> object:
    authored_check_names = {
        str(name)
        for name in getattr(authored_report, "checks", ())
        if str(name) in _AUTOMATED_BASELINE_DEFAULT_CHECK_NAMES
    }
    if not authored_check_names:
        return baseline_report

    baseline_checks = tuple(
        str(name)
        for name in getattr(baseline_report, "checks", ())
        if str(name) not in authored_check_names
    )
    baseline_failures = tuple(
        failure
        for failure in getattr(baseline_report, "failures", ())
        if str(getattr(failure, "name", "")) not in authored_check_names
    )
    baseline_warnings = tuple(
        str(warning)
        for warning in getattr(baseline_report, "warnings", ())
        if _report_warning_check_name(warning) not in authored_check_names
    )

    return _build_test_report(
        type(baseline_report),
        checks=baseline_checks,
        failures=baseline_failures,
        warnings=baseline_warnings,
        allowances=tuple(str(item) for item in getattr(baseline_report, "allowances", ())),
        allowed_isolated_parts=tuple(
            str(item) for item in getattr(baseline_report, "allowed_isolated_parts", ())
        ),
        allowed_overlaps=tuple(getattr(baseline_report, "allowed_overlaps", ())),
        disconnected_geometry_classifications=tuple(
            getattr(baseline_report, "disconnected_geometry_classifications", ())
        ),
    )


def _merge_test_reports(authored_report: object, baseline_report: object) -> object:
    filtered_baseline_report = _filter_duplicate_automated_baseline_results(
        authored_report,
        baseline_report,
    )

    check_names: list[str] = []
    seen_checks: set[str] = set()
    for report in (authored_report, filtered_baseline_report):
        for check_name in getattr(report, "checks", ()):
            normalized = str(check_name)
            if normalized in seen_checks:
                continue
            seen_checks.add(normalized)
            check_names.append(normalized)

    failures: list[object] = []
    seen_failures: set[tuple[str, str]] = set()
    for report in (authored_report, filtered_baseline_report):
        for failure in getattr(report, "failures", ()):
            key = (str(getattr(failure, "name", "")), str(getattr(failure, "details", "")))
            if key in seen_failures:
                continue
            seen_failures.add(key)
            failures.append(failure)

    warnings: list[str] = []
    seen_warnings: set[str] = set()
    for report in (authored_report, filtered_baseline_report):
        for warning in getattr(report, "warnings", ()):
            text = str(warning)
            if text in seen_warnings:
                continue
            seen_warnings.add(text)
            warnings.append(text)

    allowances: list[str] = []
    seen_allowances: set[str] = set()
    for report in (authored_report, filtered_baseline_report):
        for allowance in getattr(report, "allowances", ()):
            text = str(allowance)
            if text in seen_allowances:
                continue
            seen_allowances.add(text)
            allowances.append(text)

    allowed_isolated_parts: list[str] = []
    seen_isolated_parts: set[str] = set()
    for report in (authored_report, filtered_baseline_report):
        for part_name in getattr(report, "allowed_isolated_parts", ()):
            normalized = str(part_name)
            if normalized in seen_isolated_parts:
                continue
            seen_isolated_parts.add(normalized)
            allowed_isolated_parts.append(normalized)

    allowed_overlaps: list[object] = []
    seen_overlaps: set[tuple[str, str, str | None, str | None, str]] = set()
    for report in (authored_report, filtered_baseline_report):
        for overlap in getattr(report, "allowed_overlaps", ()):
            overlap_key = (
                str(getattr(overlap, "link_a", "")),
                str(getattr(overlap, "link_b", "")),
                (
                    None
                    if getattr(overlap, "elem_a", None) is None
                    else str(getattr(overlap, "elem_a"))
                ),
                (
                    None
                    if getattr(overlap, "elem_b", None) is None
                    else str(getattr(overlap, "elem_b"))
                ),
                str(getattr(overlap, "reason", "")),
            )
            if overlap_key in seen_overlaps:
                continue
            seen_overlaps.add(overlap_key)
            allowed_overlaps.append(overlap)

    disconnected_classifications: list[object] = []
    seen_disconnected_classifications: set[tuple[str, str, tuple[str, ...], str]] = set()
    for report in (authored_report, filtered_baseline_report):
        for item in getattr(report, "disconnected_geometry_classifications", ()):
            classification_key = (
                str(getattr(item, "part", "")),
                str(getattr(item, "classification", "")),
                tuple(str(value) for value in getattr(item, "elements", ())),
                str(getattr(item, "reason", "")),
            )
            if classification_key in seen_disconnected_classifications:
                continue
            seen_disconnected_classifications.add(classification_key)
            disconnected_classifications.append(item)

    return _build_test_report(
        sdk.TestReport,
        checks=tuple(check_names),
        failures=tuple(failures),
        warnings=tuple(warnings),
        allowances=tuple(allowances),
        allowed_isolated_parts=tuple(allowed_isolated_parts),
        allowed_overlaps=tuple(allowed_overlaps),
        disconnected_geometry_classifications=tuple(disconnected_classifications),
    )


def _raise_for_failed_test_report(report: object) -> None:
    if bool(getattr(report, "passed", False)):
        return

    failures = getattr(report, "failures", ())
    lines = ["URDF tests failed:"]
    for failure in list(failures)[:10]:
        name = getattr(failure, "name", "unknown")
        details = getattr(failure, "details", "")
        lines.append(f"- {name}: {details}")
    if len(list(failures)) > 10:
        lines.append(f"... ({len(list(failures)) - 10} more)")
    exc = ValueError("\n".join(lines))
    setattr(exc, "test_report", report)
    raise exc
