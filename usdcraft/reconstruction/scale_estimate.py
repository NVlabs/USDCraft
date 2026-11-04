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

"""Real-world scale for unit-normalized source meshes (both arms, before the input freeze).

A mesh whose file carries no unit metadata is read in metres by the format contract. Meshes that
an upstream tool normalized to a unit or two-unit bounding box then become two-metre bananas: the
AXIAL grid, every millimetre threshold, the Author's scale anchor, the authored physics and the
exported USD all inherit the wrong size, and the Author has no way to notice.

The Host cannot know an object's size from its geometry, so the scale is resolved in two steps:
a mechanical inspection flags a source as *suspected unit-normalized* (longest side within 2.5%
of 1.0 or 2.0 and no unit metadata), and one clean-context vision call over the semantic image and
the request text estimates the subject's real-world longest dimension from its category and the
visible scale cues. The quotient becomes the run's ``meters_per_unit`` — tagged
``estimated_from_semantic_image`` everywhere the frame source is recorded — and every later stage
uses it exactly as it would use a user declaration. An explicit ``--meters-per-unit`` (or a CSV
``meters_per_unit`` cell) always wins and skips the estimate.

The Provider arm asks through the Provider transport (one strict function tool); the Codex arm asks
through one ``codex exec`` vision call with an output schema. The Codex estimate runs after the
user's confirmation and before the freeze, is recorded in the request and the record, and never
enters the approval fingerprint.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ESTIMATED_SCALE_SOURCE = "estimated_from_semantic_image"
# Longest-side values an upstream normalizer produces, and the relative tolerance around them.
# Scans trimmed or re-meshed after normalization land slightly below the target, so the band is
# 2.5%; a real object whose longest side really is about 2 m loses nothing, because its
# estimate comes back near 2 m and the factor stays near 1.
UNIT_NORMALIZED_TARGETS: tuple[float, ...] = (1.0, 2.0)
UNIT_NORMALIZED_TOLERANCE = 0.025
# Frame sources under which the mesh's numbers are only assumed to be metres.
ASSUMED_METRE_SOURCES = frozenset({"format_contract"})
# Sanity band for an estimate: a hand-held gadget to a playground set.
SCALE_MIN_M = 0.02
SCALE_MAX_M = 30.0
SCALE_MAX_ATTEMPTS = 2
SCALE_TOOL_NAME = "report_subject_scale"
_CONFIDENCE = ("low", "medium", "high")


class ScaleEstimateError(RuntimeError):
    """The scale estimate could not be obtained; the caller decides how to degrade."""


@dataclass(slots=True, frozen=True)
class ScaleInspection:
    """Mechanical reading of a loaded source's scale evidence."""

    longest_extent_units: float
    meters_per_unit_source: str
    suspected_unit_normalized: bool
    matched_target: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "longest_extent_units": round(self.longest_extent_units, 6),
            "meters_per_unit_source": self.meters_per_unit_source,
            "suspected_unit_normalized": self.suspected_unit_normalized,
            "matched_target": self.matched_target,
        }


@dataclass(slots=True, frozen=True)
class ScaleEstimate:
    category: str
    longest_dimension_m: float
    confidence: str
    rationale: str
    model: str
    reasoning_effort: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "longest_dimension_m": round(self.longest_dimension_m, 4),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


def inspect_source_scale(scene: Any) -> ScaleInspection:
    """Flag a source whose longest side sits on a normalizer's target with only assumed units."""

    lo = np.asarray(scene.bounds_min_xyz, dtype=float)
    hi = np.asarray(scene.bounds_max_xyz, dtype=float)
    longest = float(np.max(hi - lo))
    source = str(scene.frame.meters_per_unit_source)
    matched: float | None = None
    if source in ASSUMED_METRE_SOURCES and longest > 0.0:
        for target in UNIT_NORMALIZED_TARGETS:
            if abs(longest - target) <= UNIT_NORMALIZED_TOLERANCE * target:
                matched = target
                break
    return ScaleInspection(
        longest_extent_units=longest,
        meters_per_unit_source=source,
        suspected_unit_normalized=matched is not None,
        matched_target=matched,
    )


SCALE_TOOL_RESPONSE = f"Call `{SCALE_TOOL_NAME}` exactly once with that estimate."
SCALE_JSON_RESPONSE = (
    "Answer with exactly one JSON object carrying `category` (string), `longest_dimension_m` "
    "(number, metres), `confidence` (`low`, `medium` or `high`) and `rationale` (string), and "
    "nothing else."
)


def scale_result_schema() -> dict[str, Any]:
    """The output schema of one Codex scale-estimate call (the tool schema's parameters)."""

    return json.loads(json.dumps(scale_tool_schema()["parameters"]))


def build_scale_estimate_task(
    *, request_text: str, label: str | None, response_instruction: str = SCALE_TOOL_RESPONSE
) -> str:
    """The one-shot task for the scale estimator (a fresh agent; positive, self-contained).

    ``response_instruction`` names how the arm returns the estimate: the Provider function tool
    or the Codex JSON final response.
    """

    label_line = f"User label: {label.strip()}\n" if label and label.strip() else ""
    return f"""You see one photograph or render of a real object and the request that names what it is.

Request: {request_text.strip()}
{label_line}
The object's 3D mesh carries no units and was normalized to a unit bounding box, so its true size
is unknown to the pipeline that will reconstruct it. Your one job is to estimate the object's
real-world size.

Report the length in metres of the longest side of the object's axis-aligned bounding box in the
orientation the image shows, including handles, arms, legs, lids, cords and every attached part
that is visible. Use the category's typical dimensions and every scale cue the image offers (a
hand, a counter, a floor tile, a door, another object of known size). Give one best estimate, not
a range; name the category you read; give a one-sentence rationale that cites the cue or the
typical size you relied on; rate your confidence low, medium or high.

{response_instruction}"""


def scale_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": SCALE_TOOL_NAME,
        "description": "Report the subject's real-world longest bounding-box side in metres.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "longest_dimension_m": {"type": "number"},
                "confidence": {"type": "string", "enum": list(_CONFIDENCE)},
                "rationale": {"type": "string"},
            },
            "required": ["category", "longest_dimension_m", "confidence", "rationale"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def parse_scale_arguments(output: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the single tool call of one estimator response."""

    calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == SCALE_TOOL_NAME
    ]
    if len(calls) != 1:
        raise ScaleEstimateError(
            f"scale estimator must call {SCALE_TOOL_NAME} exactly once (got {len(calls)} calls)"
        )
    arguments = calls[0].get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ScaleEstimateError("scale tool call arguments are invalid JSON") from exc
    return validate_scale_payload(arguments)


def validate_scale_payload(arguments: object) -> dict[str, Any]:
    """Validate one estimate object (tool arguments or a Codex JSON final response)."""

    if not isinstance(arguments, Mapping):
        raise ScaleEstimateError("scale estimate must be an object")
    try:
        value = float(arguments["longest_dimension_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScaleEstimateError("longest_dimension_m must be a number") from exc
    if not math.isfinite(value) or not SCALE_MIN_M <= value <= SCALE_MAX_M:
        raise ScaleEstimateError(
            f"longest_dimension_m {value!r} is outside the accepted band "
            f"{SCALE_MIN_M}-{SCALE_MAX_M} m"
        )
    category = " ".join(str(arguments.get("category", "")).split())
    rationale = " ".join(str(arguments.get("rationale", "")).split())
    confidence = str(arguments.get("confidence", "")).strip().lower()
    if not category:
        raise ScaleEstimateError("category must be non-empty")
    if confidence not in _CONFIDENCE:
        raise ScaleEstimateError("confidence must be low, medium or high")
    return {
        "category": category,
        "longest_dimension_m": value,
        "confidence": confidence,
        "rationale": rationale,
    }


def estimate_subject_scale(
    *,
    semantic_image_path: Path,
    request_text: str,
    label: str | None,
    config: Any,
    cancellation_event: Event | None = None,
    client: Any | None = None,
) -> ScaleEstimate:
    """One clean-context vision call; retries once on an invalid response."""

    from usdcraft.pipeline.provider import NvidiaProviderError, NvidiaResponsesClient
    from usdcraft.pipeline.provider.harness import image_path_to_data_url

    resolved_client = client or NvidiaResponsesClient(config, cancellation_event=cancellation_event)
    task = build_scale_estimate_task(request_text=request_text, label=label)
    message = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": task},
            {
                "type": "input_image",
                "image_url": image_path_to_data_url(Path(semantic_image_path)),
                "detail": "high",
            },
        ],
    }
    last_error: ScaleEstimateError | None = None
    for _attempt in range(SCALE_MAX_ATTEMPTS):
        try:
            response = resolved_client.create_response(
                input_items=[message], tools=[scale_tool_schema()]
            )
        except NvidiaProviderError as exc:
            if cancellation_event is not None and cancellation_event.is_set():
                raise
            raise ScaleEstimateError(f"provider scale-estimate call failed: {exc}") from exc
        try:
            payload = parse_scale_arguments(getattr(response, "output"))
        except ScaleEstimateError as exc:
            last_error = exc
            continue
        return ScaleEstimate(
            category=payload["category"],
            longest_dimension_m=payload["longest_dimension_m"],
            confidence=payload["confidence"],
            rationale=payload["rationale"],
            model=str(getattr(config, "model")),
            reasoning_effort=str(getattr(config, "reasoning_effort")),
        )
    assert last_error is not None
    raise last_error


def estimate_subject_scale_via_codex(
    *,
    semantic_image_path: Path,
    request_text: str,
    label: str | None,
    model: str | None,
    reasoning_effort: str | None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    process_runner: Callable[..., Any] | None = None,
) -> ScaleEstimate:
    """One clean-context Codex vision call with an output schema; retries once on an invalid reply.

    The call reuses the observation engine's Codex invocation (ephemeral, read-only sandbox, no
    user config, disabled features) with the estimate task and schema in place of the inventory's.
    The Codex arm has no time limit: the call runs until the Codex process exits.
    """

    import os

    from usdcraft.pipeline.observation import (
        ImageObservationError,
        _default_process_runner,
        build_observation_codex_command,
        resolve_observation_runtime,
    )

    runtime = resolve_observation_runtime(
        model=model, reasoning_effort=reasoning_effort, product="reconstruction", environ=environ
    )
    selected_environ = os.environ if environ is None else environ
    resolved_executable = (
        executable or selected_environ.get("USDCRAFT_CODEX_BIN", "").strip() or "codex"
    )
    runner = process_runner or _default_process_runner
    task = build_scale_estimate_task(
        request_text=request_text, label=label, response_instruction=SCALE_JSON_RESPONSE
    )
    image = Path(semantic_image_path)
    last_error: ScaleEstimateError | None = None
    for _attempt in range(SCALE_MAX_ATTEMPTS):
        workspace = Path(tempfile.mkdtemp(prefix="usdcraft-scale-estimate-"))
        try:
            codex_dir = workspace / ".codex-run"
            codex_dir.mkdir()
            shutil.copy2(image, workspace / f"input_1_{image.name}")
            (workspace / "TASK.md").write_text(task.rstrip() + "\n", encoding="utf-8")
            (codex_dir / "final.schema.json").write_text(
                json.dumps(scale_result_schema(), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            try:
                command = build_observation_codex_command(
                    runtime, workspace, executable=resolved_executable
                )
            except ImageObservationError as exc:
                raise ScaleEstimateError(str(exc)) from exc
            process_env = dict(selected_environ)
            process_env["PYTHONUNBUFFERED"] = "1"
            try:
                process = runner(
                    command,
                    stdin_text=task,
                    cwd=workspace,
                    timeout_seconds=None,
                    env=process_env,
                )
            except FileNotFoundError as exc:
                raise ScaleEstimateError(
                    "scale estimate requires the `codex` executable; install/login to Codex CLI "
                    "or set USDCRAFT_CODEX_BIN"
                ) from exc
            if int(process.returncode) != 0:
                details = (process.stderr or process.stdout or "(no output)").strip()
                last_error = ScaleEstimateError(
                    f"scale-estimate Agent exited with status {process.returncode}: "
                    f"{details[-2000:]}"
                )
                continue
            final_path = codex_dir / "final.json"
            try:
                payload = validate_scale_payload(json.loads(final_path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                last_error = ScaleEstimateError(f"scale-estimate reply is not valid JSON: {exc}")
                continue
            except ScaleEstimateError as exc:
                last_error = exc
                continue
            return ScaleEstimate(
                category=payload["category"],
                longest_dimension_m=payload["longest_dimension_m"],
                confidence=payload["confidence"],
                rationale=payload["rationale"],
                model=runtime.model_id,
                reasoning_effort=runtime.reasoning_effort,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    assert last_error is not None
    raise last_error


def meters_per_unit_from_estimate(inspection: ScaleInspection, estimate: ScaleEstimate) -> float:
    """The scale factor that makes the mesh's longest side the estimated real dimension."""

    if inspection.longest_extent_units <= 0.0:
        raise ScaleEstimateError("source has no extent to scale")
    return estimate.longest_dimension_m / inspection.longest_extent_units


def scale_report(
    inspection: ScaleInspection,
    *,
    meters_per_unit: float,
    source: str,
    estimate: ScaleEstimate | None,
) -> dict[str, Any]:
    """The `scale` entry of the runtime report: how the run's metres were decided."""

    report: dict[str, Any] = {
        "meters_per_unit": meters_per_unit,
        "source": source,
        "inspection": inspection.to_dict(),
    }
    if estimate is not None:
        report["estimate"] = estimate.to_dict()
    return report


def scale_note_for_author(inspection: ScaleInspection, estimate: ScaleEstimate) -> str:
    """The SCALE paragraph of the Author task when the run's metres come from an estimate."""

    return (
        "The source mesh carried no units and measured "
        f"{inspection.longest_extent_units:.3f} on its longest side, the signature of a mesh "
        "normalized to a unit box. Its metres were set before preprocessing from a category "
        f"estimate of the semantic image: {estimate.category}, longest side "
        f"{estimate.longest_dimension_m:.2f} m ({estimate.confidence} confidence: "
        f"{estimate.rationale}). Every AXIAL value, the brief and `check_axial` already use "
        "that scale. Treat the absolute size as an estimate that may be off by a quarter, and the "
        "mesh's proportions and relative positions as exact; do not rescale the body yourself, and "
        "say in the finish summary if the image makes that estimate look clearly wrong."
    )


__all__ = [
    "ASSUMED_METRE_SOURCES",
    "ESTIMATED_SCALE_SOURCE",
    "SCALE_TOOL_NAME",
    "UNIT_NORMALIZED_TARGETS",
    "UNIT_NORMALIZED_TOLERANCE",
    "ScaleEstimate",
    "ScaleEstimateError",
    "ScaleInspection",
    "SCALE_JSON_RESPONSE",
    "SCALE_TOOL_RESPONSE",
    "build_scale_estimate_task",
    "estimate_subject_scale",
    "estimate_subject_scale_via_codex",
    "inspect_source_scale",
    "meters_per_unit_from_estimate",
    "parse_scale_arguments",
    "scale_note_for_author",
    "scale_report",
    "scale_result_schema",
    "scale_tool_schema",
    "validate_scale_payload",
]
