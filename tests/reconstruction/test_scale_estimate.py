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
from types import SimpleNamespace

import numpy as np
import pytest

from usdcraft.reconstruction.scale_estimate import (
    ESTIMATED_SCALE_SOURCE,
    SCALE_TOOL_NAME,
    ScaleEstimate,
    ScaleEstimateError,
    build_scale_estimate_task,
    estimate_subject_scale,
    inspect_source_scale,
    meters_per_unit_from_estimate,
    parse_scale_arguments,
    scale_note_for_author,
    scale_report,
)


def _scene(longest: float, source: str = "format_contract") -> SimpleNamespace:
    return SimpleNamespace(
        bounds_min_xyz=np.array([-longest / 2, -0.3 * longest, 0.0]),
        bounds_max_xyz=np.array([longest / 2, 0.3 * longest, 0.4 * longest]),
        frame=SimpleNamespace(meters_per_unit_source=source),
    )


@pytest.mark.parametrize(
    ("longest", "source", "suspected", "target"),
    [
        (2.000, "format_contract", True, 2.0),
        (1.975, "format_contract", True, 2.0),  # within 2.5% of a two-unit box
        (1.000, "format_contract", True, 1.0),
        (0.980, "format_contract", True, 1.0),
        (0.970, "format_contract", False, None),
        (1.886, "format_contract", False, None),  # a real 1.9 m fridge stays metres
        (0.344, "format_contract", False, None),
        (2.000, "explicit", False, None),  # the user declared the scale
        (2.000, "embedded_metadata", False, None),
    ],
)
def test_inspection_flags_only_unitless_meshes_on_a_normalizer_target(
    longest: float, source: str, suspected: bool, target: float | None
) -> None:
    inspection = inspect_source_scale(_scene(longest, source))
    assert inspection.suspected_unit_normalized is suspected
    assert inspection.matched_target == target
    assert inspection.longest_extent_units == pytest.approx(longest)


def _call(arguments: object) -> dict[str, object]:
    return {"type": "function_call", "name": SCALE_TOOL_NAME, "arguments": arguments}


def test_parse_accepts_one_valid_call_and_rejects_the_rest() -> None:
    payload = parse_scale_arguments(
        [
            {"type": "reasoning"},
            _call(
                json.dumps(
                    {
                        "category": " microwave  oven ",
                        "longest_dimension_m": 0.5,
                        "confidence": "High",
                        "rationale": "countertop microwave, typical 0.45-0.55 m width",
                    }
                )
            ),
        ]
    )
    assert payload["category"] == "microwave oven" and payload["confidence"] == "high"
    assert payload["longest_dimension_m"] == 0.5
    with pytest.raises(ScaleEstimateError, match="exactly once"):
        parse_scale_arguments([_call("{}"), _call("{}")])
    with pytest.raises(ScaleEstimateError, match="outside the accepted band"):
        parse_scale_arguments(
            [
                _call(
                    {
                        "category": "banana",
                        "longest_dimension_m": 45.0,
                        "confidence": "low",
                        "rationale": "",
                    }
                )
            ]
        )
    with pytest.raises(ScaleEstimateError, match="invalid JSON"):
        parse_scale_arguments([_call("{not json")])


def test_estimate_retries_once_then_returns_the_estimate(tmp_path) -> None:
    image = tmp_path / "semantic.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    responses = [
        SimpleNamespace(output=[_call({"category": "banana", "longest_dimension_m": -1})]),
        SimpleNamespace(
            output=[
                _call(
                    {
                        "category": "banana",
                        "longest_dimension_m": 0.2,
                        "confidence": "medium",
                        "rationale": "a single ripe banana is about 20 cm long",
                    }
                )
            ]
        ),
    ]
    seen: list[dict[str, object]] = []

    class _Client:
        def create_response(self, *, input_items, tools):  # type: ignore[no-untyped-def]
            seen.append({"items": input_items, "tools": tools})
            return responses.pop(0)

    config = SimpleNamespace(model="nvidia-test-model", reasoning_effort="high")
    estimate = estimate_subject_scale(
        semantic_image_path=image,
        request_text="Reconstruct the banana shown in the semantic image and the source mesh.",
        label="banana",
        config=config,
        client=_Client(),
    )
    assert estimate == ScaleEstimate(
        category="banana",
        longest_dimension_m=0.2,
        confidence="medium",
        rationale="a single ripe banana is about 20 cm long",
        model="nvidia-test-model",
        reasoning_effort="high",
    )
    assert len(seen) == 2
    text = seen[0]["items"][0]["content"][0]["text"]  # type: ignore[index]
    assert "Reconstruct the banana" in text and "User label: banana" in text
    assert seen[0]["items"][0]["content"][1]["type"] == "input_image"  # type: ignore[index]
    assert seen[0]["tools"][0]["name"] == SCALE_TOOL_NAME  # type: ignore[index]
    assert "unit bounding box" in build_scale_estimate_task(request_text="x", label=None)

    class _AlwaysBad:
        def create_response(self, *, input_items, tools):  # type: ignore[no-untyped-def]
            return SimpleNamespace(output=[])

    with pytest.raises(ScaleEstimateError, match="exactly once"):
        estimate_subject_scale(
            semantic_image_path=image,
            request_text="x",
            label=None,
            config=config,
            client=_AlwaysBad(),
        )


def test_scale_factor_report_and_author_note() -> None:
    inspection = inspect_source_scale(_scene(1.995))
    estimate = ScaleEstimate("microwave oven", 0.5, "high", "typical countertop width", "m", "high")
    factor = meters_per_unit_from_estimate(inspection, estimate)
    assert factor == pytest.approx(0.5 / 1.995)
    report = scale_report(
        inspection, meters_per_unit=factor, source=ESTIMATED_SCALE_SOURCE, estimate=estimate
    )
    assert report["source"] == ESTIMATED_SCALE_SOURCE
    assert report["inspection"]["suspected_unit_normalized"] is True
    assert report["estimate"]["longest_dimension_m"] == 0.5
    note = scale_note_for_author(inspection, estimate)
    assert "1.995" in note and "microwave oven" in note and "0.50 m" in note
    assert "do not rescale the body yourself" in note
