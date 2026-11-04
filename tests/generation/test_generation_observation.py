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
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.prompt_snapshot import assert_matches_golden
from usdcraft.generation.observation import (
    GENERATION_OBSERVATION_MANIFEST_SCHEMA,
    ImageObservationError,
    build_generation_image_observation_task,
    run_generation_image_observation,
    run_generation_image_observation_via_provider,
)
from usdcraft.pipeline.observation import ObservationProcessResult


def _part(**overrides: object) -> dict[str, object]:
    part: dict[str, object] = {
        "name": "seat_disc",
        "visible_count": 1,
        "mobility": "fixed",
        "level": "primary_body",
        "visibility": "visible",
        "geometry": "flat round disc, about as wide as the whole stool",
        "arrangement": "round top surface",
        "affordances": ["sit on"],
        "attaches_to": None,
        "duplicate_of": None,
        "mirror_of": None,
        "carried_as": "none",
        "notes": None,
    }
    part.update(overrides)
    return part


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "parts": [_part()],
        "removals": [],
        "summary": "a simple stool",
    }
    payload.update(overrides)
    return payload


def _image(tmp_path: Path) -> Path:
    image = tmp_path / "inspiration.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    return image


def test_generation_task_uses_inspiration_semantics() -> None:
    task = build_generation_image_observation_task()
    assert_matches_golden("observation_generation", task, paths={})
    # Generation observation must not borrow Reconstruction evidence vocabulary or see renders.
    for forbidden in ("provenance role", "AXIAL", "render", "mesh"):
        assert forbidden not in task, forbidden


def test_codex_arm_publishes_with_generation_scope(tmp_path: Path) -> None:
    image = _image(tmp_path)

    def runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        assert timeout_seconds is None  # the Codex arm has no time limit
        (Path(cwd) / ".codex-run" / "final.json").write_text(
            json.dumps(_payload()), encoding="utf-8"
        )
        return ObservationProcessResult(returncode=0, stdout="{}\n", stderr="")

    result = run_generation_image_observation(
        image_path=image,
        output_dir=tmp_path / "image_observation",
        model="gpt-5-codex",
        reasoning_effort="high",
        process_runner=runner,
    )
    assert result.attempts == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == GENERATION_OBSERVATION_MANIFEST_SCHEMA
    assert manifest["scope"] == "generation_image_observation"
    assert manifest["arm"] == "codex"
    assert manifest["authority"] == "external_first_reading_not_authority"
    task = (result.root / "agent" / "TASK.md").read_text(encoding="utf-8")
    assert "inspiration image" in task


def _provider_config() -> SimpleNamespace:
    return SimpleNamespace(model="nvidia-test-model", reasoning_effort="high")


class _FakeClient:
    def __init__(self, outputs: list[list[dict[str, object]]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def create_response(self, *, input_items, tools, instructions=None):  # type: ignore[no-untyped-def]
        self.calls.append({"input_items": input_items, "tools": tools})
        return SimpleNamespace(
            response_id="resp-1",
            status="completed",
            output=self.outputs[len(self.calls) - 1],
            usage={},
        )


def _tool_call(payload: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function_call",
        "call_id": "call-1",
        "name": "submit_image_observation",
        "arguments": json.dumps(payload, ensure_ascii=False),
    }


def test_provider_arm_publishes_validated_inventory(tmp_path: Path) -> None:
    image = _image(tmp_path)
    client = _FakeClient([[_tool_call(_payload())], [_tool_call(_payload())]])
    result = run_generation_image_observation_via_provider(
        image_path=image,
        output_dir=tmp_path / "image_observation",
        config=_provider_config(),
        client=client,
    )
    assert result.attempts == 1
    assert result.inventory["parts"][0]["name"] == "seat_disc"  # type: ignore[index]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == GENERATION_OBSERVATION_MANIFEST_SCHEMA
    assert manifest["arm"] == "provider"
    assert manifest["runtime"]["model"] == "nvidia-test-model"
    assert (result.root / "agent" / "TASK.md").is_file()
    assert (result.root / "agent" / "response_attempt_1.json").is_file()
    assert manifest["review"]["status"] == "unchanged"
    assert (result.root / "agent" / "review_task.md").is_file()
    assert (result.root / "agent" / "review_response.json").is_file()
    assert (result.root / "review_merge.json").is_file()
    call, review_call = client.calls
    (tool,) = call["tools"]
    assert tool["name"] == "submit_image_observation"
    assert tool["strict"] is True
    assert tool["parameters"]["required"] == [
        "status",
        "parts",
        "removals",
        "summary",
    ]
    (message,) = call["input_items"]
    kinds = [item["type"] for item in message["content"]]
    assert kinds == ["input_text", "input_image"]


def test_provider_arm_retries_once_then_succeeds(tmp_path: Path) -> None:
    image = _image(tmp_path)
    client = _FakeClient(
        [
            [{"type": "message", "content": "no tool call"}],
            [_tool_call(_payload())],
            [_tool_call(_payload())],
        ]
    )
    result = run_generation_image_observation_via_provider(
        image_path=image,
        output_dir=tmp_path / "image_observation",
        config=_provider_config(),
        client=client,
    )
    assert result.attempts == 2
    assert (result.root / "agent" / "response_attempt_1.json").is_file()
    assert (result.root / "agent" / "response_attempt_2.json").is_file()


def test_provider_arm_fails_after_two_invalid_responses(tmp_path: Path) -> None:
    image = _image(tmp_path)
    client = _FakeClient(
        [
            [{"type": "message", "content": "still no tool call"}],
            [_tool_call({"nonsense": True})],
        ]
    )
    with pytest.raises(ImageObservationError):
        run_generation_image_observation_via_provider(
            image_path=image,
            output_dir=tmp_path / "image_observation",
            config=_provider_config(),
            client=client,
        )
    assert not (tmp_path / "image_observation").exists()
    assert not list(tmp_path.glob(".image_observation.staging-*"))


def test_provider_arm_wraps_transport_errors(tmp_path: Path) -> None:
    from usdcraft.pipeline.provider import NvidiaProviderError

    image = _image(tmp_path)

    class _FailingClient:
        def create_response(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise NvidiaProviderError("endpoint unreachable")

    with pytest.raises(ImageObservationError, match="endpoint unreachable"):
        run_generation_image_observation_via_provider(
            image_path=image,
            output_dir=tmp_path / "image_observation",
            config=_provider_config(),
            client=_FailingClient(),
        )
