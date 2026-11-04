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

import pytest

from tests.prompt_snapshot import assert_matches_golden
from usdcraft.reconstruction.observation import (
    ImageObservationError,
    ObservationProcessResult,
    build_image_observation_task,
    build_observation_codex_command,
    parse_observation_result,
    render_inventory_for_prompt,
    resolve_observation_runtime,
    run_image_observation,
)


def _part(**overrides: object) -> dict[str, object]:
    part: dict[str, object] = {
        "name": "door_panel",
        "visible_count": 1,
        "mobility": "movable",
        "level": "major_part",
        "visibility": "visible",
        "geometry": "flat rectangular panel, nearly the full front, taller than wide",
        "arrangement": "front face, hinged on the left edge",
        "affordances": ["pull open by the handle"],
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
        "summary": "one hinged door on a box body",
    }
    payload.update(overrides)
    return payload


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    semantic = tmp_path / "semantic.png"
    semantic.write_bytes(b"semantic")
    renders = []
    for azimuth in (45, 135, 225, 315):
        render = tmp_path / f"mesh_render_{azimuth:03d}.png"
        render.write_bytes(b"render")
        renders.append(render)
    return semantic, renders


def test_task_is_observation_only_over_the_semantic_image() -> None:
    task = build_image_observation_task()
    assert_matches_golden("observation_reconstruction", task, paths={})
    # The mesh is a separately produced approximation: its renders never enter this reading.
    assert "render" not in task.lower()
    assert "mesh" not in task.split("Do not inspect")[0].lower()
    # Both arms' tasks differ only in the transport named in the agent's self-description.
    provider = build_image_observation_task(agent_name="")
    assert task.replace("Codex ", "") == provider


def test_parse_accepts_a_complete_valid_inventory(tmp_path: Path) -> None:
    final = tmp_path / "final.json"
    payload = _payload(
        parts=[
            _part(name="left_leg"),
            _part(name="right_leg", mirror_of="left_leg", attaches_to="left_leg"),
        ]
    )
    _write_result(final, payload)
    assert parse_observation_result(final) == payload


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"summary": ""}, "summary"),
        ({"parts": []}, "at least one Part"),
        ({"parts": [_part(name="Bad Name")]}, "snake_case"),
        ({"parts": [_part(), _part()]}, "duplicates part name"),
        ({"parts": [_part(duplicate_of="ghost")]}, "unknown part"),
        ({"parts": [_part(mirror_of="door_panel")]}, "mirror_of is invalid"),
        ({"parts": [_part(visible_count=0)]}, "visible_count"),
        ({"parts": [_part(mobility="loose")]}, "mobility"),
        ({"parts": [_part(carried_as="loose")]}, "carried_as is invalid"),
        # attaches_to must point strictly backwards: forward and self references are ordering
        # errors, unknown names too.
        (
            {
                "parts": [
                    _part(name="body"),
                    _part(name="lid", attaches_to="hinge"),
                    _part(name="hinge", attaches_to="body"),
                ]
            },
            "earlier entry",
        ),
        ({"parts": [_part(attaches_to="door_panel")]}, "attaches_to is invalid"),
        ({"extra": True}, "unexpected fields"),
    ],
)
def test_parse_rejects_invalid_inventories(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    final = tmp_path / "final.json"
    _write_result(final, _payload(**mutation))
    with pytest.raises(ImageObservationError, match=match):
        parse_observation_result(final)


def test_parse_treats_blocked_as_degrade_and_rejects_blocked_parts(tmp_path: Path) -> None:
    final = tmp_path / "final.json"
    _write_result(final, _payload(status="blocked", parts=[], summary="image unreadable"))
    with pytest.raises(ImageObservationError, match="image unreadable"):
        parse_observation_result(final)
    _write_result(final, _payload(status="blocked"))
    with pytest.raises(ImageObservationError, match="must list no parts"):
        parse_observation_result(final)


def test_render_inventory_for_prompt_carries_every_observed_field() -> None:
    text = render_inventory_for_prompt(
        _payload(
            parts=[
                _part(notes="handle partially occluded"),
                _part(name="door_panel_mirror", mirror_of="door_panel", affordances=[]),
            ]
        )
    )
    for value in (
        "door_panel",
        "flat rectangular panel, nearly the full front, taller than wide",
        "front face, hinged on the left edge",
        "pull open by the handle",
        "handle partially occluded",
        "door_panel_mirror",
    ):
        assert value in text, value
    attached = render_inventory_for_prompt(
        _payload(parts=[_part(), _part(name="handle", attaches_to="door_panel")])
    )
    assert "handle" in attached
    # Carried contents carry their classification into the Author task; own parts stay silent.
    assert "content" not in text
    carried = render_inventory_for_prompt(
        _payload(
            parts=[
                _part(),
                _part(
                    name="goods",
                    visible_count=12,
                    level="major_part",
                    attaches_to="door_panel",
                    carried_as="displayed",
                ),
            ]
        )
    )
    assert "goods" in carried and "displayed" in carried and "content" in carried


def test_command_attaches_semantic_then_renders_and_binds_the_schema(tmp_path: Path) -> None:
    semantic, renders = _inputs(tmp_path)
    workspace = tmp_path / "agent"
    workspace.mkdir()
    from usdcraft.pipeline.observation import _write_agent_workspace

    _write_agent_workspace(
        workspace,
        image_paths=[semantic, *renders],
        build_task=lambda staged_names: f"observe {', '.join(staged_names)}",
        agent_context_schema="reconstruction_image_observation_agent_context",
        agent_context="clean_image_observation_only",
        extra_context={},
    )
    runtime = resolve_observation_runtime(model="gpt-5-codex", reasoning_effort="high")
    command = build_observation_codex_command(runtime, workspace, executable="codex-test")
    assert command[0] == "codex-test"
    assert "--output-schema" in command and "--output-last-message" in command
    assert ("--sandbox", "read-only") == tuple(
        command[command.index("--sandbox") : command.index("--sandbox") + 2]
    )
    images = [command[i + 1] for i, value in enumerate(command) if value == "--image"]
    assert [Path(image).name for image in images] == [
        "input_1_semantic.png",
        "input_2_mesh_render_045.png",
        "input_3_mesh_render_135.png",
        "input_4_mesh_render_225.png",
        "input_5_mesh_render_315.png",
    ]
    assert command[-1] == "-"


def _runner_writing(payloads: list[dict[str, object] | None]):
    calls: list[Path] = []

    def runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        assert timeout_seconds is None  # the Codex arm has no time limit
        calls.append(Path(cwd))
        payload = payloads[len(calls) - 1]
        if payload is not None:
            _write_result(Path(cwd) / ".codex-run" / "final.json", payload)
        return ObservationProcessResult(returncode=0, stdout="{}\n", stderr="")

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_run_publishes_validated_inventory_atomically(tmp_path: Path) -> None:
    semantic, renders = _inputs(tmp_path)
    runner = _runner_writing([_payload(), _payload()])  # first call + unchanged review
    result = run_image_observation(
        semantic_image_path=semantic,
        output_dir=tmp_path / "image_observation",
        model="gpt-5-codex",
        reasoning_effort="high",
        process_runner=runner,
    )
    assert result.attempts == 1
    assert result.inventory["parts"][0]["name"] == "door_panel"  # type: ignore[index]
    inventory = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    assert inventory == result.inventory
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "reconstruction_image_observation"
    assert manifest["status"] == "ready"
    assert manifest["authority"] == "external_first_reading_not_evidence"
    assert manifest["attempts"] == 1
    assert manifest["runtime"]["model"] == "gpt-5-codex"
    assert (result.root / "agent" / "TASK.md").is_file()
    assert (result.root / "agent" / ".codex-run" / "final.json").is_file()
    assert (result.root / "agent" / "context_manifest.json").is_file()
    assert manifest["review"]["status"] == "unchanged"
    assert (result.root / "review_agent" / "TASK.md").is_file()
    assert (result.root / "review_merge.json").is_file()
    review_task = (result.root / "review_agent" / "TASK.md").read_text(encoding="utf-8")
    assert "Removing an entry has exactly one channel" in " ".join(review_task.split())
    assert "=== FIRST INVENTORY (JSON) ===" in review_task


def test_run_retries_once_on_invalid_response_then_succeeds(tmp_path: Path) -> None:
    semantic, renders = _inputs(tmp_path)
    runner = _runner_writing([{"nonsense": True}, _payload(), _payload()])
    result = run_image_observation(
        semantic_image_path=semantic,
        output_dir=tmp_path / "image_observation",
        model="gpt-5-codex",
        reasoning_effort="high",
        process_runner=runner,
    )
    assert result.attempts == 2
    assert (result.root / "agent_attempt_1" / ".codex-run" / "final.json").is_file()


def test_run_fails_after_two_invalid_responses_and_retains_audit(tmp_path: Path) -> None:
    semantic, renders = _inputs(tmp_path)
    runner = _runner_writing([{"nonsense": True}, None])
    with pytest.raises(ImageObservationError):
        run_image_observation(
            semantic_image_path=semantic,
            output_dir=tmp_path / "image_observation",
            model="gpt-5-codex",
            reasoning_effort="high",
            process_runner=runner,
        )
    for attempt in (1, 2):
        assert (
            tmp_path
            / "image_observation"
            / f"agent_attempt_{attempt}"
            / ".codex-run"
            / "stdout.jsonl"
        ).is_file()
    assert not list(tmp_path.glob(".image_observation.staging-*"))


def test_run_wraps_missing_codex_executable(tmp_path: Path) -> None:
    semantic, renders = _inputs(tmp_path)

    def runner(command, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("codex")

    with pytest.raises(ImageObservationError, match="requires the `codex` executable"):
        run_image_observation(
            semantic_image_path=semantic,
            output_dir=tmp_path / "image_observation",
            model="gpt-5-codex",
            reasoning_effort="high",
            process_runner=runner,
        )


def test_schema_enforces_the_name_pattern_at_generation_time() -> None:
    """The Codex output schema carries the exact snake_case pattern the parser re-checks."""

    from usdcraft.pipeline.observation import _PART_NAME_PATTERN
    from usdcraft.reconstruction.observation import OBSERVATION_AGENT_RESULT_SCHEMA

    part_schema = OBSERVATION_AGENT_RESULT_SCHEMA["properties"]["parts"]["items"]["properties"]  # type: ignore[index]
    assert part_schema["name"]["pattern"] == _PART_NAME_PATTERN.pattern
    assert part_schema["duplicate_of"]["pattern"] == _PART_NAME_PATTERN.pattern
    assert part_schema["mirror_of"]["pattern"] == _PART_NAME_PATTERN.pattern
