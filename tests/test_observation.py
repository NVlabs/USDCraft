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

from usdcraft.pipeline.observation import (
    ImageObservationArtifacts,
    ObservationProcessResult,
    merge_reviewed_inventory,
    run_image_observation_via_codex,
)


def _part(**overrides: object) -> dict[str, object]:
    part: dict[str, object] = {
        "name": "door_panel",
        "visible_count": 1,
        "mobility": "movable",
        "level": "major_part",
        "visibility": "visible",
        "geometry": "flat rectangular panel, about half the subject height",
        "arrangement": "front face",
        "affordances": ["pull open"],
        "attaches_to": None,
        "duplicate_of": None,
        "mirror_of": None,
        "carried_as": "none",
        "notes": None,
    }
    part.update(overrides)
    return part


def _payload(parts: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "parts": parts,
        "removals": [],
        "summary": "subject",
    }
    payload.update(overrides)
    return payload


def test_merge_accepts_additions_sharpening_and_count_raises() -> None:
    base = _payload([_part(), _part(name="screw_heads", visible_count=4, level="micro_feature")])
    reviewed = _payload(
        [
            _part(
                arrangement="front face, hinged left",
                geometry="flat rectangular panel, about two thirds of the subject height",
                notes="handle glare",
            ),
            _part(name="screw_heads", visible_count=6, level="micro_feature"),
            _part(name="hinge_barrel", level="sub_feature"),
        ],
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert report["status"] == "applied"
    assert report["added"] == ["hinge_barrel"]
    assert set(report["sharpened"]) == {"door_panel", "screw_heads"}
    assert report["rejected_drops"] == []
    names = [part["name"] for part in merged["parts"]]
    assert names == ["door_panel", "screw_heads", "hinge_barrel"]
    assert merged["parts"][0]["arrangement"] == "front face, hinged left"
    assert (
        merged["parts"][0]["geometry"]
        == "flat rectangular panel, about two thirds of the subject height"
    )
    assert merged["parts"][0]["notes"] == "handle glare"
    assert merged["parts"][1]["visible_count"] == 6
    assert merged["summary"] == "subject"


def test_merge_sharpens_the_carried_as_classification() -> None:
    base = _payload([_part(), _part(name="jars", attaches_to="door_panel", carried_as="displayed")])
    reviewed = _payload(
        [_part(), _part(name="jars", attaches_to="door_panel", carried_as="enclosed")]
    )
    merged, report = merge_reviewed_inventory(base, reviewed)
    assert report["status"] == "applied" and report["sharpened"] == ["jars"]
    assert merged["parts"][1]["carried_as"] == "enclosed"


def test_merge_rejects_drops_and_then_also_rejects_that_rounds_additions() -> None:
    base = _payload([_part(name="left_leg"), _part(name="right_leg", mirror_of="left_leg")])
    # The reviewer merged the two legs into one entry: drop attempted → additions rejected too.
    reviewed = _payload([_part(name="legs", visible_count=2)])
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert sorted(report["rejected_drops"]) == ["left_leg", "right_leg"]
    assert report["added"] == []
    assert [part["name"] for part in merged["parts"]] == ["left_leg", "right_leg"]


def test_merge_freezes_identity_fields_and_rejects_count_lowering() -> None:
    base = _payload([_part(visible_count=3, mobility="movable", visibility="visible")])
    reviewed = _payload(
        [
            _part(
                visible_count=1,
                mobility="fixed",
                visibility="partially_occluded",
                duplicate_of=None,
            )
        ]
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    part = merged["parts"][0]
    assert part["visible_count"] == 3
    assert part["mobility"] == "movable"
    assert part["visibility"] == "visible"
    assert "door_panel.visible_count" in report["rejected_changes"]
    assert "door_panel.mobility" in report["rejected_changes"]
    assert "door_panel.visibility" in report["rejected_changes"]
    assert report["status"] == "unchanged"


def test_merge_freezes_attachment_declarations() -> None:
    base = _payload([_part(name="body"), _part(name="handle", attaches_to="body")])
    reviewed = _payload([_part(name="body"), _part(name="handle", attaches_to=None)])
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert merged["parts"][1]["attaches_to"] == "body"
    assert "handle.attaches_to" in report["rejected_changes"]


def test_merge_honors_explicit_named_removal_with_reason() -> None:
    base = _payload([_part(), _part(name="phantom_knob", level="sub_feature")])
    reviewed = _payload(
        [_part()],
        removals=[{"name": "phantom_knob", "reason": "no such knob anywhere in the image"}],
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert report["status"] == "applied"
    assert report["removed"] == ["phantom_knob"]
    assert report["removal_reasons"] == {"phantom_knob": "no such knob anywhere in the image"}
    assert report["rejected_drops"] == []
    assert [part["name"] for part in merged["parts"]] == ["door_panel"]
    assert merged["removals"] == []


def test_merge_rejects_contradictory_unknown_and_referenced_removals() -> None:
    base = _payload([_part(name="left_leg"), _part(name="right_leg", mirror_of="left_leg")])
    reviewed = _payload(
        [_part(name="left_leg"), _part(name="right_leg", mirror_of="left_leg")],
        removals=[
            # Referenced by the surviving right_leg mirror: rejected.
            {"name": "left_leg", "reason": "claims it is not there"},
            # Also kept in parts: contradictory, rejected.
            {"name": "right_leg", "reason": "contradiction"},
            # Unknown name: rejected.
            {"name": "ghost_part", "reason": "never existed"},
        ],
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert report["removed"] == []
    assert sorted(report["rejected_removals"]) == ["ghost_part", "left_leg", "right_leg"]
    assert [part["name"] for part in merged["parts"]] == ["left_leg", "right_leg"]


def test_merge_accepts_inferred_structural_addition() -> None:
    base = _payload([_part(name="wheel"), _part(name="cart_body")])
    reviewed = _payload(
        [
            _part(name="wheel"),
            _part(name="cart_body"),
            _part(
                name="wheel_axle",
                visibility="inferred",
                level="sub_feature",
                geometry="thin horizontal shaft between wheel and body",
                notes="the wheel cannot mount without an axle",
            ),
        ]
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert report["added"] == ["wheel_axle"]
    added = merged["parts"][-1]
    assert added["visibility"] == "inferred"


def test_merge_falls_back_to_base_when_merged_result_is_invalid() -> None:
    base = _payload([_part(name=f"part_{i:02d}") for i in range(60)])
    reviewed = _payload(
        [_part(name=f"part_{i:02d}") for i in range(60)]
        + [_part(name=f"extra_{i:02d}") for i in range(6)]
    )
    merged, report = merge_reviewed_inventory(base, reviewed)

    assert report["status"] == "merge_invalid_base_kept"
    assert len(merged["parts"]) == 60


def _engine_kwargs(tmp_path: Path, runner) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "image_paths": [tmp_path / "primary.png"],
        "output_dir": tmp_path / "image_observation",
        "build_task": lambda names: f"observe {names[0]}",
        "build_review_task": lambda names, inventory_text: f"review\n{inventory_text}",
        "product": "reconstruction",
        "manifest_schema": "reconstruction_image_observation",
        "scope": "reconstruction_image_observation",
        "agent_context_schema": "reconstruction_image_observation_agent_context",
        "agent_context": "clean_image_observation_only",
        "model": "gpt-test",
        "reasoning_effort": "high",
        "process_runner": runner,
    }


def _sequence_runner(payloads: list[object]):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        payload = payloads[len(calls)]
        calls.append(stdin_text)
        if payload is not None:
            (Path(cwd) / ".codex-run" / "final.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return ObservationProcessResult(returncode=0, stdout="{}\n", stderr="")

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_engine_review_merges_additions_into_the_published_inventory(tmp_path: Path) -> None:
    (tmp_path / "primary.png").write_bytes(b"png")
    base = _payload([_part()])
    reviewed = _payload([_part(), _part(name="hinge_barrel", level="sub_feature")])
    runner = _sequence_runner([base, reviewed])

    result = run_image_observation_via_codex(**_engine_kwargs(tmp_path, runner))  # type: ignore[arg-type]

    assert isinstance(result, ImageObservationArtifacts)
    names = [part["name"] for part in result.inventory["parts"]]  # type: ignore[index]
    assert names == ["door_panel", "hinge_barrel"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["review"]["status"] == "applied"
    assert manifest["review"]["added"] == ["hinge_barrel"]
    assert (result.root / "review_merge.json").is_file()
    # The review task received the exact first-inventory JSON.
    assert '"door_panel"' in runner.calls[1]  # type: ignore[attr-defined]


def test_engine_keeps_first_inventory_when_the_review_call_fails(tmp_path: Path) -> None:
    (tmp_path / "primary.png").write_bytes(b"png")
    base = _payload([_part()])
    runner = _sequence_runner([base, {"nonsense": True}])

    result = run_image_observation_via_codex(**_engine_kwargs(tmp_path, runner))  # type: ignore[arg-type]

    assert [part["name"] for part in result.inventory["parts"]] == ["door_panel"]  # type: ignore[index]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["review"]["status"] == "review_failed"
    assert (result.root / "review_agent" / ".codex-run" / "final.json").is_file()


def test_engine_without_review_builder_publishes_without_review_field(tmp_path: Path) -> None:
    (tmp_path / "primary.png").write_bytes(b"png")
    runner = _sequence_runner([_payload([_part()])])
    kwargs = _engine_kwargs(tmp_path, runner)
    kwargs.pop("build_review_task")

    result = run_image_observation_via_codex(**kwargs)  # type: ignore[arg-type]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "review" not in manifest
    assert len(runner.calls) == 1  # type: ignore[attr-defined]


@pytest.mark.slow
def test_codex_transport_streams_before_exit_and_retains_timeout(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    from usdcraft.pipeline.observation import _default_process_runner

    # The child cannot exit until the parent has observed both live output files.
    script = """
import pathlib, sys, time
print('{"type":"turn.started"}', flush=True)
print('waiting for response', file=sys.stderr, flush=True)
while not pathlib.Path('release').exists():
    time.sleep(0.01)
time.sleep(30)
"""
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            _default_process_runner,
            [sys.executable, "-c", script],
            stdin_text="",
            cwd=tmp_path,
            timeout_seconds=5,
            env=os.environ,
        )
        import time

        deadline = time.monotonic() + 4
        stdout = tmp_path / ".codex-run/stdout.jsonl"
        stderr = tmp_path / ".codex-run/stderr.log"
        while time.monotonic() < deadline:
            if stderr.exists() and "waiting for response" in stderr.read_text():
                break
            time.sleep(0.01)
        assert not future.done()
        assert "turn.started" in stdout.read_text()
        assert "waiting for response" in stderr.read_text()
        (tmp_path / "release").touch()
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            # Past the 5 s process deadline; the child itself would sleep 30 s more.
            future.result(timeout=15)
    assert caught.value.output == stdout.read_text()
    assert caught.value.stderr == stderr.read_text()


def test_codex_observation_and_review_have_no_time_limit(tmp_path: Path) -> None:
    import pytest

    (tmp_path / "primary.png").write_bytes(b"png")
    timeouts: list[object] = []
    inner = _sequence_runner([_payload([_part()]), _payload([_part()])])

    def runner(command, *, stdin_text, cwd, timeout_seconds, env):  # type: ignore[no-untyped-def]
        timeouts.append(timeout_seconds)
        return inner(command, stdin_text=stdin_text, cwd=cwd, timeout_seconds=None, env=env)

    run_image_observation_via_codex(**_engine_kwargs(tmp_path, runner))  # type: ignore[arg-type]

    assert timeouts == [None, None]
    with pytest.raises(TypeError):
        run_image_observation_via_codex(  # type: ignore[call-arg]
            **{**_engine_kwargs(tmp_path, runner), "timeout_seconds": 240}
        )
