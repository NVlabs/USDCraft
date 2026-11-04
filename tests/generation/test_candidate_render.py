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

from tests.generation.provider.test_harness import _call, _ScriptedClient, _workspace
from tests.generation.provider.test_tools import _patch
from usdcraft.generation.candidate_render import render_candidate
from usdcraft.generation.provider_tools import GenerationToolExecutor, build_generation_tool_schemas
from usdcraft.generation.render_service import CandidateRenderService, request_candidate_render
from usdcraft.pipeline.provider import ProviderHarness, ProviderToolResult


def _render(request):
    request.out_dir.mkdir(parents=True, exist_ok=True)
    image = request.out_dir / "candidate.png"
    Image.new("RGB", (16, 16), "red").save(image)
    return SimpleNamespace(image_paths=(image,), backend="test", capabilities={"joint_poses": True})


def test_render_preserves_camera_pose_and_reports_backend(tmp_path):
    (tmp_path / "model.urdf").write_text('<robot><joint name="slide"/></robot>')
    seen = []

    def renderer(request):
        seen.append(request)
        return _render(request)

    result = render_candidate(
        tmp_path,
        out_dir=tmp_path / "feedback",
        azimuth=-45,
        elevation=20,
        joint_positions={"slide": 0.1},
        renderer=renderer,
    )
    assert seen[0].views == (("author", -45, 20),)
    assert seen[0].joint_positions == {"slide": 0.1}
    assert result["backend"] == "test" and result["fallback_reason"] is None
    assert Path(result["images"][0]).is_file()
    assert json.loads((tmp_path / "feedback/feedback.json").read_text()) == result
    for kwargs in ({"azimuth": float("nan")}, {"elevation": 91}, {"joint_positions": {"bad": 0.1}}):
        with pytest.raises(ValueError):
            render_candidate(tmp_path, out_dir=tmp_path / "bad", renderer=renderer, **kwargs)
    assert len(seen) == 1


@pytest.mark.parametrize("render_fails", [False, True])
@pytest.mark.parametrize("with_image", [False, True])
def test_provider_same_session_inspection_repair_and_finish(
    tmp_path, monkeypatch, render_fails, with_image
):
    workspace = _workspace(tmp_path, with_image=with_image)
    if with_image:
        Image.new("RGB", (8, 8), "blue").save(workspace.input_image_path)
    requests = []

    def render(request):
        requests.append(request)
        if render_fails:
            raise RuntimeError("renderer unavailable")
        return _render(request)

    monkeypatch.setattr("usdcraft.render.browser.render_generated_model_views", render)

    def compile(_model, output, *, render):
        assert render is False
        output.mkdir(exist_ok=True)
        (output / "model.urdf").write_text('<robot><joint name="hinge"/></robot>')
        return ProviderToolResult.ok()

    executor = GenerationToolExecutor(
        workspace, check_runner=lambda _: ProviderToolResult.ok(), compile_runner=compile
    )
    view = {"azimuth": 45, "elevation": 30, "joints": [{"name": "hinge", "position": 0.4}]}
    client = _ScriptedClient(
        [
            [_call("edit", "write_model", {"content": "value = 2\n"})],
            [_call("check", "check_model", {}), _call("compile", "compile_model", {})],
            [_call("view", "render_candidate", view)],
            [_call("repair", "apply_patch", {"patch": _patch("value = 2", "value = 3")})],
            [_call("stale", "render_candidate", view)],
            [_call("check2", "check_model", {}), _call("compile2", "compile_model", {})],
            [_call("view2", "render_candidate", view)],
            [
                _call(
                    "done",
                    "finish_generation",
                    {"status": "complete", "summary": "inspected or reported limitation"},
                )
            ],
        ]
    )
    result = ProviderHarness(
        progress_label="Provider Generation",
        workspace=workspace,
        client=client,
        tool_executor=executor,
        tool_schemas=build_generation_tool_schemas(),
        max_turns=16,
    ).run(prompt="Build and inspect")
    assert result.status == "complete" and "value = 3" in workspace.model_path.read_text()
    assert len(requests) == 2  # The stale compile did not render.
    assert all(request.joint_positions == {"hinge": 0.4} for request in requests)
    history = client.inputs[3]
    if render_fails:
        assert "renderer unavailable" in history[-1]["output"]
    else:
        assert history[-1]["role"] == "user"
        assert history[-1]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert client.inputs[7][-1]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert client.inputs[7][: len(history)] == history
        image_items = history[-1]["content"][1:]
        assert len(image_items) == (2 if with_image else 1)
        assert all(item["image_url"].startswith("data:image/png;base64,") for item in image_items)
        roles = json.loads(history[-2]["output"])["attachment_roles"]
        assert roles == (["inspiration_reference", "candidate"] if with_image else ["candidate"])
        if with_image:
            from usdcraft.pipeline.provider.harness import image_path_to_data_url

            assert image_items[0]["image_url"] == image_path_to_data_url(workspace.input_image_path)
            assert image_items[0]["image_url"] != image_items[1]["image_url"]
    assert "Compile the current model revision" in client.inputs[5][-1]["output"]
    assert "render_candidate" in {tool["name"] for tool in client.tools[0]}
    events_text = (workspace.root / ".provider-run/events.jsonl").read_text()
    events = [json.loads(line) for line in events_text.splitlines()]
    write = next(event for event in events if event.get("tool_name") == "write_model")
    assert write["arguments"] == {"content": "value = 2\n"} and write["output"]["ok"]
    assert "data:image/png;base64," not in events_text


def test_codex_host_roundtrip_cli_paths_and_failure(tmp_path, monkeypatch, capsys):
    from usdcraft.cli import main

    monkeypatch.setenv("USDCRAFT_GENERATION_WORKSPACE", str(tmp_path))
    build = tmp_path / "build"
    build.mkdir()
    (build / "model.urdf").write_text('<robot><joint name="slide"/></robot>')
    seen = []

    def render(_self, request):
        seen.append(request)
        return _render(request)

    monkeypatch.setattr("usdcraft.render.browser.ViewerRenderSession.render", render)
    args = [
        "generation-render",
        str(build),
        "--output",
        str(tmp_path / "author_feedback/view"),
        "--joint",
        "slide=0.2",
        "--azimuth",
        "10",
    ]
    with CandidateRenderService(tmp_path):
        assert main(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["execution"] == "generation_host"
        assert Path(payload["images"][0]).is_file()
        assert seen[0].joint_positions == {"slide": 0.2} and seen[0].views[0][1] == 10
        assert main([*args, "--joint", "bad=1"]) == 1
        assert "Unknown joints" in capsys.readouterr().err
        for asset_dir, output in [
            (tmp_path / "other", tmp_path / "author_feedback/x"),
            (build, tmp_path / "model.py"),
        ]:
            with pytest.raises(RuntimeError, match="Render"):
                request_candidate_render(
                    asset_dir, out_dir=output, azimuth=0, elevation=0, joint_positions={}
                )
        with pytest.raises(ValueError, match="inside"):
            request_candidate_render(
                build,
                out_dir=tmp_path.parent / "outside",
                azimuth=0,
                elevation=0,
                joint_positions={},
            )
    assert main(args) == 1
    assert "unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [None, "check", "compile", "render"])
@pytest.mark.parametrize("with_image", [False, True])
def test_combined_validation_tracks_revision_and_diagnostic_failure(
    tmp_path, monkeypatch, failure, with_image
):
    workspace = _workspace(tmp_path, with_image=with_image)
    calls = []

    def check(_):
        calls.append("check")
        return (
            ProviderToolResult.error("invalid") if failure == "check" else ProviderToolResult.ok()
        )

    def compile(*args, **kwargs):
        calls.append("compile")
        return (
            ProviderToolResult.error("invalid") if failure == "compile" else ProviderToolResult.ok()
        )

    candidate = tmp_path / "candidate.png"

    def render(*args, **kwargs):
        calls.append("render")
        if failure == "render":
            raise RuntimeError("unavailable")
        return {"ok": True, "images": [str(candidate)]}

    monkeypatch.setattr("usdcraft.generation.provider_tools.render_candidate", render)
    executor = GenerationToolExecutor(workspace, check_runner=check, compile_runner=compile)
    executor.execute("write_model", {"content": "value = 2\n"})
    result = executor.execute("validate_model", {})
    expected = (
        ["check"]
        if failure == "check"
        else ["check", "compile"]
        if failure == "compile"
        else ["check", "compile", "render"]
    )
    assert calls == expected
    assert result.payload["ok"] is (failure not in {"check", "compile"})
    assert result.payload["diagnostics_complete"] is (failure is None)
    state = result.payload["validation_state"]["operations"]
    assert set(state) == {"check", "compile", "render"}
    if failure:
        assert state[failure]["status"] == "failed"
    else:
        assert all(value["status"] == "current" for value in state.values())
        assert result.image_paths == (
            (workspace.input_image_path, candidate) if with_image else (candidate,)
        )
    if failure not in {"check", "compile"}:
        assert executor.execute(
            "finish_generation", {"status": "complete", "summary": "done"}
        ).payload["ok"]
        executor.execute("write_model", {"content": "value = 3\n"})
        assert executor.validation_state()["operations"]["compile"]["status"] == "stale"
        assert not executor.execute(
            "finish_generation", {"status": "complete", "summary": "done"}
        ).payload["ok"]
