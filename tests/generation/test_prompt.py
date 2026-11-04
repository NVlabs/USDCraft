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

"""Generation Author task: structure, arm differences, boundaries and golden snapshots.

The task wording itself is pinned by the snapshots under ``tests/golden/prompts`` (rewrite them with
``USDCRAFT_UPDATE_GOLDENS=1`` after a deliberate change); the assertions here check what a snapshot
cannot express: routing to exact paths, arm parity, and what must never enter the task.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

from sdk._profiles import get_sdk_profile
from tests.prompt_snapshot import assert_matches_golden
from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.prompt import build_generation_prompt
from usdcraft.generation.workspace import create_generation_workspace
from usdcraft.pipeline.policy import (
    render_flexible_member_bullets,
    render_mechanical_authoring_bullets,
)
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS

CHECK = ("python", "-m", "usdcraft", "check", "model.py")
COMPILE = ("python", "-m", "usdcraft", "compile", "model.py", "--output", "build", "--no-render")
INVENTORY = (
    "- base: count=1, fixed, primary_body, visible\n"
    "- shade: count=1, movable, major_part, visible; attaches_to=base"
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return repo


def _image(tmp_path: Path) -> Path:
    image = tmp_path / "lamp.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    return image


def _workspace(tmp_path: Path, request: GenerationRequest, run_id: str):  # type: ignore[no-untyped-def]
    return create_generation_workspace(request, base_dir=tmp_path / "runs", run_id=run_id)


def _paths(repo: Path, workspace) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {str(workspace.root): "<workspace>", str(repo): "<repo>"}


def _body(prompt: str) -> str:
    """The task text outside the verbatim preloaded SDK documents."""

    head, _, tail = prompt.partition("PRELOADED SDK REFERENCES")
    return head + tail.split("SDK REFERENCE DISCOVERY", 1)[1]


# --- golden snapshots ------------------------------------------------------------------------


def test_codex_text_task_matches_golden(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a desk lamp with two movable arms", repo_root=repo)
    workspace = _workspace(tmp_path, request, "golden_codex_text")
    prompt = build_generation_prompt(
        request, workspace, check_command=CHECK, compile_command=COMPILE
    )
    assert_matches_golden("generation_codex_text", prompt, paths=_paths(repo, workspace))


def test_codex_image_task_with_inventory_matches_golden(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(
        prompt="a functional household refrigerator", repo_root=repo, image_path=_image(tmp_path)
    )
    workspace = _workspace(tmp_path, request, "golden_codex_image")
    prompt = build_generation_prompt(
        request,
        workspace,
        check_command=CHECK,
        compile_command=COMPILE,
        image_observation_text=INVENTORY,
    )
    assert_matches_golden("generation_codex_image", prompt, paths=_paths(repo, workspace))


def test_provider_image_task_matches_golden(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt=None, repo_root=repo, image_path=_image(tmp_path))
    workspace = _workspace(tmp_path, request, "golden_provider_image")
    prompt = build_generation_prompt(
        request,
        workspace,
        check_command="check_model",
        compile_command="compile_model",
        agent_name="Provider",
        reference_access="provider_tools",
        image_observation_text=INVENTORY,
        turn_budget=64,
    )
    assert_matches_golden("generation_provider_image", prompt, paths=_paths(repo, workspace))


# --- structure and boundaries ----------------------------------------------------------------


def test_prompt_routes_only_exact_canonical_sdk_document_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a cabinet with three drawers", repo_root=repo)
    prompt = build_generation_prompt(request, _workspace(tmp_path, request, "sdk_docs"))
    discovery = prompt.split("SDK REFERENCE DISCOVERY", 1)[1]
    inventory = discovery.split("<!-- sdk-doc-inventory:start -->", 1)[1].split(
        "<!-- sdk-doc-inventory:end -->", 1
    )[0]
    listed = re.findall(r"^- `([^`]+)`(?: \(preloaded above\))?$", inventory, re.MULTILINE)
    assert listed == [path.as_posix() for path in get_sdk_profile("sdk").docs_full]
    assert inventory.count("(preloaded above)") == len(PRELOADED_SDK_DOC_PATHS)


def test_both_arms_share_one_body_and_differ_only_in_identity_and_tools(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a desk lamp", repo_root=repo)
    workspace = _workspace(tmp_path, request, "arms")
    codex = build_generation_prompt(request, workspace)
    provider_named = build_generation_prompt(request, workspace, agent_name="Provider")
    provider = build_generation_prompt(
        request, workspace, agent_name="Provider", reference_access="provider_tools"
    )

    # Only the identity line differs between the arm names; the rest of the task is shared.
    assert codex.split("\n", 1)[0] != provider_named.split("\n", 1)[0]
    assert codex.split("\n", 1)[1] == provider_named.split("\n", 1)[1]

    # The Provider arm advertises its bounded tools; the Codex arm never does.
    codex_body, provider_body = _body(codex), _body(provider)
    for tool in ("search_docs", "find_examples", "read_file", "write_model"):
        assert tool in provider_body and tool not in codex_body, tool

    # The shared authoring policy is rendered verbatim into both arms.
    for arm in (codex, provider):
        assert render_mechanical_authoring_bullets(product="Generation") in arm
        assert render_flexible_member_bullets(product="Generation") in arm


def test_task_never_carries_runtime_identity_or_repository_context(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(
        prompt="a desk lamp",
        repo_root=repo,
        model="runtime-model-must-not-enter-task",
        reasoning_effort="runtime-effort-must-not-enter-task",
    )
    workspace = _workspace(tmp_path, request, "boundaries")
    for prompt in (
        build_generation_prompt(request, workspace),
        build_generation_prompt(
            request, workspace, agent_name="Provider", reference_access="provider_tools"
        ),
    ):
        body = _body(prompt)
        assert '"text": "a desk lamp"' in prompt
        for forbidden in (
            "runtime-model-must-not-enter-task",
            "runtime-effort-must-not-enter-task",
            "AGENTS.md",
            "Viewer",
            "Storage",
            "render_model",
        ):
            assert forbidden not in body, forbidden
        assert "Candidate views describe your output" in body
        assert "Visual-Review Agent" not in body
        assert "do not re-verify" not in body


def test_image_mode_changes_the_task_and_carries_the_inventory_only_when_given(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    text_only = GenerationRequest(prompt="a desk lamp", repo_root=repo)
    with_image = GenerationRequest(
        prompt="a desk lamp", repo_root=repo, image_path=_image(tmp_path)
    )
    text_prompt = build_generation_prompt(text_only, _workspace(tmp_path, text_only, "text"))
    image_ws = _workspace(tmp_path, with_image, "image")
    image_prompt = build_generation_prompt(with_image, image_ws)
    inventory_prompt = build_generation_prompt(
        with_image, image_ws, image_observation_text=INVENTORY
    )

    assert text_prompt != image_prompt
    assert INVENTORY not in image_prompt
    assert INVENTORY in inventory_prompt


def test_render_access_and_budget_are_transport_specific(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a desk lamp", repo_root=repo)
    workspace = _workspace(tmp_path, request, "render")
    provider = build_generation_prompt(
        request, workspace, reference_access="provider_tools", turn_budget=64
    )
    codex = build_generation_prompt(request, workspace)
    assert "`render_candidate`" in provider
    assert "usdcraft generation-render" in codex
    assert "fixed budget of 64 tool turns" in provider
    assert "turns_remaining" not in codex
    for task in (provider, codex):
        assert "Mechanical success alone" in task
        assert "PNG path alone" in task
    with pytest.raises(ValueError, match="turn budget"):
        build_generation_prompt(request, workspace, turn_budget=64)
    with pytest.raises(ValueError, match="reference access"):
        build_generation_prompt(request, workspace, reference_access="browser")


def test_preloaded_documents_are_the_canonical_files_on_both_arms(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a desk lamp", repo_root=repo)
    workspace = _workspace(tmp_path, request, "preloaded")
    provider = build_generation_prompt(
        request, workspace, agent_name="Provider", reference_access="provider_tools"
    )
    codex = build_generation_prompt(request, workspace)

    def section(prompt: str) -> str:
        return prompt.split("PRELOADED SDK REFERENCES", 1)[1].split("SDK REFERENCE DISCOVERY", 1)[0]

    provider_section, codex_section = section(provider), section(codex)
    for relative in PRELOADED_SDK_DOC_PATHS:
        assert f"===== BEGIN {relative} =====" in provider_section
        text = (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8")
        assert text.strip().splitlines()[0] in provider_section
    # Same documents; only the re-read hint names the arm's access.
    assert provider_section.split("\n\n", 1)[1] == codex_section.split("\n\n", 1)[1]
    assert "`read_file`" in provider_section.split("\n\n", 1)[0]
    assert "open one of them again" in codex_section.split("\n\n", 1)[0]


@pytest.mark.parametrize("with_image", [False, True])
def test_cache_prefix_is_identical_across_requests(tmp_path: Path, with_image: bool) -> None:
    from usdcraft.generation.prompt import generation_cache_prefix

    repo = _repo(tmp_path)
    prefixes = []
    image = _image(tmp_path) if with_image else None
    for index in range(2):
        request = GenerationRequest(
            prompt=f"instance-sentinel-{index}", repo_root=repo, image_path=image
        )
        workspace = _workspace(tmp_path, request, f"cache_{index}")
        task = build_generation_prompt(
            request,
            workspace,
            reference_access="provider_tools",
            image_observation_text="observation-sentinel" if image else None,
        )
        prefix = generation_cache_prefix(task)
        assert task.startswith(prefix)
        assert "instance-sentinel" not in prefix
        assert "observation-sentinel" not in prefix
        assert str(workspace.root) not in prefix
        assert "PRELOADED SDK REFERENCES" in prefix
        prefixes.append(prefix)
    assert prefixes[0] == prefixes[1]
