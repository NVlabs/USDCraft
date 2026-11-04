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

"""Pin the shared task and tool behavior across both arms and the image-observation switch."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.prompt_snapshot import normalize_task
from tests.reconstruction.test_reconstruction_codex_author import _evidence, _inputs
from tests.reconstruction.test_reconstruction_provider import _prompt_common
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS, preloaded_sdk_doc_paths
from usdcraft.reconstruction.prompt import build_reconstruction_author_prompt
from usdcraft.reconstruction.provider_tools import build_reconstruction_tool_schemas
from usdcraft.reconstruction.workspace import (
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)

# image_observation off/on; source rendering is always available.
VARIANTS = (False, True)
BASELINE = Path(__file__).parents[1] / "golden/prompts/reconstruction_prompt_variants.json"


def render_case(
    tmp_path: Path,
    image_observation: bool,
    arm: str,
    builder=build_reconstruction_author_prompt,
):
    repo, _, _, preflight = _inputs(tmp_path)
    preflight = replace(
        preflight,
        request=replace(preflight.request, image_observation=image_observation),
    )
    run = create_reconstruction_run_workspace(preflight)
    workspace = create_reconstruction_author_workspace(
        preflight,
        run,
        _evidence(run),
        root=run.author_workspace_base,
        author_brief=True,
    )
    common = _prompt_common(preflight, workspace)
    if not image_observation:
        common["image_observation_text"] = None
    prompt = builder(
        **common,
        check_command="check_model",
        compile_command="compile_model",
        axial_reencode_command="check_axial",
        measure_command="usdcraft recon-measure-face inputs/axial.txt",
        reference_access=arm,
    )
    paths = {str(workspace.root): "<workspace>", str(repo.resolve()): "<repo>"}
    return prompt, paths, preflight, workspace


@pytest.mark.parametrize("image_observation", VARIANTS)
@pytest.mark.parametrize("arm", ["provider_tools", "shell"])
def test_free_prompt_tools_and_foundations(
    tmp_path: Path, image_observation: bool, arm: str
) -> None:
    prompt, paths, preflight, workspace = render_case(tmp_path, image_observation, arm)
    assert workspace.attached_images == (workspace.semantic_image_path,)
    assert (workspace.root / "inputs/source_render/model.urdf").is_file()
    assert workspace.source_mesh_path.is_file()
    assert not list((workspace.root / "inputs").glob("mesh_render_*.png"))
    assert prompt.count("SOURCE INTERIOR INSPECTION\n") == 1
    assert prompt.count("FUNCTIONAL MOBILITY REVIEW\n") == 1
    for relative, path in zip(
        PRELOADED_SDK_DOC_PATHS, preloaded_sdk_doc_paths(preflight.request.repo_root), strict=True
    ):
        block = (
            f"===== BEGIN {relative} =====\n{path.read_text().rstrip()}\n===== END {relative} ====="
        )
        assert prompt.count(block) == 1
    inventory = prompt.split("PUBLIC SDK REFERENCES\n", 1)[1]
    assert inventory.count("(preloaded above)") == len(PRELOADED_SDK_DOC_PATHS)
    key = f"{arm}:image_observation={int(image_observation)}"
    baseline = json.loads(BASELINE.read_text())
    if os.environ.get("USDCRAFT_UPDATE_GOLDENS") == "1":
        baseline[key]["prompt"] = hashlib.sha256(
            normalize_task(prompt, paths=paths).encode()
        ).hexdigest()
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    expected = baseline[key]
    assert (
        hashlib.sha256(normalize_task(prompt, paths=paths).encode()).hexdigest()
        == expected["prompt"]
    )
    schemas = json.dumps(build_reconstruction_tool_schemas(), sort_keys=True)
    if os.environ.get("USDCRAFT_UPDATE_GOLDENS") == "1":
        baseline[key]["tools"] = hashlib.sha256(schemas.encode()).hexdigest()
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    names = {item["name"] for item in build_reconstruction_tool_schemas()}
    assert {"measure_mesh", "render_candidate", "compare_part", "render_source"} <= names
    assert "render_source" in prompt if arm == "provider_tools" else "recon-render-source" in prompt
    assert hashlib.sha256(schemas.encode()).hexdigest() == expected["tools"]
