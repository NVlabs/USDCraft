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

"""The physical-authoring policy is one text rendered into every Author task."""

from __future__ import annotations

from pathlib import Path

import pytest

from usdcraft.pipeline.policy import (
    render_flexible_member_bullets,
    render_mechanical_authoring_block,
    render_mechanical_authoring_bullets,
)


def test_bullets_substitute_product_and_check_command() -> None:
    generation = render_mechanical_authoring_bullets(product="Generation")
    reconstruction = render_mechanical_authoring_bullets(product="Reconstruction")
    # Full wording is covered by the rendered Author-task golden snapshots.
    assert "Generation" in generation and "Reconstruction" in reconstruction
    assert "Generation" not in reconstruction.replace("check_generation_", "")
    assert generation.replace("Generation", "X") == reconstruction.replace("Reconstruction", "X")
    tooled = render_mechanical_authoring_bullets(
        product="Reconstruction", check_command="check_model"
    )
    assert "usdcraft check" in reconstruction
    assert tooled == reconstruction.replace("usdcraft check", "check_model")
    with pytest.raises(ValueError, match="product name"):
        render_mechanical_authoring_bullets(product=" ")


def test_flexible_member_rule_binds_both_products() -> None:
    generation = render_flexible_member_bullets(product="Generation")
    reconstruction = render_flexible_member_bullets(product="Reconstruction")
    assert "Generation" in generation and "Reconstruction" in reconstruction
    assert generation.replace("Generation", "X") == reconstruction.replace("Reconstruction", "X")


def test_block_adds_the_header_and_an_optional_product_lead() -> None:
    block = render_mechanical_authoring_block(product="Reconstruction", intro="Lead sentence.")
    bare = render_mechanical_authoring_block(product="Reconstruction")
    header, body = bare.split("\n", 1)
    assert header
    assert body == (
        render_flexible_member_bullets(product="Reconstruction")
        + "\n"
        + render_mechanical_authoring_bullets(product="Reconstruction")
        + "\n"
    )
    assert block == header + "\nLead sentence.\n" + body


def test_generation_task_renders_the_shared_bullets_verbatim(tmp_path: Path) -> None:
    from usdcraft.generation.contract import GenerationRequest
    from usdcraft.generation.prompt import build_generation_prompt
    from usdcraft.generation.workspace import create_generation_workspace

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    request = GenerationRequest(prompt="a desk lamp", repo_root=repo)
    workspace = create_generation_workspace(request, base_dir=tmp_path / "runs", run_id="policy")
    prompt = build_generation_prompt(request, workspace)
    assert render_mechanical_authoring_bullets(product="Generation") in prompt
    assert render_flexible_member_bullets(product="Generation") in prompt
    assert "SIMULATOR-READY MECHANICAL AUTHORING" not in prompt.split("PRELOADED SDK REFERENCES")[0]
