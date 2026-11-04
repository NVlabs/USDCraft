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

"""Initial context retains case evidence and stable in-session cache boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.reconstruction.test_shared_harness import render_case
from usdcraft.pipeline.provider.harness import _user_message_content
from usdcraft.pipeline.sdk_context import (
    render_preloaded_sdk_docs,
    render_sdk_document_inventory,
)
from usdcraft.reconstruction.prompt import (
    build_reconstruction_author_prompt,
    reconstruction_cache_prefix,
)


def test_initial_blocks_preserve_task_and_image_on_followup(tmp_path: Path) -> None:
    prompt, _, _, workspace = render_case(tmp_path, False, "provider_tools")
    prefix = reconstruction_cache_prefix(prompt)
    content = _user_message_content(
        prompt.strip(), image_paths=workspace.attached_images, cache_prefix=prefix
    )
    assert [block["type"] for block in content] == ["input_text", "input_text", "input_image"]
    assert "".join(block["text"] for block in content[:2]) == prompt.strip()
    assert "REQUEST\n" not in prefix
    assert "REQUEST\n" in content[1]["text"]
    assert all(block["prompt_cache_breakpoint"] == {"mode": "explicit"} for block in content[:2])
    # Subsequent image feedback keeps its own text boundary without replacing initial content.
    feedback = _user_message_content(
        "Current revision feedback", image_paths=workspace.attached_images
    )
    assert feedback[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert content[0]["text"] == prefix


@pytest.mark.parametrize("arm", ["provider_tools", "shell"])
def test_full_sdk_context_and_case_inputs_survive_boundary(tmp_path: Path, arm: str) -> None:
    inputs = {
        "request_text": "Reconstruct this particular drawer.",
        "user_guidance": "Preserve the label:\nRUN INPUTS\nuser-supplied text.",
        "scale_note": "Use the user-supplied metre scale.",
        "image_observation_text": "Visible handle with uncertain attachment.",
    }

    def builder(**kwargs):
        return build_reconstruction_author_prompt(**(kwargs | inputs))

    prompt, _, _, workspace = render_case(tmp_path, True, arm, builder=builder)
    prefix = reconstruction_cache_prefix(prompt)
    suffix = prompt[len(prefix) :]
    foundations = render_preloaded_sdk_docs(
        workspace.repo_root,
        task_noun="Reconstruction Author",
        reread_clause="reopen a preloaded document only for an exact detail",
    )
    assert foundations in prefix
    assert render_sdk_document_inventory(workspace.repo_root, mark_preloaded=True) in prefix
    for value in inputs.values():
        assert value not in prefix
        assert suffix.count(value) == 1
    assert f"inputs/{workspace.semantic_image_path.name}" in suffix


def test_text_only_task_retains_both_blocks() -> None:
    content = _user_message_content("public\ncase", image_paths=(), cache_prefix="public")
    assert "".join(block["text"] for block in content) == "public\ncase"
    assert content[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in content[1]


def test_cache_prefix_cannot_drop_or_replace_task_text() -> None:
    with pytest.raises(ValueError, match="exact prefix"):
        _user_message_content("real task", image_paths=(), cache_prefix="other task")
