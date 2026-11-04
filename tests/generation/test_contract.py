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

from pathlib import Path

import pytest

from usdcraft.generation.contract import (
    GenerationImageRole,
    GenerationInputKind,
    GenerationRequest,
    generation_object_name,
)
from usdcraft.pipeline.contracts import (
    CodexCompletionStatus,
    CodexFinalMessage,
    CodexReasoningEffort,
)


def test_text_request_is_normalized(tmp_path: Path) -> None:
    request = GenerationRequest(
        prompt="  a functional desk lamp  ",
        repo_root=tmp_path,
        model="  gpt-codex  ",
        reasoning_effort="med",
    )

    assert request.prompt == "a functional desk lamp"
    assert request.repo_root == tmp_path.resolve()
    assert request.input_kind is GenerationInputKind.TEXT
    assert request.image_role is None
    assert request.model == "gpt-codex"
    assert request.reasoning_effort == CodexReasoningEffort.MEDIUM.value
    assert "timeout_seconds" not in request.to_dict()["codex"]


def test_text_image_request_has_fixed_inspiration_role(tmp_path: Path) -> None:
    image = tmp_path / "lamp.png"
    image.write_bytes(b"png")

    request = GenerationRequest(
        prompt="make a related lamp",
        repo_root=tmp_path,
        image_path=image,
    )

    assert request.input_kind is GenerationInputKind.TEXT_IMAGE
    assert request.image_role is GenerationImageRole.INSPIRATION
    assert request.image_path == image.resolve()
    assert request.to_dict()["input"] == {
        "kind": "text_image",
        "image_role": "inspiration",
        "image_path": str(image.resolve()),
    }


def test_future_reasoning_effort_is_preserved_verbatim(tmp_path: Path) -> None:
    request = GenerationRequest(
        prompt="a lamp",
        repo_root=tmp_path,
        model="future-model",
        reasoning_effort="future-tier-max",
        runtime_source="explicit_generation_request",
        selection_source="explicit_generation_request",
    )

    assert request.reasoning_effort == "future-tier-max"
    assert request.runtime.to_dict() == {
        "model_id": "future-model",
        "reasoning_effort": "future-tier-max",
        "runtime_source": "explicit_generation_request",
        "selection_source": "explicit_generation_request",
    }


def test_generation_object_name_is_single_line_bounded_and_label_first() -> None:
    assert generation_object_name(prompt="ignored prompt", label="  Desk\n  lamp  ") == "Desk lamp"
    assert generation_object_name(prompt="  Desk\n  lamp  ", label=None) == "Desk lamp"
    assert generation_object_name(prompt="x" * 121, label=None) == "x" * 117 + "..."


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_text_is_required(tmp_path: Path, prompt: str) -> None:
    with pytest.raises(ValueError, match="text prompt is required"):
        GenerationRequest(prompt=prompt, repo_root=tmp_path)


def test_image_must_be_one_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image is not a file"):
        GenerationRequest(
            prompt="lamp",
            repo_root=tmp_path,
            image_path=tmp_path / "missing.png",
        )


def test_codex_request_has_no_time_limit(tmp_path: Path) -> None:
    # Time and turn limits are Provider-only; the Codex request accepts no timeout.
    with pytest.raises(TypeError, match="timeout_seconds"):
        GenerationRequest(prompt="lamp", repo_root=tmp_path, timeout_seconds=90)  # type: ignore[call-arg]


def test_final_message_contract() -> None:
    message = CodexFinalMessage.from_payload(
        {"status": "complete", "summary": "  model authored and checked  "}
    )

    assert message.status is CodexCompletionStatus.COMPLETE
    assert message.summary == "model authored and checked"
    assert message.to_dict() == {
        "status": "complete",
        "summary": "model authored and checked",
    }

    with pytest.raises(ValueError, match="complete.*blocked"):
        CodexFinalMessage.from_payload({"status": "success", "summary": "done"})
    with pytest.raises(ValueError, match="summary is required"):
        CodexFinalMessage.from_payload({"status": "blocked", "summary": ""})
    with pytest.raises(ValueError, match="summary is required"):
        CodexFinalMessage.from_payload({"status": "blocked", "summary": 1})
