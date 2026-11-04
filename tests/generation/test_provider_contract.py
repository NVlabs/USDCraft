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

from usdcraft.generation.contract import GenerationImageRole, GenerationInputKind
from usdcraft.generation.provider_contract import ProviderGenerationRequest
from usdcraft.pipeline.contracts import ProviderRuntimeSelection


def _runtime() -> ProviderRuntimeSelection:
    return ProviderRuntimeSelection(model_id="azure/openai/gpt-5.5", reasoning_effort="high")


def test_provider_request_projects_direct_text_input(tmp_path: Path) -> None:
    request = ProviderGenerationRequest(
        prompt="a useful lamp",
        repo_root=tmp_path,
        runtime=_runtime(),
    )

    assert request.input_kind is GenerationInputKind.TEXT
    assert request.image_role is None
    assert request.to_dict()["provider"] == {
        "name": "openai",
        "model_id": "azure/openai/gpt-5.5",
        "reasoning_effort": "high",
        "runtime_source": "openai-responses-api",
        "selection_source": "repository_env_or_provider_defaults",
        "timeout_seconds": None,
    }


def test_provider_request_names_its_endpoint(tmp_path: Path) -> None:
    request = ProviderGenerationRequest(
        prompt="a useful lamp",
        repo_root=tmp_path,
        runtime=ProviderRuntimeSelection(
            model_id="gpt-test",
            reasoning_effort="high",
            runtime_source="openai-responses-api",
        ),
        endpoint="openai",
    )

    provider = request.to_dict()["provider"]
    assert (provider["name"], provider["runtime_source"]) == ("openai", "openai-responses-api")
    with pytest.raises(ValueError, match="nvidia or openai"):
        ProviderGenerationRequest(
            prompt="a useful lamp",
            repo_root=tmp_path,
            runtime=_runtime(),
            endpoint="codex-cli",  # type: ignore[arg-type]
        )


def test_provider_request_projects_inspiration_image(tmp_path: Path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"image")

    request = ProviderGenerationRequest(
        prompt="a useful lamp",
        repo_root=tmp_path,
        image_path=image,
        runtime=_runtime(),
    )

    assert request.input_kind is GenerationInputKind.TEXT_IMAGE
    assert request.image_role is GenerationImageRole.INSPIRATION
    assert request.image_path == image.resolve()


def test_provider_request_accepts_the_same_multiline_prompt_semantics_as_codex(
    tmp_path: Path,
) -> None:
    request = ProviderGenerationRequest(
        prompt="  Build a lamp\n\twith two articulated arms.  ",
        repo_root=tmp_path,
        runtime=_runtime(),
    )

    assert request.prompt == "Build a lamp\n\twith two articulated arms."


def test_provider_request_keeps_an_explicit_timeout(tmp_path: Path) -> None:
    request = ProviderGenerationRequest(
        prompt="a lamp", repo_root=tmp_path, runtime=_runtime(), timeout_seconds=45
    )

    assert request.timeout_seconds == 45.0
    assert request.to_dict()["provider"]["timeout_seconds"] == 45.0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_provider_request_rejects_invalid_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        ProviderGenerationRequest(
            prompt="a lamp",
            repo_root=tmp_path,
            runtime=_runtime(),
            timeout_seconds=timeout,
        )
