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

from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.workspace import create_generation_workspace


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    return root


def test_workspace_copies_scaffold_and_inspiration_image(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    image = tmp_path / "Reference.JPEG"
    image.write_bytes(b"image-bytes")
    request = GenerationRequest(prompt="a lamp", repo_root=repo, image_path=image)

    workspace = create_generation_workspace(
        request,
        base_dir=tmp_path / "runs",
        run_id="generation_case_001",
    )

    assert workspace.root == (tmp_path / "runs" / "generation_case_001").resolve()
    assert workspace.model_path.read_text(encoding="utf-8") == "object_model = object()\n"
    assert workspace.input_image_path == workspace.root / "inputs" / "reference.jpeg"
    assert workspace.input_image_path.read_bytes() == b"image-bytes"
    assert workspace.codex_dir.is_dir()
    assert workspace.codex_timeline_path == workspace.codex_dir / "timeline.jsonl"
    assert workspace.codex_trace_errors_path == workspace.codex_dir / "trace_errors.log"
    assert workspace.build_dir.is_dir()
    assert workspace.render_dir.is_dir()
    assert not (workspace.root / ".git").exists()

    manifest = json.loads(workspace.request_path.read_text(encoding="utf-8"))
    assert manifest["pipeline"] == "generation"
    assert manifest["input"]["kind"] == "text_image"
    assert manifest["input"]["image_role"] == "inspiration"
    assert manifest["workspace"] == {
        "model_path": "model.py",
        "input_image_path": "inputs/reference.jpeg",
    }


def test_workspace_text_request_has_no_staged_image(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a mug", repo_root=repo)

    workspace = create_generation_workspace(
        request,
        base_dir=tmp_path / "runs",
        run_id="text_case",
    )

    assert workspace.input_image_path is None
    manifest = json.loads(workspace.request_path.read_text(encoding="utf-8"))
    assert manifest["input"]["kind"] == "text"
    assert manifest["input"]["image_role"] is None
    assert manifest["workspace"]["input_image_path"] is None


def test_workspace_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a mug", repo_root=repo)
    base_dir = tmp_path / "runs"
    create_generation_workspace(request, base_dir=base_dir, run_id="same_run")

    with pytest.raises(FileExistsError, match="already exists"):
        create_generation_workspace(request, base_dir=base_dir, run_id="same_run")


@pytest.mark.parametrize("run_id", ["../escape", "with space", "", "a" * 97])
def test_workspace_rejects_unsafe_explicit_run_ids(tmp_path: Path, run_id: str) -> None:
    repo = _repo(tmp_path)
    request = GenerationRequest(prompt="a mug", repo_root=repo)

    with pytest.raises(ValueError, match="run id"):
        create_generation_workspace(
            request,
            base_dir=tmp_path / "runs",
            run_id=run_id,
        )
