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

from tests.generation.provider.test_tools import _patch, _workspace
from usdcraft.generation.provider_tools import GenerationToolExecutor
from usdcraft.pipeline.provider import ProviderToolResult
from usdcraft.pipeline.tools import MAX_MODEL_CHARACTERS, ProviderToolExecutor


@pytest.fixture(params=["shared", "generation"])
def executor(tmp_path: Path, request) -> ProviderToolExecutor:
    executor_type = ProviderToolExecutor if request.param == "shared" else GenerationToolExecutor
    options = {"finish_tool_name": "finish_generation"} if request.param == "shared" else {}
    return executor_type(
        _workspace(tmp_path),
        check_runner=lambda _: ProviderToolResult.ok(),
        compile_runner=lambda *_args, **_kwargs: ProviderToolResult.ok(),
        **options,
    )


def test_whole_file_write_recovers_stale_scaffold_and_invalidates_checks(executor) -> None:
    model = executor.workspace.model_path
    failed = executor.execute("apply_patch", {"patch": _patch("value = 0", "value = 2")})
    assert not failed.payload["ok"]
    assert "write_model" in failed.payload["repair_hint"]
    assert model.read_text() == "value = 1\n" and executor.revision == 0

    assert executor.execute("write_model", {"content": "value = 2\n"}).payload["ok"]
    executor.execute("check_model", {})
    executor.execute("compile_model", {})
    assert (
        executor.execute(
            "finish_generation", {"status": "complete", "summary": "ready"}
        ).completion_status
        == "complete"
    )
    assert executor.revision == 1
    assert (
        executor.execute("write_model", {"content": "value = 2\n"}).payload["output"]["changed"]
        is False
    )
    assert executor.check_success_revision == executor.compile_success_revision == 1

    executor.execute("write_model", {"content": "value = 3\n"})
    assert model.read_text() == "value = 3\n" and executor.revision == 2
    assert executor.check_success_revision is executor.compile_success_revision is None
    assert not executor.execute(
        "finish_generation", {"status": "complete", "summary": "ready"}
    ).payload["ok"]
    assert not executor.execute(
        "render_candidate", {"azimuth": 45, "elevation": 30, "joints": []}
    ).payload["ok"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"content": ""},
        {"content": 1},
        {"content": "x" * (MAX_MODEL_CHARACTERS + 1)},
        {"content": "value = 2", "path": "other.py"},
    ],
)
def test_invalid_writes_preserve_model_and_revision(executor, arguments: dict) -> None:
    assert not executor.execute("write_model", arguments).payload["ok"]
    assert executor.workspace.model_path.read_text() == "value = 1\n"
    assert executor.revision == 0
    assert not (executor.workspace.root / "other.py").exists()


def test_syntax_error_is_editable_feedback_and_atomic_failure_preserves_source(
    executor, monkeypatch
):
    result = executor.execute("write_model", {"content": "def broken(:\n"})
    assert result.payload["ok"] and result.payload["output"]["python_syntax"]["status"] == "error"
    assert executor.workspace.model_path.read_text() == "def broken(:\n"
    executor.execute("write_model", {"content": "value = 2\n"})

    def fail(*_args):
        raise OSError("write failed")

    monkeypatch.setattr("usdcraft.pipeline.tools._atomic_write_text", fail)
    assert not executor.execute("write_model", {"content": "value = 3\n"}).payload["ok"]
    assert executor.workspace.model_path.read_text() == "value = 2\n" and executor.revision == 2
