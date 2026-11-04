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

import os
import shutil
from pathlib import Path

import pytest

import usdcraft.pipeline.tools as provider_tools
from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.workspace import GenerationWorkspace, create_generation_workspace
from usdcraft.pipeline.provider import (
    ProviderToolExecutor,
    ProviderToolResult,
    build_provider_tool_schemas,
)


def _workspace(tmp_path: Path) -> GenerationWorkspace:
    repo = tmp_path / "repo"
    (repo / "sdk" / "_docs" / "common").mkdir(parents=True)
    (repo / "sdk" / "_examples" / "cadquery").mkdir(parents=True)
    (repo / "scaffold.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "sdk" / "_docs" / "common" / "guide.md").write_text(
        "shared guidance\n", encoding="utf-8"
    )
    (repo / "private.txt").write_text("not shared\n", encoding="utf-8")
    request = GenerationRequest(prompt="test asset", repo_root=repo)
    return create_generation_workspace(
        request,
        base_dir=tmp_path / "runs",
        run_id="provider_tools",
    )


def _patch(old: str, new: str) -> str:
    return f"*** Begin Patch\n*** Update File: model.py\n@@\n-{old}\n+{new}\n*** End Patch"


def test_base_tool_surface_is_narrow_and_render_free() -> None:
    names = {schema["name"] for schema in build_provider_tool_schemas()}

    assert names == {
        "read_file",
        "search_docs",
        "find_examples",
        "write_model",
        "apply_patch",
        "check_model",
        "compile_model",
    }
    # Product adapters add their own candidate tools without expanding this shared base.
    assert not names.intersection({"render", "render_model", "review", "probe_model", "python"})


def test_read_and_patch_are_path_limited_and_example_results_are_bounded(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    canonical_doc = workspace.repo_root / "sdk" / "_docs" / "common" / "00_quickstart.md"
    canonical_doc.write_text("Use MotionLimits for articulated motion.\n", encoding="utf-8")
    long_example = workspace.repo_root / "sdk" / "_examples" / "cadquery" / "lamp.md"
    long_example.write_text("lamp\n" + "x" * 13_000, encoding="utf-8")
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checked"),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok({"render": render}),
    )

    allowed = executor.execute("read_file", {"path": "sdk/_docs/common/guide.md"}).payload
    denied = executor.execute("read_file", {"path": "private.txt"}).payload
    doc_matches = executor.execute("search_docs", {"query": "MotionLimits", "limit": 1}).payload
    matches = executor.execute("find_examples", {"query": "lamp", "limit": 1}).payload
    wrong_target = executor.execute(
        "apply_patch",
        {
            "patch": (
                "*** Begin Patch\n*** Update File: sdk/_docs/common/guide.md\n"
                "@@\n-shared guidance\n+changed\n*** End Patch"
            )
        },
    ).payload

    assert allowed["ok"] is True
    assert "shared guidance" in allowed["output"]["content"]
    assert denied["ok"] is False
    assert "only staged model.py" in denied["error"]
    assert doc_matches["ok"] is True
    assert doc_matches["output"]["matches"] == [
        {
            "path": "sdk/_docs/common/00_quickstart.md",
            "line": 1,
            "content": "Use MotionLimits for articulated motion.",
            "truncated": False,
        }
    ]
    match = matches["output"]["matches"][0]
    assert match["path"] == "sdk/_examples/cadquery/lamp.md"
    assert match["truncated"] is True
    assert len(match["content"]) == 12_000
    assert wrong_target["ok"] is False
    assert long_example.is_file()
    assert (
        workspace.repo_root / "sdk" / "_docs" / "common" / "guide.md"
    ).read_text() == "shared guidance\n"


def test_sdk_symlink_targets_keep_stable_logical_paths(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    shared_root = tmp_path / "shared_sdk"
    shared_docs = shared_root / "docs"
    shared_examples = shared_root / "examples"
    doc = shared_docs / "common" / "00_quickstart.md"
    example = shared_examples / "cadquery" / "lamp.md"
    doc.parent.mkdir(parents=True)
    example.parent.mkdir(parents=True)
    doc.write_text("Use MotionLimits for articulated motion.\n", encoding="utf-8")
    example.write_text("A focused lamp example.\n", encoding="utf-8")

    docs_link = workspace.repo_root / "sdk" / "_docs"
    examples_link = workspace.repo_root / "sdk" / "_examples"
    shutil.rmtree(docs_link)
    shutil.rmtree(examples_link)
    docs_link.symlink_to(shared_docs, target_is_directory=True)
    examples_link.symlink_to(shared_examples, target_is_directory=True)

    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    read = executor.execute("read_file", {"path": str(doc)}).payload
    docs = executor.execute("search_docs", {"query": "MotionLimits", "limit": 1}).payload
    examples = executor.execute("find_examples", {"query": "lamp", "limit": 1}).payload

    assert read["ok"] is True
    assert read["output"]["path"] == "sdk/_docs/common/00_quickstart.md"
    assert docs["ok"] is True
    assert docs["output"]["matches"][0]["path"] == "sdk/_docs/common/00_quickstart.md"
    assert examples["ok"] is True
    assert examples["output"]["matches"][0]["path"] == "sdk/_examples/cadquery/lamp.md"


def test_search_routes_text_to_labels_without_context_substring_noise(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    docs = workspace.repo_root / "sdk/_docs"
    (docs / "cadquery").mkdir()
    (docs / "base").mkdir()
    (docs / "cadquery/35_cadquery.md").write_text(
        "# CadQuery\ncontext-solid context context\nmesh_from_cadquery(solid, name)\n"
    )
    (docs / "base/54_decals_and_labels.md").write_text(
        "# Decals and labels\nTextureCanvas text label lettering decal\nMore text label examples\n"
    )
    executor = ProviderToolExecutor(
        workspace,
        finish_tool_name="finish_generation",
        check_runner=lambda _: ProviderToolResult.ok(),
        compile_runner=lambda *_, **__: ProviderToolResult.ok(),
    )
    matches = executor.execute(
        "search_docs", {"query": "text label lettering decal create text"}
    ).payload["output"]["matches"]
    assert [match["path"] for match in matches] == ["sdk/_docs/base/54_decals_and_labels.md"]
    assert matches[0]["line"] == 2
    api = executor.execute("search_docs", {"query": "mesh_from_cadquery"}).payload
    assert api["output"]["matches"][0]["path"] == "sdk/_docs/cadquery/35_cadquery.md"


@pytest.mark.parametrize(
    "patch",
    [
        (
            "*** Begin Patch\n"
            "@@\n"
            "-value = 1\n"
            "+value = 2\n"
            "*** Update File: model.py\n"
            "@@\n"
            "-value = 1\n"
            "+value = 2\n"
            "*** End Patch"
        ),
        (
            "*** Begin Patch\n"
            "unparsed prelude\n"
            "*** Update File: model.py\n"
            "@@\n"
            "-value = 1\n"
            "+value = 2\n"
            "*** End Patch"
        ),
        _patch("value = 1", "value = 2") + "\nunparsed trailer",
    ],
)
def test_apply_patch_rejects_content_outside_the_update_block(
    tmp_path: Path,
    patch: str,
) -> None:
    workspace = _workspace(tmp_path)
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    result = executor.execute("apply_patch", {"patch": patch}).payload

    assert result["ok"] is False
    assert workspace.model_path.read_text(encoding="utf-8") == "value = 1\n"
    assert executor.revision == 0


@pytest.mark.parametrize(
    "hunks",
    [
        "@@\n+value = 2\n",
        "@@\n-value = 1\n+value = 2\n@@\n-missing = 0\n+missing = 1\n",
    ],
)
def test_failed_patch_preserves_source_revision_and_checks(tmp_path: Path, hunks: str) -> None:
    workspace = _workspace(tmp_path)
    executor = ProviderToolExecutor(
        workspace,
        finish_tool_name="finish_generation",
        check_runner=lambda _: ProviderToolResult.ok(),
        compile_runner=lambda *_, **__: ProviderToolResult.ok(),
    )
    executor.execute("check_model", {})
    executor.execute("compile_model", {})
    result = executor.execute(
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Update File: model.py\n" + hunks + "*** End Patch"},
    ).payload
    assert result["ok"] is False and result["changed"] is False
    assert result["revision"] == executor.revision == 0
    assert executor.check_success_revision == executor.compile_success_revision == 0
    assert workspace.model_path.read_text() == "value = 1\n"
    assert "hunk" in result["error"]
    assert "write_model(content)" in result["repair_hint"]


@pytest.mark.parametrize("failure", ["create", "open", "write", "flush", "close", "replace"])
def test_atomic_write_cleans_temporary_file_after_any_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    target = tmp_path / "model.py"
    target.write_text("original\n", encoding="utf-8")
    created: list[Path] = []
    real_mkstemp = provider_tools.tempfile.mkstemp

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        if failure == "create":
            raise OSError("injected create failure")
        descriptor, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        created.append(Path(name))
        return descriptor, name

    monkeypatch.setattr(provider_tools.tempfile, "mkstemp", tracked_mkstemp)

    if failure in {"open", "write", "flush", "close"}:

        class FailingHandle:
            def __init__(self, descriptor: int) -> None:
                self.descriptor = descriptor

            def __enter__(self) -> FailingHandle:
                if failure == "open":
                    raise OSError("injected open failure")
                return self

            def write(self, content: str) -> int:
                if failure == "write":
                    raise OSError("injected write failure")
                return os.write(self.descriptor, content.encode("utf-8"))

            def flush(self) -> None:
                if failure == "flush":
                    raise OSError("injected flush failure")

            def __exit__(self, *_args: object) -> None:
                os.close(self.descriptor)
                if failure == "close":
                    raise OSError("injected close failure")

        monkeypatch.setattr(
            provider_tools.os,
            "fdopen",
            lambda descriptor, **_kwargs: FailingHandle(descriptor),
        )

    if failure == "replace":
        monkeypatch.setattr(
            Path,
            "replace",
            lambda _self, _target: (_ for _ in ()).throw(OSError("injected replace failure")),
        )

    with pytest.raises(OSError, match=f"injected {failure} failure"):
        provider_tools._atomic_write_text(target, "updated\n")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert not list(tmp_path.glob(".model.py.*.tmp"))
    assert all(not path.exists() for path in created)


def test_read_file_is_page_bounded(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.model_path.write_text(
        "".join(f"line_{index} = {index}\n" for index in range(1, 1302)),
        encoding="utf-8",
    )
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.ok(),
    )

    first_page = executor.execute("read_file", {"path": "model.py"}).payload
    oversized = executor.execute("read_file", {"path": "model.py", "limit": 1201}).payload

    assert first_page["ok"] is True
    assert first_page["output"]["end_line"] == 1200
    assert first_page["output"]["truncated"] is True
    assert first_page["output"]["next_offset"] == 1201
    assert oversized["ok"] is False


def test_complete_requires_fresh_check_and_no_render_compile_after_latest_edit(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    compile_calls: list[tuple[Path, Path, bool]] = []

    def compile_runner(
        model_path: Path,
        output_dir: Path,
        *,
        render: bool,
    ) -> ProviderToolResult:
        compile_calls.append((model_path, output_dir, render))
        return ProviderToolResult.ok("compiled")

    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok("checked"),
        compile_runner=compile_runner,  # type: ignore[arg-type]
    )
    executor.execute("apply_patch", {"patch": _patch("value = 1", "value = 2")})
    executor.execute("check_model", {})
    executor.execute("compile_model", {})
    fresh = executor.execute("finish_generation", {"status": "complete", "summary": "ready"})

    assert fresh.completion_status == "complete"
    assert compile_calls == [(workspace.model_path, workspace.build_dir, False)]

    executor.execute("apply_patch", {"patch": _patch("value = 2", "value = 3")})
    stale = executor.execute("finish_generation", {"status": "complete", "summary": "stale"})
    assert stale.completion_status is None
    assert stale.payload["ok"] is False
    assert "current revision" in stale.payload["error"]


def test_blocked_is_agent_judgment_even_when_latest_compile_failed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    executor = ProviderToolExecutor(
        finish_tool_name="finish_generation",
        workspace=workspace,
        check_runner=lambda _path: ProviderToolResult.ok(),
        compile_runner=lambda _path, _output, *, render: ProviderToolResult.error(
            "unsupported SDK topology"
        ),
    )
    executor.execute("compile_model", {})

    blocked = executor.execute(
        "finish_generation",
        {"status": "blocked", "summary": "The requested closed topology is unsupported."},
    )

    assert blocked.completion_status == "blocked"
