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
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class GenerationWorkspaceRequest(Protocol):
    """Small shared request surface needed to stage either Generation arm."""

    @property
    def repo_root(self) -> Path: ...

    @property
    def image_path(self) -> Path | None: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True, frozen=True)
class GenerationWorkspace:
    run_id: str
    repo_root: Path
    root: Path
    model_path: Path
    request_path: Path
    task_path: Path
    input_image_path: Path | None
    codex_dir: Path
    codex_schema_path: Path
    codex_last_message_path: Path
    codex_stdout_path: Path
    codex_stderr_path: Path
    codex_timeline_path: Path
    codex_trace_errors_path: Path
    build_dir: Path
    render_dir: Path
    temp_dir: Path
    cache_dir: Path

    def write_task(self, text: str) -> None:
        self.task_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    def write_codex_schema(self, schema: Mapping[str, Any]) -> None:
        self.codex_schema_path.write_text(
            json.dumps(dict(schema), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_codex_logs(self, *, stdout: str, stderr: str) -> None:
        self.codex_stdout_path.write_text(stdout, encoding="utf-8")
        self.codex_stderr_path.write_text(stderr, encoding="utf-8")


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gen_{timestamp}_{uuid.uuid4().hex[:8]}"


def _normalize_run_id(run_id: str | None) -> str:
    candidate = _default_run_id() if run_id is None else str(run_id).strip()
    if not _RUN_ID_RE.fullmatch(candidate):
        raise ValueError(
            "Generation run id must be 1-96 characters using letters, digits, '.', '_' or '-'."
        )
    return candidate


def create_generation_workspace(
    request: GenerationWorkspaceRequest,
    *,
    base_dir: Path | None = None,
    scaffold_path: Path | None = None,
    run_id: str | None = None,
) -> GenerationWorkspace:
    """Create a self-contained, non-git staging workspace for one generation run."""

    resolved_run_id = _normalize_run_id(run_id)
    resolved_base = (
        Path(base_dir or request.repo_root / "data" / "local" / "generation_runs")
        .expanduser()
        .resolve()
    )
    resolved_scaffold = (
        Path(scaffold_path or request.repo_root / "scaffold.py").expanduser().resolve()
    )
    if not resolved_scaffold.is_file():
        raise FileNotFoundError(f"Generation scaffold not found: {resolved_scaffold}")

    root = resolved_base / resolved_run_id
    if root.exists():
        raise FileExistsError(f"Generation workspace already exists: {root}")

    input_dir = root / "inputs"
    codex_dir = root / ".codex-run"
    build_dir = root / "build"
    render_dir = root / "renders"
    temp_dir = root / "tmp"
    cache_dir = root / ".cache"

    try:
        for directory in (
            input_dir,
            codex_dir,
            build_dir,
            render_dir,
            temp_dir,
            cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        model_path = root / "model.py"
        shutil.copy2(resolved_scaffold, model_path)

        input_image_path: Path | None = None
        if request.image_path is not None:
            suffix = request.image_path.suffix.lower()
            input_image_path = input_dir / f"reference{suffix}"
            shutil.copy2(request.image_path, input_image_path)

        request_payload = request.to_dict()
        request_payload["run_id"] = resolved_run_id
        request_payload["workspace"] = {
            "model_path": "model.py",
            "input_image_path": (
                input_image_path.relative_to(root).as_posix()
                if input_image_path is not None
                else None
            ),
        }
        request_path = root / "request.json"
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return GenerationWorkspace(
            run_id=resolved_run_id,
            repo_root=request.repo_root,
            root=root,
            model_path=model_path,
            request_path=request_path,
            task_path=root / "TASK.md",
            input_image_path=input_image_path,
            codex_dir=codex_dir,
            codex_schema_path=codex_dir / "final.schema.json",
            codex_last_message_path=codex_dir / "final.json",
            codex_stdout_path=codex_dir / "stdout.jsonl",
            codex_stderr_path=codex_dir / "stderr.log",
            codex_timeline_path=codex_dir / "timeline.jsonl",
            codex_trace_errors_path=codex_dir / "trace_errors.log",
            build_dir=build_dir,
            render_dir=render_dir,
            temp_dir=temp_dir,
            cache_dir=cache_dir,
        )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


__all__ = ["GenerationWorkspace", "create_generation_workspace"]
