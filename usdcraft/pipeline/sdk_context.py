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

"""Canonical SDK foundations embedded in Generation and Reconstruction tasks on both arms."""

from __future__ import annotations

from pathlib import Path

PRELOADED_SDK_DOC_PATHS: tuple[str, ...] = ("sdk/_docs/common/00_quickstart.md",)
PRELOADED_MARKER = "(preloaded above)"


_PACKAGE_REPO_ROOT = Path(__file__).resolve().parents[2]


def preloaded_sdk_doc_paths(repo_root: Path) -> tuple[Path, ...]:
    """Absolute paths of the preloaded documents.

    They are read from ``repo_root``; a repository root that carries no SDK documentation (test
    fixtures, detached workspaces) falls back to the documentation shipped with this package so
    the task text is the same canonical SDK either way.
    """

    resolved: list[Path] = []
    for relative in PRELOADED_SDK_DOC_PATHS:
        path = Path(repo_root) / relative
        if not path.is_file():
            path = _PACKAGE_REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Preloaded SDK document is missing: {path}")
        resolved.append(path)
    return tuple(resolved)


def render_preloaded_sdk_docs(
    repo_root: Path,
    *,
    task_noun: str,
    reread_clause: str = "call `read_file` on one of them only to re-check an exact\nline",
) -> str:
    """The PRELOADED SDK REFERENCES task section with the canonical quickstart in full.

    ``reread_clause`` names how the arm re-opens one of the preloaded files (the Provider tool or a
    plain shell read); the section is otherwise identical for every Author.
    """

    blocks: list[str] = []
    for relative, path in zip(PRELOADED_SDK_DOC_PATHS, preloaded_sdk_doc_paths(repo_root)):
        text = path.read_text(encoding="utf-8").rstrip("\n")
        blocks.append(f"===== BEGIN {relative} =====\n{text}\n===== END {relative} =====")
    body = "\n\n".join(blocks)
    return f"""PRELOADED SDK REFERENCES
Read the shared SDK quickstart below to begin this {task_noun} task. It includes common API
signatures and tests; open further references only for the specific features you need.
The inventory marks it {PRELOADED_MARKER}; {reread_clause}.

{body}"""


def render_sdk_document_inventory(repo_root: Path, *, mark_preloaded: bool) -> str:
    """The canonical SDK documentation inventory, optionally marking the preloaded entries."""

    from sdk._profiles import get_sdk_profile

    profile = get_sdk_profile("sdk")
    preloaded = (
        {Path(relative) for relative in PRELOADED_SDK_DOC_PATHS} if mark_preloaded else set()
    )
    lines = [f"Repository root: `{Path(repo_root)}`. Paths below are relative to this root."]
    for path in profile.docs_full:
        suffix = f" {PRELOADED_MARKER}" if Path(path) in preloaded else ""
        lines.append(f"- `{path.as_posix()}`{suffix}")
    return "\n".join(lines)


__all__ = [
    "PRELOADED_MARKER",
    "PRELOADED_SDK_DOC_PATHS",
    "preloaded_sdk_doc_paths",
    "render_preloaded_sdk_docs",
    "render_sdk_document_inventory",
]
