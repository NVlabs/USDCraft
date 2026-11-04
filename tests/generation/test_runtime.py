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

import sqlite3
from pathlib import Path

import pytest

from usdcraft.pipeline.runtime import (
    CodexRuntimeUnavailableError,
    current_main_codex_runtime,
    resolve_codex_runtime,
)


def _state_database(
    codex_home: Path,
    *,
    threads: list[tuple[str, str | None, str | None]],
    edges: list[tuple[str, str]] = [],
) -> Path:
    codex_home.mkdir(parents=True, exist_ok=True)
    database = codex_home / "state_5.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            model TEXT,
            reasoning_effort TEXT
        );
        CREATE TABLE thread_spawn_edges (
            parent_thread_id TEXT NOT NULL,
            child_thread_id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'open'
        );
        """
    )
    connection.executemany(
        "INSERT INTO threads (id, model, reasoning_effort) VALUES (?, ?, ?)",
        threads,
    )
    connection.executemany(
        "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id) VALUES (?, ?)",
        edges,
    )
    connection.commit()
    connection.close()
    return database


def test_runtime_climbs_to_root_main_thread(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    _state_database(
        codex_home,
        threads=[
            ("root", "test-model-b", "ultra"),
            ("child", "gpt-mini", "low"),
            ("grandchild", "gpt-other", "max"),
        ],
        edges=[("root", "child"), ("child", "grandchild")],
    )

    runtime = current_main_codex_runtime(
        environ={"CODEX_THREAD_ID": "grandchild"},
        codex_home=codex_home,
    )

    assert runtime.to_dict() == {
        "model_id": "test-model-b",
        "reasoning_effort": "ultra",
        "runtime_source": "codex_root_thread_state",
        "selection_source": "inherited_main_codex_runtime",
    }


def test_explicit_runtime_pair_is_verbatim_and_needs_no_codex_state(tmp_path: Path) -> None:
    runtime = resolve_codex_runtime(
        model="future/model",
        reasoning_effort="Future-Tier-Max",
        environ={},
        codex_home=tmp_path / "missing",
    )

    assert runtime.model_id == "future/model"
    assert runtime.reasoning_effort == "Future-Tier-Max"
    assert runtime.runtime_source == "explicit_generation_request"


@pytest.mark.parametrize(
    ("model", "effort"),
    [("gpt-test", None), (None, "high")],
)
def test_explicit_runtime_must_be_provided_as_a_pair(
    model: str | None,
    effort: str | None,
) -> None:
    with pytest.raises(ValueError, match="provided together"):
        resolve_codex_runtime(model=model, reasoning_effort=effort, environ={})


def test_inherited_runtime_fails_closed_when_state_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(CodexRuntimeUnavailableError, match="runtime state is unavailable"):
        resolve_codex_runtime(
            model=None,
            reasoning_effort=None,
            environ={"CODEX_THREAD_ID": "root"},
            codex_home=tmp_path / "missing",
        )


def test_inherited_runtime_fails_closed_for_unreported_values(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    _state_database(codex_home, threads=[("root", "gpt-test", None)])

    with pytest.raises(CodexRuntimeUnavailableError, match="reasoning effort is unreported"):
        current_main_codex_runtime(
            environ={"CODEX_THREAD_ID": "root"},
            codex_home=codex_home,
        )


def test_thread_ancestry_cycle_fails_closed(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    _state_database(
        codex_home,
        threads=[("a", "gpt-a", "high"), ("b", "gpt-b", "low")],
        edges=[("b", "a"), ("a", "b")],
    )

    with pytest.raises(CodexRuntimeUnavailableError, match="contains a cycle"):
        current_main_codex_runtime(
            environ={"CODEX_THREAD_ID": "a"},
            codex_home=codex_home,
        )
