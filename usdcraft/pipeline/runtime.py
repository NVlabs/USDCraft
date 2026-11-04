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
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from usdcraft.pipeline.contracts import CodexRuntimeSelection

_CODEX_STATE_DATABASE = "state_5.sqlite"
_MAX_THREAD_DEPTH = 32


class CodexRuntimeUnavailableError(RuntimeError):
    pass


def _normalized_runtime_value(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise CodexRuntimeUnavailableError(f"Codex {label} is unreported.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise CodexRuntimeUnavailableError(f"Codex {label} contains control characters.")
    return normalized


def _explicit_runtime(
    model: str | None, reasoning_effort: str | None
) -> CodexRuntimeSelection | None:
    if (model is None) != (reasoning_effort is None):
        raise ValueError("--model and --reasoning must be provided together.")
    if model is None:
        return None
    raw_model = str(model or "").strip()
    raw_effort = str(reasoning_effort or "").strip()
    return CodexRuntimeSelection(
        model_id=_normalized_runtime_value(raw_model, label="model"),
        reasoning_effort=_normalized_runtime_value(raw_effort, label="reasoning effort"),
        runtime_source="explicit_generation_request",
        selection_source="explicit_generation_request",
    )


def _root_thread_id(connection: sqlite3.Connection, thread_id: str) -> str:
    current = thread_id
    seen: set[str] = set()
    for _ in range(_MAX_THREAD_DEPTH):
        if current in seen:
            raise CodexRuntimeUnavailableError("Codex thread ancestry contains a cycle.")
        seen.add(current)
        row = connection.execute(
            "SELECT parent_thread_id FROM thread_spawn_edges WHERE child_thread_id = ?",
            (current,),
        ).fetchone()
        if row is None:
            return current
        parent = str(row[0] or "").strip()
        if not parent:
            raise CodexRuntimeUnavailableError("Codex thread ancestry contains an empty parent.")
        current = parent
    raise CodexRuntimeUnavailableError(
        f"Codex thread ancestry exceeds the {_MAX_THREAD_DEPTH}-thread safety limit."
    )


def current_main_codex_runtime(
    *,
    environ: Mapping[str, str] | None = None,
    codex_home: Path | None = None,
) -> CodexRuntimeSelection:
    """Read the root user thread's model and effort from Codex's fixed state database.

    This deliberately performs no glob or directory scan. Missing or incompatible Codex state is
    a hard failure because silently substituting a configured default would create false runtime
    provenance.
    """

    env = os.environ if environ is None else environ
    thread_id = str(env.get("CODEX_THREAD_ID") or "").strip()
    if not thread_id or any(character in thread_id for character in ("/", "\\", "\x00")):
        raise CodexRuntimeUnavailableError(
            "Current Codex thread is unavailable; pass both --model and --reasoning explicitly."
        )

    home = Path(codex_home or env.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    database = (home / _CODEX_STATE_DATABASE).resolve()
    if not database.is_file():
        raise CodexRuntimeUnavailableError(
            "Codex runtime state is unavailable; pass both --model and --reasoning explicitly."
        )

    try:
        connection = sqlite3.connect(
            database.as_uri() + "?mode=ro",
            uri=True,
            timeout=0.25,
        )
        try:
            connection.execute("PRAGMA query_only = ON")
            root_thread_id = _root_thread_id(connection, thread_id)
            row = connection.execute(
                "SELECT model, reasoning_effort FROM threads WHERE id = ?",
                (root_thread_id,),
            ).fetchone()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise CodexRuntimeUnavailableError(
            "Codex runtime state could not be read; pass both --model and --reasoning explicitly."
        ) from exc

    if row is None:
        raise CodexRuntimeUnavailableError(
            "Root Codex thread runtime is unreported; pass both --model and --reasoning explicitly."
        )
    return CodexRuntimeSelection(
        model_id=_normalized_runtime_value(row[0], label="model"),
        reasoning_effort=_normalized_runtime_value(row[1], label="reasoning effort"),
        runtime_source="codex_root_thread_state",
        selection_source="inherited_main_codex_runtime",
    )


def resolve_codex_runtime(
    *,
    model: str | None,
    reasoning_effort: str | None,
    environ: Mapping[str, str] | None = None,
    codex_home: Path | None = None,
) -> CodexRuntimeSelection:
    explicit = _explicit_runtime(model, reasoning_effort)
    if explicit is not None:
        return explicit
    return current_main_codex_runtime(
        environ=environ,
        codex_home=codex_home,
    )


__all__ = [
    "CodexRuntimeUnavailableError",
    "current_main_codex_runtime",
    "resolve_codex_runtime",
]


def _codex_executable(environ: Mapping[str, str] | None) -> str:
    env = os.environ if environ is None else environ
    configured = str(env.get("USDCRAFT_CODEX_BIN") or "codex").strip() or "codex"
    resolved = shutil.which(configured, path=env.get("PATH"))
    if resolved is not None:
        return str(Path(resolved).resolve())
    if "/" in configured or "\\" in configured:
        return str(Path(configured).expanduser().resolve())
    return configured
