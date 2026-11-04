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

import ast
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

ProviderCompletionStatus = Literal["complete", "blocked"]


class ProviderAuthorWorkspace(Protocol):
    """The staged-workspace surface the Provider tools and harness rely on.

    Generation and Reconstruction stage different evidence, but both expose one editable
    ``model.py``, one build directory, the repository root for shared SDK references, and the
    workspace root where the task and the run trace live.
    """

    @property
    def repo_root(self) -> Path: ...

    @property
    def root(self) -> Path: ...

    @property
    def model_path(self) -> Path: ...

    @property
    def build_dir(self) -> Path: ...

    def write_task(self, text: str) -> None: ...


# One read covers the longest shared SDK document (sdk/_docs/common/20_core_types.md, ~1170
# lines) so a document costs one tool turn instead of three; every Provider turn carries a fixed
# 8-14 s model latency, and the text lands in the run history either way.
_DEFAULT_READ_LINES = 1200
_MAX_READ_LINES = 1200
_MAX_EXAMPLE_CONTENT_CHARACTERS = 12_000
_MAX_DOC_MATCH_CHARACTERS = 500
MAX_MODEL_CHARACTERS = 262_144


@dataclass(slots=True, frozen=True)
class ProviderToolResult:
    """Normalized authoring callback result returned to the Author."""

    success: bool
    output: Any

    @classmethod
    def ok(cls, output: Any = "") -> ProviderToolResult:
        return cls(success=True, output=output)

    @classmethod
    def error(cls, output: Any) -> ProviderToolResult:
        return cls(success=False, output=output)


class ProviderCheckRunner(Protocol):
    def __call__(self, model_path: Path) -> ProviderToolResult: ...


class ProviderCompileRunner(Protocol):
    def __call__(
        self,
        model_path: Path,
        output_dir: Path,
        *,
        render: Literal[False],
    ) -> ProviderToolResult: ...


@dataclass(slots=True, frozen=True)
class ProviderToolExecution:
    payload: Mapping[str, Any]
    completion_status: ProviderCompletionStatus | None = None
    completion_summary: str | None = None
    # Trusted local tool attachments, separate from model-authored payload paths.
    image_paths: tuple[Path, ...] = ()
    # Optional full local result for audit, separate from the compact conversation reply.
    trace_payload: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class _PatchHunk:
    lines: tuple[str, ...]


def build_provider_tool_schemas() -> tuple[dict[str, Any], ...]:
    """Return the complete, intentionally narrow NVIDIA function-tool surface."""

    schemas: tuple[dict[str, Any], ...] = (
        {
            "type": "function",
            "name": "read_file",
            "description": (
                "Read the current staged model.py or an exact shared SDK documentation/example "
                "file. Repository source outside sdk/_docs and sdk/_examples is unavailable. The "
                f"default limit ({_DEFAULT_READ_LINES} lines) reads any shared document in one call; "
                "page with offset only when the result reports truncated=true."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_READ_LINES},
                },
                "required": ["path"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "search_docs",
            "description": (
                "Search SDK words and exact API identifiers, with document-title routing. "
                "Returns one best matching line per document with exact paths and line numbers. "
                "Use a concrete operation or API name, then read its relevant section."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "find_examples",
            "description": (
                "Lexically search only the shared sdk/_examples Markdown files and return the "
                "most relevant exact paths and contents."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "write_model",
            "description": "Replace staged model.py with complete Python source. Use for first construction or a whole-file rewrite; no old-file context is needed. Send source only, without Markdown fences or read_file line labels. Other files cannot be written. Changes invalidate previous check/compile success.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "content": {"type": "string", "minLength": 1, "maxLength": MAX_MODEL_CHARACTERS}
                },
                "required": ["content"],
            },
        },
        {
            "type": "function",
            "name": "apply_patch",
            "description": (
                "Apply a small exact patch to staged model.py, using *** Begin Patch, one "
                "*** Update File: model.py block, @@ hunks and *** End Patch. Every hunk needs "
                "current context lines (space prefix) or removed lines (- prefix); + lines alone "
                "cannot locate an edit. Read the relevant region if its current text is unknown. "
                "Use write_model for first construction or whole-file replacement. This tool "
                "only updates the staged model.py; other file operations are unavailable."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "check_model",
            "description": (
                "Run the shared USDCraft model-authored tests and semantic diagnostics against "
                "the latest staged model.py. Repair every actionable failure before finishing."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "compile_model",
            "description": (
                "Run the shared USDCraft authoring compile against the latest staged model.py. "
                "Rendering is always disabled. Inspect, repair, and retry executable/export errors."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            "strict": True,
        },
    )
    # A JSON round trip prevents one caller from mutating schemas used by another batch row.
    return tuple(json.loads(json.dumps(schema)) for schema in schemas)


def build_completion_tool_schema(name: str) -> dict[str, Any]:
    """Build the common completion contract under a product-supplied tool name."""
    return {
        "type": "function",
        "name": name,
        "description": (
            "Finish only after authoring is done. `complete` requires successful check_model "
            "and compile_model calls for the current model revision. `blocked` is reserved "
            "for a concrete request or SDK limitation, never a repairable model/check/compile "
            "failure."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["complete", "blocked"]},
                "summary": {"type": "string", "minLength": 1},
            },
            "required": ["status", "summary"],
        },
        "strict": True,
    }


class ProviderToolExecutor:
    """Execute the Provider's bounded tools and track authoring-result freshness.

    A product adapter may subclass this: ``finish_tool_name`` names the completion tool its task
    refers to, ``extra_readable_files`` maps additional logical paths (staged evidence the Author
    may read) to real files, and ``execute`` may dispatch product-specific tools before deferring
    to this shared surface.
    """

    finish_tool_name: str

    def __init__(
        self,
        workspace: ProviderAuthorWorkspace,
        *,
        finish_tool_name: str,
        check_runner: ProviderCheckRunner,
        compile_runner: ProviderCompileRunner,
        extra_readable_files: Mapping[str, Path] | None = None,
    ) -> None:
        self.finish_tool_name = finish_tool_name
        self.workspace = workspace
        self.check_runner = check_runner
        self.compile_runner = compile_runner
        self.extra_readable_files: dict[str, Path] = {
            Path(name).as_posix(): Path(path).resolve()
            for name, path in dict(extra_readable_files or {}).items()
        }
        self.revision = 0
        self.check_success_revision: int | None = None
        self.compile_success_revision: int | None = None
        self._validation_results: dict[str, dict[str, Any]] = {}

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        if name == "read_file":
            return ProviderToolExecution(self._read_file(arguments))
        if name == "search_docs":
            return ProviderToolExecution(self._search_docs(arguments))
        if name == "find_examples":
            return ProviderToolExecution(self._find_examples(arguments))
        if name == "write_model":
            return ProviderToolExecution(self._write_model(arguments))
        if name == "apply_patch":
            payload = self._apply_patch(arguments)
            if not payload.get("ok"):
                payload = {
                    **payload,
                    "changed": False,
                    "repair_hint": "No edit was committed; source and revision are unchanged. "
                    "Read the current model.py region for a small exact patch. Use "
                    "write_model(content) for first construction or complete replacement; "
                    "it needs no old-source context.",
                }
            return ProviderToolExecution(payload)
        if name == "check_model":
            return ProviderToolExecution(self._check_model(arguments))
        if name == "compile_model":
            return ProviderToolExecution(self._compile_model(arguments))
        if name == self.finish_tool_name:
            return self._finish(arguments)
        return ProviderToolExecution(
            self._error(f"Unknown or unavailable Provider tool: {name!r}.")
        )

    def validate_candidate(
        self, diagnostics: Sequence[tuple[str, Callable[[], ProviderToolExecution]]]
    ) -> ProviderToolExecution:
        """Check and compile once, then collect product diagnostics without semantic gating."""
        payload: dict[str, Any] = {}
        for stage, operation in (("check", self._check_model), ("compile", self._compile_model)):
            result = operation({})
            payload[stage] = result
            if not result.get("ok"):
                return ProviderToolExecution(
                    {"ok": False, "diagnostics_complete": False, **payload}
                )
        images: list[Path] = []
        for stage, diagnostic in diagnostics:
            execution = diagnostic()
            payload[stage] = execution.payload
            images.extend(execution.image_paths)
        return ProviderToolExecution(
            {
                "ok": True,
                "diagnostics_complete": all(
                    payload[stage].get("ok", False) for stage, _ in diagnostics
                ),
                **payload,
            },
            image_paths=tuple(images),
        )

    def track_validation(
        self, name: str, result: ProviderToolExecution, stages: Mapping[str, str]
    ) -> ProviderToolExecution:
        """Attach revision coverage to product-selected operations; never add a gate."""
        observations = (
            {stage: result.payload[stage] for stage in stages.values() if stage in result.payload}
            if name == "validate_model"
            else {stages[name]: result.payload}
            if name in stages
            else {}
        )
        for stage, payload in observations.items():
            self._validation_results[stage] = {
                "revision": payload.get("revision", self.revision),
                "ok": bool(payload.get("ok")),
            }
        if name in {*stages, "validate_model", "apply_patch", "write_model", self.finish_tool_name}:
            state = self.validation_state()
            payload = {**result.payload, "validation_state": state}
            if result.completion_status:
                coverage = "; ".join(
                    f"{stage}={value['status']} (rev {value['revision']})"
                    for stage, value in state["operations"].items()
                )
                summary = (
                    f"{result.completion_summary}\nAuthor revision {self.revision} coverage: "
                    f"{coverage}. Operation success is not visual acceptance or a physics test."
                )
                payload["summary"] = summary
                result = replace(result, completion_summary=summary)
            result = replace(result, payload=payload)
        return result

    def validation_state(self, stages: tuple[str, ...] = ("check", "compile")) -> dict[str, Any]:
        """Report latest Author-operation coverage without adding completion gates."""
        operations = {}
        for stage in stages:
            latest = self._validation_results.get(stage)
            if latest is None:
                status = "not_run"
            elif not latest["ok"]:
                status = "failed"
            else:
                status = "current" if latest["revision"] == self.revision else "stale"
            operations[stage] = {
                "revision": latest["revision"] if latest else None,
                "status": status,
            }
        return {
            "basis": "author_model_revision",
            "model_revision": self.revision,
            "operations": operations,
        }

    def _read_file(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        unexpected = sorted(set(arguments) - {"path", "offset", "limit"})
        if unexpected:
            return self._error(f"Unexpected read_file arguments: {unexpected}")
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return self._error("read_file path must be a non-empty string.")
        try:
            path = self._resolve_read_path(raw_path)
        except (OSError, ValueError) as exc:
            return self._error(str(exc))
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit", _DEFAULT_READ_LINES)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            return self._error("read_file offset must be an integer >= 1.")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= _MAX_READ_LINES
        ):
            return self._error(f"read_file limit must be an integer from 1 to {_MAX_READ_LINES}.")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            display_path = self._display_path(path)
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error(f"Unable to read {raw_path!r}: {type(exc).__name__}: {exc}")
        if lines and offset > len(lines):
            return self._error("read_file offset exceeds file length.")
        start = offset - 1
        end = min(len(lines), start + limit)
        # Staged evidence files (for example the AXIAL text with its column-aligned grids) are
        # returned verbatim so their layout survives; source and documentation keep line numbers.
        raw = path.resolve() in set(self.extra_readable_files.values())
        rendered = (
            "\n".join(lines[start:end])
            if raw
            else "\n".join(f"L{index}: {lines[index - 1]}" for index in range(start + 1, end + 1))
        )
        return self._ok(
            {
                "path": display_path,
                "content": rendered,
                "line_numbers": not raw,
                "start_line": offset,
                "end_line": end,
                "total_lines": len(lines),
                "truncated": end < len(lines),
                "next_offset": end + 1 if end < len(lines) else None,
            }
        )

    def _resolve_read_path(self, raw_path: str) -> Path:
        requested = Path(raw_path).expanduser()
        if not requested.is_absolute():
            logical = requested.as_posix()
            if logical == "model.py":
                requested = self.workspace.model_path
            elif logical in self.extra_readable_files:
                requested = self.extra_readable_files[logical]
            else:
                requested = self.workspace.repo_root / requested
        resolved = requested.resolve()
        allowed_roots = (
            (self.workspace.repo_root / "sdk" / "_docs").resolve(),
            (self.workspace.repo_root / "sdk" / "_examples").resolve(),
        )
        allowed_files = {self.workspace.model_path.resolve(), *self.extra_readable_files.values()}
        if resolved not in allowed_files and not any(
            _is_relative_to(resolved, root) for root in allowed_roots
        ):
            readable = ", ".join(("staged model.py", *sorted(self.extra_readable_files)))
            raise ValueError(
                f"read_file may access only {readable} or shared sdk/_docs and sdk/_examples."
            )
        if not resolved.is_file():
            raise ValueError(f"read_file target is not a file: {raw_path}")
        return resolved

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.workspace.model_path.resolve():
            return "model.py"
        for logical, extra in self.extra_readable_files.items():
            if resolved == extra:
                return logical
        logical_roots = (
            ((self.workspace.repo_root / "sdk" / "_docs").resolve(), Path("sdk/_docs")),
            ((self.workspace.repo_root / "sdk" / "_examples").resolve(), Path("sdk/_examples")),
        )
        for resolved_root, logical_root in logical_roots:
            try:
                relative = resolved.relative_to(resolved_root)
            except ValueError:
                continue
            return (logical_root / relative).as_posix()
        raise ValueError("path is outside the staged model and shared SDK roots")

    def _search_docs(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        unexpected = sorted(set(arguments) - {"query", "limit"})
        if unexpected:
            return self._error(f"Unexpected search_docs arguments: {unexpected}")
        query = arguments.get("query")
        limit = arguments.get("limit", 5)
        if not isinstance(query, str) or not query.strip():
            return self._error("search_docs query must be a non-empty string.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10:
            return self._error("search_docs limit must be an integer from 1 to 10.")

        # Import lazily so merely importing the Provider transport does not initialize the SDK.
        from sdk._profiles import get_sdk_profile

        docs_root = (self.workspace.repo_root / "sdk" / "_docs").resolve()
        query_normalized = query.casefold().strip()
        tokens = set(re.findall(r"[a-z0-9_]+", query_normalized))
        phrase = re.compile(r"(?<!\w)" + re.escape(query_normalized) + r"(?!\w)")
        matches: list[tuple[int, str, int, str]] = []
        for relative_path in get_sdk_profile("sdk").docs_full:
            path = (self.workspace.repo_root / relative_path).resolve()
            if not _is_relative_to(path, docs_root) or not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            try:
                display_path = (Path("sdk/_docs") / path.relative_to(docs_root)).as_posix()
            except ValueError:
                continue
            title = " ".join(line for line in lines if line.startswith("#"))
            route_words = set(
                re.findall(r"[a-z0-9]+", path.stem.casefold() + " " + title.casefold())
            )
            route_score = 3 * len(tokens & route_words)
            document_matches = []
            for line_number, line in enumerate(lines, start=1):
                haystack = line.casefold()
                words = set(re.findall(r"[a-z0-9_]+", haystack))
                matched = tokens & words
                score = 8 * len(matched) + (20 if phrase.search(haystack) else 0)
                # API identifiers route more precisely than incidental prose words.
                score += 12 * sum("_" in token for token in matched)
                if score:
                    document_matches.append((score + route_score, display_path, line_number, line))
            if document_matches:
                matches.append(min(document_matches, key=lambda item: (-item[0], item[2])))
        matches.sort(key=lambda item: (-item[0], item[1], item[2]))
        return self._ok(
            {
                "query": query.strip(),
                "matches": [
                    {
                        "path": path,
                        "line": line_number,
                        "content": content[:_MAX_DOC_MATCH_CHARACTERS],
                        "truncated": len(content) > _MAX_DOC_MATCH_CHARACTERS,
                    }
                    for _score, path, line_number, content in matches[:limit]
                ],
            }
        )

    def _find_examples(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        unexpected = sorted(set(arguments) - {"query", "limit"})
        if unexpected:
            return self._error(f"Unexpected find_examples arguments: {unexpected}")
        query = arguments.get("query")
        limit = arguments.get("limit", 3)
        if not isinstance(query, str) or not query.strip():
            return self._error("find_examples query must be a non-empty string.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5:
            return self._error("find_examples limit must be an integer from 1 to 5.")
        examples_root = (self.workspace.repo_root / "sdk" / "_examples").resolve()
        query_normalized = query.casefold().strip()
        tokens = tuple(dict.fromkeys(re.findall(r"[a-z0-9_]+", query_normalized)))
        scored: list[tuple[int, str, str]] = []
        for path in examples_root.rglob("*.md"):
            resolved = path.resolve()
            if not _is_relative_to(resolved, examples_root) or not resolved.is_file():
                continue
            try:
                content = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            try:
                relative = (Path("sdk/_examples") / resolved.relative_to(examples_root)).as_posix()
            except ValueError:
                continue
            haystack = f"{relative}\n{content}".casefold()
            score = 20 if query_normalized in haystack else 0
            score += sum(min(haystack.count(token), 5) for token in tokens)
            if score:
                scored.append((score, relative, content))
        scored.sort(key=lambda item: (-item[0], item[1]))
        matches = []
        for score, example_path, content in scored[:limit]:
            truncated = len(content) > _MAX_EXAMPLE_CONTENT_CHARACTERS
            matches.append(
                {
                    "path": example_path,
                    "score": score,
                    "content": content[:_MAX_EXAMPLE_CONTENT_CHARACTERS],
                    "truncated": truncated,
                    "total_characters": len(content),
                }
            )
        return self._ok({"query": query.strip(), "matches": matches})

    def _write_model(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        content = arguments.get("content")
        if set(arguments) != {"content"} or not isinstance(content, str) or not content.strip():
            return self._error("write_model requires exactly one nonempty string: content.")
        if len(content) > MAX_MODEL_CHARACTERS:
            return self._error(f"write_model content exceeds {MAX_MODEL_CHARACTERS} characters.")
        try:
            syntax_error = None
            try:
                ast.parse(content, filename=str(self.workspace.model_path))
            except (SyntaxError, ValueError) as exc:
                syntax_error = str(exc)
            changed = content != self.workspace.model_path.read_text(encoding="utf-8")
            if changed:
                _atomic_write_text(self.workspace.model_path, content)
                self.revision += 1
                self.check_success_revision = None
                self.compile_success_revision = None
            return self._ok(
                {
                    "changed": changed,
                    "revision": self.revision,
                    "python_syntax": {"status": "error", "message": syntax_error}
                    if syntax_error
                    else {"status": "success"},
                }
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error(f"Unable to write model.py: {type(exc).__name__}: {exc}")

    def _apply_patch(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(arguments) != {"patch"}:
            return self._error("apply_patch requires exactly one string argument: patch.")
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return self._error("apply_patch patch must be a non-empty string.")
        try:
            hunks = _parse_patch(patch, model_path=self.workspace.model_path)
            previous = self.workspace.model_path.read_text(encoding="utf-8")
            updated = _apply_hunks(previous, hunks)
            syntax_error: str | None = None
            try:
                ast.parse(updated, filename=str(self.workspace.model_path))
            except SyntaxError as exc:
                syntax_error = f"{exc.msg} (line {exc.lineno})"
            changed = updated != previous
            if changed:
                _atomic_write_text(self.workspace.model_path, updated)
                self.revision += 1
                self.check_success_revision = None
                self.compile_success_revision = None
            return self._ok(
                {
                    "changed": changed,
                    "revision": self.revision,
                    "hunks_applied": len(hunks),
                    "python_syntax": (
                        {"status": "error", "message": syntax_error}
                        if syntax_error is not None
                        else {"status": "success"}
                    ),
                }
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error(f"Invalid patch: {exc}")

    def _check_model(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if arguments:
            return self._error("check_model accepts no arguments.")
        try:
            result = self.check_runner(self.workspace.model_path)
        except Exception as exc:
            result = ProviderToolResult.error(f"{type(exc).__name__}: {exc}")
        if not isinstance(result, ProviderToolResult):
            return self._error("check_runner returned an invalid ProviderToolResult.")
        if result.success:
            self.check_success_revision = self.revision
        else:
            self.check_success_revision = None
        return {
            "ok": result.success,
            "revision": self.revision,
            "output": _json_safe(result.output),
        }

    def _compile_model(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if arguments:
            return self._error("compile_model accepts no arguments.")
        try:
            result = self.compile_runner(
                self.workspace.model_path,
                self.workspace.build_dir,
                render=False,
            )
        except Exception as exc:
            result = ProviderToolResult.error(f"{type(exc).__name__}: {exc}")
        if not isinstance(result, ProviderToolResult):
            return self._error("compile_runner returned an invalid ProviderToolResult.")
        if result.success:
            self.compile_success_revision = self.revision
        else:
            self.compile_success_revision = None
        return {
            "ok": result.success,
            "revision": self.revision,
            "render": False,
            "output": _json_safe(result.output),
        }

    def _finish(self, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        finish = self.finish_tool_name
        if set(arguments) != {"status", "summary"}:
            return ProviderToolExecution(
                self._error(f"{finish} requires exactly status and summary.")
            )
        status = arguments.get("status")
        summary = arguments.get("summary")
        if status not in {"complete", "blocked"}:
            return ProviderToolExecution(
                self._error(f"{finish} status must be 'complete' or 'blocked'.")
            )
        if not isinstance(summary, str) or not summary.strip():
            return ProviderToolExecution(
                self._error(f"{finish} summary must be a non-empty string.")
            )
        if status == "complete":
            missing: list[str] = []
            if self.revision == 0:
                missing.append("at least one successful model.py edit")
            if self.check_success_revision != self.revision:
                missing.append("successful check_model for the current revision")
            if self.compile_success_revision != self.revision:
                missing.append("successful compile_model for the current revision")
            if missing:
                return ProviderToolExecution(
                    self._error("Cannot complete: missing " + "; ".join(missing) + ".")
                )
        normalized_status: ProviderCompletionStatus = status
        normalized_summary = summary.strip()
        return ProviderToolExecution(
            payload={
                "ok": True,
                "status": normalized_status,
                "summary": normalized_summary,
                "revision": self.revision,
            },
            completion_status=normalized_status,
            completion_summary=normalized_summary,
        )

    def _ok(self, output: Any) -> Mapping[str, Any]:
        return {"ok": True, "revision": self.revision, "output": _json_safe(output)}

    def _error(self, message: str) -> Mapping[str, Any]:
        return {"ok": False, "revision": self.revision, "error": message}


def _parse_patch(patch_text: str, *, model_path: Path) -> tuple[_PatchHunk, ...]:
    lines = patch_text.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("missing '*** Begin Patch' header")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("missing '*** End Patch' footer")
    if any(
        line.startswith(("*** Add File:", "*** Delete File:", "*** Move to:")) for line in lines
    ):
        raise ValueError("adding, deleting, or moving files is not supported")
    update_lines = [line for line in lines if line.startswith("*** Update File:")]
    if len(update_lines) != 1:
        raise ValueError("patch must contain exactly one Update File block")
    if lines[1] != update_lines[0]:
        raise ValueError("Update File block must immediately follow the patch header")
    target = update_lines[0].partition(":")[2].strip()
    allowed_targets = {"model.py", str(model_path), model_path.as_posix()}
    if target not in allowed_targets:
        raise ValueError("apply_patch may update only the staged model.py")

    index = lines.index(update_lines[0]) + 1
    hunks: list[_PatchHunk] = []
    while index < len(lines) - 1:
        current = lines[index]
        if not current.strip():
            index += 1
            continue
        if not current.startswith("@@"):
            raise ValueError("expected '@@' to start each patch hunk")
        index += 1
        hunk_lines: list[str] = []
        while index < len(lines) - 1:
            line = lines[index]
            if line.startswith("@@") or line.startswith("*** "):
                break
            if not line or line[0] not in {" ", "+", "-"}:
                raise ValueError("hunk lines must start with ' ', '+', or '-'")
            hunk_lines.append(line)
            index += 1
        if not hunk_lines:
            raise ValueError("patch hunk has no lines")
        hunks.append(_PatchHunk(lines=tuple(hunk_lines)))
    if not hunks:
        raise ValueError("patch contains no hunks")
    return tuple(hunks)


def _apply_hunks(source: str, hunks: tuple[_PatchHunk, ...]) -> str:
    lines = source.splitlines()
    trailing_newline = source.endswith("\n")
    search_start = 0
    for index, hunk in enumerate(hunks, start=1):
        old = [line[1:] for line in hunk.lines if line[0] in {" ", "-"}]
        new = [line[1:] for line in hunk.lines if line[0] in {" ", "+"}]
        if not old:
            raise ValueError(f"hunk {index} has no exact context or removed lines")
        position = _find_subsequence(lines, old, start=search_start)
        if position is None:
            position = _find_subsequence(lines, old, start=0)
        if position is None:
            raise ValueError(f"hunk {index} does not match current model.py")
        lines = lines[:position] + new + lines[position + len(old) :]
        search_start = position + len(new)
    rewritten = "\n".join(lines)
    if trailing_newline and rewritten:
        rewritten += "\n"
    return rewritten


def _find_subsequence(haystack: list[str], needle: list[str], *, start: int) -> int | None:
    for index in range(max(0, start), len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
        temporary.replace(path)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "MAX_MODEL_CHARACTERS",
    "ProviderAuthorWorkspace",
    "ProviderCheckRunner",
    "ProviderCompileRunner",
    "ProviderCompletionStatus",
    "ProviderToolExecution",
    "ProviderToolExecutor",
    "ProviderToolResult",
    "build_provider_tool_schemas",
]
