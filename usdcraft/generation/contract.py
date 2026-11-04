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

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from storage.artifact_contract import (
    COMPILE_REPORT_SCHEMA_VERSION,
    GENERATION_MANIFEST_SCHEMA_VERSION,
    GENERATION_VALIDATION_PROFILE,
    physics_output_contract,
)
from usdcraft.pipeline.contracts import (
    CodexReasoningEffort,
    CodexRuntimeSelection,
    _normalized_nonempty,
)
from usdcraft.values import normalize_reasoning_level


class GenerationInputKind(StrEnum):
    TEXT = "text"
    TEXT_IMAGE = "text_image"


class GenerationImageRole(StrEnum):
    INSPIRATION = "inspiration"


def generation_object_name(*, prompt: str, label: str | None) -> str:
    """Return the single-line, bounded name used for canonical display."""

    value = " ".join((label or prompt).split())
    if len(value) > 120:
        return value[:117].rstrip() + "..."
    return value


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """One Codex-only asset generation request.

    Resolved text is always required internally; image-only entry points synthesize it before
    constructing this request. A request may carry one image, whose role is fixed to inspiration
    rather than reconstruction evidence. The Codex arm has no time or turn limit, so the request
    carries none.
    """

    prompt: str
    repo_root: Path
    image_path: Path | None = None
    model: str | None = None
    reasoning_effort: CodexReasoningEffort | str = CodexReasoningEffort.HIGH
    runtime_source: str = "unreported"
    selection_source: str = "unreported"

    def __post_init__(self) -> None:
        prompt = str(self.prompt).strip()
        if not prompt:
            raise ValueError("Generation text prompt is required.")

        repo_root = Path(self.repo_root).expanduser().resolve()
        if not repo_root.is_dir():
            raise ValueError(f"USDCraft repository root is not a directory: {repo_root}")

        image_path: Path | None = None
        if self.image_path is not None:
            image_path = Path(self.image_path).expanduser().resolve()
            if not image_path.is_file():
                raise ValueError(f"Generation image is not a file: {image_path}")

        model = str(self.model).strip() if self.model is not None else None
        if model == "":
            model = None

        raw_reasoning = (
            self.reasoning_effort.value
            if isinstance(self.reasoning_effort, CodexReasoningEffort)
            else str(self.reasoning_effort)
        )
        reasoning_effort = normalize_reasoning_level(
            _normalized_nonempty(raw_reasoning, label="Codex reasoning effort")
        )
        runtime_source = _normalized_nonempty(
            self.runtime_source,
            label="Codex runtime source",
        )
        selection_source = _normalized_nonempty(
            self.selection_source,
            label="Codex selection source",
        )

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)
        object.__setattr__(self, "runtime_source", runtime_source)
        object.__setattr__(self, "selection_source", selection_source)

    @property
    def input_kind(self) -> GenerationInputKind:
        if self.image_path is None:
            return GenerationInputKind.TEXT
        return GenerationInputKind.TEXT_IMAGE

    @property
    def image_role(self) -> GenerationImageRole | None:
        if self.image_path is None:
            return None
        return GenerationImageRole.INSPIRATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline": "generation",
            "prompt": self.prompt,
            "input": {
                "kind": self.input_kind.value,
                "image_role": self.image_role.value if self.image_role is not None else None,
                "image_path": str(self.image_path) if self.image_path is not None else None,
            },
            "codex": {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "runtime_source": self.runtime_source,
                "selection_source": self.selection_source,
            },
        }

    @property
    def runtime(self) -> CodexRuntimeSelection:
        if self.model is None:
            raise ValueError("Generation execution requires a resolved Codex model.")
        return CodexRuntimeSelection(
            model_id=self.model,
            reasoning_effort=self.reasoning_effort,
            runtime_source=self.runtime_source,
            selection_source=self.selection_source,
        )


@dataclass(slots=True, frozen=True)
class GenerationPreflight:
    request: GenerationRequest
    label: str | None
    keep_workspace: bool
    scaffold_path: Path
    scaffold_sha256: str
    image_sha256: str | None
    codex_executable: str
    python_executable: str
    fingerprint: str

    @property
    def workspace_base(self) -> Path:
        return self.request.repo_root / "data" / "local" / "generation_runs"

    @property
    def records_root(self) -> Path:
        return self.request.repo_root / "data" / "records"

    @property
    def materialization_root(self) -> Path:
        return self.request.repo_root / "data" / "cache" / "record_materialization"

    @property
    def object_name(self) -> str:
        """Return the concise canonical display name without adding mutable state."""

        return generation_object_name(prompt=self.request.prompt, label=self.label)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "pipeline": "generation",
            "repo_root": str(self.request.repo_root),
            "prompt": self.request.prompt,
            "label": self.label,
            "input": {
                "kind": self.request.input_kind.value,
                "image_role": (
                    self.request.image_role.value if self.request.image_role is not None else None
                ),
                "image_path": (
                    str(self.request.image_path) if self.request.image_path is not None else None
                ),
                "image_sha256": self.image_sha256,
            },
            "codex": {
                "model_id": self.request.runtime.model_id,
                "reasoning_effort": self.request.runtime.reasoning_effort,
                "runtime_source": self.request.runtime.runtime_source,
                "selection_source": self.request.runtime.selection_source,
                "executable": self.codex_executable,
            },
            "execution": {
                "keep_workspace": self.keep_workspace,
                "python_executable": self.python_executable,
                "scaffold_path": str(self.scaffold_path),
                "scaffold_sha256": self.scaffold_sha256,
                "workspace_base": str(self.workspace_base),
                "records_root": str(self.records_root),
                "materialization_root": str(self.materialization_root),
            },
            "artifact_contract": {
                "generation_manifest_schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
                "compile_report_schema_version": COMPILE_REPORT_SCHEMA_VERSION,
                "validation_profile": GENERATION_VALIDATION_PROFILE,
            },
            "outputs": physics_output_contract(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "awaiting_approval",
            "generation_preflight": True,
            "record_created": False,
            "workspace_created": False,
            "codex_started": False,
            "confirmation_required": True,
            "fingerprint": self.fingerprint,
            "case": self.fingerprint_payload(),
        }


__all__ = [
    "GenerationPreflight",
    "GenerationImageRole",
    "GenerationInputKind",
    "GenerationRequest",
    "generation_object_name",
]
