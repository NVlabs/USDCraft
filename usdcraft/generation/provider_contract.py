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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from usdcraft.generation.contract import GenerationImageRole, GenerationInputKind
from usdcraft.pipeline.contracts import ProviderRuntimeSelection


def _nonempty(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


def _prompt(value: object) -> str:
    # Match GenerationRequest: multiline and tabbed user text is valid Generation input.
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("Generation text prompt is required.")
    return normalized


@dataclass(slots=True, frozen=True)
class ProviderGenerationRequest:
    """One direct NVIDIA Provider Generation request.

    The entry point resolves image-only text before constructing this object. Unlike the Codex
    arm, this request is execution-ready and has no approval fingerprint.
    """

    prompt: str
    repo_root: Path
    runtime: ProviderRuntimeSelection
    image_path: Path | None = None
    # Optional wall-clock deadline for one NVIDIA Responses request, never for a whole multi-turn
    # item; None (the default) means a turn runs until the provider answers.
    timeout_seconds: float | None = None
    # The Provider endpoint that serves the run; it names the request's provider.
    endpoint: Literal["nvidia", "openai"] = "openai"

    def __post_init__(self) -> None:
        if self.endpoint not in {"nvidia", "openai"}:
            raise ValueError("Provider endpoint must be nvidia or openai.")
        prompt = _prompt(self.prompt)
        repo_root = Path(self.repo_root).expanduser().resolve()
        if not repo_root.is_dir():
            raise ValueError(f"USDCraft repository root is not a directory: {repo_root}")

        image_path: Path | None = None
        if self.image_path is not None:
            image_path = Path(self.image_path).expanduser().resolve()
            if not image_path.is_file():
                raise ValueError(f"Generation image is not a file: {image_path}")

        timeout_seconds: float | None = None
        if self.timeout_seconds is not None:
            try:
                timeout_seconds = float(self.timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("Provider timeout must be a positive number.") from exc
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("Provider timeout must be a positive finite number.")

        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

    @property
    def input_kind(self) -> GenerationInputKind:
        return (
            GenerationInputKind.TEXT if self.image_path is None else GenerationInputKind.TEXT_IMAGE
        )

    @property
    def image_role(self) -> GenerationImageRole | None:
        return None if self.image_path is None else GenerationImageRole.INSPIRATION

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
            "provider": {
                "name": self.endpoint,
                **self.runtime.to_dict(),
                "timeout_seconds": self.timeout_seconds,
            },
        }


__all__ = ["ProviderGenerationRequest"]
