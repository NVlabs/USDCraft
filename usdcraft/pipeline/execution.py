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
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Literal, Protocol

from usdcraft.compile import check_model, compile_model
from usdcraft.compile.command import CompileArtifacts
from usdcraft.compile.compiler import compile_timeout_seconds
from usdcraft.compile.feedback import compile_signal_bundle_from_exception
from usdcraft.pipeline.tools import ProviderToolResult
from usdcraft.stagetime import StageTimer


def _provider_selection_source(*, model: str | None, reasoning: str | None) -> str:
    """Describe mixed Provider overrides without claiming both values came from the CLI."""

    if model is not None and reasoning is not None:
        return "explicit_provider_arguments"
    if model is not None:
        return "explicit_model_only"
    if reasoning is not None:
        return "explicit_reasoning_only"
    return "repository_env_or_provider_defaults"


def _require_isolated_provider_compile_worker(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Keep generated code out of the process that owns the NVIDIA client credential."""

    env = os.environ if environ is None else environ
    # Fail before any Provider call; compile always runs model.py in a timed ``spawn`` worker
    # process (see ``usdcraft.compile.mp_utils``), so the credential holder is never forked.
    compile_timeout_seconds(env)


def _provider_check_runner(model_path: Path) -> ProviderToolResult:
    try:
        result = check_model(model_path)
    except Exception as exc:
        bundle = compile_signal_bundle_from_exception(exc)
        return ProviderToolResult.error(bundle.to_dict())
    return ProviderToolResult.ok(
        {
            "status": "success",
            "scope": "authoring_only",
            "summary": result.signal_bundle.summary,
            "signals": [signal.to_dict() for signal in result.signal_bundle.signals],
            "warnings": list(result.warnings),
        }
    )


def _provider_compile_runner(
    model_path: Path,
    output_dir: Path,
    *,
    render: Literal[False],
) -> ProviderToolResult:
    if render is not False:
        return ProviderToolResult.error("Provider authoring compile requires render=False.")
    try:
        artifacts = compile_model(
            model_path,
            output_dir,
            clean=True,
            render=False,
            require_generation_root_floating=False,
        )
    except Exception as exc:
        bundle = compile_signal_bundle_from_exception(exc)
        return ProviderToolResult.error(bundle.to_dict())
    return ProviderToolResult.ok(
        {
            "status": "success",
            "render": False,
            "report": str(artifacts.report_path),
            "warnings": list(artifacts.warnings),
        }
    )


class AuthorCompileWorkspace(Protocol):
    @property
    def model_path(self) -> Path: ...

    @property
    def build_dir(self) -> Path: ...


def compile_author_output(
    workspace: AuthorCompileWorkspace,
    timer: StageTimer,
    *,
    compile_runner: Callable[..., CompileArtifacts],
) -> CompileArtifacts:
    """The final gate is always a fresh, render-free, floating-root compile."""
    with timer.stage("host_gate"):
        return compile_runner(
            workspace.model_path,
            workspace.build_dir,
            clean=True,
            render=False,
            require_generation_root_floating=True,
        )
