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

from usdcraft.generation.codex import run_codex
from usdcraft.generation.command import (
    GenerationApprovalError,
    GenerationBlockedError,
    GenerationFailureContext,
    GenerationInputChangedError,
    GenerationResult,
    execute_generation,
    fingerprint_generation_preflight,
    generation_failure_context,
    preflight_generation,
)
from usdcraft.generation.contract import (
    GenerationImageRole,
    GenerationInputKind,
    GenerationPreflight,
    GenerationRequest,
    generation_object_name,
)
from usdcraft.generation.default_prompt import (
    DEFAULT_IMAGE_PROMPT,
    resolve_generation_prompt,
)
from usdcraft.generation.prompt import build_generation_prompt
from usdcraft.generation.rematerialize import (
    RematerializationResult,
    RematerializationScopeError,
    rematerialize_record,
)
from usdcraft.generation.workspace import GenerationWorkspace, create_generation_workspace
from usdcraft.pipeline.codex import (
    CODEX_FINAL_SCHEMA,
    CodexExecutionError,
    CodexUnavailableError,
    ProcessResult,
    ProcessRunner,
)
from usdcraft.pipeline.contracts import (
    CodexCompletionStatus,
    CodexFinalMessage,
    CodexReasoningEffort,
    CodexRunResult,
    CodexRuntimeSelection,
)
from usdcraft.pipeline.runtime import (
    CodexRuntimeUnavailableError,
    current_main_codex_runtime,
    resolve_codex_runtime,
)

__all__ = [
    "CODEX_FINAL_SCHEMA",
    "CodexCompletionStatus",
    "CodexExecutionError",
    "CodexFinalMessage",
    "CodexReasoningEffort",
    "CodexRuntimeSelection",
    "CodexRuntimeUnavailableError",
    "CodexRunResult",
    "CodexUnavailableError",
    "DEFAULT_IMAGE_PROMPT",
    "GenerationApprovalError",
    "GenerationImageRole",
    "GenerationInputChangedError",
    "GenerationInputKind",
    "GenerationPreflight",
    "GenerationRequest",
    "GenerationBlockedError",
    "GenerationFailureContext",
    "GenerationResult",
    "GenerationWorkspace",
    "ProcessResult",
    "ProcessRunner",
    "RematerializationResult",
    "RematerializationScopeError",
    "build_generation_prompt",
    "create_generation_workspace",
    "current_main_codex_runtime",
    "execute_generation",
    "fingerprint_generation_preflight",
    "generation_failure_context",
    "generation_object_name",
    "preflight_generation",
    "rematerialize_record",
    "resolve_codex_runtime",
    "resolve_generation_prompt",
    "run_codex",
]
