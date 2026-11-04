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

"""Generation candidate rendering over the shared Provider editing and execution tools."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from usdcraft.generation.candidate_render import render_candidate
from usdcraft.pipeline.tools import (
    ProviderCheckRunner,
    ProviderCompileRunner,
    ProviderToolExecution,
    ProviderToolExecutor,
    build_completion_tool_schema,
    build_provider_tool_schemas,
)

if TYPE_CHECKING:
    from usdcraft.generation.workspace import GenerationWorkspace


def build_generation_tool_schemas() -> tuple[dict[str, Any], ...]:
    return (
        *build_provider_tool_schemas(),
        build_completion_tool_schema("finish_generation"),
        {
            "type": "function",
            "name": "validate_model",
            "description": "Default complete candidate checkpoint: check, compile and render q=0 in one call. Stops on code/export failure. ok reports check/compile success; diagnostics_complete reports render execution, not visual quality. Inspect the attached candidate before repairing. Use individual tools for targeted checks or additional views. validation_state reports author revision coverage, not visual acceptance.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "render_candidate",
            "description": "Inspect the current compiled candidate at a camera and joint pose. Returns images to this Author conversation; on image runs the original reference precedes the candidate. Compile after edits first. Angles are degrees; joint positions are radians/metres. This is output feedback, not target evidence or a physics test.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "azimuth": {"type": "number"},
                    "elevation": {"type": "number", "minimum": -90, "maximum": 90},
                    "joints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "position": {"type": "number"},
                            },
                            "required": ["name", "position"],
                        },
                    },
                },
                "required": ["azimuth", "elevation", "joints"],
            },
        },
    )


class GenerationToolExecutor(ProviderToolExecutor):
    workspace: GenerationWorkspace

    def __init__(
        self,
        workspace: GenerationWorkspace,
        *,
        check_runner: ProviderCheckRunner,
        compile_runner: ProviderCompileRunner,
    ) -> None:
        self.render_calls = 0
        super().__init__(
            workspace,
            finish_tool_name="finish_generation",
            check_runner=check_runner,
            compile_runner=compile_runner,
        )

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        return self.track_validation(
            name,
            self._execute(name, arguments),
            {
                "check_model": "check",
                "compile_model": "compile",
                "render_candidate": "render",
            },
        )

    def validation_state(self, stages: tuple[str, ...] = ()) -> dict[str, Any]:
        """Report this Author's fixed stage set; ``stages`` exists only for signature parity."""
        del stages
        return super().validation_state(("check", "compile", "render"))

    def _execute(self, name: str, arguments: Mapping[str, Any]) -> ProviderToolExecution:
        if name == "validate_model":
            if arguments:
                return ProviderToolExecution(self._error("validate_model accepts no arguments."))
            return self.validate_candidate(
                [
                    (
                        "render",
                        lambda: self._execute(
                            "render_candidate", {"azimuth": 45, "elevation": 30, "joints": []}
                        ),
                    ),
                ]
            )
        if name != "render_candidate":
            return super().execute(name, arguments)
        if self.compile_success_revision != self.revision:
            return ProviderToolExecution(
                self._error("Compile the current model revision before rendering.")
            )
        self.render_calls += 1
        try:
            if set(arguments) != {"azimuth", "elevation", "joints"}:
                raise ValueError("render_candidate requires azimuth, elevation and joints")
            positions = {}
            for joint in arguments["joints"]:
                if set(joint) != {"name", "position"} or joint["name"] in positions:
                    raise ValueError("Joint entries must have unique names and positions")
                positions[joint["name"]] = float(joint["position"])
            feedback_root = self.workspace.root / "author_feedback"
            if feedback_root.is_symlink():
                raise ValueError("Feedback directory must stay inside the workspace")
            output = feedback_root / f"view_{uuid.uuid4().hex}"
            result = render_candidate(
                self.workspace.build_dir,
                out_dir=output,
                azimuth=float(arguments["azimuth"]),
                elevation=float(arguments["elevation"]),
                joint_positions=positions,
            )
            result["revision"] = self.revision
            images = tuple(Path(p) for p in result["images"])
            reference = self.workspace.input_image_path
            roles = ["candidate"] * len(images)
            if reference is not None:
                images = (reference, *images)
                roles.insert(0, "inspiration_reference")
            result["attachment_roles"] = roles
            return ProviderToolExecution(result, image_paths=images)
        except Exception as exc:
            return ProviderToolExecution(
                self._error(f"Candidate renderer: {type(exc).__name__}: {exc}")
            )
