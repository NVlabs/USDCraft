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

"""NVIDIA Provider transport for the optional Reconstruction Subject-Isolation Agent.

`usdcraft reconstruction --mesh-cleanup` runs the same clean-context Subject-Isolation Agent as the
Codex arm: the task text, the component inventory, both isolation contact sheets, the decision
schema, the decision validation and the published manifest all come from
:mod:`usdcraft.reconstruction.isolation_agent`. Only the transport differs. One NVIDIA request
carries the staged task, the exact staged ``component_inventory.json`` bytes (the Codex Agent reads
the same file from its read-only workspace) and the two contact sheets in the Codex attachment
order; the decision returns through one strict function tool whose parameters are the shared
decision schema. The request is bounded by the Provider per-request timeout. As on the Codex arm
there is no retry: a failed request, a missing tool call or an invalid decision fails the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any, Mapping, Sequence

from usdcraft.reconstruction.isolation_agent import (
    ISOLATION_AGENT_RESULT_SCHEMA,
    IsolationRuntimeSelection,
    IsolationTransport,
    SubjectIsolationAgentError,
    SubjectIsolationArtifacts,
    run_subject_isolation_with_transport,
)
from usdcraft.reconstruction.orientation import BundleBuilder, SessionFactory
from usdcraft.reconstruction.source import SourceInput

ISOLATION_TOOL_NAME = "submit_subject_isolation"
PROVIDER_RUN_DIRNAME = ".provider-run"
PROVIDER_ISOLATION_SELECTION_SOURCE = "provider_config_for_reconstruction_isolation"
# The attachment order is the Codex command's `--image` order.
ATTACHED_IMAGE_NAMES = ("native_contact_sheet.png", "component_id_contact_sheet.png")
INVENTORY_FILENAME = "component_inventory.json"


def isolation_tool_schema() -> dict[str, object]:
    """The one strict function tool a Provider isolation call may invoke."""

    return {
        "type": "function",
        "name": ISOLATION_TOOL_NAME,
        "description": "Submit the complete subject-isolation decision exactly once.",
        "parameters": json.loads(json.dumps(ISOLATION_AGENT_RESULT_SCHEMA)),
        "strict": True,
    }


def build_provider_isolation_message(task_text: str, workspace: Path) -> dict[str, object]:
    """One user message: the task, the staged inventory file, then both contact sheets."""

    from usdcraft.pipeline.provider.harness import image_path_to_data_url

    root = Path(workspace)
    inventory_text = (root / INVENTORY_FILENAME).read_text(encoding="utf-8")
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": task_text},
        {"type": "input_text", "text": f"`{INVENTORY_FILENAME}`:\n{inventory_text}"},
    ]
    for name in ATTACHED_IMAGE_NAMES:
        content.append(
            {
                "type": "input_image",
                "image_url": image_path_to_data_url(root / name),
                "detail": "high",
            }
        )
    return {"role": "user", "content": content}


def extract_isolation_arguments(output: Sequence[Mapping[str, object]]) -> str:
    """Return the verbatim arguments of the single isolation tool call."""

    calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == ISOLATION_TOOL_NAME
    ]
    if len(calls) != 1:
        raise SubjectIsolationAgentError(
            f"Provider isolation Agent must call {ISOLATION_TOOL_NAME} exactly once "
            f"(got {len(calls)} calls)"
        )
    arguments = calls[0].get("arguments")
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(arguments, str) or not arguments.strip():
        raise SubjectIsolationAgentError(
            "Provider isolation Agent tool arguments must be JSON text"
        )
    return arguments


def run_subject_isolation_via_provider(
    source_input: SourceInput,
    output_dir: Path,
    *,
    config: Any,
    cancellation_event: Event | None = None,
    client: Any | None = None,
    bundle_builder: BundleBuilder | None = None,
    session_factory: SessionFactory | None = None,
) -> SubjectIsolationArtifacts:
    """Run and freeze one clean-context subject-isolation decision through the NVIDIA Provider."""

    from usdcraft.pipeline.provider import NvidiaProviderError, NvidiaResponsesClient

    runtime = IsolationRuntimeSelection(
        model_id=str(getattr(config, "model")),
        reasoning_effort=str(getattr(config, "reasoning_effort")),
        runtime_source="provider_repository_config",
        selection_source=PROVIDER_ISOLATION_SELECTION_SOURCE,
    )
    api_style = str(getattr(config, "api_style", "responses"))
    # The endpoint names the transport (`nvidia_provider` or `openai_provider`).
    endpoint = str(getattr(config, "endpoint", "openai"))
    timeout = getattr(config, "timeout_seconds", None)

    def invoke(workspace: Path, task_text: str) -> Mapping[str, str]:
        resolved_client = client or NvidiaResponsesClient(
            config, cancellation_event=cancellation_event
        )
        message = build_provider_isolation_message(task_text, workspace)
        try:
            response = resolved_client.create_response(
                input_items=[message], tools=[isolation_tool_schema()]
            )
        except NvidiaProviderError as exc:
            if cancellation_event is not None and cancellation_event.is_set():
                raise
            raise SubjectIsolationAgentError(
                f"Provider subject-isolation request failed: {exc}"
            ) from exc
        run_dir = workspace / PROVIDER_RUN_DIRNAME
        output = list(getattr(response, "output"))
        (run_dir / "response.json").write_text(
            json.dumps(
                {"status": getattr(response, "status", None), "output": output},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "final.json").write_text(extract_isolation_arguments(output), encoding="utf-8")
        return {"agent_response": f"{PROVIDER_RUN_DIRNAME}/response.json"}

    provider_fields: dict[str, object] = {
        "arm": "provider",
        "transport": f"{endpoint}_provider",
        "api_style": api_style,
        "response_tool": ISOLATION_TOOL_NAME,
        "inventory_delivery": "staged_file_text_in_request",
    }
    return run_subject_isolation_with_transport(
        source_input,
        output_dir,
        transport=IsolationTransport(
            agent_name="",
            run_dir_name=PROVIDER_RUN_DIRNAME,
            response_label="Provider",
            runtime=runtime,
            invoke=invoke,
            context_fields={
                **provider_fields,
                "filesystem_isolation": "request_message_only_no_filesystem_access",
            },
            agent_fields={
                **provider_fields,
                "request_timeout_seconds": timeout,
                "provider_store": False,
            },
        ),
        bundle_builder=bundle_builder,
        session_factory=session_factory,
    )


__all__ = [
    "ISOLATION_TOOL_NAME",
    "PROVIDER_RUN_DIRNAME",
    "build_provider_isolation_message",
    "extract_isolation_arguments",
    "isolation_tool_schema",
    "run_subject_isolation_via_provider",
]
