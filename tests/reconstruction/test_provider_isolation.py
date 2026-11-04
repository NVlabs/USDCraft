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

"""Optional mesh cleanup on the Provider arm: the shared Subject-Isolation Agent over the Provider transport.

Every test is offline: the NVIDIA client and the Chromium renderer are fakes.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from storage.data_validation import validate_data_format
from storage.repo import StorageRepo
from tests.reconstruction.test_reconstruction_codex_author import _evidence, _inputs
from tests.reconstruction.test_reconstruction_isolation import (
    _bundle_builder,
    _FakeChromium,
    _write_two_component_glb,
)
from tests.reconstruction.test_reconstruction_persist import (
    _artifacts as _compile_artifacts,
)
from tests.reconstruction.test_reconstruction_persist import _current_approved_evidence
from usdcraft.cli import main as cli_main
from usdcraft.pipeline.contracts import ProviderRuntimeSelection
from usdcraft.pipeline.provider import NvidiaRequestError
from usdcraft.reconstruction.command import (
    ReconstructionPreprocessingBlockedError,
    ReconstructionPreprocessingResult,
    _build_reconstruction_preprocessing,
    preflight_reconstruction,
)
from usdcraft.reconstruction.isolation import (
    SUBJECT_ISOLATION_SCHEMA,
    load_subject_isolation_decision,
)
from usdcraft.reconstruction.isolation_agent import (
    ISOLATION_AGENT_RESULT_SCHEMA,
    IsolationProcessResult,
    SubjectIsolationAgentError,
    SubjectIsolationArtifacts,
    build_subject_isolation_task,
    run_subject_isolation,
)
from usdcraft.reconstruction.persist import (
    NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING,
    persist_reconstruction,
)
from usdcraft.reconstruction.provider_batch import (
    RECONSTRUCTION_CSV_COLUMNS,
    execute_reconstruction_batch,
    load_reconstruction_batch,
)
from usdcraft.reconstruction.provider_command import (
    ProviderReconstructionResult,
    execute_provider_reconstruction,
)
from usdcraft.reconstruction.provider_isolation import (
    ISOLATION_TOOL_NAME,
    run_subject_isolation_via_provider,
)
from usdcraft.reconstruction.source import SourceInput, load_normalized_source
from usdcraft.reconstruction.workspace import create_reconstruction_run_workspace

_FAKE_CONFIG = SimpleNamespace(
    model="nvidia-test-model",
    reasoning_effort="high",
    timeout_seconds=900.0,
    api_style="responses",
    endpoint="nvidia",
    runtime_source="nvidia-responses-api",
)


class _FakeClient:
    """A fake Provider transport that answers with one scripted isolation tool call."""

    def __init__(self, reply: object | None, *, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create_response(self, *, input_items, tools):  # type: ignore[no-untyped-def]
        self.calls.append({"input_items": list(input_items), "tools": list(tools)})
        if self.error is not None:
            raise self.error
        output: list[dict[str, object]] = [{"type": "reasoning", "summary": []}]
        if self.reply is not None:
            output.append(
                {
                    "type": "function_call",
                    "name": ISOLATION_TOOL_NAME,
                    "call_id": "call_1",
                    "arguments": json.dumps(self.reply),
                }
            )
        return SimpleNamespace(status="completed", output=tuple(output))


def _decision(selection_id: str, *, status: str = "complete") -> dict[str, object]:
    return {
        "status": status,
        "decisions": [
            {
                "selection_id": selection_id,
                "disposition": "exclude_background_or_incidental",
                "reason": "detached non-subject component",
                "visual_evidence": "separate in every fixed view",
            }
        ],
        "summary": "retained the coherent primary component",
    }


def _run_provider(source: Path, output: Path, client: _FakeClient) -> SubjectIsolationArtifacts:
    return run_subject_isolation_via_provider(
        SourceInput(source),
        output,
        config=_FAKE_CONFIG,
        client=client,
        bundle_builder=_bundle_builder,
        session_factory=lambda _dist: _FakeChromium({}),
    )


def test_provider_isolation_transport_names_the_openai_endpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    raw_scene = load_normalized_source(SourceInput(source))
    reply = _decision(f"component:{raw_scene.components[1].component_id}")

    result = run_subject_isolation_via_provider(
        SourceInput(source),
        tmp_path / "isolation",
        config=SimpleNamespace(**{**vars(_FAKE_CONFIG), "endpoint": "openai"}),
        client=_FakeClient(reply),
        bundle_builder=_bundle_builder,
        session_factory=lambda _dist: _FakeChromium({}),
    )

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["agent"]["arm"] == "provider"
    assert manifest["agent"]["transport"] == "openai_provider"


def _codex_isolation(source: Path, output: Path, reply: dict[str, object]) -> dict[str, Any]:
    """Run the Codex isolation Agent over the same source and return its manifest."""

    def process_runner(command, **_kwargs):  # type: ignore[no-untyped-def]
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_text(json.dumps(reply), encoding="utf-8")
        return IsolationProcessResult(0, "{}\n", "")

    result = run_subject_isolation(
        SourceInput(source),
        output,
        model="test-model",
        reasoning_effort="high",
        executable="codex-test",
        environ={},
        bundle_builder=_bundle_builder,
        session_factory=lambda _dist: _FakeChromium({}),
        process_runner=process_runner,
    )
    return json.loads(result.manifest.read_text(encoding="utf-8"))


def test_provider_task_is_the_codex_task_without_the_transport_name() -> None:
    codex = build_subject_isolation_task(inventory_sha256="a" * 64)
    provider = build_subject_isolation_task(inventory_sha256="a" * 64, agent_name="")

    # The transport name in the self-description is the only difference between the arms.
    assert provider != codex
    assert provider == codex.replace("Codex ", "", 1)


def test_provider_isolation_sends_the_shared_task_and_evidence_and_freezes_the_decision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    _write_two_component_glb(source)
    raw_scene = load_normalized_source(SourceInput(source))
    excluded_id = raw_scene.components[1].component_id
    reply = _decision(f"component:{excluded_id}")
    client = _FakeClient(reply)

    result = _run_provider(source, tmp_path / "isolation", client)

    assert result.status == "ready"
    assert len(client.calls) == 1
    (tool,) = client.calls[0]["tools"]
    assert tool["name"] == ISOLATION_TOOL_NAME and tool["strict"] is True
    assert tool["parameters"] == ISOLATION_AGENT_RESULT_SCHEMA
    (message,) = client.calls[0]["input_items"]
    content = message["content"]
    assert [item["type"] for item in content] == [
        "input_text",
        "input_text",
        "input_image",
        "input_image",
    ]
    staged_task = (result.root / "agent" / "TASK.md").read_text(encoding="utf-8")
    inventory_bytes = result.inventory.read_bytes()
    assert content[0]["text"].rstrip() == staged_task.rstrip()
    assert (
        content[0]["text"].rstrip()
        == build_subject_isolation_task(
            inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
            agent_name="",
        ).rstrip()
    )
    assert str(source) not in content[0]["text"]
    assert content[1]["text"] == "`component_inventory.json`:\n" + inventory_bytes.decode()
    for item, sheet in zip(
        content[2:], (result.native_contact_sheet, result.component_id_contact_sheet), strict=True
    ):
        assert base64.b64decode(item["image_url"].split(",", 1)[1]) == sheet.read_bytes()

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == SUBJECT_ISOLATION_SCHEMA and manifest["status"] == "ready"
    assert manifest["selection"]["excluded_component_ids"] == [excluded_id]
    agent = manifest["agent"]
    assert agent["run"] is True and agent["context"] == "clean_subject_isolation_only"
    assert agent["arm"] == "provider" and agent["transport"] == "nvidia_provider"
    assert agent["model"] == "nvidia-test-model" and agent["reasoning_effort"] == "high"
    assert agent["request_timeout_seconds"] == 900.0
    assert agent["attached_image_count"] == 2 and agent["semantic_image"] is False
    assert "disabled_codex_features" not in agent and "sandbox" not in agent
    audit = result.root / "agent"
    assert json.loads((audit / ".provider-run" / "final.json").read_text()) == reply
    assert (audit / ".provider-run" / "final.schema.json").is_file()
    assert (audit / ".provider-run" / "response.json").is_file()
    context = json.loads((audit / "context_manifest.json").read_text())
    assert context["attached_images"] == [
        "native_contact_sheet.png",
        "component_id_contact_sheet.png",
    ]

    # The published manifest matches the Codex manifest except for the transport facts.
    codex = _codex_isolation(source, tmp_path / "codex_isolation", reply)
    assert manifest["selection"] == codex["selection"]
    assert manifest["downstream_contract"] == codex["downstream_contract"]
    transport_logs = {"agent_response", "agent_stdout", "agent_stderr"}
    assert set(manifest["artifacts"]) - transport_logs == set(codex["artifacts"]) - transport_logs
    assert manifest["artifacts"]["agent_response"]["path"] == "agent/.provider-run/response.json"

    decision = load_subject_isolation_decision(result.manifest, raw_scene)
    assert decision.excluded_component_ids == (excluded_id,)
    assert decision.upstream_agent_run is True


@pytest.mark.parametrize(
    ("reply", "error", "message"),
    [
        (lambda ids: _decision("component:unknown"), None, "unknown selection id"),
        (
            lambda ids: _decision(f"component:{ids[0]}", status="blocked"),
            None,
            "blocked.*must not exclude",
        ),
        (
            lambda ids: {"status": "complete", "decisions": []},
            None,
            "Provider isolation Agent response has unexpected fields",
        ),
        (lambda ids: None, None, f"must call {ISOLATION_TOOL_NAME} exactly once"),
        (
            lambda ids: None,
            NvidiaRequestError("HTTP 503"),
            "Provider subject-isolation request failed",
        ),
    ],
)
def test_invalid_provider_decision_fails_like_the_codex_path(
    tmp_path: Path, reply: Any, error: Exception | None, message: str
) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "isolation"
    _write_two_component_glb(source)
    ids = [item.component_id for item in load_normalized_source(SourceInput(source)).components]
    client = _FakeClient(reply(ids), error=error)

    with pytest.raises(SubjectIsolationAgentError, match=message):
        _run_provider(source, output, client)
    assert not output.exists()
    assert list(tmp_path.glob(".isolation.staging-*")) == []


def test_provider_request_carries_mesh_cleanup_and_routes_isolation_to_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source, semantic, _codex_preflight = _inputs(tmp_path)
    captured: dict[str, Any] = {}

    class _FakeConfig:
        @classmethod
        def from_repo(cls, root, **kwargs):  # type: ignore[no-untyped-def]
            return _FAKE_CONFIG

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.workspace = kwargs["workspace"]

        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                status="complete",
                summary="done",
                runtime_report=lambda: {"final_status": "complete", "turn_count": 1},
            )

    def fake_preprocessing(preflight, run, **kwargs):  # type: ignore[no-untyped-def]
        captured.setdefault("requests", []).append(preflight.request)
        captured["fingerprints"] = [*captured.get("fingerprints", []), preflight.fingerprint]
        kwargs["isolation_runner"](source_input="frozen-source", output_dir=Path("/iso"))
        return ReconstructionPreprocessingResult(
            isolation=None,
            mesh_preprocess=SimpleNamespace(),
            evidence=_evidence(run),
            image_observation=None,
        )

    def fake_isolation(**kwargs):  # type: ignore[no-untyped-def]
        captured["isolation"] = kwargs

    def fake_persist(**kwargs):  # type: ignore[no-untyped-def]
        record = repo / "data" / "records" / f"rec_{len(captured['requests'])}"
        (record / "revisions" / "rev_000001").mkdir(parents=True)
        return record

    module = "usdcraft.reconstruction.provider_command"
    monkeypatch.setattr(f"{module}.NvidiaProviderConfig", _FakeConfig)
    monkeypatch.setattr(f"{module}.NvidiaResponsesClient", lambda config, **kwargs: None)
    monkeypatch.setattr(f"{module}.ProviderHarness", _FakeHarness)
    monkeypatch.setattr(f"{module}._build_reconstruction_preprocessing", fake_preprocessing)
    monkeypatch.setattr(f"{module}.run_subject_isolation_via_provider", fake_isolation)
    monkeypatch.setattr(
        f"{module}.compile_model", lambda path, output, **kwargs: _compile_artifacts(path.parent)
    )
    monkeypatch.setattr(f"{module}.persist_reconstruction", fake_persist)
    monkeypatch.setattr(
        f"{module}.recover_pending_promotions", lambda _repo: SimpleNamespace(errors=())
    )
    common = dict(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
    )

    execute_provider_reconstruction(**common)
    execute_provider_reconstruction(**common, mesh_cleanup=True)

    default_request, cleanup_request = captured["requests"]
    # The Provider records mesh cleanup exactly as the Codex request does.
    assert default_request.mesh_cleanup is False and cleanup_request.mesh_cleanup is True
    assert default_request.to_dict()["mesh_cleanup"] == {
        "requested": False,
        "role": "optional_subject_isolation_agent_for_source_mesh_artifact_cleanup",
    }
    assert cleanup_request.to_dict()["mesh_cleanup"]["requested"] is True
    assert captured["fingerprints"][0] != captured["fingerprints"][1]
    assert captured["isolation"]["config"] is _FAKE_CONFIG
    assert captured["isolation"]["source_input"] == "frozen-source"
    assert captured["isolation"]["output_dir"] == Path("/iso")


def _provider_run(tmp_path: Path, *, mesh_cleanup: bool):  # type: ignore[no-untyped-def]
    from usdcraft.reconstruction.provider_command import resolve_provider_reconstruction_preflight

    repo, source, semantic, _codex_preflight = _inputs(tmp_path)
    preflight = resolve_provider_reconstruction_preflight(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        config=_FAKE_CONFIG,
        selection_source="test",
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
        image_observation=False,
        mesh_cleanup=mesh_cleanup,
    )
    return preflight, create_reconstruction_run_workspace(preflight, run_id="recon_iso_test")


def test_preprocessing_uses_the_supplied_isolation_transport_only_for_mesh_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, run = _provider_run(tmp_path / "on", mesh_cleanup=True)
    frozen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def fake_mesh_preprocess(source_input, upstream, *, subject_isolation):  # type: ignore[no-untyped-def]
        frozen["manifest"] = subject_isolation
        raise _Stop

    monkeypatch.setattr("usdcraft.reconstruction.command.run_mesh_preprocess", fake_mesh_preprocess)
    monkeypatch.setattr(
        "usdcraft.reconstruction.command.run_subject_isolation",
        lambda *a, **k: pytest.fail("the Codex isolation Agent must not start on the Provider"),
    )

    def ready_runner(*, source_input, output_dir):  # type: ignore[no-untyped-def]
        frozen["source"] = source_input.path
        return SubjectIsolationArtifacts(
            root=output_dir,
            status="ready",
            manifest=output_dir / "subject_isolation.json",
            inventory=output_dir / "component_inventory.json",
            native_contact_sheet=output_dir / "n.png",
            component_id_contact_sheet=output_dir / "c.png",
            retained_source=output_dir / "r.usdc",
            retained_source_manifest=output_dir / "r.json",
        )

    with pytest.raises(_Stop):
        _build_reconstruction_preprocessing(preflight, run, isolation_runner=ready_runner)
    assert frozen["source"] == run.frozen_inputs.source_path
    assert frozen["manifest"] == run.upstream_dir / "subject_isolation" / "subject_isolation.json"

    with pytest.raises(ReconstructionPreprocessingBlockedError):
        _build_reconstruction_preprocessing(
            *_provider_run(tmp_path / "blocked", mesh_cleanup=True),
            isolation_runner=lambda **kw: replace(ready_runner(**kw), status="blocked"),
        )

    # Mesh cleanup off: the Host freezes the retain-all declaration and never calls the runner.
    preflight, run = _provider_run(tmp_path / "off", mesh_cleanup=False)
    with pytest.raises(_Stop):
        _build_reconstruction_preprocessing(
            preflight,
            run,
            isolation_runner=lambda **kw: pytest.fail("mesh cleanup is off by default"),
        )
    manifest = json.loads(Path(frozen["manifest"]).read_text(encoding="utf-8"))
    assert manifest["agent"]["run"] is False
    assert manifest["agent"]["context"] == "mesh_cleanup_not_requested"


def test_persisted_provider_record_keeps_the_provider_isolation_audit(tmp_path: Path) -> None:
    StorageRepo(tmp_path).ensure_layout()
    source = tmp_path / "isolation_source.glb"
    _write_two_component_glb(source)
    excluded_id = load_normalized_source(SourceInput(source)).components[1].component_id
    isolation = _run_provider(
        source,
        tmp_path / "isolation",
        _FakeClient(_decision(f"component:{excluded_id}")),
    )
    workspace = tmp_path / "workspace"
    (workspace / ".provider-run").mkdir(parents=True)
    (workspace / "model.py").write_text("object_model = object()\n", encoding="utf-8")
    (workspace / "TASK.md").write_text("Reconstruct one subject.\n", encoding="utf-8")
    trace_dir = workspace / ".provider-run"
    (trace_dir / "events.jsonl").write_text('{"record_type":"provider.run.started"}\n')
    (trace_dir / "final.json").write_text('{"status":"complete","summary":"done"}\n')
    (trace_dir / "stderr.log").write_text("", encoding="utf-8")
    (trace_dir / "timeline.jsonl").write_text("{}\n", encoding="utf-8")
    evidence = replace(
        _current_approved_evidence(tmp_path), subject_isolation_manifest_path=isolation.manifest
    )

    record_dir = persist_reconstruction(
        repo_root=tmp_path,
        run_id="recon_provider_isolation",
        title="Test Drawer",
        request_text="",
        runtime=ProviderRuntimeSelection(model_id="nvidia-test-model", reasoning_effort="high"),
        workspace_dir=workspace,
        runtime_report={"final_status": "complete"},
        artifacts=_compile_artifacts(workspace),
        evidence=evidence,
        label=None,
        authoring=NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING,
    )

    revision = record_dir / "revisions" / "rev_000001"
    persisted = revision / "inputs" / "preprocessing" / "subject_isolation_manifest.json"
    manifest = json.loads(persisted.read_text(encoding="utf-8"))
    assert manifest["agent"]["arm"] == "provider"
    assert manifest["selection"]["excluded_component_ids"] == [excluded_id]
    audit = revision / "inputs" / "preprocessing" / "audit" / "subject_isolation"
    for relative in (
        "agent/TASK.md",
        "agent/context_manifest.json",
        "agent/.provider-run/final.json",
        "agent/.provider-run/response.json",
        "evidence/native_contact_sheet.png",
        "evidence/component_id_contact_sheet.png",
        "retained_source/normalized.usdc",
    ):
        assert (audit / relative).is_file(), relative
    assert validate_data_format(StorageRepo(tmp_path)).errors == []


@pytest.mark.parametrize("command", ["reconstruction", "reconstruction-codex"])
def test_both_single_item_arms_offer_the_same_mesh_cleanup_switch(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main([command, "--help"])
    assert exc.value.code == 0
    assert "--mesh-cleanup" in capsys.readouterr().out
    assert (
        inspect.signature(execute_provider_reconstruction).parameters["mesh_cleanup"].default
        is False
    )


def _write_rows(path: Path, cells: list[str]) -> None:
    path.write_text(
        "row_id,source_path,image_path,mesh_cleanup\n"
        + "".join(f"r{index},m.glb,i.jpg,{cell}\n" for index, cell in enumerate(cells)),
        encoding="utf-8",
    )


def test_csv_mesh_cleanup_column_is_optional_boolean_and_defaults_off(tmp_path: Path) -> None:
    assert "mesh_cleanup" in RECONSTRUCTION_CSV_COLUMNS
    path = tmp_path / "jobs.csv"
    _write_rows(path, ["true", "1", "false", "0", ""])
    rows = load_reconstruction_batch(path)
    assert [row.mesh_cleanup for row in rows] == [True, True, False, False, False]
    path.write_text("row_id,source_path,image_path\na,m.glb,i.jpg\n", encoding="utf-8")
    assert load_reconstruction_batch(path)[0].mesh_cleanup is False
    _write_rows(path, ["yes"])
    with pytest.raises(ValueError, match="mesh_cleanup must be true/false or 1/0"):
        load_reconstruction_batch(path)


@pytest.mark.parametrize("command", ["reconstruction-batch", "reconstruction-codex-batch"])
def test_validate_only_accepts_the_mesh_cleanup_column_on_both_arms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    path = tmp_path / "jobs.csv"
    _write_rows(path, ["true", ""])
    assert cli_main([command, str(path), "--validate-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2
    _write_rows(path, ["maybe"])
    assert cli_main([command, str(path), "--validate-only", "--json"]) == 1
    assert "mesh_cleanup must be true/false or 1/0" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_provider_batch_passes_mesh_cleanup_to_the_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "jobs.csv"
    _write_rows(path, ["true", ""])
    calls: dict[str, dict[str, Any]] = {}

    def fake_execute(**kwargs):  # type: ignore[no-untyped-def]
        row_id = kwargs["batch_context"]["row_id"]
        calls[row_id] = kwargs
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id=f"recon_{row_id}",
            record_id=f"rec_{row_id}",
            record_dir=tmp_path / row_id,
            model_path=None,
            workspace_dir=None,
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction", fake_execute
    )
    result = execute_reconstruction_batch(
        csv_path=path, repo_root=tmp_path, model="nvidia-test-model", reasoning="high"
    )
    assert result.succeeded == 2
    assert calls["r0"]["mesh_cleanup"] is True and calls["r1"]["mesh_cleanup"] is False


def test_codex_batch_passes_mesh_cleanup_to_the_direct_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from usdcraft.pipeline.batch import ProviderBatchRowResult
    from usdcraft.reconstruction.codex_batch import execute_codex_reconstruction_batch

    monkeypatch.setenv("USDCRAFT_CODEX_BIN", sys.executable)
    path = tmp_path / "jobs.csv"
    _write_rows(path, ["1", "0"])
    captured: dict[str, dict[str, Any]] = {}

    def supervised(arguments, **kwargs):  # type: ignore[no-untyped-def]
        captured[kwargs["row_id"]] = arguments
        return ProviderBatchRowResult(row_id=kwargs["row_id"], status="reconstruction_ready")

    monkeypatch.setattr("usdcraft.reconstruction.codex_batch.execute_with_deadline", supervised)
    result = execute_codex_reconstruction_batch(
        csv_path=path, repo_root=tmp_path, model="test-model", reasoning="low"
    )
    assert result.succeeded == 2
    assert captured["r0"]["mesh_cleanup"] is True and captured["r1"]["mesh_cleanup"] is False
    # The supervised worker hands these arguments (minus batch_context) to the real preflight.
    inspect.signature(preflight_reconstruction).bind(
        **{name: value for name, value in captured["r0"].items() if name != "batch_context"}
    )
