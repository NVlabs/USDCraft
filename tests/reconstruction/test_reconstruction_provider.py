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

"""Reconstruction Provider arm: direct execution, tools, prompt variant, persistence, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from storage.data_validation import validate_data_format
from storage.repo import StorageRepo
from tests.prompt_snapshot import assert_matches_golden
from tests.reconstruction.test_reconstruction_codex_author import _evidence, _inputs
from tests.reconstruction.test_reconstruction_persist import (
    _artifacts as _compile_artifacts,
)
from tests.reconstruction.test_reconstruction_persist import (
    _current_approved_evidence,
)
from usdcraft import cli as usdcraft_cli
from usdcraft.cli import main as cli_main
from usdcraft.pipeline.contracts import ProviderRuntimeSelection
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS
from usdcraft.pipeline.tools import ProviderToolResult
from usdcraft.reconstruction.command import (
    ReconstructionBlockedError,
    ReconstructionPreprocessingResult,
    reconstruction_failure_context,
)
from usdcraft.reconstruction.persist import (
    NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING,
    persist_reconstruction,
    provider_reconstruction_authoring,
)
from usdcraft.reconstruction.prompt import build_reconstruction_author_prompt
from usdcraft.reconstruction.provider_command import (
    execute_provider_reconstruction,
    resolve_provider_reconstruction_preflight,
)
from usdcraft.reconstruction.provider_tools import (
    ReconstructionToolExecutor,
    build_reconstruction_tool_schemas,
)
from usdcraft.reconstruction.workspace import (
    create_reconstruction_author_workspace,
    create_reconstruction_run_workspace,
)

_FAKE_CONFIG = SimpleNamespace(
    model="nvidia-test-model",
    reasoning_effort="high",
    timeout_seconds=900.0,
    api_style="responses",
    endpoint="nvidia",
    runtime_source="nvidia-responses-api",
)


@pytest.fixture(autouse=True)
def temporary_checkout_with_render_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI tests use temporary repository roots and fake every render; the checkout and
    # render-setup checks have their own tests in tests/test_setup_cli.py.
    monkeypatch.setattr(usdcraft_cli, "_checkout_root", usdcraft_cli._repo_root)
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", lambda: None)


def _provider_workspace(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A real staged Provider Author workspace over frozen inputs and fixture evidence."""

    repo, source, semantic, _codex_preflight = _inputs(tmp_path)
    preflight = resolve_provider_reconstruction_preflight(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        config=_FAKE_CONFIG,
        selection_source="explicit_provider_arguments",
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
        guidance="Keep the user-visible latch.",
    )
    run = create_reconstruction_run_workspace(preflight)
    evidence = _evidence(run)
    workspace = create_reconstruction_author_workspace(
        preflight,
        run,
        evidence,
        root=run.author_workspace_base,
        author_brief=True,
    )
    assert workspace.root == run.author_workspace_base
    return repo, preflight, run, evidence, workspace


def _prompt_common(preflight, workspace):  # type: ignore[no-untyped-def]
    return dict(
        workspace=workspace,
        user_guidance=preflight.request.guidance,
        image_observation_text="- latch: count=1, movable, control, visible",
        request_text="Reconstruct the toaster shown in the semantic image and the source mesh.",
    )


def test_provider_task_matches_golden_and_stays_within_its_boundaries(tmp_path: Path) -> None:
    """The task wording is pinned by the golden snapshot; this checks the arm's access boundary."""

    repo, preflight, _run, _evidence_bundle, workspace = _provider_workspace(tmp_path)
    common = _prompt_common(preflight, workspace)
    provider = build_reconstruction_author_prompt(
        check_command="check_model",
        compile_command="compile_model",
        axial_reencode_command="check_axial",
        reference_access="provider_tools",
        **common,
    )
    reviewed = build_reconstruction_author_prompt(
        check_command="check_model",
        compile_command="compile_model",
        axial_reencode_command="check_axial",
        reference_access="provider_tools",
        scale_note="metres were set from an estimate",
        **common,
    )
    assert_matches_golden(
        "reconstruction_provider",
        reviewed,
        paths={str(workspace.root): "<workspace>", str(repo): "<repo>"},
    )
    task_text = provider.split("PRELOADED SDK REFERENCES", 1)[0]
    for present in (
        "write_model",
        "apply_patch",
        "read_file",
        "measure_face",
        "finish_reconstruction",
    ):
        assert present in task_text, present
    for absent in (
        "here-doc",
        "rg --files",
    ):
        assert absent not in task_text, absent
    assert "SCALE\nmetres were set" in reviewed
    with pytest.raises(ValueError, match="reference access"):
        build_reconstruction_author_prompt(
            check_command="a",
            compile_command="b",
            axial_reencode_command="c",
            reference_access="browser",  # type: ignore[arg-type]
            **common,
        )


def test_reconstruction_tools_read_staged_evidence_and_gate_the_axial_diagnostic(
    tmp_path: Path,
) -> None:
    _repo, _preflight, _run, _evidence_bundle, workspace = _provider_workspace(tmp_path)
    axial_calls: list[tuple[Path, Path]] = []

    def check_runner(model_path: Path) -> ProviderToolResult:
        return ProviderToolResult.ok({"status": "success"})

    def compile_runner(model_path: Path, output_dir: Path, *, render) -> ProviderToolResult:  # type: ignore[no-untyped-def]
        candidate = output_dir / "isaac" / "model.usdc"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"usdc")
        return ProviderToolResult.ok({"status": "success"})

    def axial_runner(
        axial_path: Path, candidate_usd_path: Path, *, source_usd_path=None
    ) -> ProviderToolResult:
        axial_calls.append((axial_path, candidate_usd_path))
        return ProviderToolResult.ok({"text": "recall 0.5", "result": {"recall": 0.5}})

    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=check_runner,
        compile_runner=compile_runner,
        axial_runner=axial_runner,
    )
    names = [schema["name"] for schema in build_reconstruction_tool_schemas()]
    assert names == [
        "read_file",
        "search_docs",
        "find_examples",
        "write_model",
        "apply_patch",
        "check_model",
        "compile_model",
        "check_axial",
        "measure_face",
        "compare_part",
        "render_source",
        "measure_mesh",
        "render_candidate",
        "finish_reconstruction",
        "validate_model",
    ]

    axial = executor.execute("read_file", {"path": "inputs/axial_brief.txt"}).payload
    assert axial["ok"] is True and axial["output"]["path"] == "inputs/axial_brief.txt"
    # The full AXIAL text stays Host-side evidence: the Author reads the derived brief only.
    full = executor.execute("read_file", {"path": "inputs/axial.txt"}).payload
    assert full["ok"] is False
    # Evidence is verbatim: the AXIAL grid keeps its column alignment, no `L<n>:` prefixes.
    assert axial["output"]["content"].startswith("AXIAL BRIEF")
    assert axial["output"]["line_numbers"] is False
    model = executor.execute("read_file", {"path": "model.py"}).payload
    assert model["output"]["content"].startswith("L1: ") and model["output"]["line_numbers"]
    manifest = executor.execute("read_file", {"path": "evidence_manifest.json"}).payload
    assert manifest["ok"] is True and "allowed_evidence" in manifest["output"]["content"]
    denied = executor.execute("read_file", {"path": "inputs/semantic_image.png"}).payload
    assert denied["ok"] is False and "inputs/axial_brief.txt" in denied["error"]

    measured = executor.execute("measure_face", {"face": "-Y", "z_band": [0.0, 0.2]}).payload
    assert measured["ok"] is True and measured["face"] == "-Y"
    assert isinstance(measured["protrusions"], list) and isinstance(measured["body_islands"], list)
    assert isinstance(measured["openings"], list)
    bad_face = executor.execute("measure_face", {"face": "front", "z_band": None}).payload
    assert bad_face["ok"] is False
    early = executor.execute("check_axial", {}).payload
    assert early["ok"] is False and "compile_model" in early["error"]
    assert (
        executor.execute("finish_generation", {"status": "complete", "summary": "x"}).payload["ok"]
        is False
    )

    patch = (
        "*** Begin Patch\n*** Update File: model.py\n@@\n-object_model = build_object_model()\n"
        "+object_model = build_object_model()  # provider edit\n*** End Patch\n"
    )
    assert executor.execute("apply_patch", {"patch": patch}).payload["ok"] is True
    assert executor.execute("check_model", {}).payload["ok"] is True
    assert executor.execute("compile_model", {}).payload["ok"] is True
    reviewed = executor.execute("check_axial", {}).payload
    assert reviewed["ok"] is True and reviewed["axial_call"] == 1
    assert reviewed["output"]["text"].startswith(
        "reference mesh — resolve each numbered discrepancy"
    )
    assert reviewed["output"]["text"].endswith("\nrecall 0.5")
    assert axial_calls == [(workspace.axial_path, workspace.build_dir / "isaac" / "model.usdc")]

    finished = executor.execute(
        "finish_reconstruction", {"status": "complete", "summary": "AXIAL recall 0.5 recorded."}
    )
    assert finished.completion_status == "complete"
    assert finished.completion_summary.startswith("AXIAL recall 0.5 recorded.\nAuthor revision 1")
    assert finished.payload["validation_state"]["operations"]["axial"]["status"] == "current"


@pytest.mark.parametrize("endpoint", ["nvidia", "openai"])
@pytest.mark.parametrize(
    ("guidance", "recorded"),
    [(None, None), ("  Keep the lid closed.\n", "Keep the lid closed.")],
)
def test_persisted_provider_record_has_provider_identity(
    tmp_path: Path, guidance: str | None, recorded: str | None, endpoint: str
) -> None:
    StorageRepo(tmp_path).ensure_layout()
    workspace = tmp_path / "workspace"
    trace_dir = workspace / ".provider-run"
    trace_dir.mkdir(parents=True)
    (workspace / "model.py").write_text("object_model = object()\n", encoding="utf-8")
    (workspace / "TASK.md").write_text("Reconstruct one subject.\n", encoding="utf-8")
    (trace_dir / "events.jsonl").write_text('{"record_type":"provider.run.started"}\n')
    (trace_dir / "final.json").write_text('{"status":"complete","summary":"done"}\n')
    (trace_dir / "stderr.log").write_text("", encoding="utf-8")
    (trace_dir / "timeline.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / "stage_trace.jsonl").write_text(
        '{"event":"stage.finished","stage":"author","status":"ok"}\n', encoding="utf-8"
    )
    runtime = ProviderRuntimeSelection(
        model_id="nvidia-test-model",
        reasoning_effort="high",
        runtime_source=f"{endpoint}-responses-api",
    )

    from usdcraft.reconstruction.harness_policy import resolve_harness_policy

    evidence = _current_approved_evidence(tmp_path)
    policy = resolve_harness_policy()
    report = {"final_status": "complete", "harness": policy.to_dict(), "turn_count": 12}
    context = json.loads(evidence.author_evidence_manifest_path.read_text())
    context["harness"] = policy.to_dict()
    evidence.author_evidence_manifest_path.write_text(json.dumps(context))

    record_dir = persist_reconstruction(
        repo_root=tmp_path,
        run_id="recon_provider_run",
        title="Test Drawer",
        request_text="",
        runtime=runtime,
        workspace_dir=workspace,
        runtime_report=report,
        artifacts=_compile_artifacts(workspace),
        evidence=evidence,
        label=None,
        guidance=guidance,
        authoring=provider_reconstruction_authoring(endpoint),
    )

    assert record_dir.name.startswith("rec_reconstruct-")
    revision = record_dir / "revisions" / "rev_000001"
    record = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    manifest = json.loads((revision / "reconstruction.json").read_text(encoding="utf-8"))
    provenance = json.loads((revision / "provenance.json").read_text(encoding="utf-8"))
    # The Provider-arm identity is the endpoint that served the run.
    assert record["agent_harness"] == "provider" and record["provider"] == endpoint
    assert manifest["creator"]["provider"] == endpoint
    assert provenance["runtime"]["provider"] == endpoint
    assert provenance["runtime"]["runtime_source"] == f"{endpoint}-responses-api"
    assert record["creator"]["agent"] == "provider"
    assert record["source"] == {"run_id": "recon_provider_run"}
    assert manifest["creator"]["agent_harness"] == "provider"
    assert "guidance" in manifest["input"] and manifest["input"]["guidance"] == recorded
    assert provenance["runtime"]["harness"] == policy.to_dict()
    assert (revision / "inputs/author_context/inputs/source_mesh.usdc").is_file()
    assert provenance["runtime"]["agent_harness"] == "provider"
    assert provenance["prompting"]["system_prompt_file"] == "traces/provider_task.md"
    traces = sorted(path.name for path in (revision / "traces").iterdir())
    assert traces == [
        "provider_events.jsonl",
        "provider_final.json",
        "provider_stderr.log",
        "provider_task.md",
        "provider_timeline.jsonl",
        "stage_trace.jsonl",
    ]
    assert validate_data_format(StorageRepo(tmp_path)).errors == []


@pytest.mark.parametrize("endpoint", ["nvidia", "openai"])
def test_execute_provider_reconstruction_runs_the_direct_draft_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    repo, source, semantic, _codex_preflight = _inputs(tmp_path)
    captured: dict[str, object] = {}
    progress: list[str] = []
    config = SimpleNamespace(
        **{
            **vars(_FAKE_CONFIG),
            "endpoint": endpoint,
            "runtime_source": f"{endpoint}-responses-api",
        }
    )

    class _FakeConfig:
        @classmethod
        def from_repo(cls, root, **kwargs):  # type: ignore[no-untyped-def]
            captured["config_kwargs"] = kwargs
            return config

    class _FakeClient:
        def __init__(self, config, cancellation_event=None):  # type: ignore[no-untyped-def]
            captured["client_config"] = config

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["harness_kwargs"] = kwargs
            self.workspace = kwargs["workspace"]

        def run(self, *, prompt, image_paths=None, image_path=None, cache_prefix=None):  # type: ignore[no-untyped-def]
            captured["task"] = prompt
            captured["image_paths"] = tuple(image_paths or ())
            self.workspace.model_path.write_text(
                "object_model = object()\n# provider edit\n", encoding="utf-8"
            )
            return SimpleNamespace(
                status="complete",
                summary="AXIAL recall 0.61 recorded.",
                runtime_report=lambda: {"final_status": "complete", "turn_count": 31},
            )

    def fake_preprocessing(preflight, run, **kwargs):  # type: ignore[no-untyped-def]
        captured["observation_runner"] = kwargs.get("image_observation_runner")
        return ReconstructionPreprocessingResult(
            isolation=None,
            mesh_preprocess=SimpleNamespace(),
            evidence=_evidence(run),
            image_observation=None,
            warnings=("Image observation degraded to the Author-only reading: test",),
        )

    def fake_compile(model_path, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured["host_compile"] = kwargs
        return _compile_artifacts(model_path.parent)

    def fake_persist(**kwargs):  # type: ignore[no-untyped-def]
        captured["persist"] = kwargs
        record = repo / "data" / "records" / "rec_provider_recon"
        model = record / "revisions" / "rev_000001" / "model.py"
        model.parent.mkdir(parents=True)
        model.write_text("# canonical\n", encoding="utf-8")
        return record

    module = "usdcraft.reconstruction.provider_command"
    monkeypatch.setattr(f"{module}.NvidiaProviderConfig", _FakeConfig)
    monkeypatch.setattr(f"{module}.NvidiaResponsesClient", _FakeClient)
    monkeypatch.setattr(f"{module}.ProviderHarness", _FakeHarness)
    monkeypatch.setattr(f"{module}._build_reconstruction_preprocessing", fake_preprocessing)
    monkeypatch.setattr(f"{module}.compile_model", fake_compile)
    monkeypatch.setattr(f"{module}.persist_reconstruction", fake_persist)
    monkeypatch.setattr(
        f"{module}.recover_pending_promotions", lambda _repo: SimpleNamespace(errors=())
    )

    result = execute_provider_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        meters_per_unit=1.0,
        guidance="Keep the user-visible latch.",
        model="nvidia-test-model",
        reasoning="high",
        progress=progress.append,
    )

    assert result.status == "reconstruction_ready"
    assert result.record_id == "rec_provider_recon"
    assert result.workspace_dir is None
    assert result.provider == endpoint and result.to_dict()["provider"] == endpoint
    assert "nvidia-test-model" in progress[0]
    harness_kwargs = captured["harness_kwargs"]
    assert isinstance(harness_kwargs, dict)
    assert harness_kwargs["workspace"].root.parent.name == "reconstruction_author_runs"
    assert harness_kwargs["workspace"].root.name == result.run_id
    assert harness_kwargs["max_turns"] == 80
    assert captured["persist"]["guidance"] == "Keep the user-visible latch."
    assert captured["persist"]["runtime_report"]["author_validation"] == (
        harness_kwargs["tool_executor"].validation_state()
    )
    assert harness_kwargs["progress_label"] == "Provider Reconstruction"
    names = {schema["name"] for schema in harness_kwargs["tool_schemas"]}
    assert {"validate_model", "measure_face", "finish_reconstruction"} <= names
    assert {
        "measure_mesh",
        "render_candidate",
        "compare_part",
        "check_model",
        "compile_model",
        "check_axial",
    } <= names
    image_paths = captured["image_paths"]
    # Source rendering is requested by the Author; only the semantic image is attached initially.
    assert isinstance(image_paths, tuple) and len(image_paths) == 1
    assert image_paths[0].name.startswith("semantic_image")
    assert "render_source" in names
    task = str(captured["task"])
    assert "FREE DESIGN ORDER" in task
    assert "Keep the user-visible latch." in task
    assert callable(captured["observation_runner"])
    assert captured["host_compile"] == {
        "clean": True,
        "render": False,
        "require_generation_root_floating": True,
    }
    persisted = captured["persist"]
    assert isinstance(persisted, dict)
    assert persisted["authoring"] == provider_reconstruction_authoring(endpoint)
    assert persisted["authoring"].provider == endpoint
    if endpoint == "nvidia":
        assert persisted["authoring"] is NVIDIA_PROVIDER_RECONSTRUCTION_AUTHORING
    assert persisted["runtime"].to_dict()["model_id"] == "nvidia-test-model"
    assert persisted["runtime"].to_dict()["runtime_source"] == f"{endpoint}-responses-api"
    report = persisted["runtime_report"]
    assert report["image_observation"] == "author_only"
    assert report["harness"]["max_turns"] == 80
    assert report["preloaded_docs"] == list(PRELOADED_SDK_DOC_PATHS)
    assert "PRELOADED SDK REFERENCES" in task
    assert "PUBLIC SDK REFERENCES" in task and "AUTHOR TOOLS" in task
    assert report["turn_count"] == 31
    assert [entry["stage"] for entry in report["stage_timings"]] == [
        "freeze",
        "author",
        "host_gate",
    ]
    assert any("degraded to the Author-only reading" in warning for warning in result.warnings)
    assert not (repo / "data" / "local" / "reconstruction_runs").exists() or not any(
        (repo / "data" / "local" / "reconstruction_runs").iterdir()
    )


def test_blocked_provider_author_retains_the_workspace_for_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, semantic, _codex_preflight = _inputs(tmp_path)

    class _FakeConfig:
        @classmethod
        def from_repo(cls, root, **kwargs):  # type: ignore[no-untyped-def]
            return _FAKE_CONFIG

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            pass

        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(status="blocked", summary="unresolved evidence")

    module = "usdcraft.reconstruction.provider_command"
    monkeypatch.setattr(f"{module}.NvidiaProviderConfig", _FakeConfig)
    monkeypatch.setattr(
        f"{module}.NvidiaResponsesClient", lambda config, cancellation_event=None: None
    )
    monkeypatch.setattr(f"{module}.ProviderHarness", _FakeHarness)
    monkeypatch.setattr(
        f"{module}._build_reconstruction_preprocessing",
        lambda preflight, run, **kwargs: ReconstructionPreprocessingResult(
            isolation=None,
            mesh_preprocess=SimpleNamespace(),
            evidence=_evidence(run),
            image_observation=None,
            warnings=(),
        ),
    )
    monkeypatch.setattr(
        f"{module}.recover_pending_promotions", lambda _repo: SimpleNamespace(errors=())
    )

    with pytest.raises(ReconstructionBlockedError) as excinfo:
        execute_provider_reconstruction(
            repo_root=repo,
            source=source,
            semantic_image=semantic,
            up_axis="+Z",
            meters_per_unit=1.0,
        )

    context = reconstruction_failure_context(excinfo.value)
    assert context is not None and context.workspace_dir.is_dir()
    assert context.workspace_dir.name == context.run_id
    assert context.workspace_dir.parent.name == "reconstruction_author_runs"
    assert (context.workspace_dir / "model.py").is_file()
    assert not (repo / "data" / "records").exists() or not list(
        (repo / "data" / "records").glob("rec_*")
    )


def test_cli_reconstruction_dispatches_to_the_provider_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from usdcraft.reconstruction.provider_command import ProviderReconstructionResult

    repo = tmp_path / "repo"
    repo.mkdir()
    captured: dict[str, object] = {}

    def fake_execute(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return ProviderReconstructionResult(
            status="reconstruction_ready",
            run_id="recon_x",
            record_id="rec_x",
            record_dir=repo / "data" / "records" / "rec_x",
            model_path=repo
            / "data"
            / "records"
            / "rec_x"
            / "revisions"
            / "rev_000001"
            / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_command.execute_provider_reconstruction", fake_execute
    )

    assert (
        cli_main(
            [
                "reconstruction",
                str(repo / "source.glb"),
                "--image",
                str(repo / "semantic.png"),
                "--front-axis=-Y",
                "--label",
                "Lamp",
                "--repo-root",
                str(repo),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "reconstruction_ready"
    assert payload["provider"] == "openai"
    assert captured["front_axis"] == "-Y" and captured["label"] == "Lamp"
    # Mesh cleanup is optional on both arms and off by default.
    assert captured["mesh_cleanup"] is False
    assert "confirm_reconstruction" not in captured
    # Only OPENAI_API_KEY in `.env` selects the OpenAI endpoint, which names the run's provider.
    (repo / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    reconstruction_args = [
        "reconstruction",
        str(repo / "source.glb"),
        "--image",
        str(repo / "semantic.png"),
        "--repo-root",
        str(repo),
    ]
    assert cli_main([*reconstruction_args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "openai"
    assert cli_main(reconstruction_args) == 0
    assert "provider=openai" in capsys.readouterr().out.splitlines()
    (repo / ".env").unlink()

    captured.clear()
    assert (
        cli_main(
            [
                "reconstruction",
                str(repo / "source.glb"),
                "--image",
                str(repo / "semantic.png"),
                "--mesh-cleanup",
                "--repo-root",
                str(repo),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured["mesh_cleanup"] is True

    # The Codex arm keeps its own two-turn command under `reconstruction-codex`.
    with pytest.raises(SystemExit) as codex_help:
        cli_main(["reconstruction-codex", "--help"])
    assert codex_help.value.code == 0
    assert "--confirm-reconstruction" in capsys.readouterr().out


def test_reconstruction_progress_lines_survive_the_shared_display_sanitizer() -> None:
    from usdcraft.pipeline.display_common import (
        provider_message_fields,
        safe_provider_message,
    )
    from usdcraft.pipeline.single_display import _SingleState

    starting = (
        "Provider Reconstruction | activity=starting | model='nvidia-test-model' | "
        "effort='high' | max_turns=80"
    )
    assert safe_provider_message(starting).startswith("Provider Reconstruction | ")
    assert provider_message_fields(starting)["model"] == "'nvidia-test-model'"
    state = _SingleState()
    state.apply(SimpleNamespace(message=starting, timestamp=__import__("datetime").datetime.now()))
    assert state.model == "nvidia-test-model" and state.max_turns == 80
    state.apply(
        SimpleNamespace(
            message="Provider Reconstruction | activity=completed | status=reconstruction_ready",
            timestamp=__import__("datetime").datetime.now(),
        )
    )
    assert state.status == "success"
    assert safe_provider_message("Codex | activity=starting") == (
        'Provider Generation | activity="running"'
    )


def test_provider_estimates_scale_for_a_unit_normalized_source_before_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unitless mesh on a normalizer's 2.0 target gets its metres from one image estimate; the
    request, the frozen source frame, the Author task and the runtime report all record it."""

    from usdcraft.reconstruction.scale_estimate import ScaleEstimate, ScaleInspection

    repo, source, semantic, _codex_preflight = _inputs(tmp_path)
    captured: dict[str, object] = {}
    progress: list[str] = []

    class _FakeConfig:
        @classmethod
        def from_repo(cls, root, **kwargs):  # type: ignore[no-untyped-def]
            return _FAKE_CONFIG

    class _FakeClient:
        def __init__(self, config, cancellation_event=None):  # type: ignore[no-untyped-def]
            pass

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.workspace = kwargs["workspace"]

        def run(self, *, prompt, image_paths=None, image_path=None, cache_prefix=None):  # type: ignore[no-untyped-def]
            captured["task"] = prompt
            self.workspace.model_path.write_text(
                "object_model = object()\n# provider edit\n", encoding="utf-8"
            )
            return SimpleNamespace(
                status="complete",
                summary="done",
                runtime_report=lambda: {"final_status": "complete", "turn_count": 9},
            )

    def fake_preprocessing(preflight, run, **kwargs):  # type: ignore[no-untyped-def]
        captured["preflight_request"] = preflight.request
        return ReconstructionPreprocessingResult(
            isolation=None,
            mesh_preprocess=SimpleNamespace(),
            evidence=_evidence(run),
            image_observation=None,
            warnings=(),
        )

    def fake_persist(**kwargs):  # type: ignore[no-untyped-def]
        captured["persist"] = kwargs
        record = repo / "data" / "records" / "rec_scaled"
        model = record / "revisions" / "rev_000001" / "model.py"
        model.parent.mkdir(parents=True)
        model.write_text("# canonical\n", encoding="utf-8")
        return record

    def fake_inspect(source, **kwargs):  # type: ignore[no-untyped-def]
        captured["inspected"] = (source, kwargs)
        return ScaleInspection(
            longest_extent_units=1.995,
            meters_per_unit_source="format_contract",
            suspected_unit_normalized=True,
            matched_target=2.0,
        )

    def fake_estimate(**kwargs):  # type: ignore[no-untyped-def]
        captured["estimate_kwargs"] = kwargs
        return ScaleEstimate(
            category="microwave oven",
            longest_dimension_m=0.5,
            confidence="high",
            rationale="typical countertop width",
            model="nvidia-test-model",
            reasoning_effort="high",
        )

    module = "usdcraft.reconstruction.provider_command"
    monkeypatch.setattr(f"{module}.NvidiaProviderConfig", _FakeConfig)
    monkeypatch.setattr(f"{module}.NvidiaResponsesClient", _FakeClient)
    monkeypatch.setattr(f"{module}.ProviderHarness", _FakeHarness)
    monkeypatch.setattr(f"{module}._build_reconstruction_preprocessing", fake_preprocessing)
    monkeypatch.setattr(f"{module}.compile_model", lambda p, d, **k: _compile_artifacts(p.parent))
    monkeypatch.setattr(f"{module}.persist_reconstruction", fake_persist)
    monkeypatch.setattr(f"{module}._source_scale_inspection", fake_inspect)
    monkeypatch.setattr(f"{module}.estimate_subject_scale", fake_estimate)
    monkeypatch.setattr(
        f"{module}.recover_pending_promotions", lambda _repo: SimpleNamespace(errors=())
    )

    result = execute_provider_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        prompt="Reconstruct the microwave oven shown in the semantic image and the source mesh.",
        model="nvidia-test-model",
        reasoning="high",
        progress=progress.append,
    )

    assert result.status == "reconstruction_ready"
    assert captured["inspected"][1]["meters_per_unit"] is None  # type: ignore[index]
    estimate_kwargs = captured["estimate_kwargs"]
    assert isinstance(estimate_kwargs, dict)
    assert estimate_kwargs["request_text"].startswith("Reconstruct the microwave oven")
    assert estimate_kwargs["semantic_image_path"] == semantic.resolve()
    request = captured["preflight_request"]
    assert request.meters_per_unit == pytest.approx(0.5 / 1.995)
    assert request.meters_per_unit_source == "estimated_from_semantic_image"
    assert request.scale_estimate["category"] == "microwave oven"
    assert request.to_dict()["source"]["meters_per_unit_source"] == "estimated_from_semantic_image"
    task = captured["task"]
    assert isinstance(task, str)
    assert "\nSCALE\n" in task and "microwave oven, longest side 0.50 m" in task
    assert "do not rescale the body yourself" in task
    report = captured["persist"]["runtime_report"]["scale"]  # type: ignore[index]
    assert report["source"] == "estimated_from_semantic_image"
    assert report["meters_per_unit"] == pytest.approx(0.5 / 1.995)
    assert report["estimate"]["confidence"] == "high"
    assert any('activity="estimating scale"' in line for line in progress)
    assert any(
        'activity="scale estimated"' in line and "meters_per_unit 0.2506" in line
        for line in progress
    )
    assert not any("Scale estimate failed" in warning for warning in result.warnings)


def test_provider_keeps_metres_with_a_warning_when_the_scale_estimate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trimesh

    from usdcraft.reconstruction.scale_estimate import ScaleEstimateError, ScaleInspection

    repo, _obj_source, semantic, _codex_preflight = _inputs(tmp_path)
    # A GLB resolves its frame from the format contract, so the run proceeds without a declared
    # meters_per_unit exactly as a unit-normalized scan would.
    source = tmp_path / "normalized_source.glb"
    trimesh.creation.box(extents=(2.0, 1.0, 0.8)).export(source)
    captured: dict[str, object] = {}

    class _FakeConfig:
        @classmethod
        def from_repo(cls, root, **kwargs):  # type: ignore[no-untyped-def]
            return _FAKE_CONFIG

    class _FakeHarness:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.workspace = kwargs["workspace"]

        def run(self, *, prompt, image_paths=None, image_path=None, cache_prefix=None):  # type: ignore[no-untyped-def]
            captured["task"] = prompt
            self.workspace.model_path.write_text("object_model = object()\n", encoding="utf-8")
            return SimpleNamespace(
                status="complete", summary="done", runtime_report=lambda: {"turn_count": 3}
            )

    def fake_preprocessing(preflight, run, **kwargs):  # type: ignore[no-untyped-def]
        captured["preflight_request"] = preflight.request
        return ReconstructionPreprocessingResult(
            isolation=None,
            mesh_preprocess=SimpleNamespace(),
            evidence=_evidence(run),
            image_observation=None,
            warnings=(),
        )

    def fake_persist(**kwargs):  # type: ignore[no-untyped-def]
        captured["persist"] = kwargs
        record = repo / "data" / "records" / "rec_unscaled"
        (record / "revisions" / "rev_000001").mkdir(parents=True)
        (record / "revisions" / "rev_000001" / "model.py").write_text("#\n", encoding="utf-8")
        return record

    def fail_estimate(**kwargs):  # type: ignore[no-untyped-def]
        raise ScaleEstimateError("provider scale-estimate call failed: offline")

    module = "usdcraft.reconstruction.provider_command"
    monkeypatch.setattr(f"{module}.NvidiaProviderConfig", _FakeConfig)
    monkeypatch.setattr(
        f"{module}.NvidiaResponsesClient", lambda config, cancellation_event=None: None
    )
    monkeypatch.setattr(f"{module}.ProviderHarness", _FakeHarness)
    monkeypatch.setattr(f"{module}._build_reconstruction_preprocessing", fake_preprocessing)
    monkeypatch.setattr(f"{module}.compile_model", lambda p, d, **k: _compile_artifacts(p.parent))
    monkeypatch.setattr(f"{module}.persist_reconstruction", fake_persist)
    monkeypatch.setattr(
        f"{module}._source_scale_inspection",
        lambda source, **kwargs: ScaleInspection(2.0, "format_contract", True, 2.0),
    )
    monkeypatch.setattr(f"{module}.estimate_subject_scale", fail_estimate)
    monkeypatch.setattr(
        f"{module}.recover_pending_promotions", lambda _repo: SimpleNamespace(errors=())
    )

    result = execute_provider_reconstruction(
        repo_root=repo,
        source=source,
        semantic_image=semantic,
        up_axis="+Z",
        front_axis="-Y",
        model="nvidia-test-model",
        reasoning="high",
    )

    request = captured["preflight_request"]
    assert request.meters_per_unit is None and request.meters_per_unit_source is None
    assert "Use the staged source metre scale" in captured["task"]
    assert any(warning.startswith("Scale estimate failed") for warning in result.warnings)
    report = captured["persist"]["runtime_report"]["scale"]  # type: ignore[index]
    assert report["source"] == "format_contract" and "estimate" not in report


@pytest.mark.parametrize("axial_ok,render_ok", [(True, True), (False, True), (True, False)])
def test_validate_model_distinguishes_diagnostic_failure_from_mechanical_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axial_ok: bool, render_ok: bool
) -> None:
    from usdcraft.pipeline.tools import ProviderToolExecution

    *_, workspace = _provider_workspace(tmp_path)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda *_: ProviderToolResult.ok({}),
        compile_runner=lambda *_, **__: ProviderToolResult.ok({}),
    )
    calls = []

    def operation(name, ok):
        def run(_arguments):
            calls.append(name)
            return {"ok": ok}

        return run

    monkeypatch.setattr(executor, "_check_model", operation("check", True))
    monkeypatch.setattr(executor, "_compile_model", operation("compile", True))
    monkeypatch.setattr(executor, "_check_axial", operation("axial", axial_ok))
    monkeypatch.setattr(
        executor,
        "_author_feedback",
        lambda *_: ProviderToolExecution(operation("render", render_ok)({})),
    )
    result = executor.execute("validate_model", {}).payload
    assert calls == ["check", "compile", "axial", "render"]
    assert result["ok"] is True
    assert result["diagnostics_complete"] is (axial_ok and render_ok)
    assert result["axial"]["ok"] is axial_ok
    assert result["render"]["ok"] is render_ok


@pytest.mark.parametrize("edit_tool", ["apply_patch", "write_model"])
def test_validation_coverage_exposes_stale_feedback_without_new_completion_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit_tool: str
) -> None:
    from usdcraft.pipeline.tools import ProviderToolExecution

    *_, workspace = _provider_workspace(tmp_path)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda *_: ProviderToolResult.ok({}),
        compile_runner=lambda *_, **__: ProviderToolResult.ok({}),
    )
    monkeypatch.setattr(
        executor, "_check_axial", lambda _: {"ok": True, "revision": executor.revision}
    )
    monkeypatch.setattr(
        executor,
        "_author_feedback",
        lambda *_: ProviderToolExecution({"ok": True, "revision": executor.revision}),
    )
    patch = (
        "*** Begin Patch\n*** Update File: model.py\n@@\n-object_model = build_object_model()\n"
        "+object_model = build_object_model()  # first\n*** End Patch\n"
    )
    initial = workspace.model_path.read_text()
    if edit_tool == "write_model":
        executor.execute(edit_tool, {"content": initial + "# first\n"})
    else:
        executor.execute(edit_tool, {"patch": patch})
    checked = executor.execute("validate_model", {}).payload
    assert all(v["status"] == "current" for v in checked["validation_state"]["operations"].values())
    if edit_tool == "write_model":
        noop = executor.execute(edit_tool, {"content": workspace.model_path.read_text()}).payload
        assert noop["output"]["changed"] is False
        assert noop["validation_state"] == checked["validation_state"]
    second = patch.replace(
        "-object_model = build_object_model()\n", "-object_model = build_object_model()  # first\n"
    ).replace(
        "+object_model = build_object_model()  # first",
        "+object_model = build_object_model()  # second",
    )
    changed = executor.execute(
        edit_tool,
        {"content": initial + "# second\n"} if edit_tool == "write_model" else {"patch": second},
    ).payload
    assert all(v["status"] == "stale" for v in changed["validation_state"]["operations"].values())
    assert (
        executor.execute(
            "finish_reconstruction", {"status": "complete", "summary": "too early"}
        ).completion_status
        is None
    )
    # Mechanical completion remains possible, but older images/AXIAL are disclosed.
    executor.execute("check_model", {})
    executor.execute("compile_model", {})
    finished = executor.execute(
        "finish_reconstruction", {"status": "complete", "summary": "Built."}
    )
    assert finished.completion_status == "complete"
    assert "axial=stale (rev 1)" in finished.completion_summary
    assert "render=stale (rev 1)" in finished.completion_summary
    assert "compile=current (rev 2)" in finished.completion_summary
    # A failed latest diagnostic is not misrepresented by an earlier successful result.
    monkeypatch.setattr(
        executor, "_check_axial", lambda _: {"ok": False, "error": "fixture failure"}
    )
    failed = executor.execute("validate_model", {}).payload
    assert failed["ok"] and not failed["diagnostics_complete"]
    assert failed["validation_state"]["operations"]["axial"]["status"] == "failed"


def test_whole_model_creation_replaces_scaffold_then_accepts_local_patch(tmp_path: Path) -> None:
    *_, workspace = _provider_workspace(tmp_path)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda *_: ProviderToolResult.ok({}),
        compile_runner=lambda *_, **__: ProviderToolResult.ok({}),
    )
    content = (
        "def build_object_model():\n    return 1\n\n"
        "object_model = build_object_model()\n\n"
        "def run_tests():\n    return []\n"
    )
    # An unanchored first patch must remain recoverable with no partial source mutation.
    scaffold = workspace.model_path.read_text()
    failed = executor.execute(
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Update File: model.py\n@@\n+value = 1\n*** End Patch"},
    ).payload
    assert not failed["ok"] and not failed["changed"]
    assert workspace.model_path.read_text() == scaffold
    assert failed["validation_state"]["model_revision"] == 0
    assert "write_model" in failed["repair_hint"]
    written = executor.execute("write_model", {"content": content}).payload
    assert written["ok"] and written["validation_state"]["model_revision"] == 1
    assert workspace.model_path.read_text() == content
    assert content.count("def build_object_model") == 1
    local = executor.execute(
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Update File: model.py\n@@\n-    return 1\n+    return 2\n*** End Patch"
        },
    ).payload
    assert local["ok"] and local["validation_state"]["model_revision"] == 2
    assert workspace.model_path.read_text() == content.replace("return 1", "return 2")
    schemas = build_reconstruction_tool_schemas()
    assert sum(t["name"] == "write_model" for t in schemas) == 1


def test_validation_coverage_marks_missing_checks(tmp_path: Path) -> None:
    *_, workspace = _provider_workspace(tmp_path)
    executor = ReconstructionToolExecutor(
        workspace,
        check_runner=lambda *_: ProviderToolResult.ok({}),
        compile_runner=lambda *_, **__: ProviderToolResult.ok({}),
    )
    state = executor.validation_state()["operations"]
    assert state["render"] == {"revision": None, "status": "not_run"}
    assert state["axial"] == {"revision": None, "status": "not_run"}
