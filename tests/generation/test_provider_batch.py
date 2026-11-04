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

import csv
import threading
import time
from pathlib import Path

import pytest

from usdcraft.generation.command import GenerationResult
from usdcraft.generation.provider_batch import (
    MAX_PROVIDER_BATCH_CONCURRENCY,
    ProviderBatchInterrupted,
    execute_provider_batch,
    load_provider_batch,
)
from usdcraft.pipeline.batch_progress import ProviderBatchProgressEvent


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["row_id", "prompt", "image_path", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_batch_accepts_text_image_and_image_only_rows(tmp_path: Path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"image")
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "text", "prompt": "a lamp", "image_path": "", "label": ""},
            {
                "row_id": "image",
                "prompt": "",
                "image_path": "reference.png",
                "label": "reference object",
            },
            {
                "row_id": "both",
                "prompt": "a red lamp",
                "image_path": str(image),
                "label": "",
            },
        ],
    )

    rows = load_provider_batch(csv_path)

    assert [row.row_id for row in rows] == ["text", "image", "both"]
    assert rows[0].prompt == "a lamp" and rows[0].image_path is None
    assert rows[1].prompt is None and rows[1].image_path == image.resolve()
    assert rows[2].prompt == "a red lamp" and rows[2].image_path == image.resolve()


def test_batch_preserves_quoted_multiline_prompt(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {
                "row_id": "multiline",
                "prompt": "Build a lamp\nwith two articulated arms.",
                "image_path": "",
                "label": "",
            }
        ],
    )

    rows = load_provider_batch(csv_path)

    assert rows[0].prompt == "Build a lamp\nwith two articulated arms."


def test_batch_validation_reports_record_and_multiline_physical_end_line(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        'row_id,prompt,image_path\n,"Build a lamp\nwith two arms",\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_provider_batch(csv_path)

    message = str(exc_info.value)
    assert "data record 1" in message
    assert "ending at physical line 3" in message
    assert "empty row_id" in message


@pytest.mark.parametrize("column", ["reasoning_effort", "promt"])
def test_batch_rejects_unknown_columns_and_lists_the_allowed_set(
    tmp_path: Path, column: str
) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(f"row_id,prompt,image_path,{column}\none,a lamp,,x\n")
    with pytest.raises(ValueError) as exc_info:
        load_provider_batch(csv_path)
    message = str(exc_info.value)
    assert f"unknown columns: {column}." in message
    assert (
        "Allowed columns: row_id, prompt, image_path, label, model, reasoning, api_style."
        in message
    )
    # The model/reasoning pairing is mentioned only by the actual pairing error.
    assert "together" not in message


def test_batch_rejects_extra_csv_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\none,a lamp,,surplus\n")
    with pytest.raises(ValueError, match="more cells"):
        load_provider_batch(csv_path)


@pytest.mark.parametrize("pair", ["test-model-a,", ",low"])
def test_batch_runtime_override_must_be_a_pair(tmp_path: Path, pair: str) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(f"row_id,prompt,image_path,model,reasoning\none,a lamp,,{pair}\n")
    with pytest.raises(ValueError, match="model and reasoning must be provided together"):
        load_provider_batch(csv_path)


def test_batch_rejects_an_unknown_api_style(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path,api_style\none,a lamp,,grpc\n")
    with pytest.raises(ValueError, match="invalid api_style 'grpc'"):
        load_provider_batch(csv_path)


def test_batch_binds_row_runtime_and_api_style(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "row_id,prompt,image_path,label,model,reasoning,api_style\n"
        "a,a lamp,,Lamp,azure/openai/test-model-b,low,chat-completions\n"
        "b,a chair,,,,,\n"
    )
    first, second = load_provider_batch(csv_path)
    assert (first.model, first.reasoning, first.api_style) == (
        "azure/openai/test-model-b",
        "low",
        "chat-completions",
    )
    assert (second.model, second.reasoning, second.api_style) == (None, None, None)


def test_batch_rejects_duplicate_column_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "row_id,prompt,image_path,prompt\none,a lamp,,ignored\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate column names: 'prompt'"):
        load_provider_batch(csv_path)


def test_batch_defers_missing_image_validation_to_each_shared_primitive(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {
                "row_id": "missing_image",
                "prompt": "",
                "image_path": "missing.png",
                "label": "",
            }
        ],
    )

    rows = load_provider_batch(csv_path)

    assert rows[0].image_path == (tmp_path / "missing.png").resolve()
    assert not rows[0].image_path.is_file()


def test_batch_uses_provider_single_primitive_and_preserves_csv_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "first", "prompt": "one", "image_path": "", "label": ""},
            {"row_id": "second", "prompt": "two", "image_path": "", "label": ""},
        ],
    )
    calls: list[dict[str, object]] = []
    progress_events: list[ProviderBatchProgressEvent] = []

    def fake_execute_provider_generation(**kwargs):
        prompt = str(kwargs["prompt"])
        calls.append(dict(kwargs))
        kwargs["progress"](
            'Provider Generation | turn=1/64 | activity="running final Host compile"'
        )
        record_dir = tmp_path / f"rec_{prompt}"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        concurrency=2,
        progress=progress_events.append,
    )

    assert sorted(str(call["prompt"]) for call in calls) == ["one", "two"]
    assert all(callable(call["progress"]) for call in calls)
    assert [row.row_id for row in result.rows] == ["first", "second"]
    assert result.status == "complete"
    assert result.succeeded == 2
    assert progress_events[0].kind == "batch_started"
    assert progress_events[0].total == 2
    assert progress_events[0].effective_concurrency == 2
    assert progress_events[-1].kind == "batch_finished"
    assert progress_events[-1].succeeded == 2
    assert sum(event.kind == "row_started" for event in progress_events) == 2
    assert sum(event.kind == "row_progress" for event in progress_events) == 2
    assert sum(event.kind == "row_finished" for event in progress_events) == 2


def test_batch_reports_the_openai_endpoint_and_its_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only OPENAI_API_KEY selects the OpenAI endpoint; the batch summary names it.
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-test\nOPENAI_MODEL=openai-test-model\n", encoding="utf-8"
    )
    csv_path = tmp_path / "jobs.csv"
    _write_csv(csv_path, [{"row_id": "one", "prompt": "one", "image_path": "", "label": ""}])
    progress_events: list[ProviderBatchProgressEvent] = []

    def fake_execute_provider_generation(**kwargs):
        record_dir = tmp_path / "rec_one"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
            provider="openai",
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        concurrency=1,
        progress=progress_events.append,
    )

    assert result.provider == "openai" and result.to_dict()["provider"] == "openai"
    assert progress_events[0].kind == "batch_started"
    assert progress_events[0].model == "openai-test-model"


def test_batch_reuses_fixed_slots_without_exceeding_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "first", "prompt": "one", "image_path": "", "label": ""},
            {"row_id": "second", "prompt": "two", "image_path": "", "label": ""},
            {"row_id": "third", "prompt": "three", "image_path": "", "label": ""},
            {"row_id": "fourth", "prompt": "four", "image_path": "", "label": ""},
        ],
    )
    first_pair_ready = threading.Barrier(2)
    active = 0
    maximum_active = 0
    active_lock = threading.Lock()
    events: list[ProviderBatchProgressEvent] = []

    def fake_execute_provider_generation(**kwargs):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        if kwargs["prompt"] in {"one", "two"}:
            first_pair_ready.wait(timeout=2)
        kwargs["progress"]('Provider Generation | activity="authoring"')
        with active_lock:
            active -= 1
        record_dir = tmp_path / f"rec_{kwargs['prompt']}"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        concurrency=2,
        progress=events.append,
    )

    started = [event for event in events if event.kind == "row_started"]
    assert maximum_active == 2
    assert len(started) == 4
    assert {event.slot for event in started} == {1, 2}
    assert [row.row_id for row in result.rows] == ["first", "second", "third", "fourth"]


def test_batch_progress_callback_failure_is_presentation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [{"row_id": "one", "prompt": "lamp", "image_path": "", "label": ""}],
    )

    def fake_execute_provider_generation(**kwargs):
        kwargs["progress"]('Provider Generation | activity="authoring"')
        record_dir = tmp_path / "rec_one"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    def broken_display(_event: ProviderBatchProgressEvent) -> None:
        raise RuntimeError("display unavailable")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        progress=broken_display,
    )

    assert result.status == "complete"


def test_batch_projects_retained_success_workspace_and_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [{"row_id": "kept", "prompt": "one", "image_path": "", "label": ""}],
    )
    workspace = tmp_path / "data" / "local" / "generation_runs" / "gen_kept"

    def fake_execute_provider_generation(**kwargs):
        assert kwargs["keep_workspace"] is True
        record_dir = tmp_path / "rec_one"
        return GenerationResult(
            status="generation_ready",
            record_id="rec_one",
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=workspace,
            warnings=("cleanup pending",),
            run_id="gen_explicit",
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        keep_workspace=True,
    )

    assert result.rows[0].run_id == "gen_explicit"
    assert result.rows[0].workspace_dir == workspace
    assert result.rows[0].warnings == ("cleanup pending",)
    assert result.to_dict()["rows"][0]["warnings"] == ["cleanup pending"]


def test_batch_isolates_row_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "good", "prompt": "good", "image_path": "", "label": ""},
            {"row_id": "bad", "prompt": "bad", "image_path": "", "label": ""},
        ],
    )

    def fake_execute_provider_generation(**kwargs):
        if kwargs["prompt"] == "bad":
            error = RuntimeError("row failed")
            error.generation_run_id = "gen_failed_row"  # type: ignore[attr-defined]
            error.generation_workspace_dir = tmp_path / "failed_workspace"  # type: ignore[attr-defined]
            raise error
        record_dir = tmp_path / "rec_good"
        return GenerationResult(
            status="generation_ready",
            record_id="rec_good",
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(csv_path=csv_path, repo_root=tmp_path)

    assert result.status == "partial_failure"
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.rows[1].error == "RuntimeError: row failed"
    assert result.rows[1].run_id == "gen_failed_row"
    assert result.rows[1].workspace_dir == tmp_path / "failed_workspace"


def test_batch_isolates_worker_local_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "exit", "prompt": "exit", "image_path": "", "label": ""},
            {"row_id": "good", "prompt": "good", "image_path": "", "label": ""},
        ],
    )

    def fake_execute_provider_generation(**kwargs):
        if kwargs["prompt"] == "exit":
            raise SystemExit(17)
        record_dir = tmp_path / "rec_good"
        return GenerationResult(
            status="generation_ready",
            record_id="rec_good",
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
            run_id="gen_good",
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    result = execute_provider_batch(csv_path=csv_path, repo_root=tmp_path)

    assert result.status == "partial_failure"
    assert result.rows[0].error == "SystemExit: 17"
    assert result.rows[1].record_id == "rec_good"
    assert result.rows[1].run_id == "gen_good"


def test_batch_keyboard_interrupt_returns_immediately_with_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import usdcraft.pipeline.batch as provider_batch_module

    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "first", "prompt": "first", "image_path": "", "label": ""},
            {"row_id": "running", "prompt": "running", "image_path": "", "label": ""},
            {"row_id": "not_started", "prompt": "third", "image_path": "", "label": ""},
        ],
    )
    calls: list[str] = []
    running_started = threading.Event()
    cancellation_seen = threading.Event()
    running_finished = threading.Event()

    def fake_execute_provider_generation(**kwargs):
        prompt = str(kwargs["prompt"])
        calls.append(prompt)
        if prompt == "running":
            running_started.set()
            cancellation_event = kwargs["cancellation_event"]
            assert isinstance(cancellation_event, threading.Event)
            # Far longer than the elapsed bound below: a batch that blocked on this in-flight
            # row instead of returning would exceed it.
            if cancellation_event.wait(timeout=10):
                cancellation_seen.set()
            running_finished.set()
        record_dir = tmp_path / f"rec_{prompt}"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
            run_id=f"gen_{prompt}",
        )

    real_wait = provider_batch_module.wait
    wait_calls = 0

    def interrupt_second_wait(futures, *, return_when):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return real_wait(futures, return_when=return_when)
        assert running_started.wait(timeout=1)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )
    monkeypatch.setattr(provider_batch_module, "wait", interrupt_second_wait)

    started_at = time.monotonic()
    try:
        with pytest.raises(ProviderBatchInterrupted) as exc_info:
            execute_provider_batch(
                csv_path=csv_path,
                repo_root=tmp_path,
                concurrency=1,
            )
        elapsed = time.monotonic() - started_at
        interruption = exc_info.value
        assert elapsed < 2
        assert [row.row_id for row in interruption.partial_result.rows] == ["first"]
        assert interruption.partial_result.rows[0].run_id == "gen_first"
        assert interruption.cancelled_row_ids == ("not_started",)
        assert interruption.in_flight_row_ids == ("running",)
        assert calls == ["first", "running"]
        assert interruption.to_dict() == {
            "status": "interrupted",
            "provider": "openai",
            "csv_path": str(csv_path.resolve()),
            "total": 3,
            "completed": 1,
            "succeeded": 1,
            "failed": 0,
            "cancelled": 1,
            "in_flight": 1,
            "rows": [interruption.partial_result.rows[0].to_dict()],
            "cancelled_row_ids": ["not_started"],
            "in_flight_row_ids": ["running"],
            "error": str(interruption),
        }
    finally:
        assert cancellation_seen.wait(timeout=1)
        assert running_finished.wait(timeout=1)


def test_batch_rejects_concurrency_above_absolute_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [{"row_id": "one", "prompt": "lamp", "image_path": "", "label": ""}],
    )

    with pytest.raises(
        ValueError,
        match=rf"integer from 1 to {MAX_PROVIDER_BATCH_CONCURRENCY}",
    ):
        execute_provider_batch(
            csv_path=csv_path,
            repo_root=tmp_path,
            concurrency=MAX_PROVIDER_BATCH_CONCURRENCY + 1,
        )


def test_batch_projects_empty_item_to_the_shared_primitive_as_a_row_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "jobs.csv"
    _write_csv(
        csv_path,
        [{"row_id": "empty", "prompt": "", "image_path": "", "label": ""}],
    )

    def reject_empty(**kwargs):
        assert kwargs["prompt"] is None
        assert kwargs["image"] is None
        raise ValueError("Generation requires an explicit text prompt or one inspiration image.")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        reject_empty,
    )

    result = execute_provider_batch(csv_path=csv_path, repo_root=tmp_path)

    assert result.status == "failed"
    assert result.rows[0].error is not None
    assert "requires an explicit text prompt or one inspiration image" in result.rows[0].error


def test_batch_passes_its_batch_context_to_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "round2.csv"
    _write_csv(
        csv_path,
        [
            {"row_id": "first", "prompt": "one", "image_path": "", "label": ""},
            {"row_id": "second", "prompt": "two", "image_path": "", "label": ""},
        ],
    )
    contexts: list[dict[str, object]] = []

    def fake_execute_provider_generation(**kwargs):
        contexts.append(dict(kwargs["batch_context"]))
        record_dir = tmp_path / f"rec_{kwargs['prompt']}"
        return GenerationResult(
            status="generation_ready",
            record_id=record_dir.name,
            record_dir=record_dir,
            model_path=record_dir / "model.py",
            workspace_dir=None,
            warnings=(),
        )

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )

    execute_provider_batch(csv_path=csv_path, repo_root=tmp_path)

    contexts.sort(key=lambda item: int(item["row_index"]))  # type: ignore[arg-type]
    assert [item["row_id"] for item in contexts] == ["first", "second"]
    assert [item["row_index"] for item in contexts] == [1, 2]
    assert all(item["total"] == 2 and item["csv"] == "round2.csv" for item in contexts)
    batch_ids = {item["batch_id"] for item in contexts}
    assert len(batch_ids) == 1
    batch_id = batch_ids.pop()
    assert isinstance(batch_id, str) and batch_id.startswith("round2_") and batch_id.endswith("Z")
    assert all(item["started_at"] == contexts[0]["started_at"] for item in contexts)


def _ready(record_dir: Path) -> GenerationResult:
    return GenerationResult(
        status="generation_ready",
        record_id=record_dir.name,
        record_dir=record_dir,
        model_path=record_dir / "model.py",
        workspace_dir=None,
        warnings=(),
    )


def test_batch_row_runtime_and_api_style_reach_the_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "row_id,prompt,image_path,model,reasoning,api_style\n"
        "a,one,,azure/openai/test-model-b,low,chat-completions\n"
        "b,two,,,,\n"
    )
    calls: dict[str, dict[str, object]] = {}
    events: list[ProviderBatchProgressEvent] = []

    def fake_execute_provider_generation(**kwargs):
        calls[str(kwargs["prompt"])] = kwargs
        return _ready(tmp_path / f"rec_{kwargs['prompt']}")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )
    result = execute_provider_batch(
        csv_path=csv_path,
        repo_root=tmp_path,
        model="azure/openai/test-model-a",
        reasoning="high",
        api_style="responses",
        progress=events.append,
    )
    assert result.succeeded == 2
    first, second = calls["one"], calls["two"]
    assert (first["model"], first["reasoning"]) == ("azure/openai/test-model-b", "low")
    assert first["api_style"] == "chat-completions"
    assert (second["model"], second["reasoning"]) == ("azure/openai/test-model-a", "high")
    assert "api_style" not in second
    assert events[0].model == events[0].reasoning_effort == "mixed"


def test_batch_default_api_style_applies_to_rows_without_a_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\na,one,\n")
    calls: list[dict[str, object]] = []

    def fake_execute_provider_generation(**kwargs):
        calls.append(kwargs)
        return _ready(tmp_path / "rec_one")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )
    execute_provider_batch(csv_path=csv_path, repo_root=tmp_path, api_style="chat-completions")
    assert calls[0]["api_style"] == "chat-completions"


def test_batch_run_timeout_routes_rows_to_the_shared_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from usdcraft.generation.provider_batch import GENERATION_DEADLINE
    from usdcraft.pipeline.batch import ProviderBatchRowResult

    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\na,one,\nb,two,missing.png\n")
    calls = []

    def supervised(product, arguments, **kwargs):
        calls.append((product, arguments, kwargs))
        return ProviderBatchRowResult(
            row_id=kwargs["row_id"],
            status="timeout" if kwargs["row_id"] == "a" else "generation_ready",
            timeout_seconds=kwargs["timeout_seconds"],
        )

    monkeypatch.setattr("usdcraft.pipeline.deadline.execute_with_deadline", supervised)
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        lambda **_kwargs: pytest.fail("a supervised row runs in the worker process"),
    )
    result = execute_provider_batch(csv_path=csv_path, repo_root=tmp_path, run_timeout_seconds=90)
    assert [row.status for row in result.rows] == ["timeout", "generation_ready"]
    assert result.failed == 1
    product, arguments, kwargs = calls[0]
    assert product is GENERATION_DEADLINE
    assert kwargs["timeout_seconds"] == 90 and kwargs.get("arm", "provider") == "provider"
    assert arguments["prompt"] == "one" and arguments["batch_context"]["row_id"] == "a"
    assert calls[1][1]["image"] == (tmp_path / "missing.png").resolve()


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_batch_rejects_bad_run_timeouts_before_rows(tmp_path: Path, seconds: float) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("row_id,prompt,image_path\na,one,\n")
    with pytest.raises(ValueError, match="finite and positive"):
        execute_provider_batch(csv_path=csv_path, repo_root=tmp_path, run_timeout_seconds=seconds)


def test_generation_worker_runs_the_direct_primitive_with_the_bound_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from usdcraft.pipeline.deadline import run_worker

    control = tmp_path / "control"
    control.mkdir()
    (control / "transport.json").write_text(
        json.dumps(
            {"arm": "provider", "worker": "usdcraft.generation.provider_batch:run_generation_row"}
        )
    )
    (control / "request.json").write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "prompt": "a lamp",
                "image": "/image.png",
                "run_id": "gen_bound",
                "keep_workspace": True,
                "api_style": "chat-completions",
                "batch_context": {"row_id": "a"},
            }
        )
    )
    captured: dict[str, object] = {}

    def fake_execute_provider_generation(**kwargs):
        captured.update(kwargs)
        kwargs["progress"]("stage=author")
        return _ready(tmp_path / "rec_a")

    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        fake_execute_provider_generation,
    )
    run_worker(control)
    assert captured["repo_root"] == tmp_path and captured["image"] == Path("/image.png")
    assert captured["run_id"] == "gen_bound" and captured["api_style"] == "chat-completions"
    outcome = json.loads((control / "result.json").read_text())
    assert outcome["status"] == "generation_ready"
    assert outcome["result"]["pipeline"] == "generation"
    assert "stage=author" in (control / "progress.jsonl").read_text()


def test_generation_supervisor_reports_a_real_worker_failure_without_api_calls(
    tmp_path: Path,
) -> None:
    import json
    import os
    from threading import Event

    from usdcraft.generation.provider_batch import GENERATION_DEADLINE
    from usdcraft.pipeline.deadline import execute_with_deadline

    if os.name != "posix":
        pytest.skip("POSIX process-group deadline")
    # No .env in this repository root: the worker fails at credential resolution, before any
    # workspace or NVIDIA request.
    result = execute_with_deadline(
        GENERATION_DEADLINE,
        {"repo_root": tmp_path, "prompt": "a lamp", "keep_workspace": False},
        row_id="missing_credentials",
        timeout_seconds=120,
        progress=lambda _message: None,
        cancellation_event=Event(),
    )
    assert result.status == "failed" and result.record_id is None
    assert result.run_id.startswith("gen_")
    snapshot = json.loads(result.snapshot_path.read_text())
    assert snapshot["worker_result"]["error_type"]
    assert "result" not in snapshot["worker_result"]
    assert snapshot["workspace"].endswith(f"data/local/generation_runs/{result.run_id}")
    assert result.snapshot_path.parent.parent == tmp_path / "data/local/generation_deadlines"
