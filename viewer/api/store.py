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

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from storage.artifact_contract import (
    REQUIRED_ASSET_STRUCTURE_ARTIFACTS,
    USDZ_ARTIFACT,
    MaterializationDisplayResult,
    cheap_display_hint,
    classify_materialization_displayability,
)
from storage.identifiers import validate_record_id
from storage.repo import StorageRepo
from storage.revisions import (
    REVISION_ID_RE,
    active_cost_path,
    active_provenance_path,
    active_revision_id,
    active_traces_dir,
)
from viewer.api.run_settings import image_observation_mode, provider_api_style
from viewer.api.schemas import (
    ListMaterializationStatus,
    RecordDetailResponse,
    RecordHistoryResponse,
    RecordHistoryRevisionResponse,
    RecordSummaryResponse,
    ViewerBootstrapResponse,
    WorkbenchEntryResponse,
)
from viewer.api.validation_cache import MaterializationValidationCache

_LIST_READ_WORKERS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mtime_to_utc(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _rating(value: Any) -> int | None:
    rating = _integer(value)
    return rating if rating is not None and 1 <= rating <= 5 else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cost_totals(cost: Any) -> tuple[float | None, int | None, int | None]:
    if not isinstance(cost, dict):
        return None, None, None
    total = cost.get("total")
    if not isinstance(total, dict):
        return None, None, None
    costs = total.get("costs_usd")
    tokens = total.get("tokens")
    return (
        _float(costs.get("total")) if isinstance(costs, dict) else None,
        _integer(tokens.get("prompt_tokens")) if isinstance(tokens, dict) else None,
        _integer(tokens.get("candidates_tokens")) if isinstance(tokens, dict) else None,
    )


def _runtime_usage_tokens(runtime: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = runtime.get("usage")
    if not isinstance(usage, dict):
        return None, None
    input_tokens = _integer(usage.get("input_tokens"))
    output_tokens = _integer(usage.get("output_tokens"))
    return (
        input_tokens if input_tokens is not None and input_tokens >= 0 else None,
        output_tokens if output_tokens is not None and output_tokens >= 0 else None,
    )


def _has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(item.is_file() for item in path.rglob("*"))
    except OSError:
        return False


def _agent_harness(record: dict[str, Any]) -> Literal["codex", "provider"] | None:
    value = _string(record.get("agent_harness"))
    return cast(
        'Literal["codex", "provider"] | None',
        value if value in {"codex", "provider"} else None,
    )


class ViewerStore:
    """Small, read-only projection of canonical Generation records."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.repo = StorageRepo(self.repo_root)
        self._validation_cache = MaterializationValidationCache()

    def _materialization_display(
        self,
        record_id: str,
        record: dict[str, Any],
        *,
        use_cache: bool = True,
    ) -> MaterializationDisplayResult:
        """The full canonical contract result, reused from the validation cache when possible.

        This always returns a real ``storage.artifact_contract`` classification — never the
        cheap hint. It is the only path allowed to claim ``available``. ``use_cache=False``
        forces a fresh check (the explicit "validate this record" action).
        """

        revision_id = active_revision_id(self.repo, record_id, record=record)
        materialization_root = self.repo.layout.record_materialization_dir(record_id)
        if use_cache:
            cached = self._validation_cache.get(record_id, revision_id, materialization_root)
            if cached is not None:
                return cached
        revision_dir = self.repo.layout.record_revision_dir(record_id, revision_id)
        generation_path = self.repo.layout.record_revision_generation_path(record_id, revision_id)
        result = classify_materialization_displayability(
            materialization_root,
            generation_path,
            revision_dir=revision_dir,
        )
        self._validation_cache.put(record_id, revision_id, materialization_root, result)
        return result

    def _materialization_light(
        self,
        record_id: str,
        record: dict[str, Any],
    ) -> tuple[ListMaterializationStatus, bool, list[dict[str, str]], bool]:
        """``(status, displayable, issues, validated)`` without the full canonical contract.

        Reuses a cached full validation result when the active revision and materialization
        signature still match. Otherwise falls back to the two cheap, hash-free objective
        signals (``compile_failed``, ``missing``); when neither applies, the row is reported
        ``not_validated`` — displayable pending an explicit check, never claimed ``available``.
        """

        revision_id = active_revision_id(self.repo, record_id, record=record)
        materialization_root = self.repo.layout.record_materialization_dir(record_id)
        cached = self._validation_cache.get(record_id, revision_id, materialization_root)
        if cached is not None:
            issues = [{"path": issue.path, "message": issue.message} for issue in cached.issues]
            return cached.status, cached.displayable, issues, True
        hint_status, hint_displayable = cheap_display_hint(materialization_root)
        if hint_status is not None:
            return hint_status, hint_displayable, [], False
        return "not_validated", True, [], False

    def _read_json_dict(self, path: Path) -> dict[str, Any] | None:
        """Read one JSON object, degrading corrupt or unreadable files to ``None``."""
        try:
            payload = self.repo.read_json(path)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _record(self, record_id: str) -> dict[str, Any] | None:
        try:
            validate_record_id(record_id)
        except ValueError:
            return None
        payload = self._read_json_dict(self.repo.layout.record_metadata_path(record_id))
        if payload is None:
            return None
        pipeline = _string(payload.get("pipeline")) or "generation"
        return payload if pipeline == "generation" else None

    def _provenance(self, record_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        return self._read_json_dict(active_provenance_path(self.repo, record_id, record=record))

    def _cost(self, record_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        path = active_cost_path(self.repo, record_id, record=record)
        if not path.is_file():
            return None
        return self._read_json_dict(path)

    def _viewer_asset_updated_at(self, record_id: str) -> str | None:
        root = self.repo.layout.record_materialization_dir(record_id)
        candidates = (
            root / "model.urdf",
            root / USDZ_ARTIFACT,
            *(root / path for path in REQUIRED_ASSET_STRUCTURE_ARTIFACTS),
            root / "compile_report.json",
            root / "assets",
            root / "renders",
        )
        mtimes: list[float] = []
        for candidate in candidates:
            try:
                mtimes.append(candidate.stat().st_mtime)
            except OSError:
                continue
        return _mtime_to_utc(max(mtimes)) if mtimes else None

    def record_summary(
        self,
        record_id: str,
        *,
        fast: bool = False,
        materialization_display: MaterializationDisplayResult | None = None,
    ) -> RecordSummaryResponse | None:
        """Build one record's summary.

        ``fast=True`` (the list projection) never runs the full canonical contract check: it
        reuses a still-fresh cached validation when one exists and otherwise reports
        ``not_validated`` rather than guessing. Every other caller (``record_detail``,
        the explicit ``validate`` action) always resolves a real ``MaterializationDisplayResult``.
        """

        record = self._record(record_id)
        if record is None:
            return None

        display = cast(
            "dict[str, Any]",
            record.get("display") if isinstance(record.get("display"), dict) else {},
        )
        provenance = self._provenance(record_id, record)
        generation = cast(
            "dict[str, Any]",
            provenance.get("generation")
            if isinstance(provenance, dict) and isinstance(provenance.get("generation"), dict)
            else {},
        )
        runtime = cast(
            "dict[str, Any]",
            provenance.get("runtime")
            if isinstance(provenance, dict) and isinstance(provenance.get("runtime"), dict)
            else {},
        )
        run_summary = cast(
            "dict[str, Any]",
            provenance.get("run_summary")
            if isinstance(provenance, dict) and isinstance(provenance.get("run_summary"), dict)
            else {},
        )
        cost = self._cost(record_id, record)
        total_cost_usd, input_tokens, output_tokens = _cost_totals(cost)
        runtime_input_tokens, runtime_output_tokens = _runtime_usage_tokens(runtime)
        if input_tokens is None:
            input_tokens = runtime_input_tokens
        if output_tokens is None:
            output_tokens = runtime_output_tokens
        turn_count = _integer(run_summary.get("turn_count"))
        if turn_count is None:
            turn_count = _integer(runtime.get("turn_count"))

        active_id = active_revision_id(self.repo, record_id, record=record)
        revisions_root = self.repo.layout.record_revisions_dir(record_id)
        revision_count = (
            sum(
                1
                for path in revisions_root.iterdir()
                if path.is_dir() and REVISION_ID_RE.fullmatch(path.name)
            )
            if revisions_root.is_dir()
            else 0
        )
        traces_dir = active_traces_dir(self.repo, record_id, record=record)
        materialization = self.repo.layout.record_materialization_dir(record_id)
        materialization_validated = True
        if materialization_display is not None:
            materialization_status: ListMaterializationStatus = materialization_display.status
            materialization_displayable = materialization_display.displayable
            materialization_issues: list[Any] = [
                {"path": issue.path, "message": issue.message}
                for issue in materialization_display.issues
            ]
        elif fast:
            (
                materialization_status,
                materialization_displayable,
                materialization_issues,
                materialization_validated,
            ) = self._materialization_light(record_id, record)
        else:
            materialization_display = self._materialization_display(record_id, record)
            materialization_status = materialization_display.status
            materialization_displayable = materialization_display.displayable
            materialization_issues = [
                {"path": issue.path, "message": issue.message}
                for issue in materialization_display.issues
            ]
        raw_status = _string(record.get("status")) or _string(run_summary.get("final_status"))
        status: Literal["generation_ready"] | None = (
            "generation_ready" if raw_status == "generation_ready" else None
        )
        raw_input_kind = _string(record.get("input_kind"))
        input_kind = cast(
            'Literal["text", "text_image"] | None',
            raw_input_kind if raw_input_kind in {"text", "text_image"} else None,
        )

        batch = cast(
            "dict[str, Any]", runtime.get("batch") if isinstance(runtime.get("batch"), dict) else {}
        )
        agent_harness = _agent_harness(record)
        runtime_source = (
            _string(record.get("runtime_source"))
            or _string(runtime.get("runtime_source"))
            or _string(generation.get("runtime_source"))
        )
        return RecordSummaryResponse(
            record_id=record_id,
            runtime_source=runtime_source,
            api_style=provider_api_style(agent_harness, runtime, runtime_source),
            image_observation=image_observation_mode(runtime),
            status=status,
            input_kind=input_kind,
            batch_id=_string(batch.get("batch_id")),
            batch_row_id=_string(batch.get("row_id")),
            title=_string(display.get("title")) or record_id,
            prompt_preview=_string(display.get("prompt_preview")) or "",
            rating=_rating(record.get("rating")),
            created_at=_string(record.get("created_at")),
            updated_at=_string(record.get("updated_at")),
            viewer_asset_updated_at=self._viewer_asset_updated_at(record_id),
            provider=_string(record.get("provider"))
            or _string(generation.get("provider"))
            or _string(runtime.get("provider")),
            model_id=_string(record.get("model_id")) or _string(runtime.get("model_id")),
            agent_harness=agent_harness,
            has_traces=_has_files(traces_dir),
            thinking_level=_string(record.get("reasoning_effort"))
            or _string(runtime.get("reasoning_effort"))
            or _string(generation.get("reasoning_effort"))
            or _string(generation.get("thinking_level"))
            or _string(runtime.get("reasoning")),
            turn_count=turn_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            active_revision_id=active_id,
            revision_count=revision_count,
            materialization_status=materialization_status,
            materialization_displayable=materialization_displayable,
            materialization_validated=materialization_validated,
            materialization_issues=materialization_issues,
            has_compile_report=(materialization / "compile_report.json").is_file(),
            has_provenance=active_provenance_path(self.repo, record_id, record=record).is_file(),
            has_cost=active_cost_path(self.repo, record_id, record=record).is_file(),
        )

    def record_detail(self, record_id: str) -> RecordDetailResponse | None:
        record = self._record(record_id)
        summary = self.record_summary(record_id)
        if record is None or summary is None:
            return None
        try:
            compile_report = self.repo.read_json(
                self.repo.layout.record_materialization_compile_report_path(record_id)
            )
        except (OSError, ValueError):
            compile_report = None
        return RecordDetailResponse(
            summary=summary,
            record=record,
            compile_report=compile_report if isinstance(compile_report, dict) else None,
            provenance=self._provenance(record_id, record),
            cost=self._cost(record_id, record),
        )

    def validate(self, record_id: str) -> RecordSummaryResponse | None:
        """Explicitly run the full canonical contract check for one record and cache it.

        This is the user-invoked "validate this record" action: it does the same hash and
        dependency-closure work as opening the record's detail view, but is never triggered
        implicitly by listing or refreshing the record library.
        """

        record = self._record(record_id)
        if record is None:
            return None
        materialization_display = self._materialization_display(record_id, record, use_cache=False)
        return self.record_summary(record_id, materialization_display=materialization_display)

    def record_history(self, record_id: str) -> RecordHistoryResponse | None:
        record = self._record(record_id)
        if record is None:
            return None
        active_id = active_revision_id(self.repo, record_id, record=record)
        revisions: list[RecordHistoryRevisionResponse] = []
        root = self.repo.layout.record_revisions_dir(record_id)
        if root.is_dir():
            for revision_dir in sorted(root.iterdir(), key=lambda item: item.name):
                if not revision_dir.is_dir() or not REVISION_ID_RE.fullmatch(revision_dir.name):
                    continue
                revision = self._read_json_dict(revision_dir / "revision.json") or {}
                generation = cast(
                    "dict[str, Any]",
                    revision.get("generation")
                    if isinstance(revision.get("generation"), dict)
                    else {},
                )
                run_summary = cast(
                    "dict[str, Any]",
                    revision.get("run_summary")
                    if isinstance(revision.get("run_summary"), dict)
                    else {},
                )
                prompt_path = revision_dir / "prompt.txt"
                try:
                    prompt = (
                        prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
                    )
                except OSError:
                    prompt = ""
                revisions.append(
                    RecordHistoryRevisionResponse(
                        record_id=record_id,
                        revision_id=revision_dir.name,
                        active=revision_dir.name == active_id,
                        created_at=_string(revision.get("created_at")),
                        prompt_preview=" ".join(prompt.split())[:160],
                        provider=_string(generation.get("provider")),
                        model_id=_string(generation.get("model_id")),
                        thinking_level=_string(generation.get("reasoning_effort"))
                        or _string(generation.get("thinking_level")),
                        status=_string(run_summary.get("final_status"))
                        or _string(revision.get("status")),
                        has_cost=(revision_dir / "cost.json").is_file(),
                        has_traces=_has_files(revision_dir / "traces"),
                        has_model=(revision_dir / "model.py").is_file(),
                        has_provenance=(revision_dir / "provenance.json").is_file(),
                    )
                )
        return RecordHistoryResponse(
            record_id=record_id,
            active_revision_id=active_id,
            revisions=revisions,
        )

    def _workbench_entry(self, record_id: str) -> WorkbenchEntryResponse | None:
        """One Workbench row, or ``None`` for an unregistered, mismatched or unreadable record."""

        try:
            validate_record_id(record_id)
        except ValueError:
            return None
        payload = self._read_json_dict(self.repo.layout.record_workbench_entry_path(record_id))
        if payload is None or (_string(payload.get("record_id")) or record_id) != record_id:
            return None
        # One corrupt record must degrade to a skipped entry, never a 500 bootstrap.
        try:
            summary = self.record_summary(record_id, fast=True)
        except (OSError, ValueError):
            return None
        if summary is None:
            return None
        tags = payload.get("tags")
        return WorkbenchEntryResponse(
            record_id=record_id,
            added_at=_string(payload.get("added_at")) or "",
            label=_string(payload.get("label")),
            tags=[str(item) for item in tags if isinstance(item, str)]
            if isinstance(tags, list)
            else [],
            archived=bool(payload.get("archived", False)),
            record=summary,
        )

    def list_workbench_entries(self) -> list[WorkbenchEntryResponse]:
        records_root = self.repo.layout.records_root
        if not records_root.is_dir():
            return []
        record_ids = [path.name for path in records_root.iterdir() if path.is_dir()]
        # Rows are independent small-file reads; a few threads overlap that I/O.
        with ThreadPoolExecutor(max_workers=_LIST_READ_WORKERS) as pool:
            rows = pool.map(self._workbench_entry, record_ids)
        return sorted(
            (row for row in rows if row is not None),
            key=lambda item: (item.added_at, item.record_id),
            reverse=True,
        )

    def list_records(self) -> list[RecordSummaryResponse]:
        return [entry.record for entry in self.list_workbench_entries() if entry.record is not None]

    def bootstrap(self) -> ViewerBootstrapResponse:
        return ViewerBootstrapResponse(
            repo_root=self.repo_root.as_posix(),
            generated_at=_utc_now(),
            workbench_entries=self.list_workbench_entries(),
        )
