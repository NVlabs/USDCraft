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

"""Product-neutral coordinator for direct-execution Batch commands.

`generation-batch`, `reconstruction-batch` and `reconstruction-codex-batch` share this scheduler: a bounded thread pool with
fixed slots, per-row failure isolation, CSV-ordered results, presentation-only progress events and
the Ctrl-C snapshot protocol. Each product contributes only its CSV row contract and the function
that runs one row through its own direct primitive; nothing here knows a prompt, a mesh or
a record layout. Batch has no preflight/fingerprint exchange, and concurrency is local scheduling
rather than shared Author context.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Mapping, Protocol, TypeVar

from usdcraft.pipeline.batch_progress import (
    ProviderBatchProgressCallback,
    ProviderBatchProgressEvent,
    emit_provider_batch_progress,
)
from usdcraft.pipeline.provider.progress import ProviderProgressCallback

MAX_PROVIDER_BATCH_CONCURRENCY = 32
# The terminal statuses of the two direct Provider primitives that count as a succeeded row.
BATCH_SUCCESS_STATUSES = frozenset({"generation_ready", "reconstruction_ready"})


class BatchRow(Protocol):
    """The one attribute the coordinator needs from a product's validated CSV row."""

    @property
    def row_id(self) -> str: ...


RowT = TypeVar("RowT", bound=BatchRow)


class BatchFailureContext(Protocol):
    @property
    def run_id(self) -> str: ...

    @property
    def workspace_dir(self) -> Path: ...


@dataclass(slots=True, frozen=True)
class ProviderBatchRowResult:
    row_id: str
    status: str
    record_id: str | None = None
    record_dir: Path | None = None
    run_id: str | None = None
    workspace_dir: Path | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    elapsed_seconds: float | None = None
    timeout_seconds: float | None = None
    snapshot_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "status": self.status,
            "record_id": self.record_id,
            "record_dir": str(self.record_dir) if self.record_dir is not None else None,
            "run_id": self.run_id,
            "workspace_dir": (str(self.workspace_dir) if self.workspace_dir is not None else None),
            "warnings": list(self.warnings),
            "error": self.error,
            **(
                {"elapsed_seconds": self.elapsed_seconds}
                if self.elapsed_seconds is not None
                else {}
            ),
            **(
                {"timeout_seconds": self.timeout_seconds}
                if self.timeout_seconds is not None
                else {}
            ),
            **(
                {"snapshot_path": str(self.snapshot_path)} if self.snapshot_path is not None else {}
            ),
        }


@dataclass(slots=True, frozen=True)
class ProviderBatchResult:
    csv_path: Path
    rows: tuple[ProviderBatchRowResult, ...]
    # `codex-cli` for the Codex arm; the Provider arm's endpoint (`openai` or `nvidia`).
    provider: str = "openai"

    @property
    def succeeded(self) -> int:
        return sum(row.status in BATCH_SUCCESS_STATUSES for row in self.rows)

    @property
    def failed(self) -> int:
        return len(self.rows) - self.succeeded

    @property
    def status(self) -> str:
        if self.failed == 0:
            return "complete"
        if self.succeeded == 0:
            return "failed"
        return "partial_failure"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "csv_path": str(self.csv_path),
            "total": len(self.rows),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "rows": [row.to_dict() for row in self.rows],
        }


class ProviderBatchInterrupted(KeyboardInterrupt):
    """A user interruption with an immutable snapshot of finalized Batch work.

    ``partial_result`` contains only rows whose futures had finalized when cancellation was
    requested. Python cannot safely terminate an already-running thread, so those row IDs receive
    a cooperative stop signal and are reported separately from futures canceled before starting.
    """

    def __init__(
        self,
        *,
        partial_result: ProviderBatchResult,
        total: int,
        cancelled_row_ids: tuple[str, ...],
        in_flight_row_ids: tuple[str, ...],
    ) -> None:
        self.partial_result = partial_result
        self.csv_path = partial_result.csv_path
        self.total = total
        self.cancelled_row_ids = cancelled_row_ids
        self.in_flight_row_ids = in_flight_row_ids
        super().__init__(
            "Provider batch interrupted after "
            f"{len(partial_result.rows)}/{total} rows finalized; "
            f"{len(cancelled_row_ids)} canceled and {len(in_flight_row_ids)} still in flight."
        )

    def to_dict(self) -> dict[str, Any]:
        partial = self.partial_result
        return {
            "status": "interrupted",
            "provider": partial.provider,
            "csv_path": str(self.csv_path),
            "total": self.total,
            "completed": len(partial.rows),
            "succeeded": partial.succeeded,
            "failed": partial.failed,
            "cancelled": len(self.cancelled_row_ids),
            "in_flight": len(self.in_flight_row_ids),
            "rows": [row.to_dict() for row in partial.rows],
            "cancelled_row_ids": list(self.cancelled_row_ids),
            "in_flight_row_ids": list(self.in_flight_row_ids),
            "error": str(self),
        }


# ---------------------------------------------------------------------------------------------
# CSV reading shared by both products.


def optional_text(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def resolve_csv_relative_path(value: str, *, csv_dir: Path) -> Path:
    """Resolve a CSV cell path; relative paths are relative to the CSV's own directory."""

    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else csv_dir / candidate).resolve()


def iter_batch_csv(
    csv_path: Path, *, required_columns: frozenset[str]
) -> Iterator[tuple[str, dict[str, str], str]]:
    """Yield ``(location, raw_row, row_id)`` for each validated data record of a Batch CSV.

    Validates the file, duplicate headers, required columns, non-empty unique control-free
    ``row_id`` values, and raises ``ValueError`` on an empty CSV. Product loaders add their own
    column semantics on top.
    """

    resolved_csv = Path(csv_path).expanduser().resolve()
    if not resolved_csv.is_file():
        raise ValueError(f"Provider batch CSV is not a file: {resolved_csv}")

    seen_ids: set[str] = set()
    yielded = False
    with resolved_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        duplicate_headers = sorted(
            header for header, count in Counter(fieldnames).items() if count > 1
        )
        if duplicate_headers:
            raise ValueError(
                "Provider batch CSV has duplicate column names: "
                + ", ".join(repr(header) for header in duplicate_headers)
            )
        missing = sorted(required_columns - set(fieldnames))
        if missing:
            raise ValueError(
                "Provider batch CSV is missing required columns: " + ", ".join(missing)
            )

        for record_number, raw_row in enumerate(reader, start=1):
            location = f"data record {record_number} (ending at physical line {reader.line_num})"
            row_id = optional_text(raw_row.get("row_id"))
            if row_id is None:
                raise ValueError(f"Provider batch {location} has an empty row_id.")
            if any(ord(character) < 32 or ord(character) == 127 for character in row_id):
                raise ValueError(f"Provider batch {location} row_id contains control characters.")
            if row_id in seen_ids:
                raise ValueError(f"Provider batch {location} duplicates row_id {row_id!r}.")
            seen_ids.add(row_id)
            yielded = True
            yield location, raw_row, row_id

    if not yielded:
        raise ValueError("Provider batch CSV has no data rows.")


PROVIDER_API_STYLES = ("responses", "chat-completions")


def reject_unknown_columns(
    location: str, raw_row: Mapping[Any, object], *, allowed_columns: tuple[str, ...]
) -> None:
    """Reject surplus cells and every column outside a product's CSV contract.

    A typo must never silently fall back to a default, so an unknown column is an error that names
    the unknown columns and the complete allowed set.
    """

    if None in raw_row:
        raise ValueError(f"Provider batch {location}: more cells than CSV columns.")
    unknown = set(raw_row) - set(allowed_columns)
    if unknown:
        raise ValueError(
            f"Provider batch {location}: unknown columns: {', '.join(sorted(map(str, unknown)))}. "
            f"Allowed columns: {', '.join(allowed_columns)}."
        )


@dataclass(slots=True, frozen=True)
class RowRuntimeOverride:
    """The optional per-row Provider runtime cells shared by both products' CSV contracts."""

    model: str | None = None
    reasoning: str | None = None
    api_style: str | None = None


def row_runtime_override(location: str, raw_row: Mapping[str, object]) -> RowRuntimeOverride:
    """Read the paired ``model``/``reasoning`` override and the optional ``api_style`` cell."""

    model = optional_text(raw_row.get("model"))
    reasoning = optional_text(raw_row.get("reasoning"))
    api_style = optional_text(raw_row.get("api_style"))
    if api_style is not None and api_style not in PROVIDER_API_STYLES:
        raise ValueError(
            f"Provider batch {location}: invalid api_style {api_style!r}; "
            f"use {' or '.join(PROVIDER_API_STYLES)}."
        )
    if (model is None) != (reasoning is None):
        raise ValueError(
            f"Provider batch {location}: model and reasoning must be provided together "
            "for a per-row runtime override."
        )
    return RowRuntimeOverride(model=model, reasoning=reasoning, api_style=api_style)


class RuntimeOverrideRow(Protocol):
    @property
    def row_id(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def reasoning(self) -> str | None: ...


def resolve_provider_row_runtimes(
    repo_root: Path,
    rows: tuple[RuntimeOverrideRow, ...],
    *,
    model: str | None,
    reasoning: str | None,
) -> dict[str, tuple[str, str]] | None:
    """Resolve every row's effective NVIDIA runtime before any row starts.

    Returns ``None`` when no row carries a CSV runtime pair (the batch-wide runtime applies).
    """

    if not any(row.model is not None for row in rows):
        return None
    from usdcraft.pipeline.provider import resolve_nvidia_runtime_identity

    runtime_by_row: dict[str, tuple[str, str]] = {}
    for row in rows:
        runtime = resolve_nvidia_runtime_identity(
            repo_root,
            model=row.model if row.model is not None else model,
            reasoning_effort=row.reasoning if row.model is not None else reasoning,
        )
        runtime_by_row[row.row_id] = (runtime.model, runtime.reasoning_effort)
    return runtime_by_row


# ---------------------------------------------------------------------------------------------
# Scheduling.

RowExecutor = Callable[
    [RowT, ProviderProgressCallback, Event, Mapping[str, Any]], ProviderBatchRowResult
]
"""Run one validated row through the product's direct Provider primitive.

The callable receives the row, a progress callback for that row's Provider messages, the shared
cooperative cancellation event and the row's batch context (``batch_id``, ``csv``, ``row_id``,
``row_index``, ``total``, ``started_at``), and returns the row's result. It may raise; the
coordinator turns any exception into a failed row result using ``failure_context``.
"""


def batch_context_for_row(
    *, batch_id: str, csv_path: Path, row_id: str, row_index: int, total: int, started_at: str
) -> dict[str, Any]:
    """The batch membership a primitive records in its runtime report (for Viewer grouping)."""

    return {
        "batch_id": batch_id,
        "csv": csv_path.name,
        "row_id": row_id,
        "row_index": row_index,
        "total": total,
        "started_at": started_at,
    }


FailureContextResolver = Callable[[BaseException], BatchFailureContext | None]


def _failed_row(
    row_id: str,
    exc: BaseException,
    *,
    failure_context: FailureContextResolver,
) -> ProviderBatchRowResult:
    try:
        context = failure_context(exc)
    except BaseException:
        context = None
    try:
        error = f"{type(exc).__name__}: {exc}"
    except BaseException:
        error = type(exc).__name__
    return ProviderBatchRowResult(
        row_id=row_id,
        status="failed",
        run_id=context.run_id if context is not None else None,
        workspace_dir=context.workspace_dir if context is not None else None,
        error=error,
    )


def _execute_row(
    row: BatchRow,
    *,
    row_index: int,
    slot: int,
    execute_row: RowExecutor[Any],
    failure_context: FailureContextResolver,
    progress: ProviderBatchProgressCallback | None,
    cancellation_event: Event,
    batch_context: Mapping[str, Any],
) -> ProviderBatchRowResult:
    def report_batch_event(event: ProviderBatchProgressEvent) -> None:
        # This callback runs inside a worker. Even BaseException subclasses raised by a
        # presentation sink are worker-local and must not terminate Batch scheduling.
        try:
            emit_provider_batch_progress(progress, event)
        except BaseException:
            pass

    def report_provider_progress(message: str) -> None:
        report_batch_event(
            ProviderBatchProgressEvent(
                kind="row_progress",
                timestamp=datetime.now().astimezone(),
                slot=slot,
                row_index=row_index,
                row_id=row.row_id,
                message=message,
            ),
        )

    report_batch_event(
        ProviderBatchProgressEvent(
            kind="row_started",
            timestamp=datetime.now().astimezone(),
            slot=slot,
            row_index=row_index,
            row_id=row.row_id,
        )
    )
    try:
        row_result = execute_row(row, report_provider_progress, cancellation_event, batch_context)
    except BaseException as exc:
        row_result = _failed_row(row.row_id, exc, failure_context=failure_context)
        report_batch_event(
            ProviderBatchProgressEvent(
                kind="row_finished",
                timestamp=datetime.now().astimezone(),
                slot=slot,
                row_index=row_index,
                row_id=row.row_id,
                status=row_result.status,
                error_type=type(exc).__name__,
            ),
        )
        return row_result
    report_batch_event(
        ProviderBatchProgressEvent(
            kind="row_finished",
            timestamp=datetime.now().astimezone(),
            slot=slot,
            row_index=row_index,
            row_id=row.row_id,
            status=row_result.status,
            record_id=row_result.record_id,
        ),
    )
    return row_result


def run_provider_batch(
    *,
    csv_path: Path,
    rows: tuple[RowT, ...],
    repo_root: Path,
    execute_row: RowExecutor[RowT],
    failure_context: FailureContextResolver,
    provider: str = "openai",
    model: str | None = None,
    reasoning: str | None = None,
    runtime_by_row: Mapping[str, tuple[str, str]] | None = None,
    concurrency: int = 1,
    progress: ProviderBatchProgressCallback | None = None,
) -> ProviderBatchResult:
    """Run every validated row through ``execute_row`` on a fixed pool of ``concurrency`` slots."""

    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or not 1 <= concurrency <= MAX_PROVIDER_BATCH_CONCURRENCY
    ):
        raise ValueError(
            "Provider batch concurrency must be an integer from 1 to "
            f"{MAX_PROVIDER_BATCH_CONCURRENCY}."
        )
    if not rows:
        raise ValueError("Provider batch CSV has no data rows.")
    resolved_csv = Path(csv_path).expanduser().resolve()
    resolved_repo = Path(repo_root).expanduser().resolve()
    effective_concurrency = min(concurrency, len(rows))
    if runtime_by_row is not None:
        models = {runtime_by_row[row.row_id][0] for row in rows}
        efforts = {runtime_by_row[row.row_id][1] for row in rows}
        model = next(iter(models)) if len(models) == 1 else "mixed"
        reasoning = next(iter(efforts)) if len(efforts) == 1 else "mixed"
    elif provider != "codex-cli":
        # Provider arm (`nvidia` or `openai` endpoint): one runtime identity from `.env`.
        from usdcraft.pipeline.provider import resolve_nvidia_runtime_identity

        runtime_identity = resolve_nvidia_runtime_identity(
            resolved_repo,
            model=model,
            reasoning_effort=reasoning,
        )
        model, reasoning = runtime_identity.model, runtime_identity.reasoning_effort
    ordered_results: list[ProviderBatchRowResult | None] = [None] * len(rows)
    batch_started = datetime.now().astimezone()
    batch_id = f"{resolved_csv.stem}_{batch_started.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
    started_at_text = batch_started.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emit_provider_batch_progress(
        progress,
        ProviderBatchProgressEvent(
            kind="batch_started",
            timestamp=batch_started,
            total=len(rows),
            effective_concurrency=effective_concurrency,
            model=model,
            reasoning_effort=reasoning,
        ),
    )

    executor = ThreadPoolExecutor(max_workers=effective_concurrency)
    cancellation_event = Event()
    futures: dict[Future[ProviderBatchRowResult], tuple[int, int]] = {}
    next_row_index = 0

    def submit_row(index: int, slot: int) -> None:
        future = executor.submit(
            _execute_row,
            rows[index],
            row_index=index + 1,
            slot=slot,
            execute_row=execute_row,
            failure_context=failure_context,
            progress=progress,
            cancellation_event=cancellation_event,
            batch_context=batch_context_for_row(
                batch_id=batch_id,
                csv_path=resolved_csv,
                row_id=rows[index].row_id,
                row_index=index + 1,
                total=len(rows),
                started_at=started_at_text,
            ),
        )
        futures[future] = (index, slot)

    def collect_future(
        future: Future[ProviderBatchRowResult],
        *,
        index: int,
        slot: int,
    ) -> ProviderBatchRowResult:
        def failed_result(exc: BaseException) -> ProviderBatchRowResult:
            row_result = _failed_row(rows[index].row_id, exc, failure_context=failure_context)
            emit_provider_batch_progress(
                progress,
                ProviderBatchProgressEvent(
                    kind="row_finished",
                    timestamp=datetime.now().astimezone(),
                    slot=slot,
                    row_index=index + 1,
                    row_id=rows[index].row_id,
                    status=row_result.status,
                    error_type=type(exc).__name__,
                ),
            )
            return row_result

        # Future.exception() returns a worker-raised BaseException as a value. This lets a
        # worker-local KeyboardInterrupt become a row failure while a signal delivered to the
        # coordinator during this call still propagates normally.
        try:
            worker_exception = future.exception()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            return failed_result(exc)
        if worker_exception is not None:
            return failed_result(worker_exception)
        try:
            return future.result()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            return failed_result(exc)

    try:
        for slot in range(1, effective_concurrency + 1):
            submit_row(next_row_index, slot)
            next_row_index += 1

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: futures[item][1]):
                index, slot = futures.pop(future)
                ordered_results[index] = collect_future(future, index=index, slot=slot)
                if next_row_index < len(rows):
                    submit_row(next_row_index, slot)
                    next_row_index += 1
    except KeyboardInterrupt as exc:
        cancelled_indexes: set[int] = set(range(next_row_index, len(rows)))
        in_flight_indexes: set[int] = set()
        completed_futures: list[tuple[Future[ProviderBatchRowResult], int, int]] = []
        for future, (index, slot) in tuple(futures.items()):
            if future.cancelled():
                cancelled_indexes.add(index)
            elif future.done():
                completed_futures.append((future, index, slot))
            elif future.cancel():
                cancelled_indexes.add(index)
            elif future.done():
                # The worker may have completed between the first state check and cancel().
                completed_futures.append((future, index, slot))
            else:
                in_flight_indexes.add(index)
        cancellation_event.set()
        # Do not use the executor context manager here: __exit__ calls shutdown(wait=True).
        # Running Python threads cannot be killed safely. They receive the shared cooperative stop
        # signal above; pending futures are canceled and the coordinator returns immediately.
        executor.shutdown(wait=False, cancel_futures=True)
        for future, index, slot in completed_futures:
            ordered_results[index] = collect_future(future, index=index, slot=slot)
        partial_result = ProviderBatchResult(
            csv_path=resolved_csv,
            provider=provider,
            rows=tuple(result for result in ordered_results if result is not None),
        )
        raise ProviderBatchInterrupted(
            partial_result=partial_result,
            total=len(rows),
            cancelled_row_ids=tuple(rows[index].row_id for index in sorted(cancelled_indexes)),
            in_flight_row_ids=tuple(rows[index].row_id for index in sorted(in_flight_indexes)),
        ) from exc
    except BaseException:
        cancellation_event.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    result = ProviderBatchResult(
        csv_path=resolved_csv,
        provider=provider,
        rows=tuple(result for result in ordered_results if result is not None),
    )
    emit_provider_batch_progress(
        progress,
        ProviderBatchProgressEvent(
            kind="batch_finished",
            timestamp=datetime.now().astimezone(),
            total=len(rows),
            succeeded=result.succeeded,
            failed=result.failed,
            status=result.status,
        ),
    )
    return result


__all__ = [
    "BATCH_SUCCESS_STATUSES",
    "MAX_PROVIDER_BATCH_CONCURRENCY",
    "BatchRow",
    "PROVIDER_API_STYLES",
    "ProviderBatchInterrupted",
    "ProviderBatchResult",
    "ProviderBatchRowResult",
    "RowExecutor",
    "RowRuntimeOverride",
    "batch_context_for_row",
    "iter_batch_csv",
    "optional_text",
    "reject_unknown_columns",
    "resolve_csv_relative_path",
    "resolve_provider_row_runtimes",
    "row_runtime_override",
    "run_provider_batch",
]
