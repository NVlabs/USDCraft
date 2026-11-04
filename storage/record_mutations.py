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

import fcntl
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from storage.identifiers import validate_record_id
from storage.materialization_mutations import (
    MaterializationReplacementRecoveryError,
    recover_materialization_replacement,
)
from storage.record_locks import record_lock
from storage.repo import StorageRepo

_DELETION_MANIFEST = "transaction.json"
PROMOTION_MANIFEST = "promotion.json"
PROMOTION_ACTIVITY_LOCK = "promotion.lock"
PROMOTION_COORDINATION_LOCK = "promotion-recovery.lock"
PROMOTION_MANIFEST_SCHEMA_VERSION = 2
_PROMOTION_STATES = ("staging", "committing", "committed")


class RecordNotFoundError(LookupError):
    """Raised when a canonical record does not exist."""


class RecordDeletionRecoveryError(RuntimeError):
    """Raised when an interrupted deletion cannot be resumed safely."""


@dataclass(slots=True, frozen=True)
class RatingUpdateResult:
    record_id: str
    rating: int | None
    updated_at: str


@dataclass(slots=True, frozen=True)
class RecordDeletionResult:
    record_id: str
    materialization_deleted: bool
    cleanup_pending: bool


@dataclass(slots=True, frozen=True)
class DeletionRecoveryResult:
    completed_record_ids: tuple[str, ...] = ()
    rolled_back_record_ids: tuple[str, ...] = ()
    cleanup_pending_record_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class _DeletionRecoveryOutcome:
    status: Literal["completed", "rolled_back", "pending"]
    deletion_committed: bool
    materialization_deleted: bool
    cleanup_pending: bool
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _rename_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_deletion_manifest(
    repo: StorageRepo,
    staging_dir: Path,
    *,
    record_id: str,
) -> tuple[dict[str, Any], str | None]:
    path = staging_dir / _DELETION_MANIFEST
    if not path.is_file():
        return {}, f"missing deletion manifest at {path}"
    try:
        payload = repo.read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"invalid deletion manifest at {path}: {exc}"
    if not isinstance(payload, dict):
        return {}, f"invalid deletion manifest at {path}: expected an object"
    if payload.get("schema_version") != 1:
        return {}, f"invalid deletion manifest at {path}: schema_version must be 1"
    if payload.get("record_id") != record_id:
        return {}, (
            f"invalid deletion manifest at {path}: record_id must be {record_id!r}, "
            f"got {payload.get('record_id')!r}"
        )
    if payload.get("state") not in {"prepared", "record_staged", "committed"}:
        return {}, f"invalid deletion manifest at {path}: state is not recognized"
    if not isinstance(payload.get("materialization_expected"), bool):
        return {}, (
            f"invalid deletion manifest at {path}: materialization_expected must be a boolean"
        )
    return payload, None


def _write_deletion_manifest(
    repo: StorageRepo,
    staging_dir: Path,
    *,
    record_id: str,
    state: Literal["prepared", "record_staged", "committed"],
    materialization_expected: bool,
) -> None:
    repo.write_json(
        staging_dir / _DELETION_MANIFEST,
        {
            "schema_version": 1,
            "record_id": record_id,
            "state": state,
            "materialization_expected": materialization_expected,
            "updated_at": _utc_now(),
        },
    )


def _cleanup_staging(
    staging_dir: Path,
    *,
    deletion_committed: bool,
    materialization_deleted: bool,
    warning: str | None = None,
) -> _DeletionRecoveryOutcome:
    try:
        _remove_tree(staging_dir)
    except OSError as exc:
        message = f"could not clean deletion transaction {staging_dir}: {exc}"
        if warning:
            message = f"{warning}; {message}"
        return _DeletionRecoveryOutcome(
            status="pending",
            deletion_committed=deletion_committed,
            materialization_deleted=materialization_deleted,
            cleanup_pending=True,
            error=message,
        )
    return _DeletionRecoveryOutcome(
        status="completed" if deletion_committed else "rolled_back",
        deletion_committed=deletion_committed,
        materialization_deleted=materialization_deleted,
        cleanup_pending=False,
        error=warning,
    )


def _recover_pending_deletion_locked(
    repo: StorageRepo,
    record_id: str,
) -> _DeletionRecoveryOutcome:
    """Resume or roll back one deterministic deletion transaction with its record lock held."""

    staging_dir = repo.layout.record_deletion_staging_dir(record_id)
    if not staging_dir.is_dir():
        return _DeletionRecoveryOutcome(
            status="rolled_back",
            deletion_committed=False,
            materialization_deleted=False,
            cleanup_pending=False,
        )

    manifest, manifest_warning = _read_deletion_manifest(
        repo,
        staging_dir,
        record_id=record_id,
    )
    if manifest_warning:
        return _DeletionRecoveryOutcome(
            status="pending",
            deletion_committed=False,
            materialization_deleted=False,
            cleanup_pending=True,
            error=manifest_warning,
        )

    record_dir = repo.layout.record_dir(record_id)
    materialization_dir = repo.layout.record_materialization_dir(record_id)
    staged_record_dir = staging_dir / "record"
    staged_materialization_dir = staging_dir / "materialization"
    canonical_record_exists = _path_exists(record_dir)
    staged_record_exists = _path_exists(staged_record_dir)
    canonical_materialization_exists = _path_exists(materialization_dir)
    staged_materialization_exists = _path_exists(staged_materialization_dir)
    materialization_expected = bool(manifest.get("materialization_expected", False))

    if canonical_record_exists and staged_record_exists:
        return _DeletionRecoveryOutcome(
            status="pending",
            deletion_committed=False,
            materialization_deleted=staged_materialization_exists,
            cleanup_pending=True,
            error=(
                f"deletion transaction {staging_dir} conflicts with an existing canonical record"
            ),
        )

    if canonical_record_exists:
        if staged_materialization_exists:
            if canonical_materialization_exists:
                return _DeletionRecoveryOutcome(
                    status="pending",
                    deletion_committed=False,
                    materialization_deleted=False,
                    cleanup_pending=True,
                    error=(
                        f"deletion transaction {staging_dir} has both staged and canonical "
                        "materializations"
                    ),
                )
            try:
                _rename_path(staged_materialization_dir, materialization_dir)
            except OSError as exc:
                return _DeletionRecoveryOutcome(
                    status="pending",
                    deletion_committed=False,
                    materialization_deleted=False,
                    cleanup_pending=True,
                    error=f"could not roll back materialization for {record_id}: {exc}",
                )
        return _cleanup_staging(
            staging_dir,
            deletion_committed=False,
            materialization_deleted=False,
        )

    if canonical_materialization_exists and staged_materialization_exists:
        return _DeletionRecoveryOutcome(
            status="pending",
            deletion_committed=True,
            materialization_deleted=True,
            cleanup_pending=True,
            error=(
                f"deletion transaction {staging_dir} has both staged and canonical materializations"
            ),
        )

    if canonical_materialization_exists:
        try:
            _rename_path(materialization_dir, staged_materialization_dir)
        except OSError as exc:
            return _DeletionRecoveryOutcome(
                status="pending",
                deletion_committed=True,
                materialization_deleted=False,
                cleanup_pending=True,
                error=f"could not stage materialization for {record_id}: {exc}",
            )
        staged_materialization_exists = True

    return _cleanup_staging(
        staging_dir,
        deletion_committed=True,
        materialization_deleted=materialization_expected or staged_materialization_exists,
    )


def recover_pending_deletions(repo: StorageRepo) -> DeletionRecoveryResult:
    """Recover transactions found only under the dedicated deletion staging root."""

    root = repo.layout.record_deletion_staging_root
    if not root.is_dir():
        return DeletionRecoveryResult()

    completed: list[str] = []
    rolled_back: list[str] = []
    pending: list[str] = []
    errors: list[str] = []
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return DeletionRecoveryResult(errors=(f"could not scan deletion transactions: {exc}",))

    for candidate in candidates:
        if not candidate.is_dir():
            errors.append(f"unexpected deletion transaction entry: {candidate}")
            continue
        try:
            record_id = validate_record_id(candidate.name)
        except ValueError as exc:
            errors.append(f"invalid deletion transaction {candidate}: {exc}")
            continue
        try:
            with record_lock(repo, record_id):
                outcome = _recover_pending_deletion_locked(repo, record_id)
        except Exception as exc:
            pending.append(record_id)
            errors.append(f"could not recover deletion for {record_id}: {exc}")
            continue

        if outcome.status == "completed":
            completed.append(record_id)
        elif outcome.status == "rolled_back":
            rolled_back.append(record_id)
        else:
            pending.append(record_id)
        if outcome.error:
            errors.append(outcome.error)

    return DeletionRecoveryResult(
        completed_record_ids=tuple(completed),
        rolled_back_record_ids=tuple(rolled_back),
        cleanup_pending_record_ids=tuple(pending),
        errors=tuple(errors),
    )


@dataclass(slots=True, frozen=True)
class PromotionRecoveryResult:
    rolled_back_record_ids: tuple[str, ...] = ()
    cleaned_staging_dirs: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@contextmanager
def promotion_staging_coordination(repo: StorageRepo) -> Iterator[None]:
    """Serialize staging publication with recovery discovery.

    The lock lives beside, rather than inside, the dedicated staging root so it can be
    acquired before a new candidate directory becomes visible to recovery.
    """

    root = repo.layout.promotion_staging_root
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / PROMOTION_COORDINATION_LOCK
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_promotion_manifest(
    repo: StorageRepo,
    staging_dir: Path,
    *,
    record_id: str,
    transaction_id: str,
    state: Literal["staging", "committing", "committed"],
) -> None:
    staged_record_id, separator, suffix = staging_dir.name.rpartition("_")
    if (
        separator != "_"
        or staged_record_id != record_id
        or transaction_id != suffix
        or not _valid_promotion_transaction_id(transaction_id)
    ):
        raise ValueError(
            f"Promotion transaction {transaction_id!r} does not match staging directory "
            f"{staging_dir.name!r} for record {record_id!r}"
        )
    repo.write_json(
        staging_dir / PROMOTION_MANIFEST,
        {
            "schema_version": PROMOTION_MANIFEST_SCHEMA_VERSION,
            "record_id": record_id,
            "transaction_id": transaction_id,
            "state": state,
            "updated_at": _utc_now(),
        },
    )


def _valid_promotion_transaction_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {8, 32}
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_promotion_manifest(
    repo: StorageRepo, staging_dir: Path
) -> tuple[dict[str, Any] | None, str | None]:
    path = staging_dir / PROMOTION_MANIFEST
    if not path.is_file():
        return None, None
    try:
        payload = repo.read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid promotion manifest at {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid promotion manifest at {path}"
    schema_version = payload.get("schema_version")
    if (
        schema_version not in {1, PROMOTION_MANIFEST_SCHEMA_VERSION}
        or payload.get("state") not in _PROMOTION_STATES
        or not isinstance(payload.get("record_id"), str)
    ):
        return None, f"invalid promotion manifest at {path}"
    try:
        record_id = validate_record_id(payload["record_id"])
    except ValueError as exc:
        return None, f"invalid promotion manifest at {path}: {exc}"
    staged_record_id, separator, suffix = staging_dir.name.rpartition("_")
    valid_suffix = _valid_promotion_transaction_id(suffix)
    if (
        record_id != payload["record_id"]
        or separator != "_"
        or staged_record_id != record_id
        or not valid_suffix
    ):
        return None, (
            f"invalid promotion manifest at {path}: record_id {record_id!r} "
            f"does not match staging directory {staging_dir.name!r}"
        )
    if schema_version == PROMOTION_MANIFEST_SCHEMA_VERSION:
        transaction_id = payload.get("transaction_id")
        if transaction_id != suffix or not _valid_promotion_transaction_id(transaction_id):
            return None, (
                f"invalid promotion manifest at {path}: transaction_id "
                f"{transaction_id!r} does not match staging directory {staging_dir.name!r}"
            )
    return payload, None


def _canonical_promotion_transaction_id(repo: StorageRepo, metadata_path: Path) -> str | None:
    if not metadata_path.is_file():
        return None
    try:
        payload = repo.read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    namespace = payload.get("usdcraft")
    if not isinstance(namespace, dict):
        return None
    transaction_id = namespace.get("promotion_transaction_id")
    return transaction_id if _valid_promotion_transaction_id(transaction_id) else None


def _rollback_partial_promotion(
    repo: StorageRepo,
    record_id: str,
    transaction_id: str,
) -> Literal["rolled_back", "superseded", "conflict"]:
    """Remove only canonical state owned by one interrupted promotion transaction."""

    with record_lock(repo, record_id):
        record_dir = repo.layout.record_dir(record_id)
        materialization_dir = repo.layout.record_materialization_dir(record_id)
        existing: list[tuple[Path, str | None]] = []
        if _path_exists(record_dir):
            existing.append(
                (
                    record_dir,
                    _canonical_promotion_transaction_id(
                        repo,
                        repo.layout.record_metadata_path(record_id),
                    ),
                )
            )
        if _path_exists(materialization_dir):
            existing.append(
                (
                    materialization_dir,
                    _canonical_promotion_transaction_id(
                        repo,
                        repo.layout.record_materialization_compile_report_path(record_id),
                    ),
                )
            )

        if not existing:
            return "rolled_back"
        if any(owner is None for _, owner in existing):
            # Missing/corrupt ownership may be an incompletely removed path from
            # this transaction. Retain the manifest so recovery can be retried or
            # inspected instead of silently abandoning an orphan.
            return "conflict"
        ownership = [owner == transaction_id for _, owner in existing]
        if not any(ownership):
            # A later same-record transaction owns the canonical state. The stale
            # transaction may be forgotten, but its successor must not be touched.
            return "superseded"
        if not all(ownership):
            # Mixed ownership is inconsistent and cannot be repaired destructively.
            return "conflict"

        # Remove the cache before its canonical source record. If either operation
        # fails, the retained manifest makes the remaining owned path retryable.
        if _path_exists(materialization_dir):
            _remove_tree(materialization_dir)
        if _path_exists(record_dir):
            _remove_tree(record_dir)
        return "rolled_back"


def recover_pending_promotions(repo: StorageRepo) -> PromotionRecoveryResult:
    """Roll back interrupted Generation promotions left under the promotion staging root.

    A promotion whose process died before reporting success never becomes canonical: a
    manifest in ``committing`` state rolls back its partially committed record, while
    ``staging``/``committed`` leftovers only need their staging directory removed. Recovery
    enumerates only the dedicated ``data/local/promotion_staging/`` root.
    """

    root = repo.layout.promotion_staging_root
    if not root.is_dir():
        return PromotionRecoveryResult()

    rolled_back: list[str] = []
    cleaned: list[str] = []
    errors: list[str] = []
    recoverable: list[tuple[Path, Any]] = []
    try:
        with promotion_staging_coordination(repo):
            candidates = sorted(root.iterdir(), key=lambda path: path.name)
            for candidate in candidates:
                if candidate.is_symlink() or not candidate.is_dir():
                    errors.append(f"unexpected promotion staging entry: {candidate}")
                    continue
                try:
                    activity_lock = (candidate / PROMOTION_ACTIVITY_LOCK).open("a+b")
                except OSError as exc:
                    errors.append(f"could not open promotion activity lock in {candidate}: {exc}")
                    continue
                try:
                    fcntl.flock(activity_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    # A live promotion still holds this staging directory; leave it alone.
                    activity_lock.close()
                    continue
                recoverable.append((candidate, activity_lock))
    except OSError as exc:
        for _, activity_lock in recoverable:
            activity_lock.close()
        return PromotionRecoveryResult(errors=(f"could not scan promotion staging: {exc}",))

    for candidate, activity_lock in recoverable:
        try:
            manifest, manifest_error = _read_promotion_manifest(repo, candidate)
            if manifest_error is not None:
                errors.append(manifest_error)
                continue
            try:
                cleanup_candidate = True
                if manifest is not None and manifest["state"] == "committing":
                    record_id = str(manifest["record_id"])
                    transaction_id = manifest.get("transaction_id")
                    if not isinstance(transaction_id, str):
                        canonical_exists = _path_exists(repo.layout.record_dir(record_id)) or (
                            _path_exists(repo.layout.record_materialization_dir(record_id))
                        )
                        if canonical_exists:
                            cleanup_candidate = False
                            errors.append(
                                f"could not safely recover promotion staging {candidate} "
                                "without a transaction id: canonical ownership is not recorded"
                            )
                        else:
                            rolled_back.append(record_id)
                    else:
                        outcome = _rollback_partial_promotion(
                            repo,
                            record_id,
                            transaction_id,
                        )
                        if outcome == "rolled_back":
                            rolled_back.append(record_id)
                        elif outcome == "conflict":
                            cleanup_candidate = False
                            errors.append(
                                f"could not safely recover promotion staging {candidate}: "
                                "canonical component ownership is missing or mixed"
                            )
                if cleanup_candidate:
                    _remove_tree(candidate)
                    cleaned.append(candidate.name)
            except (OSError, ValueError) as exc:
                errors.append(f"could not recover promotion staging {candidate}: {exc}")
        finally:
            activity_lock.close()

    return PromotionRecoveryResult(
        rolled_back_record_ids=tuple(rolled_back),
        cleaned_staging_dirs=tuple(cleaned),
        errors=tuple(errors),
    )


MUTABLE_PIPELINES = frozenset({"generation", "reconstruction"})


@dataclass(slots=True)
class RecordMutationStore:
    """Narrow canonical mutations used by Workbench controls."""

    repo: StorageRepo

    def update_rating(
        self, record_id: str, rating: int | None, *, pipeline: str = "generation"
    ) -> RatingUpdateResult:
        record_id = validate_record_id(record_id)
        self._validate_rating(rating)
        with record_lock(self.repo, record_id):
            record = self._load_record(record_id, pipeline=pipeline)
            updated_at = _utc_now()
            record["rating"] = rating
            record["updated_at"] = updated_at
            self.repo.write_json(self.repo.layout.record_metadata_path(record_id), record)
        return RatingUpdateResult(
            record_id=record_id,
            rating=rating,
            updated_at=updated_at,
        )

    def delete_record(
        self, record_id: str, *, pipeline: str = "generation"
    ) -> RecordDeletionResult:
        record_id = validate_record_id(record_id)
        with record_lock(self.repo, record_id):
            replacement_staging = self.repo.layout.materialization_replacement_staging_dir(
                record_id
            )
            if replacement_staging.exists() or replacement_staging.is_symlink():
                try:
                    replacement_recovery = recover_materialization_replacement(
                        self.repo,
                        record_id,
                    )
                except MaterializationReplacementRecoveryError as exc:
                    raise RecordDeletionRecoveryError(
                        f"Materialization replacement recovery failed for {record_id}: {exc}"
                    ) from exc
                if replacement_recovery.cleanup_pending:
                    raise RecordDeletionRecoveryError(
                        f"Materialization replacement cleanup is still pending for {record_id}"
                    )
            staging_dir = self.repo.layout.record_deletion_staging_dir(record_id)
            if staging_dir.exists():
                recovered = _recover_pending_deletion_locked(self.repo, record_id)
                if recovered.deletion_committed:
                    return RecordDeletionResult(
                        record_id=record_id,
                        materialization_deleted=recovered.materialization_deleted,
                        cleanup_pending=recovered.cleanup_pending,
                    )
                if recovered.cleanup_pending:
                    raise RecordDeletionRecoveryError(
                        recovered.error or f"Deletion recovery is still pending for {record_id}"
                    )

            self._load_record(record_id, pipeline=pipeline)
            record_dir = self.repo.layout.record_dir(record_id)
            materialization_dir = self.repo.layout.record_materialization_dir(record_id)
            staged_record_dir = staging_dir / "record"
            staged_materialization_dir = staging_dir / "materialization"
            materialization_present = _path_exists(materialization_dir)
            staging_dir.mkdir(parents=True)

            record_staged = False
            materialization_staged = False
            try:
                _write_deletion_manifest(
                    self.repo,
                    staging_dir,
                    record_id=record_id,
                    state="prepared",
                    materialization_expected=materialization_present,
                )
                _rename_path(record_dir, staged_record_dir)
                record_staged = True
                _write_deletion_manifest(
                    self.repo,
                    staging_dir,
                    record_id=record_id,
                    state="record_staged",
                    materialization_expected=materialization_present,
                )
                if materialization_present:
                    _rename_path(materialization_dir, staged_materialization_dir)
                    materialization_staged = True
                _write_deletion_manifest(
                    self.repo,
                    staging_dir,
                    record_id=record_id,
                    state="committed",
                    materialization_expected=materialization_present,
                )
            except BaseException as exc:
                rollback_errors: list[str] = []
                if materialization_staged:
                    try:
                        _rename_path(staged_materialization_dir, materialization_dir)
                        materialization_staged = False
                    except OSError as rollback_exc:
                        rollback_errors.append(f"materialization rollback failed: {rollback_exc}")
                if record_staged:
                    try:
                        _rename_path(staged_record_dir, record_dir)
                        record_staged = False
                    except OSError as rollback_exc:
                        rollback_errors.append(f"record rollback failed: {rollback_exc}")
                if not record_staged and not materialization_staged:
                    shutil.rmtree(staging_dir, ignore_errors=True)
                for message in rollback_errors:
                    exc.add_note(message)
                raise

            cleanup_pending = False
            try:
                _remove_tree(staging_dir)
            except OSError:
                cleanup_pending = True

        return RecordDeletionResult(
            record_id=record_id,
            materialization_deleted=materialization_present,
            cleanup_pending=cleanup_pending,
        )

    def _load_record(self, record_id: str, *, pipeline: str) -> dict[str, Any]:
        """Load one canonical record of the named pipeline; any other record is "not found".

        Rating and deletion are the two Workbench mutations; both Generation and Reconstruction
        records accept them, but each route names its own pipeline so a Generation route can never
        touch a Reconstruction record or the reverse.
        """

        if pipeline not in MUTABLE_PIPELINES:
            raise ValueError(f"unsupported pipeline for record mutations: {pipeline!r}")
        record = self.repo.read_json(self.repo.layout.record_metadata_path(record_id))
        if (
            not isinstance(record, dict)
            or record.get("record_id") != record_id
            or record.get("pipeline", "generation") != pipeline
        ):
            raise RecordNotFoundError(record_id)
        return record

    @staticmethod
    def _validate_rating(rating: int | None) -> None:
        if rating is None:
            return
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise ValueError("rating must be an integer from 1 to 5, or null")


__all__ = [
    "MUTABLE_PIPELINES",
    "DeletionRecoveryResult",
    "PROMOTION_ACTIVITY_LOCK",
    "PROMOTION_COORDINATION_LOCK",
    "PROMOTION_MANIFEST",
    "PROMOTION_MANIFEST_SCHEMA_VERSION",
    "PromotionRecoveryResult",
    "RatingUpdateResult",
    "RecordDeletionRecoveryError",
    "RecordDeletionResult",
    "RecordMutationStore",
    "RecordNotFoundError",
    "recover_pending_deletions",
    "recover_pending_promotions",
    "promotion_staging_coordination",
    "write_promotion_manifest",
]
