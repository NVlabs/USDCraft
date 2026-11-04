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
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ContextManager

from storage.collections import CollectionStore
from storage.identifiers import validate_record_id
from storage.record_locks import record_lock
from storage.record_mutations import (
    PROMOTION_ACTIVITY_LOCK,
    promotion_staging_coordination,
    write_promotion_manifest,
)
from storage.repo import StorageRepo

PromotionManifestWriter = Callable[..., None]
RecordLockFactory = Callable[[StorageRepo, str], ContextManager[None]]


@dataclass(slots=True)
class CanonicalPromotion:
    """One recoverable canonical record/materialization publication transaction.

    Product-specific persistence code owns the contents of ``record_dir`` and
    ``materialization_dir``. This object owns only publication, conflict handling, rollback and
    recovery identity, so Generation and Reconstruction cannot acquire different atomicity rules.
    """

    repo: StorageRepo
    record_id: str
    transaction_id: str
    root: Path
    record_dir: Path
    materialization_dir: Path
    final_record_dir: Path
    final_materialization_dir: Path
    activity_lock: BinaryIO


def begin_canonical_promotion(
    repo: StorageRepo,
    record_id: str,
    *,
    transaction_id: str | None = None,
    manifest_writer: PromotionManifestWriter = write_promotion_manifest,
) -> CanonicalPromotion:
    """Create and lock one recovery-visible staging transaction."""

    record_id = validate_record_id(record_id)
    selected_transaction_id = transaction_id or uuid.uuid4().hex
    root = repo.layout.promotion_staging_root / f"{record_id}_{selected_transaction_id}"
    record_dir = root / "record"
    materialization_dir = root / "materialization"
    final_record_dir = repo.layout.record_dir(record_id)
    final_materialization_dir = repo.layout.record_materialization_dir(record_id)

    with promotion_staging_coordination(repo):
        root.mkdir()
        activity_lock = (root / PROMOTION_ACTIVITY_LOCK).open("a+b")
        fcntl.flock(activity_lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest_writer(
                repo,
                root,
                record_id=record_id,
                transaction_id=selected_transaction_id,
                state="staging",
            )
        except BaseException:
            activity_lock.close()
            raise

    return CanonicalPromotion(
        repo=repo,
        record_id=record_id,
        transaction_id=selected_transaction_id,
        root=root,
        record_dir=record_dir,
        materialization_dir=materialization_dir,
        final_record_dir=final_record_dir,
        final_materialization_dir=final_materialization_dir,
        activity_lock=activity_lock,
    )


def commit_canonical_promotion(
    promotion: CanonicalPromotion,
    *,
    added_at: str,
    label: str | None,
    manifest_writer: PromotionManifestWriter = write_promotion_manifest,
    record_lock_factory: RecordLockFactory = record_lock,
) -> Path:
    """Atomically publish one fully staged record plus its materialization."""

    if not promotion.record_dir.is_dir():
        raise FileNotFoundError(f"Canonical promotion has no staged record: {promotion.record_dir}")
    if not promotion.materialization_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical promotion has no staged materialization: {promotion.materialization_dir}"
        )

    promotion.final_record_dir.parent.mkdir(parents=True, exist_ok=True)
    promotion.final_materialization_dir.parent.mkdir(parents=True, exist_ok=True)
    record_committed = False
    materialization_committed = False
    with record_lock_factory(promotion.repo, promotion.record_id):
        pending_deletion = promotion.repo.layout.record_deletion_staging_dir(promotion.record_id)
        if (
            promotion.final_record_dir.exists()
            or promotion.final_materialization_dir.exists()
            or pending_deletion.exists()
        ):
            raise FileExistsError(f"Refusing to overwrite existing record {promotion.record_id}")
        manifest_writer(
            promotion.repo,
            promotion.root,
            record_id=promotion.record_id,
            transaction_id=promotion.transaction_id,
            state="committing",
        )
        try:
            promotion.record_dir.replace(promotion.final_record_dir)
            record_committed = True
            promotion.materialization_dir.replace(promotion.final_materialization_dir)
            materialization_committed = True
            CollectionStore(promotion.repo).append_workbench_entry(
                record_id=promotion.record_id,
                added_at=added_at,
                label=label,
                tags=[],
            )
            manifest_writer(
                promotion.repo,
                promotion.root,
                record_id=promotion.record_id,
                transaction_id=promotion.transaction_id,
                state="committed",
            )
        except BaseException:
            # Best-effort rollback must never mask the commit failure. The retained transaction
            # manifest lets storage recovery finish anything suppressed here.
            try:
                CollectionStore(promotion.repo).remove_workbench_entries(promotion.record_id)
            except Exception:
                pass
            if materialization_committed:
                shutil.rmtree(promotion.final_materialization_dir, ignore_errors=True)
            if record_committed:
                shutil.rmtree(promotion.final_record_dir, ignore_errors=True)
            raise
    return promotion.final_record_dir


def finish_canonical_promotion(
    promotion: CanonicalPromotion,
    *,
    succeeded: bool,
) -> None:
    """Release the activity lock and discard only a successfully committed staging root."""

    promotion.activity_lock.close()
    if succeeded:
        shutil.rmtree(promotion.root, ignore_errors=True)


__all__ = [
    "CanonicalPromotion",
    "PromotionManifestWriter",
    "RecordLockFactory",
    "begin_canonical_promotion",
    "commit_canonical_promotion",
    "finish_canonical_promotion",
]
