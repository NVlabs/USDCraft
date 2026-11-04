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
import threading
from pathlib import Path
from typing import Literal

from storage.collections import CollectionStore
from storage.record_mutations import (
    PROMOTION_ACTIVITY_LOCK,
    PROMOTION_MANIFEST,
    PromotionRecoveryResult,
    promotion_staging_coordination,
    recover_pending_promotions,
    write_promotion_manifest,
)
from storage.repo import StorageRepo


def _staging_dir(repo: StorageRepo, record_id: str, suffix: str = "abcd1234") -> Path:
    staging = repo.layout.promotion_staging_root / f"{record_id}_{suffix}"
    staging.mkdir(parents=True)
    return staging


def _transaction_id(staging: Path) -> str:
    return staging.name.rpartition("_")[2]


def _write_manifest(
    repo: StorageRepo,
    staging: Path,
    record_id: str,
    state: Literal["staging", "committing", "committed"],
) -> None:
    write_promotion_manifest(
        repo,
        staging,
        record_id=record_id,
        transaction_id=_transaction_id(staging),
        state=state,
    )


def _write_partial_commit(
    repo: StorageRepo,
    record_id: str,
    *,
    transaction_id: str = "ffffffffffffffffffffffffffffffff",
) -> None:
    repo.write_json(
        repo.layout.record_metadata_path(record_id),
        {
            "schema_version": 3,
            "record_id": record_id,
            "pipeline": "generation",
            "usdcraft": {"promotion_transaction_id": transaction_id},
        },
    )
    materialization = repo.layout.record_materialization_dir(record_id)
    materialization.mkdir(parents=True)
    (materialization / "model.urdf").write_text("<robot/>", encoding="utf-8")
    repo.write_json(
        materialization / "compile_report.json",
        {"usdcraft": {"promotion_transaction_id": transaction_id}},
    )
    CollectionStore(repo).append_workbench_entry(
        record_id=record_id,
        added_at="2026-07-31T00:00:00Z",
    )


def test_recovery_rolls_back_interrupted_committing_promotion(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_interrupted"
    staging = _staging_dir(repo, record_id)
    _write_manifest(repo, staging, record_id, "committing")
    _write_partial_commit(repo, record_id, transaction_id=_transaction_id(staging))

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == (record_id,)
    assert result.errors == ()
    assert not repo.layout.record_dir(record_id).exists()
    assert not repo.layout.record_materialization_dir(record_id).exists()
    assert not staging.exists()


def test_recovery_only_cleans_staging_for_pre_commit_and_committed_states(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_completed"
    _write_partial_commit(repo, record_id)

    staged = _staging_dir(repo, record_id, "aaaa0001")
    _write_manifest(repo, staged, record_id, "staging")
    committed = _staging_dir(repo, record_id, "aaaa0002")
    _write_manifest(repo, committed, record_id, "committed")

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert set(result.cleaned_staging_dirs) == {staged.name, committed.name}
    assert not staged.exists()
    assert not committed.exists()
    # The canonical record was fully committed by its own run; recovery must not touch it.
    assert repo.layout.record_metadata_path(record_id).is_file()
    assert repo.layout.record_materialization_dir(record_id).is_dir()


def test_recovery_skips_staging_held_by_a_live_promotion(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_live"
    staging = _staging_dir(repo, record_id)
    _write_manifest(repo, staging, record_id, "staging")

    with (staging / PROMOTION_ACTIVITY_LOCK).open("a+b") as live_handle:
        fcntl.flock(live_handle.fileno(), fcntl.LOCK_EX)
        result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert result.cleaned_staging_dirs == ()
    assert result.errors == ()
    assert staging.exists()


def test_recovery_reports_invalid_manifest_without_touching_canonical_state(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_garbage"
    _write_partial_commit(repo, record_id)
    staging = _staging_dir(repo, record_id)
    (staging / "promotion.json").write_text("{not json", encoding="utf-8")

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert result.errors and "invalid promotion manifest" in result.errors[0]
    assert staging.exists()
    assert repo.layout.record_metadata_path(record_id).is_file()


def test_recovery_rejects_manifest_bound_to_a_different_staging_directory(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    canonical_record_id = "rec_prefix"
    _write_partial_commit(repo, canonical_record_id)
    # The canonical record id is a prefix of this staging name; binding must be exact.
    staging = _staging_dir(repo, "rec_prefix_collision")
    repo.write_json(
        staging / PROMOTION_MANIFEST,
        {
            "schema_version": 2,
            "record_id": canonical_record_id,
            "transaction_id": _transaction_id(staging),
            "state": "committing",
        },
    )

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert result.errors and "does not match staging directory" in result.errors[0]
    assert staging.exists()
    assert repo.layout.record_metadata_path(canonical_record_id).is_file()
    assert repo.layout.record_materialization_dir(canonical_record_id).is_dir()


def test_recovery_cannot_observe_candidate_before_activity_lock(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_publication_lock"
    staging = repo.layout.promotion_staging_root / f"{record_id}_abcd1234"
    result_holder: list[PromotionRecoveryResult] = []
    started = threading.Event()

    def recover() -> None:
        started.set()
        result_holder.append(recover_pending_promotions(repo))

    with promotion_staging_coordination(repo):
        staging.mkdir()
        live_handle = (staging / PROMOTION_ACTIVITY_LOCK).open("a+b")
        fcntl.flock(live_handle.fileno(), fcntl.LOCK_EX)
        _write_manifest(repo, staging, record_id, "staging")
        worker = threading.Thread(target=recover, daemon=True)
        worker.start()
        assert started.wait(timeout=2)
        worker.join(timeout=0.05)
        assert worker.is_alive()

    worker.join(timeout=2)
    live_handle.close()

    assert not worker.is_alive()
    assert len(result_holder) == 1
    result = result_holder[0]
    assert result.rolled_back_record_ids == ()
    assert result.cleaned_staging_dirs == ()
    assert result.errors == ()
    assert staging.is_dir()
    assert (staging / PROMOTION_MANIFEST).is_file()


def test_recovery_removes_manifestless_staging_leftovers(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    staging = _staging_dir(repo, "rec_manifestless")
    (staging / "record").mkdir()

    result = recover_pending_promotions(repo)

    assert result.cleaned_staging_dirs == (staging.name,)
    assert result.rolled_back_record_ids == ()
    assert not staging.exists()


def test_recovery_does_not_delete_newer_same_record_transaction(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_same_id_successor"
    stale = _staging_dir(repo, record_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    _write_manifest(repo, stale, record_id, "committing")
    successor_transaction_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    _write_partial_commit(
        repo,
        record_id,
        transaction_id=successor_transaction_id,
    )

    result = recover_pending_promotions(repo)

    assert result.errors == ()
    assert result.rolled_back_record_ids == ()
    assert result.cleaned_staging_dirs == (stale.name,)
    assert repo.layout.record_metadata_path(record_id).is_file()
    assert repo.layout.record_materialization_compile_report_path(record_id).is_file()
    assert (
        repo.read_json(repo.layout.record_metadata_path(record_id))["usdcraft"][
            "promotion_transaction_id"
        ]
        == successor_transaction_id
    )


def test_recovery_preserves_unowned_materialization_when_record_is_absent(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_materialization_successor"
    stale = _staging_dir(repo, record_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    _write_manifest(repo, stale, record_id, "committing")
    successor_transaction_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_json(
        materialization / "compile_report.json",
        {"usdcraft": {"promotion_transaction_id": successor_transaction_id}},
    )

    result = recover_pending_promotions(repo)

    assert result.errors == ()
    assert result.rolled_back_record_ids == ()
    assert result.cleaned_staging_dirs == (stale.name,)
    assert materialization.is_dir()


def test_recovery_retains_evidence_for_canonical_path_with_unknown_ownership(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_unknown_owner"
    staging = _staging_dir(repo, record_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    _write_manifest(repo, staging, record_id, "committing")
    materialization = repo.layout.record_materialization_dir(record_id)
    materialization.mkdir(parents=True)
    (materialization / "orphan.txt").write_text("rollback was incomplete", encoding="utf-8")

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert result.cleaned_staging_dirs == ()
    assert result.errors and "ownership is missing or mixed" in result.errors[0]
    assert staging.is_dir()
    assert materialization.is_dir()


def test_committing_manifest_without_transaction_never_deletes_unowned_canonical_state(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_no_transaction"
    staging = _staging_dir(repo, record_id)
    repo.write_json(
        staging / PROMOTION_MANIFEST,
        {
            "schema_version": 1,
            "record_id": record_id,
            "state": "committing",
        },
    )
    _write_partial_commit(repo, record_id)

    result = recover_pending_promotions(repo)

    assert result.rolled_back_record_ids == ()
    assert result.errors and "canonical ownership is not recorded" in result.errors[0]
    assert staging.is_dir()
    assert repo.layout.record_metadata_path(record_id).is_file()
    assert repo.layout.record_materialization_dir(record_id).is_dir()
