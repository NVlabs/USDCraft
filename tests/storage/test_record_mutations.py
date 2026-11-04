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

from pathlib import Path

import pytest

import storage.record_mutations as record_mutations
from storage.data_validation import validate_data_format
from storage.record_mutations import (
    RecordMutationStore,
    RecordNotFoundError,
    recover_pending_deletions,
)
from storage.repo import StorageRepo


def _write_record(
    repo: StorageRepo,
    record_id: str,
) -> None:
    repo.write_json(
        repo.layout.record_metadata_path(record_id),
        {
            "schema_version": 3,
            "record_id": record_id,
            "pipeline": "generation",
            "rating": None,
            "updated_at": "2026-07-13T00:00:00Z",
            "lineage": {
                "origin_record_id": record_id,
                "parent_record_id": None,
            },
        },
    )
    repo.write_text(repo.layout.record_dir(record_id) / "payload.txt", record_id)


def test_rating_updates_only_record_metadata_and_supports_clear(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_rating"
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "unchanged")
    mutations = RecordMutationStore(repo)

    updated = mutations.update_rating(record_id, 4)
    assert updated.rating == 4
    record = repo.read_json(repo.layout.record_metadata_path(record_id))
    assert record["rating"] == 4
    assert record["updated_at"] == updated.updated_at
    assert (repo.layout.record_dir(record_id) / "payload.txt").read_text() == record_id
    assert (materialization / "model.urdf").read_text() == "unchanged"

    cleared = mutations.update_rating(record_id, None)
    assert cleared.rating is None
    assert repo.read_json(repo.layout.record_metadata_path(record_id))["rating"] is None

    for invalid in (0, 6, True, 2.5, "3"):
        with pytest.raises(ValueError, match="rating"):
            mutations.update_rating(record_id, invalid)  # type: ignore[arg-type]

    with pytest.raises(RecordNotFoundError):
        mutations.update_rating("rec_missing", 3)


def test_mutations_reject_mismatched_record_payload(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    _write_record(repo, "rec_directory")
    record_path = repo.layout.record_metadata_path("rec_directory")
    payload = repo.read_json(record_path)
    payload["record_id"] = "rec_different"
    repo.write_json(record_path, payload)

    with pytest.raises(RecordNotFoundError):
        RecordMutationStore(repo).update_rating("rec_directory", 3)
    with pytest.raises(RecordNotFoundError):
        RecordMutationStore(repo).delete_record("rec_directory")
    assert repo.layout.record_dir("rec_directory").is_dir()


def test_data_validation_rejects_out_of_range_or_boolean_rating(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_invalid_rating"
    _write_record(repo, record_id)
    record_path = repo.layout.record_metadata_path(record_id)

    for invalid_rating in (0, 6, True):
        payload = repo.read_json(record_path)
        payload["rating"] = invalid_rating
        repo.write_json(record_path, payload)
        result = validate_data_format(repo, record_ids=[record_id])
        assert any(
            "rating must be an integer from 1 to 5, or null" in error for error in result.errors
        )


def test_delete_renames_record_and_materialization_and_preserves_local_run(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_delete"
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    retained = repo.layout.generation_run_dir("run_retained") / "trace.jsonl"
    repo.write_text(retained, "trace")

    result = RecordMutationStore(repo).delete_record(record_id)

    assert result.materialization_deleted is True
    assert result.cleanup_pending is False
    assert not repo.layout.record_dir(record_id).exists()
    assert not materialization.exists()
    assert retained.read_text() == "trace"
    assert list(repo.layout.record_deletion_staging_root.iterdir()) == []


def test_delete_without_materialization_reports_false(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    _write_record(repo, "rec_no_materialization")

    result = RecordMutationStore(repo).delete_record("rec_no_materialization")

    assert result.materialization_deleted is False
    assert result.cleanup_pending is False


def test_delete_does_not_scan_or_parse_unrelated_records(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    _write_record(repo, "rec_target")
    bad_record = repo.layout.record_metadata_path("rec_bad_neighbor")
    bad_record.parent.mkdir(parents=True)
    bad_record.write_text("{bad json", encoding="utf-8")

    result = RecordMutationStore(repo).delete_record("rec_target")

    assert result.record_id == "rec_target"
    assert not repo.layout.record_dir("rec_target").exists()
    assert bad_record.read_text(encoding="utf-8") == "{bad json"


def test_delete_rolls_back_record_when_materialization_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_rollback"
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    original_rename = record_mutations._rename_path

    def fail_materialization(source: Path, destination: Path) -> None:
        if source == materialization:
            raise OSError("simulated materialization rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(record_mutations, "_rename_path", fail_materialization)

    with pytest.raises(OSError, match="simulated"):
        RecordMutationStore(repo).delete_record(record_id)

    assert (repo.layout.record_dir(record_id) / "payload.txt").read_text() == record_id
    assert (materialization / "model.urdf").read_text() == "compiled"
    assert list(repo.layout.record_deletion_staging_root.iterdir()) == []


def test_delete_commits_when_staging_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_cleanup_pending"
    _write_record(repo, record_id)
    repo.write_text(repo.layout.record_materialization_dir(record_id) / "model.urdf", "compiled")

    def fail_cleanup(path: Path) -> None:
        raise OSError(f"cannot clean {path}")

    monkeypatch.setattr(record_mutations, "_remove_tree", fail_cleanup)

    result = RecordMutationStore(repo).delete_record(record_id)

    assert result.cleanup_pending is True
    assert not repo.layout.record_dir(record_id).exists()
    assert not repo.layout.record_materialization_dir(record_id).exists()
    assert repo.layout.record_deletion_staging_dir(record_id).is_dir()


def test_recovery_finishes_crash_after_record_was_staged(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_crash_recovery"
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    staging = repo.layout.record_deletion_staging_dir(record_id)
    staging.mkdir(parents=True)
    repo.write_json(
        staging / "transaction.json",
        {
            "schema_version": 1,
            "record_id": record_id,
            "state": "record_staged",
            "materialization_expected": True,
        },
    )
    repo.layout.record_dir(record_id).replace(staging / "record")

    recovery = recover_pending_deletions(repo)

    assert recovery.completed_record_ids == (record_id,)
    assert recovery.rolled_back_record_ids == ()
    assert recovery.cleanup_pending_record_ids == ()
    assert not repo.layout.record_dir(record_id).exists()
    assert not materialization.exists()
    assert not staging.exists()


def test_recovery_rolls_back_valid_prepared_transaction(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_prepared_recovery"
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    staging = repo.layout.record_deletion_staging_dir(record_id)
    staging.mkdir(parents=True)
    repo.write_json(
        staging / "transaction.json",
        {
            "schema_version": 1,
            "record_id": record_id,
            "state": "prepared",
            "materialization_expected": True,
        },
    )

    recovery = recover_pending_deletions(repo)

    assert recovery.completed_record_ids == ()
    assert recovery.rolled_back_record_ids == (record_id,)
    assert recovery.cleanup_pending_record_ids == ()
    assert repo.layout.record_dir(record_id).is_dir()
    assert materialization.is_dir()
    assert not staging.exists()


@pytest.mark.parametrize(
    ("record_id", "manifest"),
    (
        ("rec_missing_manifest", None),
        (
            "rec_invalid_manifest",
            {
                "schema_version": 99,
                "record_id": "rec_invalid_manifest",
                "state": "record_staged",
                "materialization_expected": True,
            },
        ),
    ),
)
def test_recovery_does_not_touch_transactions_without_a_valid_manifest(
    tmp_path: Path,
    record_id: str,
    manifest: dict[str, object] | None,
) -> None:
    repo = StorageRepo(tmp_path)
    _write_record(repo, record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    staging = repo.layout.record_deletion_staging_dir(record_id)
    staging.mkdir(parents=True)
    if manifest is not None:
        repo.write_json(staging / "transaction.json", manifest)
    repo.layout.record_dir(record_id).replace(staging / "record")

    recovery = recover_pending_deletions(repo)

    assert recovery.completed_record_ids == ()
    assert recovery.cleanup_pending_record_ids == (record_id,)
    assert recovery.errors
    assert (staging / "record").is_dir()
    assert materialization.is_dir()
    assert not repo.layout.record_dir(record_id).exists()


def test_recovery_retries_committed_tombstone_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_cleanup_retry"
    _write_record(repo, record_id)
    repo.write_text(repo.layout.record_materialization_dir(record_id) / "model.urdf", "compiled")
    original_remove_tree = record_mutations._remove_tree

    def fail_cleanup(path: Path) -> None:
        raise OSError(f"cannot clean {path}")

    monkeypatch.setattr(record_mutations, "_remove_tree", fail_cleanup)
    deleted = RecordMutationStore(repo).delete_record(record_id)
    assert deleted.cleanup_pending is True
    assert repo.layout.record_deletion_staging_dir(record_id).is_dir()

    monkeypatch.setattr(record_mutations, "_remove_tree", original_remove_tree)
    recovery = recover_pending_deletions(repo)

    assert recovery.completed_record_ids == (record_id,)
    assert recovery.cleanup_pending_record_ids == ()
    assert not repo.layout.record_deletion_staging_dir(record_id).exists()


def test_delete_retry_resumes_when_initial_rollback_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_retry_recovery"
    _write_record(repo, record_id)
    record_dir = repo.layout.record_dir(record_id)
    materialization = repo.layout.record_materialization_dir(record_id)
    repo.write_text(materialization / "model.urdf", "compiled")
    staging = repo.layout.record_deletion_staging_dir(record_id)
    original_rename = record_mutations._rename_path

    def fail_stage_and_rollback(source: Path, destination: Path) -> None:
        if source == materialization:
            raise OSError("simulated materialization staging failure")
        if source == staging / "record" and destination == record_dir:
            raise OSError("simulated record rollback failure")
        original_rename(source, destination)

    monkeypatch.setattr(record_mutations, "_rename_path", fail_stage_and_rollback)
    with pytest.raises(OSError, match="materialization staging") as exc_info:
        RecordMutationStore(repo).delete_record(record_id)
    assert any("record rollback failed" in note for note in exc_info.value.__notes__)
    assert not record_dir.exists()
    assert materialization.exists()
    assert (staging / "record").is_dir()

    monkeypatch.setattr(record_mutations, "_rename_path", original_rename)
    retried = RecordMutationStore(repo).delete_record(record_id)

    assert retried.materialization_deleted is True
    assert retried.cleanup_pending is False
    assert not record_dir.exists()
    assert not materialization.exists()
    assert not staging.exists()


def test_mutations_accept_reconstruction_records_only_through_their_own_pipeline(
    tmp_path: Path,
) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_reconstruct_rating"
    repo.write_json(
        repo.layout.record_metadata_path(record_id),
        {
            "schema_version": 3,
            "record_id": record_id,
            "pipeline": "reconstruction",
            "rating": None,
            "updated_at": "2026-09-05T00:00:00Z",
        },
    )
    store = RecordMutationStore(repo)

    # The Generation-bound call never touches a Reconstruction record.
    with pytest.raises(RecordNotFoundError):
        store.update_rating(record_id, 4)
    with pytest.raises(RecordNotFoundError):
        store.delete_record(record_id)
    assert repo.read_json(repo.layout.record_metadata_path(record_id))["rating"] is None

    result = store.update_rating(record_id, 4, pipeline="reconstruction")
    assert result.rating == 4
    assert repo.read_json(repo.layout.record_metadata_path(record_id))["rating"] == 4

    deleted = store.delete_record(record_id, pipeline="reconstruction")
    assert deleted.record_id == record_id and deleted.materialization_deleted is False
    assert not repo.layout.record_dir(record_id).exists()

    with pytest.raises(ValueError, match="unsupported pipeline"):
        store.update_rating(record_id, 1, pipeline="edit")
