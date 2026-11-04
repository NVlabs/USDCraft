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

import os
from pathlib import Path

import pytest

from storage.artifact_contract import MaterializationDisplayResult
from viewer.api.validation_cache import MaterializationValidationCache, materialization_signature


def test_materialization_signature_counts_regular_files_recursively(tmp_path: Path) -> None:
    root = tmp_path / "materialization"
    (root / "assets" / "nested").mkdir(parents=True)
    (root / "model.urdf").write_bytes(b"a" * 10)
    (root / "assets" / "mesh.glb").write_bytes(b"b" * 30)
    (root / "assets" / "nested" / "deep.txt").write_bytes(b"c" * 5)
    initial = materialization_signature(root)
    assert [entry[0] for entry in initial] == [
        "assets/mesh.glb",
        "assets/nested/deep.txt",
        "model.urdf",
    ]
    assert sum(entry[1] for entry in initial) == 45
    assert materialization_signature(root) == initial
    (root / "assets" / "nested" / "deep.txt").write_bytes(b"c" * 6)
    assert materialization_signature(root) != initial
    (root / "extra.json").write_bytes(b"{}")
    assert len(materialization_signature(root)) == 4
    assert materialization_signature(tmp_path / "absent") is None


def test_materialization_signature_does_not_follow_directory_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "materialization"
    root.mkdir()
    (root / "model.urdf").write_bytes(b"a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 1000)
    os.symlink(outside, root / "linked_dir", target_is_directory=True)
    os.symlink(outside / "big.bin", root / "linked_file.bin")

    # A symlinked directory is not walked; a symlinked file counts like ``Path.is_file()`` does.
    assert len(materialization_signature(root)) == 2
    assert sum(entry[1] for entry in materialization_signature(root)) == 1001


@pytest.mark.parametrize("change", ["rename", "older_edit", "preserved_mtime", "replace"])
def test_cache_invalidates_metadata_collisions_without_content_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    older, newer = tmp_path / "older", tmp_path / "newer"
    older.write_bytes(b"aa")
    newer.write_bytes(b"bb")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(3_000_000_000, 3_000_000_000))
    initial = materialization_signature(tmp_path)
    cache = MaterializationValidationCache()
    result = MaterializationDisplayResult(status="available", issues=())
    cache.put("rec_test", "rev_000001", tmp_path, result)
    assert cache.get("rec_test", "rev_000001", tmp_path) is result
    if change == "rename":
        older.rename(tmp_path / "renamed")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"zz")
        os.utime(replacement, ns=(1_000_000_000, 1_000_000_000))
        replacement.replace(older)
    else:
        older.write_bytes(b"zz")
        stamp = 2_000_000_000 if change == "older_edit" else 1_000_000_000
        os.utime(older, ns=(stamp, stamp))

    def unexpected_read(*args, **kwargs):
        pytest.fail("cache signature must not read file contents")

    monkeypatch.setattr(Path, "open", unexpected_read)
    assert materialization_signature(tmp_path) != initial
    assert cache.get("rec_test", "rev_000001", tmp_path) is None


def test_unreadable_scan_does_not_reuse_partial_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args, **kwargs):
        raise PermissionError("cannot enumerate")

    monkeypatch.setattr(os, "scandir", unreadable)
    assert materialization_signature(tmp_path) is None


@pytest.mark.parametrize("pipeline", ["generation", "reconstruction"])
def test_listing_after_a_materialization_change_reports_not_validated(
    tmp_path: Path, pipeline: str
) -> None:
    from storage.repo import StorageRepo
    from tests.viewer.test_cli import _write_generation_record_with_invalid_materialization
    from tests.viewer.test_reconstruction_records import _write_reconstruction_record
    from viewer.api import reconstruction_records, store

    if pipeline == "generation":
        _write_generation_record_with_invalid_materialization(tmp_path)
        viewer_store = store.ViewerStore(tmp_path)
    else:
        _write_reconstruction_record(tmp_path)
        viewer_store = reconstruction_records.ReconstructionRecordStore(tmp_path)
    record_id = f"rec_{pipeline}_001"
    root = StorageRepo(tmp_path).layout.record_materialization_dir(record_id)
    extra = root / "extra.txt"
    extra.write_text("aa")
    viewer_store.validate(record_id)
    assert viewer_store.list_records()[0].materialization_validated is True
    extra.rename(root / "renamed.txt")

    # Listing only compares the stat signature; it reports the change instead of re-checking.
    summary = viewer_store.list_records()[0]
    assert summary.materialization_status == "not_validated"
    assert summary.materialization_validated is False
