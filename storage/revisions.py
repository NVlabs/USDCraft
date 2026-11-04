# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

import hashlib
import re
from pathlib import Path
from typing import Any

from storage.identifiers import validate_record_id
from storage.repo import StorageRepo

INITIAL_REVISION_ID = "rev_000001"
REVISION_ID_RE = re.compile(r"^rev_[0-9]{6}$")


def validate_revision_id(value: str | None, *, label: str = "revision_id") -> str:
    revision_id = str(value or "").strip()
    if not revision_id:
        raise ValueError(f"{label} is required.")
    if not REVISION_ID_RE.fullmatch(revision_id):
        raise ValueError(f"{label} must match rev_000001 style six-digit IDs.")
    return revision_id


def active_revision_id(
    repo: StorageRepo,
    record_id: str,
    record: dict[str, Any] | None = None,
) -> str:
    record_id = validate_record_id(record_id)
    payload = record
    if payload is None:
        payload = repo.read_json(repo.layout.record_metadata_path(record_id))
    if isinstance(payload, dict) and payload.get("active_revision_id") is not None:
        return validate_revision_id(str(payload["active_revision_id"]))
    return INITIAL_REVISION_ID


def revision_relative_path(revision_id: str, filename: str) -> str:
    return f"revisions/{validate_revision_id(revision_id)}/{filename}"


def active_artifact_path(
    repo: StorageRepo,
    record_id: str,
    artifact_key: str,
    *,
    record: dict[str, Any] | None = None,
) -> Path | None:
    record_id = validate_record_id(record_id)
    payload = record
    if payload is None:
        payload = repo.read_json(repo.layout.record_metadata_path(record_id))
    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    value = artifacts.get(artifact_key) if isinstance(artifacts, dict) else None
    if isinstance(value, str) and value.strip():
        relative = Path(value)
        if not relative.is_absolute() and ".." not in relative.parts:
            return repo.layout.record_dir(record_id) / relative
    fallback_names = {
        "prompt_txt": "prompt.txt",
        "request_txt": "request.txt",
        "model_py": "model.py",
        "provenance_json": "provenance.json",
        "cost_json": "cost.json",
        "inputs_dir": "inputs",
        "input_manifest_json": "inputs/manifest.json",
        "traces_dir": "traces",
        "generation_json": "generation.json",
        "reconstruction_json": "reconstruction.json",
    }
    filename = fallback_names.get(artifact_key)
    if filename is None:
        return None
    revision_id = active_revision_id(repo, record_id, record=payload)
    return repo.layout.record_revision_dir(record_id, revision_id) / filename


def _active_path(
    repo: StorageRepo,
    record_id: str,
    artifact_key: str,
    filename: str,
    record: dict[str, Any] | None,
) -> Path:
    path = active_artifact_path(repo, record_id, artifact_key, record=record)
    if path is not None:
        return path
    return (
        repo.layout.record_revision_dir(
            record_id,
            active_revision_id(repo, record_id, record=record),
        )
        / filename
    )


def active_prompt_path(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "prompt_txt", "prompt.txt", record)


def active_request_path(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "request_txt", "request.txt", record)


def active_model_path(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "model_py", "model.py", record)


def active_provenance_path(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "provenance_json", "provenance.json", record)


def active_cost_path(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "cost_json", "cost.json", record)


def active_inputs_dir(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "inputs_dir", "inputs", record)


def active_traces_dir(
    repo: StorageRepo, record_id: str, *, record: dict[str, Any] | None = None
) -> Path:
    return _active_path(repo, record_id, "traces_dir", "traces", record)


def revision_artifacts_payload(*, revision_id: str, has_cost_file: bool) -> dict[str, str | None]:
    return {
        "prompt_txt": revision_relative_path(revision_id, "prompt.txt"),
        "model_py": revision_relative_path(revision_id, "model.py"),
        "provenance_json": revision_relative_path(revision_id, "provenance.json"),
        "cost_json": revision_relative_path(revision_id, "cost.json") if has_cost_file else None,
        "inputs_dir": revision_relative_path(revision_id, "inputs"),
        "traces_dir": revision_relative_path(revision_id, "traces"),
    }


def build_revision_payload(
    *,
    record_id: str,
    revision_id: str,
    created_at: str,
    prompt_text: str,
    prompt_kind: str,
    source: dict[str, Any],
    generation: dict[str, Any],
    artifacts: dict[str, Any],
    hashes: dict[str, Any],
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_id": validate_record_id(record_id),
        "revision_id": validate_revision_id(revision_id),
        "created_at": created_at,
        "prompt_kind": prompt_kind,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "source": dict(source),
        "generation": dict(generation),
        "artifacts": dict(artifacts),
        "hashes": dict(hashes),
        "run_summary": dict(run_summary or {}),
    }
