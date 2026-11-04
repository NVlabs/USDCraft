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

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from storage.artifact_contract import (
    COMPILE_REPORT_SCHEMA_VERSION,
    GENERATION_MANIFEST_SCHEMA_VERSION,
    GENERATION_VALIDATION_PROFILE,
    physics_output_contract,
    validate_materialization_contract,
)
from storage.identifiers import validate_record_id
from storage.materialization_mutations import (
    MaterializationReplacement,
    abort_materialization_replacement,
    begin_materialization_replacement,
    commit_materialization_replacement,
    recover_materialization_replacement,
)
from storage.record_locks import record_lock
from storage.repo import StorageRepo
from storage.revisions import validate_revision_id
from usdcraft.compile.command import compile_model


class RematerializationScopeError(ValueError):
    """Raised when a record is outside the explicit current Generation repair scope."""


@dataclass(slots=True, frozen=True)
class RematerializationResult:
    record_id: str
    revision_id: str
    model_path: Path
    materialization_dir: Path
    report_path: Path
    transaction_id: str
    warnings: tuple[str, ...]
    recovered: bool = False
    cleanup_pending: bool = False


@dataclass(slots=True, frozen=True)
class _CurrentGenerationSource:
    record_id: str
    revision_id: str
    record_dir: Path
    revision_dir: Path
    model_path: Path
    inputs_dir: Path
    generation_manifest: dict[str, Any]
    promotion_transaction_id: str
    model_sha256: str
    source_fingerprint: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RematerializationScopeError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RematerializationScopeError(f"{label} is invalid at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RematerializationScopeError(f"{label} must be a JSON object: {path}")
    return payload


def _promotion_transaction_id(payload: Mapping[str, Any], *, label: str) -> str:
    namespace = payload.get("usdcraft")
    if not isinstance(namespace, Mapping):
        raise RematerializationScopeError(f"{label}.usdcraft must be an object")
    value = namespace.get("promotion_transaction_id")
    if not (
        isinstance(value, str)
        and len(value) in {8, 32}
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise RematerializationScopeError(f"{label}.usdcraft.promotion_transaction_id is invalid")
    return value


def _source_fingerprint(paths: tuple[Path, ...], inputs_dir: Path) -> str:
    digest = hashlib.sha256()
    all_paths = list(paths)
    if inputs_dir.is_dir():
        all_paths.extend(sorted(path for path in inputs_dir.rglob("*") if path.is_file()))
    for path in all_paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
        else:
            digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return not (
        path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value)
    )


def _load_current_generation_source(
    repo: StorageRepo,
    record_id: str,
) -> _CurrentGenerationSource:
    record_path = repo.layout.record_metadata_path(record_id)
    record = _json_object(record_path, label="record.json")
    if (
        record.get("schema_version") != 3
        or record.get("record_id") != record_id
        or record.get("pipeline") != "generation"
        or record.get("status") != "generation_ready"
    ):
        raise RematerializationScopeError(
            "Rematerialization supports only canonical schema-3 Generation records in "
            "generation_ready state"
        )

    revision_id = validate_revision_id(record.get("active_revision_id"))
    revision_dir = repo.layout.record_revision_dir(record_id, revision_id)
    model_path = repo.layout.record_revision_model_path(record_id, revision_id)
    prompt_path = repo.layout.record_revision_prompt_path(record_id, revision_id)
    inputs_dir = repo.layout.record_revision_inputs_dir(record_id, revision_id)
    revision_path = repo.layout.record_revision_metadata_path(record_id, revision_id)
    provenance_path = repo.layout.record_revision_provenance_path(record_id, revision_id)
    generation_path = repo.layout.record_revision_generation_path(record_id, revision_id)

    artifacts = record.get("artifacts")
    expected_model = f"revisions/{revision_id}/model.py"
    if not isinstance(artifacts, Mapping) or artifacts.get("model_py") != expected_model:
        raise RematerializationScopeError(
            f"record.json artifacts.model_py must bind the active model as {expected_model!r}"
        )
    if not model_path.is_file():
        raise RematerializationScopeError(f"Active canonical model.py is missing: {model_path}")

    revision = _json_object(revision_path, label="revision.json")
    if (
        revision.get("schema_version") != 1
        or revision.get("record_id") != record_id
        or revision.get("revision_id") != revision_id
        or revision.get("pipeline") != "generation"
        or revision.get("status") != "generation_ready"
    ):
        raise RematerializationScopeError("Active revision metadata is not generation_ready")
    provenance = _json_object(provenance_path, label="provenance.json")
    if (
        provenance.get("schema_version") != 3
        or provenance.get("record_id") != record_id
        or provenance.get("pipeline") != "generation"
    ):
        raise RematerializationScopeError(
            "Active provenance is not canonical schema-3 Generation provenance"
        )

    generation = _json_object(generation_path, label="generation.json")
    if (
        generation.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION
        or generation.get("pipeline") != "generation"
        or generation.get("status") != "generation_ready"
        or generation.get("record_id") != record_id
        or generation.get("revision_id") != revision_id
        or generation.get("validation_profile") != GENERATION_VALIDATION_PROFILE
        or generation.get("outputs") != physics_output_contract()
    ):
        raise RematerializationScopeError(
            "Rematerialization supports only schema-3 generation manifests with the current "
            "output contract"
        )

    if not prompt_path.is_file():
        raise RematerializationScopeError(f"Canonical prompt is missing: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    if not prompt:
        raise RematerializationScopeError(f"Canonical prompt is empty: {prompt_path}")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    input_payload = generation.get("input")
    input_kind = record.get("input_kind")
    if input_kind not in {"text", "text_image"} or not isinstance(input_payload, Mapping):
        raise RematerializationScopeError("Canonical Generation input metadata is invalid")
    if input_payload.get("kind") != input_kind:
        raise RematerializationScopeError("generation.json input.kind does not match record.json")
    if input_payload.get("prompt_sha256") != prompt_sha256:
        raise RematerializationScopeError("generation.json prompt_sha256 does not match prompt.txt")
    expected_image_role = "inspiration" if input_kind == "text_image" else None
    if input_payload.get("image_role") != expected_image_role:
        raise RematerializationScopeError(
            f"generation.json input.image_role must be {expected_image_role!r}"
        )
    if input_kind == "text":
        if (
            input_payload.get("image_path") is not None
            or input_payload.get("image_sha256") is not None
        ):
            raise RematerializationScopeError("Text-only Generation must not declare an image")
    else:
        image_value = input_payload.get("image_path")
        if not _safe_relative_path(image_value):
            raise RematerializationScopeError(
                "generation.json input.image_path must be revision-relative"
            )
        image_path = revision_dir / str(image_value)
        try:
            image_path.resolve().relative_to(revision_dir.resolve())
        except (OSError, ValueError) as exc:
            raise RematerializationScopeError(
                "generation.json input.image_path escapes the active revision"
            ) from exc
        if not image_path.is_file() or input_payload.get("image_sha256") != _sha256(image_path):
            raise RematerializationScopeError(
                "generation.json inspiration image is missing or has the wrong hash"
            )

    materialization = repo.layout.record_materialization_dir(record_id)
    if not materialization.is_dir() or materialization.is_symlink():
        raise RematerializationScopeError(
            f"Current materialization directory is missing or unsafe: {materialization}"
        )
    old_report = _json_object(
        repo.layout.record_materialization_compile_report_path(record_id),
        label="compile_report.json",
    )
    if (
        old_report.get("schema_version") != COMPILE_REPORT_SCHEMA_VERSION
        or old_report.get("status") != "success"
        or old_report.get("validation_profile") != GENERATION_VALIDATION_PROFILE
    ):
        raise RematerializationScopeError(
            "Rematerialization supports only schema-2 generation compile reports"
        )

    owners = (
        _promotion_transaction_id(record, label="record.json"),
        _promotion_transaction_id(generation, label="generation.json"),
        _promotion_transaction_id(old_report, label="compile_report.json"),
    )
    if len(set(owners)) != 1:
        raise RematerializationScopeError(
            "Canonical promotion_transaction_id ownership is inconsistent across the record"
        )
    promotion_transaction_id = owners[0]

    model_sha256 = _sha256(model_path)
    record_hashes = record.get("hashes")
    revision_hashes = revision.get("hashes")
    if (
        not isinstance(record_hashes, Mapping)
        or record_hashes.get("model_py_sha256") != model_sha256
        or record_hashes.get("prompt_sha256") != prompt_sha256
    ):
        raise RematerializationScopeError("record.json source hashes do not match active source")
    if (
        not isinstance(revision_hashes, Mapping)
        or revision_hashes.get("model_py_sha256") != model_sha256
        or revision_hashes.get("prompt_sha256") != prompt_sha256
        or revision.get("prompt_sha256") != prompt_sha256
    ):
        raise RematerializationScopeError("revision.json source hashes do not match active source")
    if generation.get("model_sha256") != model_sha256:
        raise RematerializationScopeError("generation.json model_sha256 does not match model.py")

    expected_report = f"../../../../cache/record_materialization/{record_id}/compile_report.json"
    if generation.get("compile_report") != expected_report:
        raise RematerializationScopeError(
            f"generation.json compile_report must be {expected_report!r}"
        )
    fingerprint_paths = (
        record_path,
        revision_path,
        provenance_path,
        generation_path,
        model_path,
        prompt_path,
    )
    return _CurrentGenerationSource(
        record_id=record_id,
        revision_id=revision_id,
        record_dir=repo.layout.record_dir(record_id),
        revision_dir=revision_dir,
        model_path=model_path,
        inputs_dir=inputs_dir,
        generation_manifest=generation,
        promotion_transaction_id=promotion_transaction_id,
        model_sha256=model_sha256,
        source_fingerprint=_source_fingerprint(fingerprint_paths, inputs_dir),
    )


def _copy_execution_source(source: _CurrentGenerationSource, execution_dir: Path) -> Path:
    execution_dir.mkdir()
    model_copy = execution_dir / "model.py"
    shutil.copy2(source.model_path, model_copy)
    if source.inputs_dir.is_dir():
        shutil.copytree(source.inputs_dir, execution_dir / "inputs")
    if _sha256(model_copy) != source.model_sha256:
        raise RuntimeError("Staged model.py copy does not match the canonical source hash")
    return model_copy


def _bind_compile_report(
    repo: StorageRepo,
    *,
    source: _CurrentGenerationSource,
    replacement: MaterializationReplacement,
) -> tuple[str, ...]:
    report_path = replacement.replacement_dir / "compile_report.json"
    report = _json_object(report_path, label="replacement compile_report.json")
    report["record_id"] = source.record_id
    report["model_path"] = (
        f"data/records/{source.record_id}/revisions/{source.revision_id}/model.py"
    )
    report["usdcraft"] = {
        "promotion_transaction_id": source.promotion_transaction_id,
        "rematerialization_transaction_id": replacement.transaction_id,
        "source_revision_id": source.revision_id,
        "source_model_sha256": source.model_sha256,
    }
    repo.write_json(report_path, report)
    warnings = report.get("warnings")
    return tuple(str(value) for value in warnings) if isinstance(warnings, list) else ()


def _result_from_canonical(
    repo: StorageRepo,
    source: _CurrentGenerationSource,
    *,
    transaction_id: str,
    recovered: bool,
    cleanup_pending: bool,
) -> RematerializationResult:
    materialization = repo.layout.record_materialization_dir(source.record_id)
    report_path = repo.layout.record_materialization_compile_report_path(source.record_id)
    report = _json_object(report_path, label="compile_report.json")
    warnings = report.get("warnings")
    return RematerializationResult(
        record_id=source.record_id,
        revision_id=source.revision_id,
        model_path=source.model_path,
        materialization_dir=materialization,
        report_path=report_path,
        transaction_id=transaction_id,
        warnings=(tuple(str(value) for value in warnings) if isinstance(warnings, list) else ()),
        recovered=recovered,
        cleanup_pending=cleanup_pending,
    )


def rematerialize_record(
    *,
    repo_root: Path,
    record_id: str,
    render: bool = False,
) -> RematerializationResult:
    """Rebuild one current canonical Generation cache from an isolated source copy."""

    repo = StorageRepo(Path(repo_root).expanduser().resolve())
    record_id = validate_record_id(record_id)
    with record_lock(repo, record_id):
        deletion_staging = repo.layout.record_deletion_staging_dir(record_id)
        if deletion_staging.exists() or deletion_staging.is_symlink():
            raise RematerializationScopeError(
                f"Record deletion recovery is pending; rematerialization refuses to race it: "
                f"{deletion_staging}"
            )

        recovery = recover_materialization_replacement(repo, record_id)
        source = _load_current_generation_source(repo, record_id)
        if recovery.status == "committed":
            contract = validate_materialization_contract(
                repo.layout.record_materialization_dir(record_id),
                source.generation_manifest,
                phase="canonical",
                revision_dir=source.revision_dir,
            )
            if not contract.ok:
                raise RuntimeError(
                    "Recovered canonical materialization violates the current contract: "
                    + contract.summary()
                )
            return _result_from_canonical(
                repo,
                source,
                transaction_id=str(recovery.transaction_id),
                recovered=True,
                cleanup_pending=recovery.cleanup_pending,
            )

        transaction_id = uuid.uuid4().hex
        replacement = begin_materialization_replacement(repo, record_id, transaction_id)
        execution_dir = replacement.staging_dir / "execution"
        try:
            staged_model = _copy_execution_source(source, execution_dir)
            compile_model(
                staged_model,
                replacement.replacement_dir,
                render=render,
                require_generation_root_floating=True,
            )
            warnings = _bind_compile_report(
                repo,
                source=source,
                replacement=replacement,
            )
            source_paths = (
                repo.layout.record_metadata_path(record_id),
                repo.layout.record_revision_metadata_path(record_id, source.revision_id),
                repo.layout.record_revision_provenance_path(record_id, source.revision_id),
                repo.layout.record_revision_generation_path(record_id, source.revision_id),
                source.model_path,
                repo.layout.record_revision_prompt_path(record_id, source.revision_id),
            )
            if _source_fingerprint(source_paths, source.inputs_dir) != source.source_fingerprint:
                raise RuntimeError("Canonical record source changed during rematerialization")
            staged_contract = validate_materialization_contract(
                replacement.replacement_dir,
                source.generation_manifest,
                phase="canonical",
                revision_dir=source.revision_dir,
            )
            if not staged_contract.ok:
                raise RuntimeError(
                    "Replacement materialization violates the current canonical contract: "
                    + staged_contract.summary()
                )
            committed = commit_materialization_replacement(repo, replacement)
        except BaseException as exc:
            try:
                cleanup_pending = abort_materialization_replacement(repo, replacement)
            except BaseException as cleanup_exc:
                exc.add_note(f"Materialization replacement recovery is pending: {cleanup_exc}")
            else:
                if cleanup_pending:
                    exc.add_note(
                        "Materialization replacement staging cleanup is pending at "
                        f"{replacement.staging_dir}"
                    )
            raise

        return RematerializationResult(
            record_id=record_id,
            revision_id=source.revision_id,
            model_path=source.model_path,
            materialization_dir=repo.layout.record_materialization_dir(record_id),
            report_path=repo.layout.record_materialization_compile_report_path(record_id),
            transaction_id=transaction_id,
            warnings=warnings,
            cleanup_pending=committed.cleanup_pending,
        )


__all__ = [
    "RematerializationResult",
    "RematerializationScopeError",
    "rematerialize_record",
]
