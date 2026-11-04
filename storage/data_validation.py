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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from storage import artifact_contract
from storage.identifiers import validate_record_id
from storage.repo import StorageRepo
from storage.revisions import REVISION_ID_RE

_RUNTIME_PROJECTION_KEYS = (
    "model_id",
    "reasoning_effort",
    "runtime_source",
    "selection_source",
)
# Each authoring harness publishes under its own provider identities: the Codex arm is always
# `codex-cli`; the Provider arm records its endpoint (`nvidia` or `openai`).
_HARNESS_PROVIDERS: dict[str, tuple[str, ...]] = {
    "codex": ("codex-cli",),
    "provider": ("nvidia", "openai"),
}


@dataclass(slots=True, frozen=True)
class DataFormatValidationResult:
    errors: list[str]
    record_count: int = 0
    record_errors: dict[str, list[str]] = field(default_factory=dict)
    materializations: dict[str, artifact_contract.MaterializationDisplayResult] = field(
        default_factory=dict
    )

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_data_format(
    repo: StorageRepo,
    *,
    require_records: bool = False,
    record_ids: list[str] | None = None,
    pipeline: Literal["generation", "reconstruction"] | None = None,
) -> DataFormatValidationResult:
    """Validate canonical Generation and Reconstruction record contracts.

    Cache and local staging data are deliberately outside this format contract, except for
    the successful materialization referenced by each promoted record. ``None`` selects raw
    record directories; an explicit empty list selects nothing. A pipeline filter skips readable
    records of the other product but reports unreadable/unknown headers. Explicit ids of the wrong
    product are errors. Each selected materialization is checked once and its display projection
    is returned alongside the data-format errors for callers such as the validation CLI.
    """

    validator = _CanonicalDataValidator(repo, record_ids=record_ids, pipeline=pipeline)
    validator.validate(require_records=require_records)
    return DataFormatValidationResult(
        errors=validator.errors,
        record_count=validator.record_count,
        record_errors=validator.record_errors,
        materializations=validator.materializations,
    )


class _CanonicalDataValidator:
    def __init__(
        self,
        repo: StorageRepo,
        *,
        record_ids: list[str] | None,
        pipeline: Literal["generation", "reconstruction"] | None,
    ) -> None:
        self.repo = repo
        self.requested_ids = None if record_ids is None else list(dict.fromkeys(record_ids))
        self.pipeline = pipeline
        self.record_errors: dict[str, list[str]] = {}
        self.materializations: dict[str, artifact_contract.MaterializationDisplayResult] = {}
        self.errors: list[str] = []
        self.record_count = 0

    def validate(self, *, require_records: bool) -> None:
        records_root = self.repo.layout.records_root
        if self.requested_ids is not None:
            record_ids = self.requested_ids
        elif records_root.is_dir():
            record_ids = sorted(path.name for path in records_root.iterdir() if path.is_dir())
        else:
            record_ids = []

        if require_records and not record_ids:
            self._error(records_root, "at least one canonical record is required")
            return

        for record_id in record_ids:
            error_start, count_before = len(self.errors), self.record_count
            try:
                validate_record_id(record_id)
            except ValueError as exc:
                self._error(records_root / record_id, str(exc))
            else:
                self._validate_record(record_id)
            if self.record_count != count_before or len(self.errors) != error_start:
                self.record_errors[record_id] = self.errors[error_start:]

    def _validate_record(self, record_id: str) -> None:
        record_dir = self.repo.layout.record_dir(record_id)
        record_path = self.repo.layout.record_metadata_path(record_id)
        if not record_dir.is_dir():
            self._error(record_dir, "record not found")
            return
        record = self._json_object(record_path)
        if record is None:
            return
        if self.pipeline is not None:
            actual_pipeline = record.get("pipeline")
            if actual_pipeline not in ("generation", "reconstruction"):
                self._error(record_path, "pipeline must be generation or reconstruction")
                return
            if actual_pipeline != self.pipeline:
                if self.requested_ids is not None:
                    self._error(
                        record_path, f"expected pipeline {self.pipeline}, got {actual_pipeline}"
                    )
                return
        self.record_count += 1

        if record.get("pipeline") == "reconstruction":
            self._validate_reconstruction_record(record_id, record, record_path)
            return

        self._equal(record_path, record.get("schema_version"), 3, "schema_version")
        self._equal(record_path, record.get("record_id"), record_id, "record_id")
        self._equal(record_path, record.get("pipeline"), "generation", "pipeline")
        self._equal(record_path, record.get("status"), "generation_ready", "status")
        input_kind = record.get("input_kind")
        if input_kind not in {"text", "text_image"}:
            self._error(record_path, "input_kind must be 'text' or 'text_image'")
        rating = record.get("rating")
        if rating is not None and (
            isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5
        ):
            self._error(record_path, "rating must be an integer from 1 to 5, or null")
        runtime_projection: dict[str, str] = {}
        for key in _RUNTIME_PROJECTION_KEYS:
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                self._error(record_path, f"{key} must be a non-empty string")
            else:
                runtime_projection[key] = value
        authoring_projection = self._authoring_identity(record_path, record)
        self._validate_authoring_projection(
            record_path,
            record.get("creator"),
            authoring_projection,
            "creator",
        )

        revision_id = record.get("active_revision_id")
        if not isinstance(revision_id, str) or not REVISION_ID_RE.fullmatch(revision_id):
            self._error(record_path, "active_revision_id must match rev_000001 style IDs")
            return

        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            self._error(record_path, "artifacts must be an object")
        else:
            expected = {
                "prompt_txt": f"revisions/{revision_id}/prompt.txt",
                "model_py": f"revisions/{revision_id}/model.py",
                "provenance_json": f"revisions/{revision_id}/provenance.json",
                "inputs_dir": f"revisions/{revision_id}/inputs",
                "traces_dir": f"revisions/{revision_id}/traces",
            }
            for key, value in expected.items():
                self._equal(record_path, artifacts.get(key), value, f"artifacts.{key}")

        revision_dir = self.repo.layout.record_revision_dir(record_id, revision_id)
        revision_path = revision_dir / "revision.json"
        revision = self._json_object(revision_path)
        provenance = self._json_object(revision_dir / "provenance.json")
        generation = self._json_object(revision_dir / "generation.json")
        prompt_path = revision_dir / "prompt.txt"
        model_path = revision_dir / "model.py"
        for path, label in ((prompt_path, "prompt"), (model_path, "model")):
            if not path.is_file():
                self._error(path, f"missing {label} artifact")
        if prompt_path.is_file() and not prompt_path.read_text(encoding="utf-8").strip():
            self._error(prompt_path, "prompt must not be empty")

        if revision is not None:
            self._equal(
                revision_dir / "revision.json", revision.get("record_id"), record_id, "record_id"
            )
            self._equal(
                revision_dir / "revision.json",
                revision.get("revision_id"),
                revision_id,
                "revision_id",
            )
            self._equal(
                revision_dir / "revision.json", revision.get("pipeline"), "generation", "pipeline"
            )
            self._equal(
                revision_dir / "revision.json",
                revision.get("status"),
                "generation_ready",
                "status",
            )
            self._validate_runtime_projection(
                revision_dir / "revision.json",
                revision.get("generation"),
                runtime_projection,
                "generation",
            )
            self._validate_authoring_projection(
                revision_dir / "revision.json",
                revision.get("generation"),
                authoring_projection,
                "generation",
                include_agent=False,
            )
        if provenance is not None:
            provenance_path = revision_dir / "provenance.json"
            self._equal(
                provenance_path,
                provenance.get("schema_version"),
                artifact_contract.PROVENANCE_SCHEMA_VERSION,
                "schema_version",
            )
            self._equal(
                provenance_path,
                provenance.get("record_id"),
                record_id,
                "record_id",
            )
            self._equal(
                provenance_path,
                provenance.get("pipeline"),
                "generation",
                "pipeline",
            )
            runtime = provenance.get("runtime")
            self._validate_authoring_projection(
                provenance_path,
                runtime,
                authoring_projection,
                "runtime",
            )
            self._validate_runtime_projection(
                provenance_path,
                runtime,
                runtime_projection,
                "runtime",
            )
        if generation is not None:
            generation_path = revision_dir / "generation.json"
            for key, expected_value in (
                ("schema_version", artifact_contract.GENERATION_MANIFEST_SCHEMA_VERSION),
                ("validation_profile", artifact_contract.GENERATION_VALIDATION_PROFILE),
                ("outputs", artifact_contract.physics_output_contract()),
                ("pipeline", "generation"),
                ("status", "generation_ready"),
                ("record_id", record_id),
                ("revision_id", revision_id),
            ):
                self._equal(generation_path, generation.get(key), expected_value, key)
            input_payload = generation.get("input")
            if not isinstance(input_payload, dict):
                self._error(generation_path, "input must be an object")
            else:
                self._equal(generation_path, input_payload.get("kind"), input_kind, "input.kind")
                expected_role = "inspiration" if input_kind == "text_image" else None
                self._equal(
                    generation_path,
                    input_payload.get("image_role"),
                    expected_role,
                    "input.image_role",
                )
                if input_kind == "text_image":
                    self._validate_image(revision_dir, generation_path, input_payload)
            creator = generation.get("creator")
            self._validate_authoring_projection(
                generation_path,
                creator,
                authoring_projection,
                "creator",
            )
            self._validate_runtime_projection(
                generation_path,
                creator,
                runtime_projection,
                "creator",
            )

        if model_path.is_file():
            hashes = record.get("hashes")
            expected_hash = hashes.get("model_py_sha256") if isinstance(hashes, dict) else None
            actual_hash = self._sha256(model_path)
            if expected_hash != actual_hash:
                self._error(record_path, "hashes.model_py_sha256 does not match model.py")

        workbench_path = self.repo.layout.record_workbench_entry_path(record_id)
        workbench = self._json_object(workbench_path)
        if workbench is not None:
            self._equal(workbench_path, workbench.get("record_id"), record_id, "record_id")

        self._validate_materialization(
            record_id,
            generation_manifest=generation,
            revision_dir=revision_dir,
        )

    def _validate_reconstruction_record(
        self,
        record_id: str,
        record: dict[str, Any],
        record_path: Path,
    ) -> None:
        self._equal(record_path, record.get("schema_version"), 3, "schema_version")
        self._equal(record_path, record.get("record_id"), record_id, "record_id")
        self._equal(record_path, record.get("status"), "reconstruction_ready", "status")
        input_kind = record.get("input_kind")
        if input_kind not in {"mesh_image", "mesh_image_text"}:
            self._error(
                record_path,
                "input_kind must be 'mesh_image' or 'mesh_image_text'",
            )
        rating = record.get("rating")
        if rating is not None and (
            isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5
        ):
            self._error(record_path, "rating must be an integer from 1 to 5, or null")

        runtime_projection: dict[str, str] = {}
        for key in _RUNTIME_PROJECTION_KEYS:
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                self._error(record_path, f"{key} must be a non-empty string")
            else:
                runtime_projection[key] = value
        expected_identity = self._authoring_identity(record_path, record)
        self._validate_authoring_projection(
            record_path,
            record.get("creator"),
            expected_identity,
            "creator",
        )
        source_projection = record.get("source")
        if not isinstance(source_projection, dict):
            self._error(record_path, "source must be an object")
        else:
            run_id = source_projection.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                self._error(record_path, "source.run_id must be a non-empty string")
            if set(source_projection) != {"run_id"}:
                self._error(record_path, "source must contain exactly run_id")

        revision_id = record.get("active_revision_id")
        if not isinstance(revision_id, str) or not REVISION_ID_RE.fullmatch(revision_id):
            self._error(record_path, "active_revision_id must match rev_000001 style IDs")
            return
        relative_base = f"revisions/{revision_id}"
        expected_artifacts = {
            "request_txt": f"{relative_base}/request.txt",
            "model_py": f"{relative_base}/model.py",
            "provenance_json": f"{relative_base}/provenance.json",
            "input_manifest_json": f"{relative_base}/inputs/manifest.json",
            "inputs_dir": f"{relative_base}/inputs",
            "traces_dir": f"{relative_base}/traces",
            "reconstruction_json": f"{relative_base}/reconstruction.json",
        }
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            self._error(record_path, "artifacts must be an object")
        else:
            self._equal(record_path, artifacts, expected_artifacts, "artifacts")

        revision_dir = self.repo.layout.record_revision_dir(record_id, revision_id)
        revision_path = revision_dir / "revision.json"
        provenance_path = revision_dir / "provenance.json"
        reconstruction_path = revision_dir / "reconstruction.json"
        input_manifest_path = revision_dir / "inputs" / "manifest.json"
        revision = self._json_object(revision_path)
        provenance = self._json_object(provenance_path)
        reconstruction = self._json_object(reconstruction_path)
        input_manifest = self._json_object(input_manifest_path)
        request_path = revision_dir / "request.txt"
        model_path = revision_dir / "model.py"
        for path, label in ((request_path, "request"), (model_path, "model")):
            if not path.is_file():
                self._error(path, f"missing {label} artifact")

        if revision is not None:
            for key, expected in (
                ("record_id", record_id),
                ("revision_id", revision_id),
                ("pipeline", "reconstruction"),
                ("status", "reconstruction_ready"),
            ):
                self._equal(revision_path, revision.get(key), expected, key)
            self._validate_runtime_projection(
                revision_path,
                revision.get("reconstruction"),
                runtime_projection,
                "reconstruction",
            )
            self._equal(
                revision_path,
                revision.get("source"),
                source_projection,
                "source",
            )
        if provenance is not None:
            self._equal(
                provenance_path,
                provenance.get("schema_version"),
                artifact_contract.PROVENANCE_SCHEMA_VERSION,
                "schema_version",
            )
            for key, expected in (
                ("record_id", record_id),
                ("pipeline", "reconstruction"),
            ):
                self._equal(provenance_path, provenance.get(key), expected, key)
            self._validate_runtime_projection(
                provenance_path,
                provenance.get("runtime"),
                runtime_projection,
                "runtime",
            )
            self._validate_authoring_projection(
                provenance_path,
                provenance.get("runtime"),
                expected_identity,
                "runtime",
            )
            self._equal(
                provenance_path,
                provenance.get("source"),
                source_projection,
                "source",
            )
        if reconstruction is not None:
            for key, expected_value in (
                ("schema_version", artifact_contract.RECONSTRUCTION_MANIFEST_SCHEMA_VERSION),
                ("pipeline", "reconstruction"),
                ("status", "reconstruction_ready"),
                ("record_id", record_id),
                ("revision_id", revision_id),
                (
                    "validation_profile",
                    artifact_contract.RECONSTRUCTION_VALIDATION_PROFILE,
                ),
                ("outputs", artifact_contract.physics_output_contract()),
            ):
                self._equal(reconstruction_path, reconstruction.get(key), expected_value, key)
            creator = reconstruction.get("creator")
            self._validate_runtime_projection(
                reconstruction_path,
                creator,
                runtime_projection,
                "creator",
            )
            self._validate_authoring_projection(
                reconstruction_path,
                creator,
                expected_identity,
                "creator",
            )
            input_payload = reconstruction.get("input")
            if not isinstance(input_payload, dict):
                self._error(reconstruction_path, "input must be an object")
            else:
                self._equal(
                    reconstruction_path,
                    input_payload.get("kind"),
                    input_kind,
                    "input.kind",
                )

        if request_path.is_file() and model_path.is_file():
            hashes = record.get("hashes")
            if not isinstance(hashes, dict):
                self._error(record_path, "hashes must be an object")
            else:
                request_text = request_path.read_text(encoding="utf-8").rstrip("\n")
                expected_hashes = {
                    "request_sha256": hashlib.sha256(request_text.encode()).hexdigest(),
                    "model_py_sha256": self._sha256(model_path),
                    "input_manifest_sha256": (
                        self._sha256(input_manifest_path) if input_manifest_path.is_file() else None
                    ),
                }
                for key, expected_digest in expected_hashes.items():
                    self._equal(record_path, hashes.get(key), expected_digest, f"hashes.{key}")

        if input_manifest is not None:
            self._validate_reconstruction_evidence_projection(
                record_path,
                record.get("evidence"),
                input_manifest,
                input_manifest_path,
            )

        workbench_path = self.repo.layout.record_workbench_entry_path(record_id)
        workbench = self._json_object(workbench_path)
        if workbench is not None:
            self._equal(workbench_path, workbench.get("record_id"), record_id, "record_id")

        materialization = self.repo.layout.record_materialization_dir(record_id)
        result = artifact_contract.classify_reconstruction_materialization_displayability(
            materialization,
            reconstruction,
            revision_dir=revision_dir,
        )
        self.materializations[record_id] = result
        for issue in result.issues:
            if issue.path in {
                "reconstruction.json",
                "model.py",
                "request.txt",
            } or issue.path.startswith("inputs/"):
                path = revision_dir / issue.path
            else:
                path = materialization / issue.path
            self._error(path, issue.message)

    def _validate_reconstruction_evidence_projection(
        self,
        record_path: Path,
        evidence: Any,
        input_manifest: dict[str, Any],
        input_manifest_path: Path,
    ) -> None:
        if not isinstance(evidence, dict):
            self._error(record_path, "evidence must be an object")
            return
        expected_relative = input_manifest_path.relative_to(record_path.parent).as_posix()
        self._equal(
            record_path,
            evidence.get("manifest_path"),
            expected_relative,
            "evidence.manifest_path",
        )
        self._equal(
            record_path,
            evidence.get("manifest_sha256"),
            self._sha256(input_manifest_path),
            "evidence.manifest_sha256",
        )
        raw_files = input_manifest.get("files")
        if not isinstance(raw_files, list):
            return
        files = {
            item.get("key"): item
            for item in raw_files
            if isinstance(item, dict) and isinstance(item.get("key"), str)
        }
        for group_key, manifest_group_key in (
            ("author_inputs", "author_inputs"),
            ("host_preprocessing_inputs", "host_preprocessing_inputs"),
        ):
            keys = input_manifest.get(manifest_group_key)
            expected = []
            if isinstance(keys, list):
                for key in keys:
                    descriptor = files.get(key)
                    if descriptor is None:
                        continue
                    expected.append(
                        {
                            field: value
                            for field, value in descriptor.items()
                            if field != "author_input"
                        }
                    )
            self._equal(
                record_path,
                evidence.get(group_key),
                expected,
                f"evidence.{group_key}",
            )

    def _validate_runtime_projection(
        self,
        path: Path,
        payload: Any,
        expected: dict[str, str],
        label: str,
    ) -> None:
        if not isinstance(payload, dict):
            self._error(path, f"{label} must be an object")
            return
        for key in _RUNTIME_PROJECTION_KEYS:
            expected_value = expected.get(key)
            actual = payload.get(key)
            if expected_value is None:
                if not isinstance(actual, str) or not actual.strip():
                    self._error(path, f"{label}.{key} must be a non-empty string")
            else:
                self._equal(path, actual, expected_value, f"{label}.{key}")

    def _authoring_identity(self, record_path: Path, record: dict[str, Any]) -> dict[str, str]:
        """Return the record's authoring identity, reporting an unknown harness or provider."""

        agent_harness = record.get("agent_harness")
        allowed_providers = _HARNESS_PROVIDERS.get(str(agent_harness))
        if allowed_providers is None:
            self._error(record_path, "agent_harness must be 'codex' or 'provider'")
            agent_harness, expected_provider = "codex", _HARNESS_PROVIDERS["codex"][0]
        else:
            provider = record.get("provider")
            if provider in allowed_providers:
                expected_provider = str(provider)
            else:
                expected_provider = allowed_providers[0]
                choices = " or ".join(repr(value) for value in allowed_providers)
                self._error(
                    record_path,
                    f"provider must be {choices} for agent_harness {agent_harness!r}, "
                    f"got {provider!r}",
                )
        return {
            "agent": str(agent_harness),
            "agent_harness": str(agent_harness),
            "provider": expected_provider,
        }

    def _validate_authoring_projection(
        self,
        path: Path,
        payload: Any,
        expected: dict[str, Any],
        label: str,
        *,
        include_agent: bool = True,
    ) -> None:
        if not isinstance(payload, dict):
            self._error(path, f"{label} must be an object")
            return
        if include_agent:
            self._equal(path, payload.get("agent"), expected["agent"], f"{label}.agent")
        for key in ("agent_harness", "provider"):
            self._equal(path, payload.get(key), expected[key], f"{label}.{key}")

    def _validate_image(
        self,
        revision_dir: Path,
        generation_path: Path,
        input_payload: dict[str, Any],
    ) -> None:
        image_value = input_payload.get("image_path")
        if not isinstance(image_value, str) or not image_value:
            self._error(generation_path, "input.image_path is required for text_image")
            return
        relative = Path(image_value)
        if relative.is_absolute() or ".." in relative.parts:
            self._error(generation_path, "input.image_path must be a safe revision-relative path")
            return
        image_path = revision_dir / relative
        if not image_path.is_file():
            self._error(image_path, "inspiration image is missing")
            return
        if input_payload.get("image_sha256") != self._sha256(image_path):
            self._error(generation_path, "input.image_sha256 does not match the inspiration image")

    def _validate_materialization(
        self,
        record_id: str,
        *,
        generation_manifest: dict[str, Any] | None,
        revision_dir: Path,
    ) -> None:
        materialization = self.repo.layout.record_materialization_dir(record_id)
        result = artifact_contract.classify_materialization_displayability(
            materialization,
            generation_manifest,
            revision_dir=revision_dir,
        )
        self.materializations[record_id] = result
        for issue in result.issues:
            path = (
                revision_dir / issue.path
                if issue.path in {"generation.json", "model.py", "prompt.txt"}
                else materialization / issue.path
            )
            self._error(path, issue.message)

    def _json_object(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            self._error(path, "missing JSON object")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._error(path, f"invalid JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self._error(path, "expected a JSON object")
            return None
        return payload

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _equal(self, path: Path, actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            self._error(path, f"{label} must be {expected!r}, got {actual!r}")

    def _error(self, path: Path, message: str) -> None:
        try:
            display = path.relative_to(self.repo.root).as_posix()
        except ValueError:
            display = path.as_posix()
        self.errors.append(f"{display}: {message}")


__all__ = ["DataFormatValidationResult", "validate_data_format"]
