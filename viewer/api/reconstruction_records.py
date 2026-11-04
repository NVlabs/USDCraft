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
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from threading import Lock
from typing import Any, Literal, Mapping, cast
from urllib.parse import quote

from fastapi import HTTPException
from pydantic import ValidationError

from storage.artifact_contract import (
    RECONSTRUCTION_INPUT_MANIFEST_SCHEMA,
    REQUIRED_ASSET_STRUCTURE_ARTIFACTS,
    USDZ_ARTIFACT,
    MaterializationDisplayResult,
    cheap_display_hint,
    classify_reconstruction_materialization_displayability,
)
from storage.identifiers import validate_record_id
from storage.repo import StorageRepo
from storage.revisions import (
    REVISION_ID_RE,
    active_artifact_path,
    active_cost_path,
    active_model_path,
    active_provenance_path,
    active_request_path,
    active_revision_id,
    active_traces_dir,
)
from viewer.api.file_resolver import is_materialized_viewer_path_allowed
from viewer.api.reconstruction_presentation import recorded_duration
from viewer.api.run_settings import (
    image_observation_mode,
    mesh_cleanup_status,
    provider_api_style,
    recorded_guidance,
    recorded_max_turns,
    request_prompt_kind,
)
from viewer.api.schemas import (
    AuthorTimelineEventResponse,
    AuthorTimelineResponse,
    AxialDiagnosticResponse,
    AxialExcessSliceResponse,
    AxialGeometryResponse,
    ListMaterializationStatus,
    ReconstructionRecordAssetResponse,
    ReconstructionRecordDetailResponse,
    ReconstructionRecordEvidenceResponse,
    ReconstructionRecordSummaryResponse,
)
from viewer.api.validation_cache import (
    MaterializationSignature,
    MaterializationValidationCache,
    materialization_signature,
)

_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})
_MESH_SUFFIXES = frozenset({".glb", ".gltf", ".obj", ".usd", ".usda", ".usdc", ".usdz"})
_TEXT_SUFFIXES = frozenset({".md", ".txt"})
_JSON_SUFFIXES = frozenset({".json"})
_LIST_READ_WORKERS = 8
_SOURCE_ROLES = frozenset(
    {
        "input_mesh",
        "normalized_source",
        "source",
        "source_mesh",
        "source_snapshot",
    }
)
_PRIMARY_INPUT_KEYS = (
    "original_source",
    "semantic_image",
    "axial",
)


def _string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _rating(value: Any) -> int | None:
    rating = _integer(value)
    return rating if rating is not None and 1 <= rating <= 5 else None


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


def _mtime_to_utc(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dict(repo: StorageRepo, path: Path) -> dict[str, Any] | None:
    try:
        value = repo.read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _has_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(item.is_file() for item in path.rglob("*"))
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_kind(path: Path) -> Literal["image", "mesh", "text", "json", "binary"]:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _MESH_SUFFIXES:
        return "mesh"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    if suffix in _JSON_SUFFIXES:
        return "json"
    return "binary"


def _safe_relative(value: str) -> Path | None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        return None
    return path


def _contained(target: Path, root: Path) -> Path | None:
    resolved = target.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _primitive_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"label", "name", "path", "role", "sha256"}
        and isinstance(value, (bool, dict, float, int, list, str, type(None)))
    }


class ReconstructionRecordStore:
    """Read-only projection of canonical Reconstruction records.

    Input discovery is manifest-driven. The store never crawls an input directory to invent
    evidence, and every served declaration must resolve inside the active revision.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.repo = StorageRepo(self.repo_root)
        self._validation_cache = MaterializationValidationCache()
        self._list_summary_lock = Lock()
        self._list_summary_cache: dict[
            str,
            tuple[
                dict[str, Any], MaterializationSignature | None, ReconstructionRecordSummaryResponse
            ],
        ] = {}

    def _record(self, record_id: str) -> dict[str, Any] | None:
        try:
            validate_record_id(record_id)
        except ValueError:
            return None
        record = _json_dict(self.repo, self.repo.layout.record_metadata_path(record_id))
        return record if record is not None and record.get("pipeline") == "reconstruction" else None

    def _revision_dir(self, record_id: str, record: dict[str, Any]) -> Path:
        revision_id = active_revision_id(self.repo, record_id, record=record)
        return self.repo.layout.record_revision_dir(record_id, revision_id).resolve()

    def _declared_record_path(
        self,
        record_id: str,
        record: dict[str, Any],
        value: Any,
    ) -> Path | None:
        relative = _safe_relative(value) if isinstance(value, str) else None
        if relative is None:
            return None
        record_root = self.repo.layout.record_dir(record_id).resolve()
        target = _contained(record_root / relative, record_root)
        return target if target is not None and target.is_file() else None

    def _manifest(
        self,
        record_id: str,
        record: dict[str, Any],
        *,
        reconstruction: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Path | None]:
        """The evidence manifest; ``reconstruction`` reuses an already-read `reconstruction.json`."""

        revision_dir = self._revision_dir(record_id, record)
        if reconstruction is None:
            reconstruction, _ = self._reconstruction_manifest(record_id, record)
        reconstruction_input = cast(
            "dict[str, Any]",
            reconstruction.get("input")
            if isinstance(reconstruction, dict) and isinstance(reconstruction.get("input"), dict)
            else {},
        )
        relative = _safe_relative(str(reconstruction_input.get("manifest_path") or ""))
        if relative is None:
            return None, None
        path = _contained(revision_dir / relative, revision_dir)
        if path is None:
            return None, None
        manifest = _json_dict(self.repo, path)
        if (
            manifest is not None
            and manifest.get("schema") == RECONSTRUCTION_INPUT_MANIFEST_SCHEMA
            and manifest.get("pipeline") == "reconstruction"
            and manifest.get("path_base") == "revision"
        ):
            return manifest, path
        return None, None

    def _reconstruction_manifest(
        self,
        record_id: str,
        record: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, Path | None]:
        revision_dir = self._revision_dir(record_id, record)
        artifacts = cast(
            "dict[str, Any]",
            record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {},
        )
        candidates: list[Path] = []
        declared = self._declared_record_path(
            record_id,
            record,
            artifacts.get("reconstruction_json"),
        )
        if declared is not None:
            candidates.append(declared)
        candidates.append(revision_dir / "reconstruction.json")
        for candidate in candidates:
            resolved = candidate.resolve()
            if _contained(resolved, revision_dir) is None:
                continue
            payload = _json_dict(self.repo, resolved)
            if payload is not None:
                return payload, resolved
        return None, None

    def _request_text(
        self,
        record_id: str,
        record: dict[str, Any],
        reconstruction: dict[str, Any] | None,
    ) -> str | None:
        input_payload = cast(
            "dict[str, Any]",
            reconstruction.get("input")
            if isinstance(reconstruction, dict) and isinstance(reconstruction.get("input"), dict)
            else {},
        )
        request_relative = _safe_relative(str(input_payload.get("request_path") or ""))
        if request_relative is None:
            return None
        revision_dir = self._revision_dir(record_id, record)
        target = _contained(revision_dir / request_relative, revision_dir)
        if target is None or not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except OSError:
            return None

    def _resolve_evidence_path(
        self,
        declared: str,
        *,
        revision_dir: Path,
        directory_cache: dict[Path, tuple[Path, Path] | None] | None = None,
    ) -> tuple[Path, Path] | None:
        """``(resolved_target, path_relative_to_revision)`` for one declared input, or ``None``.

        ``directory_cache`` memoizes, across the files of one manifest, the resolved form of every
        declared directory beneath ``revision_dir`` (see ``_resolve_declared_directory``). A
        regular file that is not itself a symlink then resolves to its resolved parent plus its
        own name, so the whole manifest costs one ``lstat`` per directory and per file instead of
        one ``realpath`` per file. Symlinks and other non-regular entries keep the full per-file
        resolution.
        """

        relative = _safe_relative(declared)
        if relative is None or not relative.parts or relative.parts[0] != "inputs":
            return None
        lexical = revision_dir / relative
        if directory_cache is not None:
            try:
                is_regular_file = S_ISREG(os.lstat(lexical).st_mode)
            except OSError:
                return None
            if is_regular_file:
                parent = self._resolve_declared_directory(
                    relative.parent,
                    revision_dir=revision_dir,
                    directory_cache=directory_cache,
                )
                if parent is None:
                    return None
                resolved_parent, relative_parent = parent
                return resolved_parent / relative.name, relative_parent / relative.name
        target = _contained(lexical, revision_dir)
        if target is None or not target.is_file():
            return None
        return target, target.relative_to(revision_dir)

    @staticmethod
    def _resolve_declared_directory(
        relative_dir: Path,
        *,
        revision_dir: Path,
        directory_cache: dict[Path, tuple[Path, Path] | None],
    ) -> tuple[Path, Path] | None:
        """``(resolved_dir, dir_relative_to_revision)`` for a directory declared below the revision.

        ``revision_dir`` is already resolved, so a real (non-symlink) directory resolves to its
        resolved parent plus its own name; only a symlinked directory needs ``realpath``, and the
        result must still sit inside ``revision_dir``. Results are memoized per relative path.
        """

        if not relative_dir.parts:
            return revision_dir, relative_dir
        if relative_dir in directory_cache:
            return directory_cache[relative_dir]
        result: tuple[Path, Path] | None = None
        try:
            mode = os.lstat(revision_dir / relative_dir).st_mode
        except OSError:
            mode = None
        if mode is not None and S_ISDIR(mode):
            parent = ReconstructionRecordStore._resolve_declared_directory(
                relative_dir.parent,
                revision_dir=revision_dir,
                directory_cache=directory_cache,
            )
            if parent is not None:
                result = (parent[0] / relative_dir.name, parent[1] / relative_dir.name)
        elif mode is not None:
            resolved = _contained(revision_dir / relative_dir, revision_dir)
            if resolved is not None and resolved.is_dir():
                result = (resolved, resolved.relative_to(revision_dir))
        directory_cache[relative_dir] = result
        return result

    def _evidence(
        self,
        record_id: str,
        record: dict[str, Any],
        manifest: dict[str, Any] | None,
    ) -> list[ReconstructionRecordEvidenceResponse]:
        if manifest is None:
            return []
        revision_dir = self._revision_dir(record_id, record)
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            return []
        evidence: list[ReconstructionRecordEvidenceResponse] = []
        seen: set[Path] = set()
        directory_cache: dict[Path, tuple[Path, Path] | None] = {}
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            declared = item.get("path")
            if not isinstance(declared, str):
                continue
            resolved = self._resolve_evidence_path(
                declared,
                revision_dir=revision_dir,
                directory_cache=directory_cache,
            )
            if resolved is None:
                continue
            target, relative = resolved
            if target in seen:
                continue
            seen.add(target)
            role = _string(item.get("role")) or "reconstruction_input"
            name = _string(item.get("name")) or _string(item.get("key")) or target.stem
            label = _string(item.get("label")) or name.replace("_", " ").title()
            declared_sha = _string(item.get("sha256"))
            evidence.append(
                ReconstructionRecordEvidenceResponse(
                    role=role,
                    name=name,
                    label=label,
                    relative_path=relative.as_posix(),
                    file_url=(
                        f"/api/reconstruction-records/{quote(record_id)}/files/"
                        f"{quote(relative.as_posix(), safe='/')}"
                    ),
                    media_kind=_media_kind(target),
                    byte_count=target.stat().st_size,
                    sha256=declared_sha or _sha256(target),
                    metadata=_primitive_metadata(item),
                )
            )
        return evidence

    @staticmethod
    def _primary_inputs(
        evidence: list[ReconstructionRecordEvidenceResponse],
    ) -> list[ReconstructionRecordEvidenceResponse]:
        """Project only the three user-facing Reconstruction input groups.

        The canonical manifest intentionally retains a larger Host audit closure. Those files
        remain available to strict manifest-based resolution, but are not additional inputs to
        present in the Recon Viewer.
        """

        by_key: dict[str, ReconstructionRecordEvidenceResponse] = {}
        for item in evidence:
            key = item.metadata.get("key")
            if isinstance(key, str) and key not in by_key:
                by_key[key] = item
        return [by_key[key] for key in _PRIMARY_INPUT_KEYS if key in by_key]

    @staticmethod
    def _primary_input_channel_count(
        evidence: list[ReconstructionRecordEvidenceResponse],
    ) -> int:
        keys = {key for item in evidence if isinstance((key := item.metadata.get("key")), str)}
        return sum(key in keys for key in _PRIMARY_INPUT_KEYS)

    @staticmethod
    def _is_source_evidence(item: ReconstructionRecordEvidenceResponse) -> bool:
        normalized_role = item.role.lower().replace("-", "_").replace(" ", "_")
        key = item.metadata.get("key")
        return item.media_kind == "mesh" and (
            key == "normalized_source"
            or normalized_role in _SOURCE_ROLES
            or "source_mesh" in normalized_role
            or "source_snapshot" in normalized_role
            or normalized_role == "canonical_upstream_source_snapshot"
        )

    @classmethod
    def _source_evidence(
        cls,
        evidence: list[ReconstructionRecordEvidenceResponse],
    ) -> ReconstructionRecordEvidenceResponse | None:
        normalized = next(
            (item for item in evidence if item.metadata.get("key") == "normalized_source"),
            None,
        )
        return normalized or next(
            (item for item in evidence if cls._is_source_evidence(item)), None
        )

    def _source_overlay_base_url(
        self,
        record_id: str,
        manifest: dict[str, Any] | None,
        evidence: list[ReconstructionRecordEvidenceResponse],
    ) -> tuple[str | None, str | None]:
        channels = cast(
            "dict[str, Any]",
            manifest.get("channels")
            if isinstance(manifest, dict) and isinstance(manifest.get("channels"), dict)
            else {},
        )
        overlay = cast(
            "dict[str, Any]",
            channels.get("source_overlay")
            if isinstance(channels.get("source_overlay"), dict)
            else {},
        )
        relative = _safe_relative(str(overlay.get("base_path") or ""))
        if relative is None or not relative.parts or relative.parts[0] != "inputs":
            return None, None
        urdf_key = _string(overlay.get("urdf_key"))
        glb_key = _string(overlay.get("glb_key"))
        urdf = next(
            (item for item in evidence if item.metadata.get("key") == urdf_key),
            None,
        )
        glb = next(
            (item for item in evidence if item.metadata.get("key") == glb_key),
            None,
        )
        if (
            urdf is None
            or glb is None
            or Path(urdf.relative_path).parent != relative
            or not Path(glb.relative_path).is_relative_to(relative)
        ):
            return None, None
        revision_material = f"{urdf.sha256 or ''}:{glb.sha256 or ''}".encode()
        revision_key = hashlib.sha256(revision_material).hexdigest()[:20]
        return (
            f"/api/reconstruction-records/{quote(record_id)}/files/"
            f"{quote(relative.as_posix(), safe='/')}",
            revision_key,
        )

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
        revision_dir = self._revision_dir(record_id, record)
        reconstruction_path = self.repo.layout.record_revision_reconstruction_path(
            record_id,
            revision_id,
        )
        result = classify_reconstruction_materialization_displayability(
            materialization_root,
            reconstruction_path,
            revision_dir=revision_dir,
        )
        self._validation_cache.put(record_id, revision_id, materialization_root, result)
        return result

    def _materialization_light(
        self,
        record_id: str,
        record: dict[str, Any],
        *,
        signature: MaterializationSignature | None = None,
    ) -> tuple[ListMaterializationStatus, bool, list[dict[str, str]], bool]:
        """``(status, displayable, issues, validated)`` without the full canonical contract.

        Reuses a cached full validation result when the active revision and materialization
        signature still match. Otherwise falls back to the two cheap, hash-free objective
        signals (``compile_failed``, ``missing``); when neither applies, the row is reported
        ``not_validated`` — displayable pending an explicit check, never claimed ``available``.
        ``signature`` reuses a materialization walk the caller already did for this request.
        """

        revision_id = active_revision_id(self.repo, record_id, record=record)
        materialization_root = self.repo.layout.record_materialization_dir(record_id)
        cached = self._validation_cache.get(
            record_id, revision_id, materialization_root, signature=signature
        )
        if cached is not None:
            issues = [{"path": issue.path, "message": issue.message} for issue in cached.issues]
            return cached.status, cached.displayable, issues, True
        hint_status, hint_displayable = cheap_display_hint(materialization_root)
        if hint_status is not None:
            return hint_status, hint_displayable, [], False
        return "not_validated", True, [], False

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

    def _candidate_asset(
        self,
        record_id: str,
        record: dict[str, Any],
        *,
        display: MaterializationDisplayResult | None = None,
        available: bool | None = None,
    ) -> ReconstructionRecordAssetResponse:
        materialization = self.repo.layout.record_materialization_dir(record_id)
        urdf = materialization / "model.urdf"
        if available is None:
            resolved_display = display or self._materialization_display(record_id, record)
            available = resolved_display.displayable
        revision_key = None
        if available:
            stat = urdf.stat()
            revision_key = f"{stat.st_size:x}-{stat.st_mtime_ns:x}"
        return ReconstructionRecordAssetResponse(
            available=available,
            label="Reconstructed candidate",
            base_file_url=f"/api/reconstruction-records/{quote(record_id)}/materialized"
            if available
            else None,
            format="urdf",
            revision_key=revision_key,
        )

    def detail(self, record_id: str) -> ReconstructionRecordDetailResponse | None:
        record = self._record(record_id)
        if record is None:
            return None
        reconstruction, _ = self._reconstruction_manifest(record_id, record)
        manifest, _ = self._manifest(record_id, record, reconstruction=reconstruction)
        evidence = self._evidence(record_id, record, manifest)
        source_input = self._source_evidence(evidence)
        source_base_url, source_revision_key = self._source_overlay_base_url(
            record_id,
            manifest,
            evidence,
        )
        source = ReconstructionRecordAssetResponse(
            available=source_input is not None and source_base_url is not None,
            label="Post-isolation normalized q0 source"
            if source_input and source_input.metadata.get("key") == "normalized_source"
            else source_input.label
            if source_input
            else "Input source mesh",
            file_url=source_input.file_url if source_input else None,
            base_file_url=source_base_url,
            format="urdf" if source_base_url else None,
            revision_key=source_revision_key,
        )
        materialization_display = self._materialization_display(record_id, record)
        candidate = self._candidate_asset(
            record_id,
            record,
            display=materialization_display,
        )
        summary = self._summary(
            record_id,
            record,
            evidence=evidence,
            source=source,
            candidate=candidate,
            materialization_display=materialization_display,
        )
        provenance = _json_dict(
            self.repo,
            active_provenance_path(self.repo, record_id, record=record),
        )
        compile_report = _json_dict(
            self.repo,
            self.repo.layout.record_materialization_compile_report_path(record_id),
        )
        request_text = self._request_text(record_id, record, reconstruction)
        guidance_status, guidance = recorded_guidance(reconstruction)
        return ReconstructionRecordDetailResponse(
            summary=summary,
            source=source,
            candidate=candidate,
            inputs=self._primary_inputs(evidence),
            record=record,
            reconstruction_manifest=reconstruction,
            evidence_manifest=manifest,
            request_text=request_text,
            request_prompt=request_prompt_kind(request_text),
            guidance_status=guidance_status,
            guidance=guidance,
            provenance=provenance,
            compile_report=compile_report,
        )

    def author_timeline(self, record_id: str) -> AuthorTimelineResponse | None:
        """Per-turn tool events of a Provider run (read-only reduction of `provider_events.jsonl`).

        Returns None for a record without a Provider trace (Codex arm, or the trace was not
        retained). Response times are attached to the turn whose tool call followed them.
        """

        record_id = validate_record_id(record_id)
        record = self._record(record_id)
        if record is None:
            return None
        traces = active_traces_dir(self.repo, record_id, record=record)
        events_path = traces / "provider_events.jsonl" if traces is not None else None
        if events_path is None or not events_path.is_file():
            return None
        events: list[AuthorTimelineEventResponse] = []
        response_elapsed: dict[int, float] = {}
        try:
            with events_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    kind = event.get("record_type")
                    turn = _integer(event.get("turn"))
                    if kind == "provider.response.completed" and turn is not None:
                        elapsed = event.get("elapsed_seconds")
                        if isinstance(elapsed, (int, float)):
                            response_elapsed[turn] = float(elapsed)
                        continue
                    if kind != "provider.tool.completed" or turn is None:
                        continue
                    recall: float | None = None
                    output = event.get("output")
                    if event.get("tool_name") == "check_axial" and isinstance(output, dict):
                        inner = output.get("output")
                        result = inner.get("result") if isinstance(inner, dict) else None
                        comparison = (
                            result.get("evidence_comparison") if isinstance(result, dict) else None
                        )
                        value = comparison.get("recall") if isinstance(comparison, dict) else None
                        if isinstance(value, (int, float)):
                            recall = float(value)
                    tool_elapsed = event.get("elapsed_seconds")
                    events.append(
                        AuthorTimelineEventResponse(
                            turn=turn,
                            tool=_string(event.get("tool_name")) or "unknown",
                            ok=event.get("ok") is True,
                            revision=_integer(event.get("revision")),
                            tool_elapsed_seconds=(
                                float(tool_elapsed)
                                if isinstance(tool_elapsed, (int, float))
                                else None
                            ),
                            response_elapsed_seconds=response_elapsed.get(turn),
                            recall=recall,
                        )
                    )
        except OSError:
            return None
        return AuthorTimelineResponse(record_id=record_id, source="provider_events", events=events)

    def axial_geometry(self, record_id: str) -> AxialGeometryResponse | None:
        """Bounds of the worst exterior-excess slices and control markers of the last check_axial."""

        record_id = validate_record_id(record_id)
        record = self._record(record_id)
        if record is None:
            return None
        traces = active_traces_dir(self.repo, record_id, record=record)
        events_path = traces / "provider_events.jsonl" if traces is not None else None
        if events_path is None or not events_path.is_file():
            return None
        last_result: dict[str, Any] | None = None
        try:
            with events_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if (
                        not isinstance(event, dict)
                        or event.get("record_type") != "provider.tool.completed"
                        or event.get("tool_name") != "check_axial"
                    ):
                        continue
                    output = event.get("output")
                    inner = output.get("output") if isinstance(output, dict) else None
                    result = inner.get("result") if isinstance(inner, dict) else None
                    if (
                        isinstance(output, dict)
                        and output.get("ok") is True
                        and isinstance(result, dict)
                    ):
                        last_result = result
        except OSError:
            return None
        if last_result is None:
            return None
        grid = cast(
            "dict[str, Any]",
            last_result.get("grid") if isinstance(last_result.get("grid"), dict) else {},
        )
        audit = cast(
            "dict[str, Any]",
            last_result.get("provider_audit")
            if isinstance(last_result.get("provider_audit"), dict)
            else {},
        )
        excess = cast(
            "dict[str, Any]",
            audit.get("exterior_excess") if isinstance(audit.get("exterior_excess"), dict) else {},
        )

        def floats(value: Any) -> list[float] | None:
            if not isinstance(value, list) or len(value) != 3:
                return None
            if not all(isinstance(item, (int, float)) for item in value):
                return None
            return [float(item) for item in value]

        bounds_min = floats(grid.get("bounds_min_xyz_m"))
        bounds_max = floats(grid.get("bounds_max_xyz_m"))
        if bounds_min is None or bounds_max is None:
            return None
        slices: list[AxialExcessSliceResponse] = []
        for item in (
            excess.get("worst_slices", []) if isinstance(excess.get("worst_slices"), list) else []
        ):
            if not isinstance(item, dict):
                continue
            lo, hi = floats(item.get("bounds_min_xyz_m")), floats(item.get("bounds_max_xyz_m"))
            if lo is None or hi is None:
                continue
            slices.append(
                AxialExcessSliceResponse(
                    stack_slice=_integer(item.get("stack_slice")) or 0,
                    side=_string(item.get("side")) or "?",
                    excess_cells=_integer(item.get("excess_cells")) or 0,
                    bounds_min_xyz_m=lo,
                    bounds_max_xyz_m=hi,
                )
            )
        pitch = audit.get("stack_pitch_m")
        return AxialGeometryResponse(
            record_id=record_id,
            source="provider_events",
            grid_bounds_min_xyz_m=bounds_min,
            grid_bounds_max_xyz_m=bounds_max,
            stack_pitch_m=float(pitch) if isinstance(pitch, (int, float)) else None,
            excess_slices=slices,
        )

    def _axial_diagnostic(
        self, record_id: str, record: dict[str, Any], runtime: Mapping[str, Any]
    ) -> AxialDiagnosticResponse | None:
        """The record's `check_axial` summary recorded at promotion (`runtime.axial_diagnostic`)."""

        payload = runtime.get("axial_diagnostic")
        if not isinstance(payload, Mapping):
            return None
        try:
            return AxialDiagnosticResponse(**dict(payload))
        except ValidationError:
            return None

    def _summary(
        self,
        record_id: str,
        record: dict[str, Any],
        *,
        evidence: list[ReconstructionRecordEvidenceResponse] | None = None,
        source: ReconstructionRecordAssetResponse | None = None,
        candidate: ReconstructionRecordAssetResponse | None = None,
        materialization_display: MaterializationDisplayResult | None = None,
        fast: bool = False,
    ) -> ReconstructionRecordSummaryResponse:
        """Build one record's summary.

        ``fast=True`` (the list projection) never runs the full canonical contract check: it
        reuses a still-fresh cached validation when one exists and otherwise reports
        ``not_validated`` rather than guessing. Every other caller (``detail``, ``summary``, the
        explicit validate action) always resolves a real ``MaterializationDisplayResult``.

        The fast projection also keeps a process-lifetime cache of each row, keyed on the parsed
        `record.json` content (rating and `updated_at` live there) and the stat-only signature of
        the materialization directory. Everything else a row reads sits in an immutable promoted
        revision, so that key cannot go stale without a real change; the four materialization
        fields and ``candidate_available`` are re-derived on every call so a validation completed
        since (opening the record, an explicit validate) shows up in the list immediately.
        """

        materialization_validated = True
        materialization_status_value: ListMaterializationStatus
        materialization_displayable_value: bool
        materialization_issues_value: list[Any]
        light: tuple[ListMaterializationStatus, bool, list[dict[str, str]], bool] | None = None
        signature: MaterializationSignature | None = None
        if fast:
            signature = materialization_signature(
                self.repo.layout.record_materialization_dir(record_id)
            )
            light = self._materialization_light(record_id, record, signature=signature)
            with self._list_summary_lock:
                cached = self._list_summary_cache.get(record_id)
            if cached is not None and cached[0] == record and cached[1] == signature:
                status, displayable, issues, validated = light
                return cached[2].model_copy(
                    update={
                        "candidate_available": displayable,
                        "materialization_status": status,
                        "materialization_displayable": displayable,
                        "materialization_validated": validated,
                        "materialization_issues": issues,
                    }
                )
        reconstruction, _ = self._reconstruction_manifest(record_id, record)
        if evidence is None or source is None or candidate is None:
            manifest, _ = self._manifest(record_id, record, reconstruction=reconstruction)
            evidence = self._evidence(record_id, record, manifest)
            source_item = self._source_evidence(evidence)
            source_base_url, _ = self._source_overlay_base_url(record_id, manifest, evidence)
            source = ReconstructionRecordAssetResponse(
                available=source_item is not None and source_base_url is not None,
                label=source_item.label if source_item else "Input source mesh",
            )
            if fast:
                assert light is not None
                (
                    materialization_status_value,
                    materialization_displayable_value,
                    materialization_issues_value,
                    materialization_validated,
                ) = light
                candidate = self._candidate_asset(
                    record_id,
                    record,
                    available=materialization_displayable_value,
                )
            else:
                materialization_display = self._materialization_display(record_id, record)
                candidate = self._candidate_asset(
                    record_id,
                    record,
                    display=materialization_display,
                )
        if materialization_display is not None:
            materialization_status_value = materialization_display.status
            materialization_displayable_value = materialization_display.displayable
            materialization_issues_value = [
                {"path": issue.path, "message": issue.message}
                for issue in materialization_display.issues
            ]
        elif not fast:
            materialization_display = self._materialization_display(record_id, record)
            materialization_status_value = materialization_display.status
            materialization_displayable_value = materialization_display.displayable
            materialization_issues_value = [
                {"path": issue.path, "message": issue.message}
                for issue in materialization_display.issues
            ]
        materialization = self.repo.layout.record_materialization_dir(record_id)
        display = cast(
            "dict[str, Any]",
            record.get("display") if isinstance(record.get("display"), dict) else {},
        )
        creator = cast(
            "dict[str, Any]",
            record.get("creator") if isinstance(record.get("creator"), dict) else {},
        )
        reconstruction_creator = cast(
            "dict[str, Any]",
            reconstruction.get("creator")
            if isinstance(reconstruction, dict) and isinstance(reconstruction.get("creator"), dict)
            else {},
        )
        provenance_path = active_provenance_path(self.repo, record_id, record=record)
        provenance = _json_dict(self.repo, provenance_path) or {}
        runtime = cast(
            "dict[str, Any]",
            provenance.get("runtime") if isinstance(provenance.get("runtime"), dict) else {},
        )
        run_summary = cast(
            "dict[str, Any]",
            provenance.get("run_summary")
            if isinstance(provenance.get("run_summary"), dict)
            else {},
        )
        cost_path = active_cost_path(self.repo, record_id, record=record)
        cost = _json_dict(self.repo, cost_path)
        total_cost_usd, input_tokens, output_tokens = _cost_totals(cost)
        runtime_input_tokens, runtime_output_tokens = _runtime_usage_tokens(runtime)
        if input_tokens is None:
            input_tokens = runtime_input_tokens
        if output_tokens is None:
            output_tokens = runtime_output_tokens
        turn_count = _integer(run_summary.get("turn_count"))
        if turn_count is None:
            turn_count = _integer(runtime.get("turn_count"))
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
        raw_status = _string(record.get("status"))
        raw_input_kind = _string(record.get("input_kind"))
        input_kind = cast(
            'Literal["mesh_image", "mesh_image_text"] | None',
            raw_input_kind if raw_input_kind in {"mesh_image", "mesh_image_text"} else None,
        )
        reasoning_effort = (
            _string(record.get("reasoning_effort"))
            or _string(creator.get("reasoning_effort"))
            or _string(reconstruction_creator.get("reasoning_effort"))
            or _string(runtime.get("reasoning_effort"))
            or _string(runtime.get("reasoning"))
        )
        raw_agent_harness = _string(record.get("agent_harness"))
        agent_harness = cast(
            'Literal["codex", "provider"] | None',
            raw_agent_harness if raw_agent_harness in {"codex", "provider"} else None,
        )
        runtime_source = (
            _string(record.get("runtime_source"))
            or _string(creator.get("runtime_source"))
            or _string(reconstruction_creator.get("runtime_source"))
            or _string(runtime.get("runtime_source"))
        )
        scale = cast(
            "dict[str, Any]", runtime.get("scale") if isinstance(runtime.get("scale"), dict) else {}
        )
        estimate = cast(
            "dict[str, Any]",
            scale.get("estimate") if isinstance(scale.get("estimate"), dict) else {},
        )
        scale_source = _string(scale.get("source"))
        batch = cast(
            "dict[str, Any]", runtime.get("batch") if isinstance(runtime.get("batch"), dict) else {}
        )
        duration = recorded_duration(
            runtime,
            (reconstruction.get("created_at") if isinstance(reconstruction, dict) else None)
            or record.get("created_at"),
        )
        summary = ReconstructionRecordSummaryResponse(
            duration_seconds=duration,
            batch_id=_string(batch.get("batch_id")),
            batch_row_id=_string(batch.get("row_id")),
            record_id=record_id,
            agent_harness=agent_harness,
            runtime_source=runtime_source,
            api_style=provider_api_style(agent_harness, runtime, runtime_source),
            image_observation=image_observation_mode(runtime),
            max_turns=recorded_max_turns(agent_harness, runtime),
            mesh_cleanup=mesh_cleanup_status(runtime),
            scale_source="declared" if scale_source == "explicit" else scale_source,
            scale_meters_per_unit=_float(scale.get("meters_per_unit")),
            scale_estimate_category=_string(estimate.get("category")),
            scale_estimate_longest_m=_float(estimate.get("longest_dimension_m")),
            axial_diagnostic=self._axial_diagnostic(record_id, record, runtime),
            status="reconstruction_ready" if raw_status == "reconstruction_ready" else None,
            input_kind=input_kind,
            title=_string(display.get("title")) or record_id,
            prompt_preview=_string(display.get("prompt_preview")) or "",
            rating=_rating(record.get("rating")),
            created_at=_string(record.get("created_at")),
            updated_at=_string(record.get("updated_at")),
            viewer_asset_updated_at=self._viewer_asset_updated_at(record_id),
            provider=_string(record.get("provider"))
            or _string(creator.get("provider"))
            or _string(reconstruction_creator.get("provider"))
            or _string(runtime.get("provider")),
            model_id=_string(record.get("model_id"))
            or _string(creator.get("model_id"))
            or _string(reconstruction_creator.get("model_id"))
            or _string(runtime.get("model_id")),
            reasoning_effort=reasoning_effort,
            thinking_level=reasoning_effort,
            turn_count=turn_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            active_revision_id=active_revision_id(self.repo, record_id, record=record),
            revision_count=revision_count,
            input_count=self._primary_input_channel_count(evidence),
            source_available=source.available,
            candidate_available=candidate.available,
            materialization_status=materialization_status_value,
            materialization_displayable=materialization_displayable_value,
            materialization_validated=materialization_validated,
            materialization_issues=materialization_issues_value,
            has_compile_report=(materialization / "compile_report.json").is_file(),
            has_provenance=provenance_path.is_file(),
            has_cost=cost_path.is_file(),
            has_traces=_has_files(active_traces_dir(self.repo, record_id, record=record)),
        )
        if fast:
            with self._list_summary_lock:
                self._list_summary_cache[record_id] = (record, signature, summary)
        return summary

    def summary(self, record_id: str) -> ReconstructionRecordSummaryResponse | None:
        record = self._record(record_id)
        return self._summary(record_id, record) if record is not None else None

    def validate(self, record_id: str) -> ReconstructionRecordSummaryResponse | None:
        """Explicitly run the full canonical contract check for one record and cache it.

        This is the user-invoked "validate this record" action: it does the same hash and
        dependency-closure work as opening the record's detail view, but is never triggered
        implicitly by listing or refreshing the record library.
        """

        record = self._record(record_id)
        if record is None:
            return None
        materialization_display = self._materialization_display(record_id, record, use_cache=False)
        return self._summary_with_display(record_id, record, materialization_display)

    def _summary_with_display(
        self,
        record_id: str,
        record: dict[str, Any],
        materialization_display: MaterializationDisplayResult,
    ) -> ReconstructionRecordSummaryResponse:
        manifest, _ = self._manifest(record_id, record)
        evidence = self._evidence(record_id, record, manifest)
        source_item = self._source_evidence(evidence)
        source_base_url, _ = self._source_overlay_base_url(record_id, manifest, evidence)
        source = ReconstructionRecordAssetResponse(
            available=source_item is not None and source_base_url is not None,
            label=source_item.label if source_item else "Input source mesh",
        )
        candidate = self._candidate_asset(record_id, record, display=materialization_display)
        return self._summary(
            record_id,
            record,
            evidence=evidence,
            source=source,
            candidate=candidate,
            materialization_display=materialization_display,
        )

    def _list_summary(self, record_id: str) -> ReconstructionRecordSummaryResponse | None:
        """One list row, or ``None`` for a non-Reconstruction, unregistered or unreadable record."""

        record = self._record(record_id)
        if record is None:
            return None
        workbench = _json_dict(self.repo, self.repo.layout.record_workbench_entry_path(record_id))
        if workbench is None or workbench.get("record_id") != record_id:
            return None
        try:
            return self._summary(record_id, record, fast=True)
        except (OSError, ValueError):
            return None

    def list_records(self) -> list[ReconstructionRecordSummaryResponse]:
        root = self.repo.layout.records_root
        if not root.is_dir():
            return []
        record_ids = [path.name for path in root.iterdir() if path.is_dir()]
        # Rows are independent small-file reads; a few threads overlap that I/O.
        with ThreadPoolExecutor(max_workers=_LIST_READ_WORKERS) as pool:
            rows = pool.map(self._list_summary, record_ids)
        return sorted(
            (row for row in rows if row is not None),
            key=lambda item: (item.created_at or "", item.record_id),
            reverse=True,
        )

    def declared_file(self, record_id: str, file_path: str) -> Path:
        record = self._record(record_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Reconstruction record not found: {record_id}"
            )
        requested = _safe_relative(file_path)
        if requested is None:
            raise HTTPException(status_code=400, detail="Invalid file path")
        manifest, _ = self._manifest(record_id, record)
        evidence = self._evidence(record_id, record, manifest)
        allowed = {Path(item.relative_path).parts for item in evidence}
        if requested.parts not in allowed:
            raise HTTPException(status_code=404, detail=f"Input file not declared: {file_path}")
        revision_dir = self._revision_dir(record_id, record)
        target = _contained(revision_dir / requested, revision_dir)
        if target is None or not target.is_file():
            raise HTTPException(status_code=404, detail=f"Input file not found: {file_path}")
        return target

    def text_file(self, record_id: str, file_path: str) -> Path:
        """Resolve one exact canonical text artifact without broad revision access.

        Reconstruction inputs keep their stronger manifest allowlist. The remaining aliases are
        the small read-only surface needed by the canonical model inspector; notably, source code
        remains readable when a materialization is unavailable, while ``model.urdf`` continues to
        use the materialization displayability contract.
        """

        record = self._record(record_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Reconstruction record not found: {record_id}"
            )
        requested = _safe_relative(file_path)
        if requested is None:
            raise HTTPException(status_code=400, detail="Invalid file path")
        if requested.parts and requested.parts[0] == "inputs":
            return self.declared_file(record_id, file_path)
        if requested.parts == ("model.urdf",):
            return self.materialized_file(record_id, file_path)

        revision_dir = self._revision_dir(record_id, record)
        if requested.parts == ("model.py",):
            candidate: Path | None = active_model_path(
                self.repo,
                record_id,
                record=record,
            )
            root = revision_dir
        elif requested.parts == ("request.txt",):
            candidate = active_request_path(
                self.repo,
                record_id,
                record=record,
            )
            root = revision_dir
        elif requested.parts == ("provenance.json",):
            candidate = active_provenance_path(
                self.repo,
                record_id,
                record=record,
            )
            root = revision_dir
        elif requested.parts == ("reconstruction.json",):
            candidate = active_artifact_path(
                self.repo,
                record_id,
                "reconstruction_json",
                record=record,
            )
            root = revision_dir
        elif requested.parts == ("compile_report.json",):
            root = self.repo.layout.record_materialization_dir(record_id).resolve()
            candidate = root / "compile_report.json"
        else:
            raise HTTPException(status_code=404, detail=f"Text file not allowlisted: {file_path}")

        target = _contained(candidate, root) if candidate is not None else None
        if target is None or not target.is_file():
            raise HTTPException(status_code=404, detail=f"Text file not found: {file_path}")
        return target

    def materialized_file(self, record_id: str, file_path: str) -> Path:
        record = self._record(record_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail=f"Reconstruction record not found: {record_id}"
            )
        requested = _safe_relative(file_path)
        if requested is None:
            raise HTTPException(status_code=400, detail="Invalid file path")
        materialization = self.repo.layout.record_materialization_dir(record_id).resolve()
        display = self._materialization_display(record_id, record)
        report = _json_dict(self.repo, materialization / "compile_report.json")
        if not is_materialized_viewer_path_allowed(
            requested,
            display=display,
            report=report,
        ):
            raise HTTPException(status_code=404, detail=f"Materialized file not found: {file_path}")
        target = _contained(materialization / requested, materialization)
        if target is None or not target.is_file():
            raise HTTPException(status_code=404, detail=f"Materialized file not found: {file_path}")
        return target


__all__ = ["ReconstructionRecordStore"]
