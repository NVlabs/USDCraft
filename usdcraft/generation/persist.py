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

import json
import platform
import re
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from storage.artifact_contract import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    GENERATION_VALIDATION_PROFILE,
    physics_output_contract,
    validate_materialization_contract,
)
from storage.canonical_promotion import (
    CanonicalPromotion,
    begin_canonical_promotion,
    commit_canonical_promotion,
    finish_canonical_promotion,
)
from storage.record_locks import record_lock
from storage.record_mutations import write_promotion_manifest
from storage.repo import StorageRepo
from storage.revisions import build_revision_payload, revision_artifacts_payload
from usdcraft.compile.command import CompileArtifacts
from usdcraft.generation.contract import generation_object_name
from usdcraft.pipeline.files import _format_utc, _git_commit, _utc_now, _write_json
from usdcraft.pipeline.hashing import sha256_file, sha256_text

REVISION_ID = "rev_000001"


class RuntimeSelection(Protocol):
    """Small shared projection required by canonical Generation persistence."""

    def to_dict(self) -> dict[str, str]: ...


@dataclass(slots=True, frozen=True)
class GenerationAuthoringIdentity:
    """Identify the authoring arm without changing the shared Generation contract."""

    agent_harness: Literal["codex", "provider"]
    # The Codex arm is `codex-cli`; the Provider arm records its endpoint (`nvidia` or `openai`).
    provider: Literal["codex-cli", "nvidia", "openai"]

    def __post_init__(self) -> None:
        if self.agent_harness not in {"codex", "provider"}:
            raise ValueError("agent_harness must be 'codex' or 'provider'.")
        if self.agent_harness == "codex" and self.provider != "codex-cli":
            raise ValueError("'codex' authoring requires provider 'codex-cli'.")
        if self.agent_harness == "provider" and self.provider not in {"nvidia", "openai"}:
            raise ValueError("'provider' authoring requires provider 'nvidia' or 'openai'.")

    @property
    def agent(self) -> Literal["codex", "provider"]:
        return self.agent_harness

    @property
    def trace_prefix(self) -> Literal["codex", "provider"]:
        return self.agent_harness

    def to_dict(self) -> dict[str, str]:
        return {
            "agent": self.agent,
            "agent_harness": self.agent_harness,
            "provider": self.provider,
        }


CODEX_AUTHORING_IDENTITY = GenerationAuthoringIdentity(
    agent_harness="codex",
    provider="codex-cli",
)
NVIDIA_PROVIDER_AUTHORING_IDENTITY = GenerationAuthoringIdentity(
    agent_harness="provider",
    provider="nvidia",
)
OPENAI_PROVIDER_AUTHORING_IDENTITY = GenerationAuthoringIdentity(
    agent_harness="provider",
    provider="openai",
)


def provider_authoring_identity(
    endpoint: Literal["nvidia", "openai"],
) -> GenerationAuthoringIdentity:
    """The Provider-arm authoring identity for the endpoint that served the run."""

    if endpoint == "openai":
        return OPENAI_PROVIDER_AUTHORING_IDENTITY
    if endpoint == "nvidia":
        return NVIDIA_PROVIDER_AUTHORING_IDENTITY
    raise ValueError("Provider endpoint must be nvidia or openai.")


def _slug(value: str, *, limit: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "asset")[:limit].rstrip("-")


def _record_id(prompt: str, run_id: str, created_at: datetime) -> str:
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    digest = sha256_text(prompt)[:8]
    run_suffix = re.sub(r"[^a-zA-Z0-9]+", "", run_id)[-8:] or "run"
    return f"rec_{_slug(prompt)}_{timestamp}_{digest}_{run_suffix}"


def _default_trace_sources(
    workspace_dir: Path,
    identity: GenerationAuthoringIdentity,
) -> tuple[dict[str, Path], dict[str, Path]]:
    prefix = identity.trace_prefix
    run_dir = workspace_dir / f".{prefix}-run"
    event_source = "stdout.jsonl" if prefix == "codex" else "events.jsonl"
    primary = {
        f"{prefix}_task.md": workspace_dir / "TASK.md",
        f"{prefix}_events.jsonl": run_dir / event_source,
        f"{prefix}_final.json": run_dir / "final.json",
        f"{prefix}_stderr.log": run_dir / "stderr.log",
    }
    debug = {
        f"{prefix}_timeline.jsonl": run_dir / "timeline.jsonl",
        f"{prefix}_trace_errors.log": run_dir / "trace_errors.log",
        "stage_trace.jsonl": workspace_dir / "stage_trace.jsonl",
        "image_observation.json": workspace_dir / "image_observation.json",
        "image_observation_manifest.json": workspace_dir / "image_observation_manifest.json",
    }
    return primary, debug


def persist_generation(
    *,
    repo_root: Path,
    run_id: str,
    prompt: str,
    image_path: Path | None,
    label: str | None,
    runtime: RuntimeSelection,
    workspace_dir: Path,
    runtime_report: dict[str, Any],
    artifacts: CompileArtifacts,
    authoring_identity: GenerationAuthoringIdentity = CODEX_AUTHORING_IDENTITY,
) -> Path:
    """Promote a validated staging workspace into one canonical Generation record."""

    repo_root = Path(repo_root).resolve()
    prompt = prompt.rstrip()
    for path in (
        artifacts.urdf_path,
        artifacts.usdc_path,
        artifacts.newton_usdc_path,
        artifacts.asset_usda_path,
        artifacts.physics_usdc_path,
        artifacts.physx_usda_path,
        artifacts.mujoco_usda_path,
        artifacts.usdz_path,
        artifacts.report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Validated compile artifact disappeared: {path}")
    model_path = workspace_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Validated model disappeared: {model_path}")
    source_contract = validate_materialization_contract(
        artifacts.output_dir,
        {
            "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
            "pipeline": "generation",
            "status": "generation_ready",
            "validation_profile": GENERATION_VALIDATION_PROFILE,
            "outputs": physics_output_contract(),
        },
        phase="compile_output",
    )
    if not source_contract.ok:
        raise ValueError(
            "Validated compile output violates the Generation artifact contract: "
            + source_contract.summary()
        )

    created_at = _utc_now()
    record_id = _record_id(prompt, run_id, created_at)
    repo = StorageRepo(repo_root)
    # The manifest plus a held activity flock make an interrupted promotion recoverable:
    # `recover_pending_promotions` rolls back any staging dir whose lock is no longer held.
    promotion = begin_canonical_promotion(
        repo,
        record_id,
        manifest_writer=write_promotion_manifest,
    )
    promotion_succeeded = False
    try:
        promoted_record = _stage_and_commit_promotion(
            promotion=promotion,
            repo=repo,
            repo_root=repo_root,
            run_id=run_id,
            prompt=prompt,
            image_path=image_path,
            label=label,
            runtime=runtime,
            workspace_dir=workspace_dir,
            runtime_report=runtime_report,
            artifacts=artifacts,
            authoring_identity=authoring_identity,
            created_at=created_at,
            record_id=record_id,
        )
        promotion_succeeded = True
        return promoted_record
    finally:
        # Failed promotions retain their manifest and staged evidence for deterministic
        # recovery. Cleanup after a completed commit is bookkeeping and cannot turn the
        # successful canonical mutation into a CLI failure.
        finish_canonical_promotion(promotion, succeeded=promotion_succeeded)


def _stage_and_commit_promotion(
    *,
    promotion: CanonicalPromotion,
    repo: StorageRepo,
    repo_root: Path,
    run_id: str,
    prompt: str,
    image_path: Path | None,
    label: str | None,
    runtime: RuntimeSelection,
    workspace_dir: Path,
    runtime_report: dict[str, Any],
    artifacts: CompileArtifacts,
    authoring_identity: GenerationAuthoringIdentity,
    created_at: datetime,
    record_id: str,
) -> Path:
    promotion_transaction_id = promotion.transaction_id
    record_dir = promotion.record_dir
    materialization_dir = promotion.materialization_dir
    revision_dir = record_dir / "revisions" / REVISION_ID
    now = _format_utc(created_at)
    model_path = workspace_dir / "model.py"
    revision_dir.mkdir(parents=True)
    (revision_dir / "inputs").mkdir()
    (revision_dir / "traces").mkdir()
    shutil.copy2(model_path, revision_dir / "model.py")
    (revision_dir / "prompt.txt").write_text(prompt.rstrip() + "\n", encoding="utf-8")

    persisted_image: str | None = None
    image_sha256: str | None = None
    if image_path is not None:
        destination = revision_dir / "inputs" / f"inspiration{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        persisted_image = str(destination.relative_to(revision_dir))
        image_sha256 = sha256_file(destination)

    persisted_trace_sources, persisted_debug_trace_sources = _default_trace_sources(
        workspace_dir,
        authoring_identity,
    )
    for name, source in persisted_trace_sources.items():
        if source.is_file():
            shutil.copy2(source, revision_dir / "traces" / name)

    debug_copy_errors: list[dict[str, str]] = []
    for name, source in persisted_debug_trace_sources.items():
        if not source.is_file():
            continue
        destination = revision_dir / "traces" / name
        try:
            shutil.copy2(source, destination)
        except Exception as exc:
            # These two sidecars are explicitly non-contractual diagnostics. Losing one must not
            # turn an otherwise valid asset into a failed promotion.
            with suppress(OSError):
                destination.unlink(missing_ok=True)
            debug_copy_errors.append(
                {
                    "observed_at_utc": now,
                    "stage": "promotion.debug_trace_copy",
                    "error_type": type(exc).__name__,
                    "message": f"{name}: {exc}",
                }
            )
    if debug_copy_errors:
        error_path = revision_dir / "traces" / f"{authoring_identity.trace_prefix}_trace_errors.log"
        with suppress(OSError):
            with error_path.open("a", encoding="utf-8") as stream:
                for error in debug_copy_errors:
                    stream.write(json.dumps(error, ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
    feedback_root = workspace_dir / "author_feedback"
    if feedback_root.is_dir() and not feedback_root.is_symlink():
        feedback_index = []
        for source in sorted(feedback_root.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            if not source.resolve().is_relative_to(feedback_root.resolve()):
                continue
            relative = source.relative_to(feedback_root)
            destination = revision_dir / "traces" / "author_feedback" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            feedback_index.append({"path": relative.as_posix(), "sha256": sha256_file(destination)})
        _write_json(
            revision_dir / "traces" / "author_feedback_index.json", {"files": feedback_index}
        )
    trace_available = any(path.is_file() for path in (revision_dir / "traces").iterdir())

    shutil.copytree(artifacts.output_dir, materialization_dir)
    compile_report_path = materialization_dir / "compile_report.json"
    compile_report = json.loads(compile_report_path.read_text(encoding="utf-8"))
    compile_report["record_id"] = record_id
    compile_report["model_path"] = f"data/records/{record_id}/revisions/{REVISION_ID}/model.py"
    compile_report["usdcraft"] = {
        "promotion_transaction_id": promotion_transaction_id,
    }
    _write_json(compile_report_path, compile_report)

    model_sha256 = sha256_file(revision_dir / "model.py")
    input_kind = "text_image" if image_path is not None else "text"
    runtime_projection = runtime.to_dict()
    authoring_projection = authoring_identity.to_dict()
    generation_manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "pipeline": "generation",
        "status": "generation_ready",
        "record_id": record_id,
        "revision_id": REVISION_ID,
        "run_id": run_id,
        "input": {
            "kind": input_kind,
            "prompt_sha256": sha256_text(prompt),
            "image_role": "inspiration" if image_path is not None else None,
            "image_path": persisted_image,
            "image_sha256": image_sha256,
        },
        "creator": {**authoring_projection, **runtime_projection},
        "model_sha256": model_sha256,
        "validation_profile": GENERATION_VALIDATION_PROFILE,
        "outputs": physics_output_contract(),
        "compile_report": "../../../../cache/record_materialization/"
        + record_id
        + "/compile_report.json",
        "created_at": now,
        "usdcraft": {"promotion_transaction_id": promotion_transaction_id},
    }
    _write_json(revision_dir / "generation.json", generation_manifest)
    staged_contract = validate_materialization_contract(
        materialization_dir,
        generation_manifest,
        phase="canonical",
        revision_dir=revision_dir,
    )
    if not staged_contract.ok:
        raise ValueError(
            "Staged canonical record violates the Generation artifact contract: "
            + staged_contract.summary()
        )

    task_path = workspace_dir / "TASK.md"
    provenance = {
        "schema_version": 3,
        "record_id": record_id,
        "pipeline": "generation",
        "runtime": {
            **runtime_report,
            **runtime_projection,
            **authoring_projection,
        },
        "prompting": {
            "system_prompt_file": (f"traces/{authoring_identity.trace_prefix}_task.md"),
            "system_prompt_sha256": sha256_file(task_path) if task_path.is_file() else None,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "git_commit": _git_commit(repo_root),
        },
        "run_summary": {"final_status": "generation_ready"},
    }
    _write_json(revision_dir / "provenance.json", provenance)
    generation_projection = {
        "agent_harness": authoring_identity.agent_harness,
        "provider": authoring_identity.provider,
        **runtime_projection,
    }
    revision_artifacts = revision_artifacts_payload(
        revision_id=REVISION_ID,
        has_cost_file=False,
    )
    revision_artifacts["generation_json"] = f"revisions/{REVISION_ID}/generation.json"
    revision = build_revision_payload(
        record_id=record_id,
        revision_id=REVISION_ID,
        created_at=now,
        prompt_text=prompt,
        prompt_kind="single_prompt",
        source={"run_id": run_id},
        generation=generation_projection,
        artifacts=revision_artifacts,
        hashes={"prompt_sha256": sha256_text(prompt), "model_py_sha256": model_sha256},
        run_summary={"final_status": "generation_ready"},
    )
    revision.update({"pipeline": "generation", "status": "generation_ready"})
    _write_json(revision_dir / "revision.json", revision)

    title = generation_object_name(prompt=prompt, label=label)
    record = {
        "schema_version": 3,
        "record_id": record_id,
        "created_at": now,
        "updated_at": now,
        "rating": None,
        "kind": "structured_3d_asset",
        "pipeline": "generation",
        "status": "generation_ready",
        "input_kind": input_kind,
        "prompt_kind": "single_prompt",
        "category_slug": None,
        "source": {"run_id": run_id},
        "agent_harness": authoring_identity.agent_harness,
        "provider": authoring_identity.provider,
        **runtime_projection,
        "display": {"title": title, "prompt_preview": prompt[:240]},
        "artifacts": {
            "prompt_txt": f"revisions/{REVISION_ID}/prompt.txt",
            "prompt_series_json": None,
            "model_py": f"revisions/{REVISION_ID}/model.py",
            "provenance_json": f"revisions/{REVISION_ID}/provenance.json",
            "cost_json": None,
            "inputs_dir": f"revisions/{REVISION_ID}/inputs",
            "traces_dir": f"revisions/{REVISION_ID}/traces",
        },
        "hashes": {
            "prompt_sha256": sha256_text(prompt),
            "model_py_sha256": model_sha256,
        },
        "collections": ["workbench"],
        "active_revision_id": REVISION_ID,
        "lineage": {"origin_record_id": record_id, "parent_record_id": None},
        "creator": {
            "mode": "external_agent",
            **authoring_projection,
            "trace_available": trace_available,
        },
        "usdcraft": {"promotion_transaction_id": promotion_transaction_id},
    }
    _write_json(record_dir / "record.json", record)

    return commit_canonical_promotion(
        promotion,
        added_at=now,
        label=label,
        manifest_writer=write_promotion_manifest,
        record_lock_factory=record_lock,
    )
