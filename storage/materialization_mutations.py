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

import ctypes
import errno
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from storage.identifiers import validate_record_id
from storage.record_locks import record_lock
from storage.repo import StorageRepo

MATERIALIZATION_REPLACEMENT_MANIFEST = "transaction.json"
MATERIALIZATION_REPLACEMENT_SCHEMA_VERSION = 1
_REPLACEMENT_DIR = "replacement"
_STATES = ("building", "prepared", "committed")
_RENAME_EXCHANGE = 2
_AT_FDCWD = -100


class MaterializationReplacementRecoveryError(RuntimeError):
    """Raised when a prior materialization replacement cannot be recovered safely."""


@dataclass(slots=True, frozen=True)
class MaterializationReplacement:
    record_id: str
    transaction_id: str
    staging_dir: Path
    replacement_dir: Path


@dataclass(slots=True, frozen=True)
class MaterializationReplacementResult:
    record_id: str
    transaction_id: str
    cleanup_pending: bool = False


@dataclass(slots=True, frozen=True)
class MaterializationReplacementRecovery:
    status: Literal["none", "rolled_back", "committed"]
    transaction_id: str | None = None
    cleanup_pending: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _valid_transaction_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_manifest(
    repo: StorageRepo,
    staging_dir: Path,
    *,
    record_id: str,
    transaction_id: str,
    state: Literal["building", "prepared", "committed"],
) -> None:
    repo.write_json(
        staging_dir / MATERIALIZATION_REPLACEMENT_MANIFEST,
        {
            "schema_version": MATERIALIZATION_REPLACEMENT_SCHEMA_VERSION,
            "record_id": record_id,
            "transaction_id": transaction_id,
            "state": state,
            "updated_at": _utc_now(),
        },
    )


def _read_manifest(
    repo: StorageRepo,
    staging_dir: Path,
    *,
    record_id: str,
) -> dict[str, Any]:
    manifest_path = staging_dir / MATERIALIZATION_REPLACEMENT_MANIFEST
    if not manifest_path.is_file():
        raise MaterializationReplacementRecoveryError(
            f"Missing materialization replacement manifest: {manifest_path}"
        )
    try:
        payload = repo.read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationReplacementRecoveryError(
            f"Invalid materialization replacement manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise MaterializationReplacementRecoveryError(
            f"Invalid materialization replacement manifest at {manifest_path}: expected an object"
        )
    if (
        payload.get("schema_version") != MATERIALIZATION_REPLACEMENT_SCHEMA_VERSION
        or payload.get("record_id") != record_id
        or not _valid_transaction_id(payload.get("transaction_id"))
        or payload.get("state") not in _STATES
    ):
        raise MaterializationReplacementRecoveryError(
            f"Invalid materialization replacement manifest at {manifest_path}"
        )
    return payload


def _report_transaction_id(materialization: Path) -> str | None:
    report_path = materialization / "compile_report.json"
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    namespace = payload.get("usdcraft") if isinstance(payload, dict) else None
    transaction_id = (
        namespace.get("rematerialization_transaction_id") if isinstance(namespace, dict) else None
    )
    return transaction_id if _valid_transaction_id(transaction_id) else None


def _atomic_exchange_paths(first: Path, second: Path) -> None:
    """Exchange same-filesystem paths with Darwin ``renamex_np`` or Linux ``renameat2``.

    Unsupported platforms and filesystems fail before mutation, leaving the canonical path
    unchanged.
    """

    library = ctypes.CDLL(None, use_errno=True)
    first_bytes = os.fsencode(first)
    second_bytes = os.fsencode(second)
    if sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "renamex_np is unavailable for atomic directory exchange")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(first_bytes, second_bytes, _RENAME_EXCHANGE)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable for atomic directory exchange")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            first_bytes,
            _AT_FDCWD,
            second_bytes,
            _RENAME_EXCHANGE,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            f"Atomic materialization directory exchange is unsupported on {sys.platform}",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{first} <-> {second}",
        )


def _cleanup_transaction_dir(staging_dir: Path) -> None:
    """Remove payloads before the manifest so interrupted cleanup remains recoverable."""

    manifest_path = staging_dir / MATERIALIZATION_REPLACEMENT_MANIFEST
    expected = {manifest_path, staging_dir / _REPLACEMENT_DIR, staging_dir / "execution"}
    unexpected = [path for path in staging_dir.iterdir() if path not in expected]
    if unexpected:
        raise OSError(
            errno.ENOTEMPTY,
            "Materialization replacement staging contains unexpected entries: "
            + ", ".join(str(path) for path in unexpected),
        )
    for payload_dir in (staging_dir / _REPLACEMENT_DIR, staging_dir / "execution"):
        if payload_dir.is_symlink():
            payload_dir.unlink()
        elif payload_dir.is_dir():
            shutil.rmtree(payload_dir)
        elif payload_dir.exists():
            payload_dir.unlink()
    if manifest_path.is_file() or manifest_path.is_symlink():
        manifest_path.unlink()
    staging_dir.rmdir()


def begin_materialization_replacement(
    repo: StorageRepo,
    record_id: str,
    transaction_id: str,
) -> MaterializationReplacement:
    """Create one dedicated replacement transaction with the record lock already held."""

    record_id = validate_record_id(record_id)
    if not _valid_transaction_id(transaction_id):
        raise ValueError("materialization replacement transaction_id must be 32 lowercase hex")
    staging_dir = repo.layout.materialization_replacement_staging_dir(record_id)
    if staging_dir.exists() or staging_dir.is_symlink():
        raise FileExistsError(
            f"Pending materialization replacement must be recovered first: {staging_dir}"
        )
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    try:
        _write_manifest(
            repo,
            staging_dir,
            record_id=record_id,
            transaction_id=transaction_id,
            state="building",
        )
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return MaterializationReplacement(
        record_id=record_id,
        transaction_id=transaction_id,
        staging_dir=staging_dir,
        replacement_dir=staging_dir / _REPLACEMENT_DIR,
    )


def commit_materialization_replacement(
    repo: StorageRepo,
    replacement: MaterializationReplacement,
) -> MaterializationReplacementResult:
    """Atomically exchange a validated replacement with the canonical cache directory."""

    record_id = validate_record_id(replacement.record_id)
    canonical = repo.layout.record_materialization_dir(record_id)
    if not canonical.is_dir() or canonical.is_symlink():
        raise FileNotFoundError(f"Canonical materialization is missing: {canonical}")
    if not replacement.replacement_dir.is_dir() or replacement.replacement_dir.is_symlink():
        raise FileNotFoundError(
            f"Validated replacement materialization is missing: {replacement.replacement_dir}"
        )
    if _report_transaction_id(replacement.replacement_dir) != replacement.transaction_id:
        raise ValueError("Replacement compile report is not bound to its transaction")

    _write_manifest(
        repo,
        replacement.staging_dir,
        record_id=record_id,
        transaction_id=replacement.transaction_id,
        state="prepared",
    )
    _atomic_exchange_paths(canonical, replacement.replacement_dir)

    # The exchange is the commit point. A failed manifest write or cleanup cannot turn the
    # validated canonical replacement back into a reported failure; recovery recognizes the
    # transaction owner in the canonical report on the next explicit invocation.
    try:
        _write_manifest(
            repo,
            replacement.staging_dir,
            record_id=record_id,
            transaction_id=replacement.transaction_id,
            state="committed",
        )
    except OSError:
        pass
    try:
        _cleanup_transaction_dir(replacement.staging_dir)
    except OSError:
        cleanup_pending = True
    else:
        # Once the whole transaction directory is gone there is nothing left to recover,
        # even if the post-commit manifest update itself could not be persisted.
        cleanup_pending = False
    return MaterializationReplacementResult(
        record_id=record_id,
        transaction_id=replacement.transaction_id,
        cleanup_pending=cleanup_pending,
    )


def abort_materialization_replacement(
    repo: StorageRepo,
    replacement: MaterializationReplacement,
) -> bool:
    """Discard a pre-commit replacement; return whether cleanup remains pending."""

    canonical = repo.layout.record_materialization_dir(replacement.record_id)
    if _report_transaction_id(canonical) == replacement.transaction_id:
        raise MaterializationReplacementRecoveryError(
            "Refusing to abort a materialization replacement after its atomic commit point"
        )
    try:
        _cleanup_transaction_dir(replacement.staging_dir)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return False


def recover_materialization_replacement(
    repo: StorageRepo,
    record_id: str,
) -> MaterializationReplacementRecovery:
    """Recover only the explicit record's dedicated replacement transaction."""

    record_id = validate_record_id(record_id)
    with record_lock(repo, record_id):
        staging_dir = repo.layout.materialization_replacement_staging_dir(record_id)
        if not staging_dir.is_dir() or staging_dir.is_symlink():
            if staging_dir.exists() or staging_dir.is_symlink():
                raise MaterializationReplacementRecoveryError(
                    f"Invalid materialization replacement staging path: {staging_dir}"
                )
            return MaterializationReplacementRecovery(status="none")
        manifest_path = staging_dir / MATERIALIZATION_REPLACEMENT_MANIFEST
        if not manifest_path.exists():
            try:
                has_entries = next(staging_dir.iterdir(), None) is not None
            except OSError as exc:
                raise MaterializationReplacementRecoveryError(
                    f"Could not inspect manifestless replacement staging {staging_dir}: {exc}"
                ) from exc
            if has_entries:
                raise MaterializationReplacementRecoveryError(
                    f"Missing materialization replacement manifest: {manifest_path}"
                )
            try:
                staging_dir.rmdir()
            except OSError:
                return MaterializationReplacementRecovery(
                    status="rolled_back",
                    cleanup_pending=True,
                )
            # Without a manifest there is no safe way to attribute an arbitrary older
            # rematerialization marker in the canonical report to this empty cleanup tail.
            # Removing the empty dedicated directory is safe; the explicit invocation then
            # performs a fresh rebuild.
            return MaterializationReplacementRecovery(status="rolled_back")
        if manifest_path.is_symlink():
            raise MaterializationReplacementRecoveryError(
                f"Invalid materialization replacement manifest symlink: {manifest_path}"
            )
        manifest = _read_manifest(repo, staging_dir, record_id=record_id)
        transaction_id = str(manifest["transaction_id"])
        state = str(manifest["state"])
        replacement_dir = staging_dir / _REPLACEMENT_DIR
        canonical = repo.layout.record_materialization_dir(record_id)
        canonical_owner = _report_transaction_id(canonical)
        replacement_owner = _report_transaction_id(replacement_dir)

        committed = canonical_owner == transaction_id
        not_committed = replacement_owner == transaction_id and canonical_owner != transaction_id
        if state == "building":
            if committed:
                raise MaterializationReplacementRecoveryError(
                    f"Building replacement unexpectedly owns canonical materialization {record_id}"
                )
            outcome: Literal["rolled_back", "committed"] = "rolled_back"
        elif committed:
            if replacement_owner == transaction_id:
                raise MaterializationReplacementRecoveryError(
                    f"Replacement transaction {transaction_id} owns both canonical and staged data"
                )
            outcome = "committed"
        elif state == "prepared" and (
            not_committed or (canonical_owner != transaction_id and not replacement_dir.exists())
        ):
            outcome = "rolled_back"
        else:
            raise MaterializationReplacementRecoveryError(
                f"Cannot safely recover materialization replacement {transaction_id} for "
                f"{record_id}: state={state!r}, canonical_owner={canonical_owner!r}, "
                f"replacement_owner={replacement_owner!r}"
            )

        cleanup_pending = False
        try:
            _cleanup_transaction_dir(staging_dir)
        except OSError:
            cleanup_pending = True
        return MaterializationReplacementRecovery(
            status=outcome,
            transaction_id=transaction_id,
            cleanup_pending=cleanup_pending,
        )


__all__ = [
    "MATERIALIZATION_REPLACEMENT_MANIFEST",
    "MATERIALIZATION_REPLACEMENT_SCHEMA_VERSION",
    "MaterializationReplacement",
    "MaterializationReplacementRecovery",
    "MaterializationReplacementRecoveryError",
    "MaterializationReplacementResult",
    "abort_materialization_replacement",
    "begin_materialization_replacement",
    "commit_materialization_replacement",
    "recover_materialization_replacement",
]
