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
from threading import Lock

from storage.artifact_contract import MaterializationDisplayResult

MaterializationSignature = tuple[tuple[str, int, int, int], ...]
"""Sorted ``(relative_path, size, mtime_ns, ctime_ns)`` entries; no file content reads."""


def materialization_signature(root: Path) -> MaterializationSignature | None:
    """File metadata fingerprint, or ``None`` for a missing or unreadable tree.

    Paths detect renames; per-file timestamps detect edits even when another file has a newer
    timestamp. This remains a stat heuristic, not a content-integrity guarantee. Real directories
    are traversed, directory symlinks are not, and file symlinks retain the existing stat behavior.
    """

    if not root.is_dir():
        return None
    files: list[tuple[str, int, int, int]] = []
    pending = [(os.fspath(root), "")]
    try:
        while pending:
            directory, relative = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = f"{relative}/{entry.name}" if relative else entry.name
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((entry.path, path))
                        continue
                    if not entry.is_file():
                        continue
                    stat = entry.stat()
                    files.append((path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    except OSError:
        return None
    return tuple(sorted(files))


class MaterializationValidationCache:
    """Process-lifetime cache of a full canonical contract validation result.

    Keyed on ``(record_id, active_revision_id, materialization_signature)``. Promoted source
    revisions are immutable. Per-file metadata detects ordinary derived-file edits without
    hashing contents or re-running dependency-closure checks on list requests. Filesystem
    metadata is not a cryptographic proof that contents are unchanged.

    This cache never weakens or replaces canonical validation: it only remembers the exact
    ``MaterializationDisplayResult`` a caller already computed via
    ``storage.artifact_contract.classify_*_materialization_displayability`` and reuses it while
    the signature it was computed under still matches.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[
            str, tuple[str, MaterializationSignature, MaterializationDisplayResult]
        ] = {}

    def get(
        self,
        record_id: str,
        revision_id: str,
        materialization_root: Path,
        *,
        signature: MaterializationSignature | None = None,
    ) -> MaterializationDisplayResult | None:
        """The cached result, or ``None``.

        ``signature`` lets a caller that already walked ``materialization_root`` this request
        reuse that walk instead of repeating it; it must be the current
        ``materialization_signature(materialization_root)``.
        """

        if signature is None:
            signature = materialization_signature(materialization_root)
        if signature is None:
            return None
        with self._lock:
            entry = self._entries.get(record_id)
        if entry is None:
            return None
        cached_revision, cached_signature, result = entry
        if cached_revision != revision_id or cached_signature != signature:
            return None
        return result

    def put(
        self,
        record_id: str,
        revision_id: str,
        materialization_root: Path,
        result: MaterializationDisplayResult,
    ) -> None:
        signature = materialization_signature(materialization_root)
        if signature is None:
            self.invalidate(record_id)
            return
        with self._lock:
            self._entries[record_id] = (revision_id, signature, result)

    def invalidate(self, record_id: str) -> None:
        with self._lock:
            self._entries.pop(record_id, None)


__all__ = [
    "MaterializationValidationCache",
    "MaterializationSignature",
    "materialization_signature",
]
