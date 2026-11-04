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

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from storage.identifiers import validate_record_id
from storage.repo import StorageRepo


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True, frozen=True)
class WorkbenchEntry:
    record_id: str
    added_at: str
    label: str | None = None
    tags: list[str] = field(default_factory=list)
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class WorkbenchCollection:
    entries: list[WorkbenchEntry] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "collection": "workbench",
            "updated_at": self.updated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _entry_from_payload(payload: dict[str, Any], fallback_record_id: str) -> WorkbenchEntry | None:
    record_id = str(payload.get("record_id") or fallback_record_id).strip()
    added_at = str(payload.get("added_at") or "").strip()
    try:
        validate_record_id(record_id)
    except ValueError:
        return None
    if not added_at:
        return None
    raw_tags = payload.get("tags")
    tags = [tag for tag in raw_tags if isinstance(tag, str)] if isinstance(raw_tags, list) else []
    label = payload.get("label")
    return WorkbenchEntry(
        record_id=record_id,
        added_at=added_at,
        label=label if isinstance(label, str) else None,
        tags=tags,
        archived=bool(payload.get("archived", False)),
    )


@dataclass(slots=True)
class CollectionStore:
    repo: StorageRepo

    def load_workbench(self) -> dict[str, Any] | None:
        entries: list[WorkbenchEntry] = []
        root = self.repo.layout.records_root
        if root.is_dir():
            for record_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                payload = self.repo.read_json(
                    self.repo.layout.record_workbench_entry_path(record_dir.name)
                )
                if not isinstance(payload, dict):
                    continue
                entry = _entry_from_payload(payload, record_dir.name)
                if entry is not None and entry.record_id == record_dir.name:
                    entries.append(entry)
        if not entries:
            return None
        entries.sort(key=lambda entry: (entry.added_at, entry.record_id))
        return WorkbenchCollection(entries=entries).to_dict()

    def ensure_workbench(self) -> dict[str, Any]:
        return self.load_workbench() or WorkbenchCollection().to_dict()

    def append_workbench_entry(
        self,
        *,
        record_id: str,
        added_at: str,
        label: str | None = None,
        tags: list[str] | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        record_id = validate_record_id(record_id)
        entry = WorkbenchEntry(
            record_id=record_id,
            added_at=added_at,
            label=label,
            tags=list(tags or []),
            archived=archived,
        )
        self.repo.write_json(
            self.repo.layout.record_workbench_entry_path(record_id), entry.to_dict()
        )
        return self.ensure_workbench()

    def remove_workbench_entries(self, record_id: str) -> bool:
        path = self.repo.layout.record_workbench_entry_path(validate_record_id(record_id))
        if not path.exists():
            return False
        path.unlink()
        return True
