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

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

# Field value sets, used to narrow unvalidated payload strings in ``from_dict``.
_Severity: TypeAlias = Literal["failure", "warning", "note"]
_Source: TypeAlias = Literal["compiler", "tests", "harness"]
_Group: TypeAlias = Literal["build", "qc", "design", "hygiene"]
_Status: TypeAlias = Literal["success", "failure"]


@dataclass(slots=True, frozen=True)
class CompileSignal:
    severity: Literal["failure", "warning", "note"]
    kind: str
    code: str
    summary: str
    details: str = ""
    blocking: bool = False
    source: Literal["compiler", "tests", "harness"] = "compiler"
    group: Literal["build", "qc", "design", "hygiene"] = "qc"
    check_name: str | None = None
    dedupe_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "code": self.code,
            "summary": self.summary,
            "details": self.details,
            "blocking": self.blocking,
            "source": self.source,
            "group": self.group,
            "check_name": self.check_name,
            "dedupe_key": self.dedupe_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileSignal":
        return cls(
            severity=cast(_Severity, str(payload.get("severity", "warning"))),
            kind=str(payload.get("kind", "unknown")),
            code=str(payload.get("code", "UNKNOWN")),
            summary=str(payload.get("summary", "")),
            details=str(payload.get("details", "")),
            blocking=bool(payload.get("blocking", False)),
            source=cast(_Source, str(payload.get("source", "compiler"))),
            group=cast(_Group, str(payload.get("group", "qc"))),
            check_name=(
                None if payload.get("check_name") is None else str(payload.get("check_name"))
            ),
            dedupe_key=(
                None if payload.get("dedupe_key") is None else str(payload.get("dedupe_key"))
            ),
        )


@dataclass(slots=True, frozen=True)
class CompileSignalBundle:
    status: Literal["success", "failure"]
    summary: str
    signals: tuple[CompileSignal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "signals": [signal.to_dict() for signal in self.signals],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileSignalBundle":
        raw_signals = payload.get("signals")
        signals: list[CompileSignal] = []
        if isinstance(raw_signals, list):
            for item in raw_signals:
                if isinstance(item, dict):
                    signals.append(CompileSignal.from_dict(item))
        return cls(
            status=cast(_Status, str(payload.get("status", "success"))),
            summary=str(payload.get("summary", "")),
            signals=tuple(signals),
        )


@dataclass(slots=True, frozen=True)
class CompileReport:
    """Compile result plus non-blocking warnings and informational backend notes."""

    urdf_xml: str
    warnings: list[str]
    signal_bundle: CompileSignalBundle
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_metrics: dict[str, object] = field(default_factory=dict)
    # Exporter backend-mapping facts (one per kind, aggregated over joints). Recorded in the
    # compile report only; not part of the Author feedback or the CLI warnings.
    notes: list[str] = field(default_factory=list)
