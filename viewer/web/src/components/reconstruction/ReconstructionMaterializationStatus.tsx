// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

import { type JSX } from "react";

import { ArtifactContractWarning } from "@/components/layout/ArtifactContractWarning";
import { Badge } from "@/components/ui/badge";
import type { ReconstructionRecordSummary } from "@/lib/types";

export function ReconstructionMaterializationStatus({
  record,
}: {
  record: ReconstructionRecordSummary;
}): JSX.Element {
  const blocking = !record.materialization_displayable;
  const variant =
    record.materialization_status === "available"
      ? "success"
      : record.materialization_status === "compile_failed"
        ? "destructive"
        : "warning";
  return (
    <div className="rounded-md border border-[var(--border-default)] bg-[var(--surface-0)] p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[9px] font-medium uppercase tracking-[0.06em] text-[var(--text-quaternary)]">
          Candidate materialization
        </span>
        <Badge variant={variant}>{record.materialization_status}</Badge>
      </div>
      {record.materialization_status === "contract_invalid" ? (
        <div className="mt-2">
          <ArtifactContractWarning issues={record.materialization_issues} />
        </div>
      ) : blocking ? (
        <div className="mt-2 text-[9px] leading-4 text-[var(--text-tertiary)]">
          <p>
            {record.materialization_status === "compile_failed"
              ? "The compile report records failure; candidate preview is blocked."
              : "The required Viewer model is missing; candidate preview is blocked."}
          </p>
          {record.materialization_issues.length > 0 ? (
            <ul className="mt-1.5 max-h-24 space-y-1 overflow-y-auto rounded bg-[var(--surface-1)] p-2 font-mono">
              {record.materialization_issues.map((issue, index) => (
                <li key={`${issue.path}:${issue.message}:${index}`}>
                  <span className="font-semibold">{issue.path}</span>: {issue.message}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
