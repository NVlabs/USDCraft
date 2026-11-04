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

import { AlertTriangle, ChevronUp } from "lucide-react";
import { type JSX } from "react";

import type { MaterializationIssue } from "@/lib/types";

type ArtifactContractWarningProps = {
  issues: MaterializationIssue[];
};

function IssueList({ issues }: ArtifactContractWarningProps): JSX.Element {
  if (issues.length === 0) {
    return (
      <p className="mt-2 text-[10.5px] leading-4 text-[var(--text-tertiary)]">
        No structured contract details were reported.
      </p>
    );
  }

  return (
    <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto rounded-md border border-amber-600/20 bg-amber-500/[0.04] p-2 font-mono text-[10px] leading-4 text-[var(--text-secondary)]">
      {issues.map((issue, index) => (
        <li key={`${issue.path}:${issue.message}:${index}`}>
          <span className="font-semibold">{issue.path}</span>: {issue.message}
        </li>
      ))}
    </ul>
  );
}

export function ArtifactContractWarning({
  issues,
}: ArtifactContractWarningProps): JSX.Element {
  return (
    <details className="group relative inline-block max-w-full">
      <summary
        className="inline-flex cursor-pointer list-none items-center gap-1.5 rounded-md border border-amber-600/15 bg-[var(--surface-0)]/95 px-2.5 py-1 text-[11px] font-medium text-amber-700 shadow-sm backdrop-blur-sm outline-none transition hover:bg-amber-500/[0.08] focus-visible:ring-2 focus-visible:ring-amber-500/30 [&::-webkit-details-marker]:hidden"
        aria-label="Artifact contract is invalid. Click to view details."
      >
        <AlertTriangle className="size-3 shrink-0 opacity-80" />
        <span>Artifact contract is invalid</span>
        <ChevronUp className="size-3 shrink-0 opacity-60 transition-transform group-open:rotate-180" />
      </summary>

      <div className="absolute bottom-full left-0 z-50 mb-2 w-max min-w-72 max-w-[min(32rem,calc(100vw-2rem))] rounded-lg border border-amber-600/20 bg-[var(--surface-0)] p-3 shadow-[0_10px_32px_rgba(15,23,42,0.16)]">
        <p className="text-[11px] font-semibold text-[var(--text-primary)]">
          Artifact contract is invalid
        </p>
        <p className="mt-1 text-[10.5px] leading-4 text-[var(--text-tertiary)]">
          A displayable model is available. Canonical artifact validation reported the following
          issue{issues.length === 1 ? "" : "s"}.
        </p>
        <IssueList issues={issues} />
      </div>
    </details>
  );
}
