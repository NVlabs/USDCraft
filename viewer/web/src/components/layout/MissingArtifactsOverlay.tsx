// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
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

import type { MaterializationIssue, MaterializationStatus } from "@/lib/types";

type MissingArtifactsOverlayProps = {
  status: Extract<MaterializationStatus, "missing" | "compile_failed">;
  issues: MaterializationIssue[];
  hasCompileReport: boolean;
  detail?: string | null;
  compact?: boolean;
  productLabel?: "Generation" | "Reconstruction";
};

export function MissingArtifactsOverlay({
  status,
  issues,
  hasCompileReport,
  detail = null,
  compact = false,
  productLabel = "Generation",
}: MissingArtifactsOverlayProps): JSX.Element {
  const compileFailed = status === "compile_failed";
  const title = compileFailed ? "Artifact compilation failed" : "Viewer model files are missing";
  const description = compileFailed
    ? `The ${productLabel} compile report records a failed artifact build, so this record has no model the read-only viewer can display.`
    : "The saved model.urdf entry required to display this record is missing.";

  return (
    <div className="flex h-full w-full items-center justify-center p-4">
      <div
        className={[
          "w-full rounded-xl border border-[var(--border-default)] bg-[var(--surface-0)] shadow-[0_16px_40px_rgba(15,23,42,0.14)]",
          compact ? "max-w-sm p-4" : "max-w-xl p-5",
        ].join(" ")}
      >
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex h-2.5 w-2.5 rounded-full ${compileFailed ? "bg-[var(--destructive)]" : "bg-[#d97706]"}`}
          />
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--text-tertiary)]">
            {compileFailed ? "Compilation Failed" : "Materialization Required"}
          </p>
        </div>

        <h3 className={compact ? "mt-3 text-[15px] font-semibold text-[var(--text-primary)]" : "mt-3 text-[18px] font-semibold text-[var(--text-primary)]"}>
          {title}
        </h3>
        <p className="mt-2 text-[12px] leading-5 text-[var(--text-secondary)]">{description}</p>
        {detail ? <p className="mt-2 text-[11px] leading-5 text-[var(--text-tertiary)]">{detail}</p> : null}
        {issues.length > 0 ? (
          <div className="mt-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--text-tertiary)]">
              {compileFailed ? "Compilation details" : "Materialization details"}
            </p>
            <ul className="mt-1.5 max-h-32 space-y-1 overflow-y-auto rounded-md border border-[var(--border-default)] bg-[var(--surface-1)] p-2 font-mono text-[10px] leading-4 text-[var(--text-secondary)]">
              {issues.map((issue, index) => (
                <li key={`${issue.path}:${issue.message}:${index}`}>
                  <span className="font-semibold">{issue.path}</span>: {issue.message}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {compileFailed && hasCompileReport ? (
          <p className="mt-3 text-[11px] leading-5 text-[var(--text-tertiary)]">
            compile_report.json remains available in Read-only Artifacts for diagnosis.
          </p>
        ) : null}

        <p className="mt-4 text-[11px] leading-5 text-[var(--text-tertiary)]">
          The viewer is read-only and only displays assets already produced by the {productLabel} pipeline.
        </p>
      </div>
    </div>
  );
}
