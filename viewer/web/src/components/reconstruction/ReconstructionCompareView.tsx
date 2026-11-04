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

import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useCallback, type JSX } from "react";

import { ViewportPanel } from "@/components/layout/ViewportPanel";
import { defaultRenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { ReconstructionRecordDetail, ReconstructionRecordSummary } from "@/lib/types";
import { agentHarnessLabel } from "@/lib/utils";
import { reconstructionRecordQueryOptions } from "@/lib/viewer-queries";

const Q0 = new Map<string, number>();
const COMPARE_RENDER_OPTIONS = { ...defaultRenderOptions, autoAnimate: false };

function paneCaption(summary: ReconstructionRecordSummary): string {
  const diagnostic = summary.axial_diagnostic;
  return [
    agentHarnessLabel(summary.agent_harness),
    diagnostic?.recall != null ? `recall ${diagnostic.recall.toFixed(2)}` : null,
    diagnostic?.exterior_excess_fraction != null
      ? `excess ${(diagnostic.exterior_excess_fraction * 100).toFixed(1)}%`
      : null,
    summary.rating != null ? `${summary.rating}★` : null,
  ]
    .filter((item): item is string => Boolean(item))
    .join(" · ");
}

function ComparePane({
  detail,
  label,
}: {
  detail: ReconstructionRecordDetail | null;
  label: string;
}): JSX.Element {
  const noop = useCallback(() => {}, []);
  return (
    <div className="relative flex min-w-0 flex-1 flex-col border-[var(--border-default)] first:border-r">
      <div className="pointer-events-none absolute left-3 top-3 z-[7] max-w-[85%] rounded-md border border-[var(--border-default)] bg-[var(--surface-0)]/85 px-2.5 py-1.5 backdrop-blur">
        <p className="text-[9px] font-medium uppercase tracking-[0.06em] text-[var(--text-quaternary)]">
          {label}
        </p>
        <p className="truncate text-[10.5px] text-[var(--text-primary)]">
          {detail ? detail.summary.title || detail.summary.record_id : "Loading…"}
        </p>
        {detail ? (
          <p className="mt-0.5 truncate font-mono text-[9px] text-[var(--text-tertiary)]">
            {paneCaption(detail.summary)}
          </p>
        ) : null}
      </div>
      <div className="min-h-0 flex-1">
        {detail ? (
          <ViewportPanel
            baseFileUrl={detail.candidate.available ? detail.candidate.base_file_url : null}
            assetRevisionKey={detail.candidate.revision_key}
            selectionKey={detail.summary.record_id}
            jointPoseSignal={Q0}
            renderOptions={COMPARE_RENDER_OPTIONS}
            onUrdfSpecChange={noop}
            allowLocalUsdFile={false}
          />
        ) : null}
      </div>
    </div>
  );
}

/** Two Reconstruction candidates side by side at q=0, each in its own viewport. */
export function ReconstructionCompareView({
  left,
  rightRecordId,
  onClose,
}: {
  left: ReconstructionRecordDetail;
  rightRecordId: string;
  onClose: () => void;
}): JSX.Element {
  const rightQuery = useQuery(reconstructionRecordQueryOptions(rightRecordId));
  const rightError = rightQuery.error instanceof Error ? rightQuery.error.message : null;
  return (
    <section className="relative flex h-full min-h-0 bg-[var(--surface-2)]" data-reconstruction-compare>
      <ComparePane detail={left} label="Selected" />
      {rightError ? (
        <div className="flex min-w-0 flex-1 items-center justify-center px-6 text-center text-[11px] text-[var(--destructive)]">
          {rightError}
        </div>
      ) : (
        <ComparePane detail={rightQuery.data ?? null} label="Compared with" />
      )}
      <button
        type="button"
        onClick={onClose}
        className="absolute right-3 top-3 z-[7] flex items-center gap-1 rounded-md border border-[var(--border-default)] bg-[var(--surface-0)]/85 px-2 py-1 text-[10px] text-[var(--text-secondary)] backdrop-blur hover:text-[var(--text-primary)]"
        aria-label="Close comparison"
      >
        <X className="size-3" />
        Close compare
      </button>
    </section>
  );
}
