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

import { AlertTriangle, Boxes, X } from "lucide-react";
import { useMemo, type JSX } from "react";

import { useViewer, useViewerDispatch } from "@/lib/viewer-context";
import { RecordList } from "@/components/browser/RecordList";
import { RecordFilters } from "@/components/browser/RecordFilters";
import { RecordSearch } from "@/components/browser/RecordSearch";
import {
  availableWorkbenchModels,
  filterWorkbenchEntries,
} from "@/lib/workbench-filters";
import type { WorkbenchEntry } from "@/lib/types";

const EMPTY_WORKBENCH_ENTRIES: WorkbenchEntry[] = [];

export function RecordBrowser(): JSX.Element {
  const {
    bootstrap,
    error,
    inputKindFilter,
    loading,
    modelFilter,
    ratingFilter,
    searchQuery,
    selectedRecordId,
    sourceFilter,
    workbenchNotice,
  } = useViewer();
  const dispatch = useViewerDispatch();
  const workbenchEntries = bootstrap?.workbench_entries ?? EMPTY_WORKBENCH_ENTRIES;

  const entries = useMemo(() => {
    return filterWorkbenchEntries(workbenchEntries, {
      searchQuery,
      sourceFilter,
      modelFilter,
      inputKindFilter,
      ratingFilter,
    });
  }, [inputKindFilter, modelFilter, ratingFilter, searchQuery, sourceFilter, workbenchEntries]);
  const availableModels = useMemo(
    () => availableWorkbenchModels(workbenchEntries),
    [workbenchEntries],
  );
  const totalVisibleEntries = workbenchEntries.filter(
    (entry) => !entry.archived && entry.record,
  ).length;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-[var(--border-default)] bg-[linear-gradient(180deg,var(--surface-0),var(--surface-0)_75%,var(--surface-2))] px-3 pb-2.5 pt-3">
        <div className="mb-2 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="panel-kicker flex items-center gap-1.5 text-[var(--accent)]">
              <span className="h-px w-3 bg-[var(--accent)]" aria-hidden="true" />
              Generation
            </p>
            <p className="mt-1 text-[13px] font-semibold tracking-[-0.025em] text-[var(--text-primary)]">
              Asset library
            </p>
            <p className="mt-0.5 text-[9px] leading-3.5 text-[var(--text-tertiary)]">
              Models created from text and images
            </p>
          </div>
          <span className="flex shrink-0 items-center gap-1.5 rounded-full border border-[var(--border-default)] bg-[var(--surface-1)] px-2 py-1 font-mono text-[9px] tabular-nums text-[var(--text-secondary)]">
            <Boxes className="size-2.5 text-[var(--accent)]" />
            {entries.length === totalVisibleEntries ? entries.length : `${entries.length}/${totalVisibleEntries}`}
          </span>
        </div>
        <RecordSearch />
        <RecordFilters
          models={availableModels.models}
          hasUnknownModel={availableModels.hasUnknown}
        />
        {workbenchNotice ? (
          <div
            className="mt-2 flex items-start gap-2 rounded-md border border-amber-500/25 bg-amber-500/[0.07] p-2 text-[9.5px] leading-4 text-amber-700"
            role="alert"
          >
            <AlertTriangle className="mt-0.5 size-3 shrink-0" />
            <span className="min-w-0 flex-1">{workbenchNotice}</span>
            <button
              type="button"
              onClick={() => dispatch({ type: "SET_WORKBENCH_NOTICE", payload: null })}
              className="rounded p-0.5 text-amber-700/70 outline-none hover:text-amber-800 focus-visible:ring-2 focus-visible:ring-amber-500/30"
              aria-label="Dismiss Workbench notice"
            >
              <X className="size-3" />
            </button>
          </div>
        ) : null}
      </div>

      <div className="min-h-0 flex-1">
        {loading ? (
          <div className="flex h-40 flex-col items-center justify-center gap-2 text-[11px] text-[var(--text-quaternary)]">
            <span className="size-5 animate-pulse rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)]" />
            Loading asset library…
          </div>
        ) : error ? (
          <div className="p-4 text-[11px] leading-5 text-[var(--destructive)]">{error}</div>
        ) : (
          <RecordList
            entries={entries}
            selectedRecordId={selectedRecordId}
            onSelect={(recordId) => dispatch({ type: "SELECT_RECORD", payload: recordId })}
          />
        )}
      </div>
    </div>
  );
}
