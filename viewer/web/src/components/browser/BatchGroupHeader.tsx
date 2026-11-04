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

import { ChevronDown, ChevronRight, Layers } from "lucide-react";
import { useCallback, useState, type JSX } from "react";

import { formatBatchStart, parseBatchId, type BatchGroup } from "@/lib/batch-groups";
import { cn } from "@/lib/utils";

export type BatchExpansion = {
  /** Expanded when toggled open, or (by default) when the group holds the selected record. */
  isExpanded: (batchId: string, containsSelected: boolean) => boolean;
  toggle: (batchId: string, currentlyExpanded: boolean) => void;
};

/** Manual open/closed overrides per batch; groups without an override follow the selection. */
export function useBatchExpansion(): BatchExpansion {
  const [overrides, setOverrides] = useState<ReadonlyMap<string, boolean>>(new Map());
  const isExpanded = useCallback(
    (batchId: string, containsSelected: boolean) => overrides.get(batchId) ?? containsSelected,
    [overrides],
  );
  const toggle = useCallback((batchId: string, currentlyExpanded: boolean) => {
    setOverrides((current) => new Map(current).set(batchId, !currentlyExpanded));
  }, []);
  return { isExpanded, toggle };
}

export function groupContainsSelected<T extends { record_id: string }>(
  group: BatchGroup<T>,
  selectedRecordId: string | null,
): boolean {
  return selectedRecordId !== null && group.records.some((record) => record.record_id === selectedRecordId);
}

export function BatchGroupHeader({
  batchId,
  detail,
  expanded,
  containsSelected,
  onToggle,
  className,
}: {
  batchId: string;
  detail: string;
  expanded: boolean;
  containsSelected: boolean;
  onToggle: () => void;
  className?: string;
}): JSX.Element {
  const label = parseBatchId(batchId);
  const started = formatBatchStart(label);
  const Chevron = expanded ? ChevronDown : ChevronRight;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={expanded}
      data-batch-group={batchId}
      title={batchId}
      className={cn(
        "flex w-full items-center gap-2 border-b border-l-2 border-b-[var(--border-subtle)] bg-[var(--surface-1)] px-3 py-2 text-left outline-none transition-colors hover:bg-[var(--surface-2)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]",
        containsSelected ? "border-l-[var(--accent)]" : "border-l-transparent",
        className,
      )}
    >
      <Chevron className="size-3 shrink-0 text-[var(--text-tertiary)]" />
      <Layers className="size-3 shrink-0 text-[var(--accent)]" />
      <span className="min-w-0 flex-1">
        <span className="flex items-baseline gap-2">
          <span className="truncate font-mono text-[10px] font-medium text-[var(--text-primary)]">
            {label.name}
          </span>
          {started ? (
            <span className="shrink-0 font-mono text-[9px] text-[var(--text-quaternary)]">{started}</span>
          ) : null}
        </span>
        <span className="block truncate text-[9.5px] text-[var(--text-tertiary)]">{detail}</span>
      </span>
      <span className="shrink-0 rounded-full border border-[var(--border-default)] px-1.5 py-0.5 font-mono text-[9px] tabular-nums text-[var(--text-secondary)]">
        {detail.split(" ")[0]}
      </span>
    </button>
  );
}
