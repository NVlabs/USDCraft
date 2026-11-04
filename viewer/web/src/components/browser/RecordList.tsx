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

import { Boxes } from "lucide-react";
import { useMemo, type JSX } from "react";

import {
  BatchGroupHeader,
  groupContainsSelected,
  useBatchExpansion,
} from "@/components/browser/BatchGroupHeader";
import { RecordListItem } from "@/components/browser/RecordListItem";
import { ScrollArea } from "@/components/ui/scroll-area";
import { batchStats, describeBatchStats, groupByBatch, visibleRecords } from "@/lib/batch-groups";
import type { WorkbenchEntry } from "@/lib/types";

type RecordListProps = {
  entries: WorkbenchEntry[];
  selectedRecordId: string | null;
  onSelect: (recordId: string) => void;
};

export function RecordList({ entries, selectedRecordId, onSelect }: RecordListProps): JSX.Element {
  const expansion = useBatchExpansion();
  const items = useMemo(() => groupByBatch(entries, (entry) => entry.record?.batch_id), [entries]);
  const isGroupExpanded = (group: Extract<(typeof items)[number], { kind: "group" }>): boolean =>
    expansion.isExpanded(group.batchId, groupContainsSelected(group, selectedRecordId));
  const ordered = visibleRecords(items, isGroupExpanded);
  const neighbourOnDelete = (entry: WorkbenchEntry): string | null => {
    const index = ordered.indexOf(entry);
    return (ordered[index + 1] ?? ordered[index - 1] ?? null)?.record_id ?? null;
  };

  if (entries.length === 0) {
    return (
      <div className="flex h-48 flex-col items-center justify-center px-6 text-center">
        <span className="mb-3 grid size-9 place-items-center rounded-xl border border-[var(--border-default)] bg-[var(--surface-1)] text-[var(--accent)]">
          <Boxes className="size-4" />
        </span>
        <p className="text-[11px] font-medium text-[var(--text-secondary)]">No crafted assets found</p>
        <p className="mt-1 max-w-52 text-[9.5px] leading-4 text-[var(--text-quaternary)]">
          New canonical Generation records will appear here.
        </p>
      </div>
    );
  }

  const renderEntry = (entry: WorkbenchEntry): JSX.Element => (
    <RecordListItem
      key={entry.record_id}
      recordId={entry.record_id}
      record={entry.record}
      isSelected={selectedRecordId === entry.record_id}
      nextRecordIdOnDelete={neighbourOnDelete(entry)}
      onSelect={onSelect}
    />
  );

  return (
    <ScrollArea className="h-full">
      <div className="py-1">
        {items.map((item) => {
          if (item.kind === "record") return renderEntry(item.record);
          const containsSelected = groupContainsSelected(item, selectedRecordId);
          const expanded = expansion.isExpanded(item.batchId, containsSelected);
          return (
            <div key={`batch:${item.batchId}`}>
              <BatchGroupHeader
                batchId={item.batchId}
                detail={describeBatchStats(
                  batchStats(item.records, (entry) => entry.record?.rating),
                )}
                expanded={expanded}
                containsSelected={containsSelected}
                onToggle={() => expansion.toggle(item.batchId, expanded)}
              />
              {expanded ? (
                <div className="border-l-2 border-l-[var(--border-default)]">
                  {item.records.map(renderEntry)}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
