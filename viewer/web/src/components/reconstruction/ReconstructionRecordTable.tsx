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

import type { JSX } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import type { BatchGroup, BatchListItem } from "@/lib/batch-groups";
import { scaleSourceLabel } from "@/lib/reconstruction-filters";
import { reconstructionDuration } from "@/lib/reconstruction-duration";
import type { ReconstructionRecordSummary } from "@/lib/types";
import { agentHarnessLabel, cn } from "@/lib/utils";

const DATE_FORMATTER = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDate(value: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "--" : DATE_FORMATTER.format(date);
}

export type ReconstructionTableCell = {
  key: string;
  value: string;
  tone?: "pass" | "warn" | "muted";
};

/** The dense table cells of one record, in column order (pure; unit-tested). */
export function reconstructionTableCells(record: ReconstructionRecordSummary): ReconstructionTableCell[] {
  const diagnostic = record.axial_diagnostic;
  const recall = diagnostic?.recall;
  const excess = diagnostic?.exterior_excess_fraction;
  return [
    {
      key: "source",
      value: agentHarnessLabel(record.agent_harness) ?? "--",
      tone: record.agent_harness ? undefined : "muted",
    },
    { key: "model", value: record.model_id ?? "--", tone: record.model_id ? undefined : "muted" },
    {
      key: "scale",
      value: scaleSourceLabel(record.scale_source)?.replace("scale: ", "") ?? "--",
      tone: record.scale_source === "format_contract" ? "warn" : undefined,
    },
    {
      key: "recall",
      value: recall != null ? `${Math.round(recall * 100)}%` : "--",
      tone: recall == null ? "muted" : recall < 0.6 ? "warn" : "pass",
    },
    {
      key: "excess",
      value: excess != null ? `${(excess * 100).toFixed(1)}%` : "--",
      tone: excess == null ? "muted" : excess > 0.016 ? "warn" : "pass",
    },
    { key: "rating", value: record.rating != null ? `${record.rating}★` : "--", tone: record.rating != null ? undefined : "muted" },
    { key: "updated", value: formatDate(record.updated_at ?? record.created_at), tone: "muted" },
  ];
}

export const RECONSTRUCTION_TABLE_COLUMNS: ReadonlyArray<{ key: string; label: string; numeric?: boolean }> = [
  { key: "title", label: "Asset" },
  { key: "source", label: "Author" },
  { key: "model", label: "Model" },
  { key: "scale", label: "Scale" },
  { key: "recall", label: "Recall", numeric: true },
  { key: "excess", label: "Excess", numeric: true },
  { key: "rating", label: "Rating", numeric: true },
  { key: "updated", label: "Updated" },
];

export function ReconstructionRecordTable({
  items,
  selectedRecordId,
  onSelect,
  isGroupExpanded,
  renderGroupHeader,
}: {
  items: readonly BatchListItem<ReconstructionRecordSummary>[];
  selectedRecordId: string | null;
  onSelect: (recordId: string) => void;
  isGroupExpanded: (group: BatchGroup<ReconstructionRecordSummary>) => boolean;
  renderGroupHeader: (group: BatchGroup<ReconstructionRecordSummary>, className?: string) => JSX.Element;
}): JSX.Element {
  const renderRow = (record: ReconstructionRecordSummary, grouped: boolean): JSX.Element => {
    const selected = record.record_id === selectedRecordId;
    return (
      <tr
        key={record.record_id}
        onClick={() => onSelect(record.record_id)}
        aria-selected={selected}
        data-reconstruction-record={record.record_id}
        className={cn(
          "cursor-pointer border-b border-[var(--border-subtle)] transition-colors",
          selected ? "bg-[var(--accent-soft)]" : "hover:bg-[var(--surface-1)]",
          grouped && "border-l-2 border-l-[var(--border-default)]",
        )}
      >
        <td className="max-w-[16rem] px-2 py-1.5">
          <span className="line-clamp-2 block text-[10.5px] font-medium text-[var(--text-primary)]">
            {record.title || record.prompt_preview || record.record_id}
          </span>
          {reconstructionDuration(record) && (
            <span title={reconstructionDuration(record)!.title} className="mt-1 inline-block rounded border border-[var(--border-default)] px-1 py-0.5 font-mono text-[8px] text-[var(--text-secondary)]">
              {reconstructionDuration(record)!.text}
            </span>
          )}
        </td>
        {reconstructionTableCells(record).map((cell, index) => (
          <td
            key={cell.key}
            className={cn(
              "whitespace-nowrap px-2 py-1.5 font-mono",
              RECONSTRUCTION_TABLE_COLUMNS[index + 1]?.numeric && "text-right tabular-nums",
              cell.tone === "warn"
                ? "font-medium text-[var(--destructive)]"
                : cell.tone === "pass"
                  ? "font-medium text-[var(--success)]"
                  : cell.tone === "muted"
                    ? "text-[var(--text-quaternary)]"
                    : "text-[var(--text-secondary)]",
            )}
          >
            {cell.value}
          </td>
        ))}
      </tr>
    );
  };
  return (
    <ScrollArea className="h-full">
      <table className="w-full border-collapse text-left text-[10px]" data-reconstruction-table>
        <thead className="sticky top-0 z-[1] bg-[var(--surface-0)]">
          <tr>
            {RECONSTRUCTION_TABLE_COLUMNS.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cn(
                  "whitespace-nowrap border-b border-[var(--border-default)] px-2 py-1.5 text-[9px] font-medium uppercase tracking-[0.06em] text-[var(--text-quaternary)]",
                  column.numeric && "text-right",
                )}
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.flatMap((item) => {
            if (item.kind === "record") return [renderRow(item.record, false)];
            const rows = [
              <tr key={`batch:${item.batchId}`}>
                <td colSpan={RECONSTRUCTION_TABLE_COLUMNS.length} className="p-0">
                  {renderGroupHeader(item, "py-1.5")}
                </td>
              </tr>,
            ];
            if (isGroupExpanded(item)) {
              rows.push(...item.records.map((record) => renderRow(record, true)));
            }
            return rows;
          })}
        </tbody>
      </table>
    </ScrollArea>
  );
}
