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

import { useQueryClient } from "@tanstack/react-query";
import { Box, Boxes, LayoutList, ScanLine, Search, Table2, Trash2 } from "lucide-react";
import { useMemo, useRef, useState, type JSX } from "react";

import {
  BatchGroupHeader,
  groupContainsSelected,
  useBatchExpansion,
} from "@/components/browser/BatchGroupHeader";
import { StarRating, type StarValue } from "@/components/browser/StarRating";
import { ReconstructionRecordFilters } from "@/components/reconstruction/ReconstructionRecordFilters";
import { ReconstructionRecordTable } from "@/components/reconstruction/ReconstructionRecordTable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { deleteReconstructionRecord, saveReconstructionRecordRating } from "@/lib/api";
import { useViewerReadOnly } from "@/lib/use-viewer-read-only";
import {
  batchStats,
  describeBatchStats,
  groupByBatch,
  type BatchGroup,
} from "@/lib/batch-groups";
import {
  DEFAULT_RECONSTRUCTION_FILTERS,
  filterReconstructionRecords,
  recallOf,
  reconstructionFilterOptions,
  scaleSourceLabel,
  type ReconstructionFilters,
} from "@/lib/reconstruction-filters";
import { reconstructionDuration } from "@/lib/reconstruction-duration";
import type { ReconstructionRecordSummary } from "@/lib/types";
import { agentHarnessLabel, cn } from "@/lib/utils";
import { viewerQueryKeys } from "@/lib/viewer-queries";

const DATE_FORMATTER = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });

function formatDate(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : DATE_FORMATTER.format(date);
}

export function reconstructionRecordListMetadata(record: ReconstructionRecordSummary): string[] {
  return [
    agentHarnessLabel(record.agent_harness),
    record.model_id,
    record.reasoning_effort ? `${record.reasoning_effort} reasoning` : null,
    formatDate(record.updated_at ?? record.created_at),
  ].filter((item): item is string => Boolean(item));
}

export type ReconstructionQualityMetric = {
  key: "recall" | "excess";
  label: string;
  value: string;
  detail: string | null;
  tone: "pass" | "warn" | "muted";
};

/** Compact, user-facing AXIAL quality readings. The 60% recall split is presentation only. */
export function reconstructionRecordQuality(
  record: ReconstructionRecordSummary,
): ReconstructionQualityMetric[] {
  const diagnostic = record.axial_diagnostic;
  const recall = diagnostic?.recall;
  const excess = diagnostic?.exterior_excess_fraction;
  return [
    {
      key: "recall",
      label: "AXIAL recall",
      value: recall != null ? `${Math.round(recall * 100)}%` : "No reading",
      detail: recall != null ? (recall >= 0.6 ? "60%+" : "Below 60%") : null,
      tone: recall == null ? "muted" : recall >= 0.6 ? "pass" : "warn",
    },
    {
      key: "excess",
      label: "Exterior excess",
      value: excess != null ? `${(excess * 100).toFixed(1)}%` : "No reading",
      detail: excess != null ? (excess > 0.016 ? "Outside source" : "No excess") : null,
      tone: excess == null ? "muted" : excess > 0.016 ? "warn" : "pass",
    },
  ];
}

function qualityToneClass(tone: ReconstructionQualityMetric["tone"]): string {
  if (tone === "pass") {
    return "border-[var(--accent)] bg-[var(--success-soft)] text-[var(--success)]";
  }
  if (tone === "warn") {
    return "border-[var(--destructive)] bg-[var(--destructive-soft)] text-[var(--destructive)]";
  }
  return "border-[var(--border-default)] bg-[var(--surface-1)] text-[var(--text-quaternary)]";
}

/** Group summary line under a batch header: count, mean recall and ratings. */
export function reconstructionBatchDetail(records: readonly ReconstructionRecordSummary[]): string {
  return describeBatchStats(batchStats(records, (record) => record.rating, recallOf));
}

/** Replace one summary in the cached Reconstruction record list (a rating changed). */
export function replaceReconstructionRecord(
  records: readonly ReconstructionRecordSummary[] | undefined,
  updated: ReconstructionRecordSummary,
): ReconstructionRecordSummary[] {
  if (!records) return [updated];
  let replaced = false;
  const next = records.map((record) => {
    if (record.record_id !== updated.record_id) return record;
    replaced = true;
    return updated;
  });
  return replaced ? next : [...next, updated];
}

/** Drop one record from the cached Reconstruction record list (it was deleted). */
export function withoutReconstructionRecord(
  records: readonly ReconstructionRecordSummary[] | undefined,
  recordId: string,
): ReconstructionRecordSummary[] {
  return (records ?? []).filter((record) => record.record_id !== recordId);
}

type RowMutation = "rating" | "delete";

function RecordRow({
  record,
  selected,
  onSelect,
}: {
  record: ReconstructionRecordSummary;
  selected: boolean;
  onSelect: () => void;
}): JSX.Element {
  const queryClient = useQueryClient();
  const readOnly = useViewerReadOnly();
  const [mutationPending, setMutationPending] = useState<RowMutation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const mutationRef = useRef<RowMutation | null>(null);
  const recordId = record.record_id;
  const quality = reconstructionRecordQuality(record);
  const duration = reconstructionDuration(record);
  const scaleLabel = scaleSourceLabel(record.scale_source)?.replace("scale: ", "") ?? "unknown";

  const handleRating = async (rating: StarValue): Promise<void> => {
    if (deleteConfirmOpen || mutationRef.current) return;
    const nextRating = record.rating === rating ? null : rating;
    mutationRef.current = "rating";
    setMutationPending("rating");
    setError(null);
    try {
      const updated = await saveReconstructionRecordRating(recordId, nextRating);
      queryClient.setQueryData<ReconstructionRecordSummary[]>(
        viewerQueryKeys.reconstructionRecords(),
        (records) => replaceReconstructionRecord(records, updated),
      );
      void queryClient.invalidateQueries({
        queryKey: viewerQueryKeys.reconstructionRecord(recordId),
      });
    } catch (mutationError) {
      setError(mutationError instanceof Error ? mutationError.message : "Failed to save rating");
    } finally {
      mutationRef.current = null;
      setMutationPending(null);
    }
  };

  const handleDelete = async (): Promise<void> => {
    if (mutationRef.current) return;
    mutationRef.current = "delete";
    setMutationPending("delete");
    setError(null);
    try {
      await deleteReconstructionRecord(recordId);
      queryClient.setQueryData<ReconstructionRecordSummary[]>(
        viewerQueryKeys.reconstructionRecords(),
        (records) => withoutReconstructionRecord(records, recordId),
      );
      queryClient.removeQueries({ queryKey: viewerQueryKeys.reconstructionRecord(recordId) });
      void queryClient.invalidateQueries({ queryKey: viewerQueryKeys.reconstructionRecords() });
    } catch (mutationError) {
      setError(mutationError instanceof Error ? mutationError.message : "Failed to delete record");
      mutationRef.current = null;
      setMutationPending(null);
    }
  };

  return (
    <div
      className={cn(
        "border-b border-l-2 border-b-[var(--border-subtle)] px-3 py-3 transition-colors",
        selected
          ? "border-l-[var(--accent)] bg-[var(--accent-soft)]"
          : "border-l-transparent hover:bg-[var(--surface-1)]",
      )}
      data-reconstruction-record={recordId}
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          onClick={onSelect}
          aria-current={selected ? "true" : undefined}
          className="flex min-w-0 flex-1 items-start gap-2 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]"
        >
          <Box
            className={cn(
              "mt-0.5 size-3 shrink-0 text-[var(--text-quaternary)]",
              selected && "text-[var(--accent)]",
            )}
          />
          <span className="min-w-0 flex-1">
            <span className="line-clamp-2 block text-[11.5px] font-medium leading-[1.5] tracking-[-0.01em] text-[var(--text-primary)]">
              {record.title || record.prompt_preview || recordId}
            </span>
            <span className="mt-1 line-clamp-2 block text-[9.5px] leading-[1.45] text-[var(--text-tertiary)]">
              {reconstructionRecordListMetadata(record).join(" · ")}
            </span>
            <span className="mt-1.5 flex flex-wrap gap-1">
              <span className="rounded border border-[var(--border-default)] bg-[var(--surface-1)] px-1.5 py-0.5 font-mono text-[8px] uppercase tracking-[0.05em] text-[var(--text-secondary)]">
                {scaleLabel} scale
              </span>
              {duration && (
                <span title={duration.title} className="rounded border border-[var(--border-default)] bg-[var(--surface-1)] px-1.5 py-0.5 font-mono text-[8px] text-[var(--text-secondary)]">
                  {duration.text}
                </span>
              )}
            </span>
          </span>
        </button>
        {readOnly ? null : (
          <button
            type="button"
            disabled={mutationPending !== null}
            onClick={() => setDeleteConfirmOpen((open) => !open)}
            aria-expanded={deleteConfirmOpen}
            aria-label={`Delete record ${recordId}`}
            className="rounded-sm p-0.5 text-[var(--text-quaternary)] outline-none transition-colors hover:text-[var(--destructive)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)] disabled:cursor-wait disabled:opacity-60"
          >
            <Trash2 className="size-3" />
          </button>
        )}
      </div>

      {!deleteConfirmOpen ? (
        <div className="mt-2.5 ml-5">
          <div className="grid grid-cols-3 gap-1.5" aria-label="Reconstruction quality">
            {quality.map((metric) => (
              <div
                key={metric.key}
                className={cn(
                  "min-w-0 rounded-md border px-1.5 py-1.5",
                  qualityToneClass(metric.tone),
                )}
                title={
                  metric.key === "recall"
                    ? "AXIAL source-volume recall; 60% is the list review threshold"
                    : "Candidate geometry measured outside the source volume"
                }
              >
                <span className="block truncate font-mono text-[7.5px] font-medium uppercase tracking-[0.06em] opacity-80">
                  {metric.label}
                </span>
                <span
                  className={cn(
                    "mt-0.5 block truncate font-mono font-medium tabular-nums",
                    metric.key === "recall" ? "text-[14px] leading-4" : "text-[9.5px] leading-4",
                  )}
                >
                  {metric.value}
                </span>
                {metric.detail ? (
                  <span className="block truncate text-[7.5px] leading-3 opacity-75">
                    {metric.detail}
                  </span>
                ) : null}
              </div>
            ))}
          </div>
          <div className="mt-2 flex items-center justify-between gap-2">
            <span className="font-mono text-[8px] uppercase tracking-[0.06em] text-[var(--text-quaternary)]">
              {readOnly ? "Rating" : "Your rating"}
            </span>
            <StarRating
              rating={record.rating}
              disabled={mutationPending !== null || readOnly}
              pending={mutationPending === "rating"}
              onRate={(value) => void handleRating(value)}
            />
          </div>
        </div>
      ) : (
        <div
          className="mt-2 rounded-md border border-[rgba(209,52,21,0.24)] bg-[rgba(209,52,21,0.05)] p-2"
          role="alertdialog"
          aria-busy={mutationPending === "delete"}
          onKeyDown={(event) => {
            if (event.key === "Escape" && mutationPending !== "delete") {
              event.preventDefault();
              setDeleteConfirmOpen(false);
            }
          }}
        >
          <p className="text-[10.5px] font-medium text-[var(--destructive)]">
            Permanently delete this record?
          </p>
          <p className="mt-1 break-all font-mono text-[9px] leading-3.5 text-[var(--text-tertiary)]">
            {recordId}
          </p>
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              disabled={mutationPending === "delete"}
              onClick={() => void handleDelete()}
              className="rounded-md bg-[var(--destructive)] px-2 py-1 text-[10px] font-medium text-white outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)] disabled:cursor-wait disabled:opacity-60"
            >
              {mutationPending === "delete" ? "Deleting…" : "Delete"}
            </button>
            <button
              type="button"
              disabled={mutationPending === "delete"}
              onClick={() => setDeleteConfirmOpen(false)}
              className="rounded-md border border-[var(--border-default)] px-2 py-1 text-[10px] text-[var(--text-secondary)] outline-none hover:bg-[var(--surface-2)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)] disabled:opacity-60"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {error ? (
        <p className="mt-1 ml-5 text-[9.5px] leading-4 text-[var(--destructive)]" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function ReconstructionRecordBrowser({
  records,
  selectedRecordId,
  loading,
  error,
  onSelect,
}: {
  records: readonly ReconstructionRecordSummary[];
  selectedRecordId: string | null;
  loading: boolean;
  error: string | null;
  onSelect: (recordId: string) => void;
}): JSX.Element {
  const [filters, setFilters] = useState<ReconstructionFilters>(DEFAULT_RECONSTRUCTION_FILTERS);
  const [view, setView] = useState<"list" | "table">("list");
  const options = useMemo(() => reconstructionFilterOptions(records), [records]);
  const visibleRecords = useMemo(
    () => filterReconstructionRecords(records, filters),
    [filters, records],
  );
  const items = useMemo(
    () => groupByBatch(visibleRecords, (record) => record.batch_id),
    [visibleRecords],
  );
  const expansion = useBatchExpansion();
  const groupExpanded = (group: BatchGroup<ReconstructionRecordSummary>): boolean =>
    expansion.isExpanded(group.batchId, groupContainsSelected(group, selectedRecordId));
  const renderGroupHeader = (
    group: BatchGroup<ReconstructionRecordSummary>,
    className?: string,
  ): JSX.Element => {
    const containsSelected = groupContainsSelected(group, selectedRecordId);
    const expanded = expansion.isExpanded(group.batchId, containsSelected);
    return (
      <BatchGroupHeader
        batchId={group.batchId}
        detail={reconstructionBatchDetail(group.records)}
        expanded={expanded}
        containsSelected={containsSelected}
        onToggle={() => expansion.toggle(group.batchId, expanded)}
        className={className}
      />
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-[var(--border-default)] bg-[linear-gradient(180deg,var(--surface-0),var(--surface-0)_75%,var(--surface-2))] px-3 pb-2.5 pt-3">
        <div className="mb-2 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="panel-kicker flex items-center gap-1.5 text-[var(--accent)]">
              <span className="h-px w-3 bg-[var(--accent)]" aria-hidden="true" />
              Reconstruction
            </p>
            <p className="mt-1 text-[13px] font-semibold tracking-[-0.025em] text-[var(--text-primary)]">
              Asset quality review
            </p>
            <p className="mt-0.5 text-[9px] leading-3.5 text-[var(--text-tertiary)]">
              Compare model fit, scale and rating
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <span className="flex items-center gap-1.5 rounded-full border border-[var(--border-default)] bg-[var(--surface-1)] px-2 py-1 font-mono text-[9px] tabular-nums text-[var(--text-secondary)]">
              <Boxes className="size-2.5 text-[var(--accent)]" />
              {visibleRecords.length === records.length
                ? visibleRecords.length
                : `${visibleRecords.length}/${records.length}`}
            </span>
            <div className="flex rounded-md border border-[var(--border-default)] bg-[var(--surface-1)] p-0.5" aria-label="List layout">
              {(["list", "table"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  aria-pressed={view === option}
                  aria-label={option === "list" ? "Card list" : "Dense table"}
                  onClick={() => setView(option)}
                  className={cn(
                    "rounded p-1 transition-colors",
                    view === option
                      ? "bg-[var(--accent-soft)] text-[var(--text-primary)]"
                      : "text-[var(--text-quaternary)] hover:text-[var(--text-secondary)]",
                  )}
                >
                  {option === "list" ? <LayoutList className="size-3" /> : <Table2 className="size-3" />}
                </button>
              ))}
            </div>
          </div>
        </div>
        <label className="relative block">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3 -translate-y-1/2 text-[var(--text-quaternary)]" />
          <input
            type="search"
            value={filters.searchQuery}
            onChange={(event) => setFilters({ ...filters, searchQuery: event.target.value })}
            placeholder="Search records…"
            aria-label="Search Reconstruction records"
            className="h-8 w-full rounded-md border border-[var(--border-default)] bg-[var(--surface-1)] pl-7 pr-2 text-[10px] text-[var(--text-primary)] shadow-[inset_0_1px_0_rgba(0,0,0,0.08)] outline-none placeholder:text-[var(--text-quaternary)] focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
          />
        </label>
        <ReconstructionRecordFilters filters={filters} options={options} onChange={setFilters} />
      </div>
      <div className="min-h-0 flex-1">
        {loading ? (
          <div className="flex h-40 flex-col items-center justify-center gap-2 text-[11px] text-[var(--text-quaternary)]">
            <span className="size-5 animate-pulse rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)]" />
            Loading reconstruction records…
          </div>
        ) : error ? (
          <div className="p-4 text-[11px] leading-5 text-[var(--destructive)]">{error}</div>
        ) : visibleRecords.length === 0 ? (
          <div className="flex h-48 flex-col items-center justify-center px-6 text-center">
            <span className="mb-3 grid size-9 place-items-center rounded-xl border border-[var(--border-default)] bg-[var(--surface-1)] text-[var(--accent)]">
              <ScanLine className="size-4" />
            </span>
            <p className="text-[11px] font-medium text-[var(--text-secondary)]">
              {records.length === 0 ? "No reconstructed assets yet" : "No records match"}
            </p>
            <p className="mt-1 max-w-52 text-[9.5px] leading-4 text-[var(--text-quaternary)]">
              {records.length === 0
                ? "Reconstruction records will appear here."
                : "Try clearing or widening the active filters."}
            </p>
          </div>
        ) : view === "table" ? (
          <ReconstructionRecordTable
            items={items}
            selectedRecordId={selectedRecordId}
            onSelect={onSelect}
            isGroupExpanded={groupExpanded}
            renderGroupHeader={renderGroupHeader}
          />
        ) : (
          <ScrollArea className="h-full">
            <div className="py-1">
              {items.map((item) => {
                if (item.kind === "record") {
                  return (
                    <RecordRow
                      key={item.record.record_id}
                      record={item.record}
                      selected={item.record.record_id === selectedRecordId}
                      onSelect={() => onSelect(item.record.record_id)}
                    />
                  );
                }
                return (
                  <div key={`batch:${item.batchId}`}>
                    {renderGroupHeader(item)}
                    {groupExpanded(item) ? (
                      <div className="border-l-2 border-l-[var(--border-default)]">
                        {item.records.map((record) => (
                          <RecordRow
                            key={record.record_id}
                            record={record}
                            selected={record.record_id === selectedRecordId}
                            onSelect={() => onSelect(record.record_id)}
                          />
                        ))}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}
