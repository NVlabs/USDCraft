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

import type { ReconstructionRecordSummary } from "@/lib/types";

export const UNKNOWN_VALUE_FILTER = "__unknown__";

export type ReconstructionSourceFilter = "codex" | "provider" | null;
export type ReconstructionRatingFilter = 1 | 2 | 3 | 4 | 5 | "unrated" | null;
/** AXIAL recall bands of the last `check_axial` call; `none` = no diagnostic on the record. */
export type ReconstructionRecallFilter = "high" | "mid" | "low" | "none" | null;
export type ReconstructionSort = "updated" | "recall_desc" | "recall_asc" | "rating_desc";

export type ReconstructionFilters = {
  searchQuery: string;
  sourceFilter: ReconstructionSourceFilter;
  modelFilter: string | null;
  scaleSourceFilter: string | null;
  batchFilter: string | null;
  ratingFilter: ReconstructionRatingFilter;
  recallFilter: ReconstructionRecallFilter;
  sort: ReconstructionSort;
};

export const DEFAULT_RECONSTRUCTION_FILTERS: ReconstructionFilters = {
  searchQuery: "",
  sourceFilter: null,
  modelFilter: null,
  scaleSourceFilter: null,
  batchFilter: null,
  ratingFilter: null,
  recallFilter: null,
  sort: "updated",
};

export const RECALL_HIGH_MIN = 0.9;
export const RECALL_MID_MIN = 0.7;

export function hasActiveReconstructionFilter(filters: ReconstructionFilters): boolean {
  return (
    filters.sourceFilter !== null ||
    filters.modelFilter !== null ||
    filters.scaleSourceFilter !== null ||
    filters.batchFilter !== null ||
    filters.ratingFilter !== null ||
    filters.recallFilter !== null
  );
}

function matchesValue(value: string | null | undefined, filter: string | null): boolean {
  if (!filter) return true;
  if (filter === UNKNOWN_VALUE_FILTER) return !value;
  return value === filter;
}

function matchesSearch(record: ReconstructionRecordSummary, query: string): boolean {
  if (!query) return true;
  return [record.record_id, record.title, record.prompt_preview, record.model_id]
    .filter((value): value is string => Boolean(value))
    .some((value) => value.toLowerCase().includes(query));
}

function matchesRating(
  record: ReconstructionRecordSummary,
  filter: ReconstructionRatingFilter,
): boolean {
  if (filter == null) return true;
  if (filter === "unrated") return record.rating == null;
  return record.rating === filter;
}

export function recallOf(record: ReconstructionRecordSummary): number | null {
  const value = record.axial_diagnostic?.recall;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function recallBand(record: ReconstructionRecordSummary): Exclude<
  ReconstructionRecallFilter,
  null
> {
  const recall = recallOf(record);
  if (recall == null) return "none";
  if (recall >= RECALL_HIGH_MIN) return "high";
  if (recall >= RECALL_MID_MIN) return "mid";
  return "low";
}

function timestamp(record: ReconstructionRecordSummary): number {
  const value = record.updated_at ?? record.created_at;
  const parsed = value ? Date.parse(value) : Number.NaN;
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function sortReconstructionRecords(
  records: readonly ReconstructionRecordSummary[],
  sort: ReconstructionSort,
): ReconstructionRecordSummary[] {
  const sorted = [...records];
  const byUpdated = (a: ReconstructionRecordSummary, b: ReconstructionRecordSummary): number =>
    timestamp(b) - timestamp(a);
  switch (sort) {
    case "recall_desc":
      sorted.sort((a, b) => (recallOf(b) ?? -1) - (recallOf(a) ?? -1) || byUpdated(a, b));
      break;
    case "recall_asc":
      sorted.sort(
        (a, b) =>
          (recallOf(a) ?? Number.POSITIVE_INFINITY) - (recallOf(b) ?? Number.POSITIVE_INFINITY) ||
          byUpdated(a, b),
      );
      break;
    case "rating_desc":
      sorted.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0) || byUpdated(a, b));
      break;
    default:
      sorted.sort(byUpdated);
  }
  return sorted;
}

export function filterReconstructionRecords(
  records: readonly ReconstructionRecordSummary[],
  filters: ReconstructionFilters,
): ReconstructionRecordSummary[] {
  const query = filters.searchQuery.trim().toLowerCase();
  const visible = records.filter(
    (record) =>
      matchesSearch(record, query) &&
      (!filters.sourceFilter || record.agent_harness === filters.sourceFilter) &&
      matchesValue(record.model_id, filters.modelFilter) &&
      matchesValue(record.scale_source, filters.scaleSourceFilter) &&
      matchesValue(record.batch_id, filters.batchFilter) &&
      matchesRating(record, filters.ratingFilter) &&
      (!filters.recallFilter || recallBand(record) === filters.recallFilter),
  );
  return sortReconstructionRecords(visible, filters.sort);
}

export type ReconstructionFilterOptions = {
  models: string[];
  hasUnknownModel: boolean;
  scaleSources: string[];
  batches: string[];
};

function distinct(values: (string | null | undefined)[]): string[] {
  return Array.from(new Set(values.filter((value): value is string => Boolean(value)))).sort(
    (left, right) => left.localeCompare(right),
  );
}

export function reconstructionFilterOptions(
  records: readonly ReconstructionRecordSummary[],
): ReconstructionFilterOptions {
  return {
    models: distinct(records.map((record) => record.model_id)),
    hasUnknownModel: records.some((record) => !record.model_id),
    scaleSources: distinct(records.map((record) => record.scale_source)),
    // Newest batch first: batch ids end with their UTC start stamp.
    batches: distinct(records.map((record) => record.batch_id)).sort((a, b) =>
      b.localeCompare(a),
    ),
  };
}

export function scaleSourceLabel(source: string | null | undefined): string | null {
  switch (source) {
    case "estimated_from_semantic_image":
      return "scale: estimated";
    case "declared":
    case "explicit":
      return "scale: declared";
    case "embedded_metadata":
      return "scale: embedded";
    case "format_contract":
      return "scale: assumed metres";
    default:
      return null;
  }
}
