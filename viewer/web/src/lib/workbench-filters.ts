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

import type {
  InputKindFilter,
  RatingFilter,
  SourceFilter,
  WorkbenchEntry,
} from "@/lib/types";

export const UNKNOWN_MODEL_FILTER = "__unknown_model__";

export type WorkbenchFilters = {
  searchQuery: string;
  sourceFilter: SourceFilter;
  modelFilter: string | null;
  inputKindFilter: InputKindFilter;
  ratingFilter: RatingFilter;
};

function matchesSearch(entry: WorkbenchEntry, query: string): boolean {
  if (!entry.record) return false;
  if (!query) return true;
  const record = entry.record;
  return [
    entry.record_id,
    entry.label,
    ...entry.tags,
    record.title,
    record.prompt_preview,
    record.model_id,
  ].some((value) => value?.toLowerCase().includes(query));
}

function matchesModel(entry: WorkbenchEntry, modelFilter: string | null): boolean {
  if (!modelFilter) return true;
  if (modelFilter === UNKNOWN_MODEL_FILTER) return !entry.record?.model_id;
  return entry.record?.model_id === modelFilter;
}

function matchesRating(entry: WorkbenchEntry, ratingFilter: RatingFilter): boolean {
  if (ratingFilter == null) return true;
  if (ratingFilter === "unrated") return entry.record?.rating == null;
  return entry.record?.rating === ratingFilter;
}

export function filterWorkbenchEntries(
  entries: WorkbenchEntry[],
  filters: WorkbenchFilters,
): WorkbenchEntry[] {
  const query = filters.searchQuery.trim().toLowerCase();
  return entries.filter(
    (entry) =>
      !entry.archived &&
      entry.record != null &&
      matchesSearch(entry, query) &&
      (!filters.sourceFilter || entry.record.agent_harness === filters.sourceFilter) &&
      matchesModel(entry, filters.modelFilter) &&
      (!filters.inputKindFilter || entry.record.input_kind === filters.inputKindFilter) &&
      matchesRating(entry, filters.ratingFilter),
  );
}

export function reconcileWorkbenchSelection(
  entries: WorkbenchEntry[],
  filters: WorkbenchFilters,
  selectedRecordId: string | null,
): string | null {
  const visibleEntries = filterWorkbenchEntries(entries, filters);
  if (selectedRecordId && visibleEntries.some((entry) => entry.record_id === selectedRecordId)) {
    return selectedRecordId;
  }
  return visibleEntries[0]?.record_id ?? null;
}

export function availableWorkbenchModels(entries: WorkbenchEntry[]): {
  models: string[];
  hasUnknown: boolean;
} {
  const visibleRecords = entries
    .filter((entry) => !entry.archived)
    .map((entry) => entry.record)
    .filter((record) => record != null);
  return {
    models: Array.from(
      new Set(
        visibleRecords
          .map((record) => record.model_id)
          .filter((model): model is string => Boolean(model)),
      ),
    ).sort((left, right) => left.localeCompare(right)),
    hasUnknown: visibleRecords.some((record) => !record.model_id),
  };
}
