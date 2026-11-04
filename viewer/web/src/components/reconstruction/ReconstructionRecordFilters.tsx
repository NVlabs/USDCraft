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

import { ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { useState, type JSX } from "react";

import {
  DEFAULT_RECONSTRUCTION_FILTERS,
  UNKNOWN_VALUE_FILTER,
  hasActiveReconstructionFilter,
  type ReconstructionFilterOptions,
  type ReconstructionFilters,
  type ReconstructionRatingFilter,
  type ReconstructionRecallFilter,
  type ReconstructionSort,
  type ReconstructionSourceFilter,
} from "@/lib/reconstruction-filters";

const SELECT_CLASS =
  "h-7 w-full rounded-md border border-[var(--border-default)] bg-[var(--surface-1)] px-2 text-[9.5px] text-[var(--text-secondary)] outline-none transition focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]";

function FilterLabel({ htmlFor, children }: { htmlFor: string; children: string }): JSX.Element {
  return (
    <label
      htmlFor={htmlFor}
      className="mb-1 block font-mono text-[8.5px] font-medium uppercase tracking-[0.06em] text-[var(--text-tertiary)]"
    >
      {children}
    </label>
  );
}

function ValueSelect({
  id,
  label,
  value,
  values,
  allLabel,
  hasUnknown = false,
  onChange,
}: {
  id: string;
  label: string;
  value: string | null;
  values: string[];
  allLabel: string;
  hasUnknown?: boolean;
  onChange: (value: string | null) => void;
}): JSX.Element {
  return (
    <div>
      <FilterLabel htmlFor={id}>{label}</FilterLabel>
      <select
        id={id}
        className={SELECT_CLASS}
        value={value ?? "all"}
        onChange={(event) => onChange(event.target.value === "all" ? null : event.target.value)}
      >
        <option value="all">{allLabel}</option>
        {values.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
        {hasUnknown ? <option value={UNKNOWN_VALUE_FILTER}>Unknown</option> : null}
      </select>
    </div>
  );
}

export function ReconstructionRecordFilters({
  filters,
  options,
  onChange,
}: {
  filters: ReconstructionFilters;
  options: ReconstructionFilterOptions;
  onChange: (filters: ReconstructionFilters) => void;
}): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const update = (patch: Partial<ReconstructionFilters>): void =>
    onChange({ ...filters, ...patch });
  const activeFilterCount = [
    filters.sourceFilter,
    filters.modelFilter,
    filters.scaleSourceFilter,
    filters.batchFilter,
    filters.ratingFilter,
    filters.recallFilter,
  ].filter((value) => value !== null).length;
  const hasActiveFilters = hasActiveReconstructionFilter(filters);

  return (
    <div className="mt-2 border-t border-[var(--border-subtle)] pt-1.5">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          className="flex h-7 min-w-0 flex-1 items-center gap-1.5 rounded-md px-1.5 text-[9.5px] font-medium text-[var(--text-tertiary)] outline-none transition hover:bg-[var(--surface-1)] hover:text-[var(--text-secondary)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]"
          aria-expanded={expanded}
          aria-controls="reconstruction-record-filters"
        >
          <SlidersHorizontal className="size-3 text-[var(--accent)]" />
          <span>Filters</span>
          {activeFilterCount > 0 ? (
            <span className="rounded-full bg-[var(--accent-soft)] px-1.5 font-mono text-[8px] text-[var(--accent-strong)]">
              {activeFilterCount} active
            </span>
          ) : null}
          <ChevronDown
            className={`ml-auto size-3 transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </button>
        {hasActiveFilters ? (
          <button
            type="button"
            onClick={() =>
              onChange({
                ...DEFAULT_RECONSTRUCTION_FILTERS,
                searchQuery: filters.searchQuery,
                sort: filters.sort,
              })
            }
            className="flex items-center gap-1 rounded px-1 py-0.5 text-[9.5px] text-[var(--text-tertiary)] hover:bg-[var(--surface-2)] hover:text-[var(--text-secondary)]"
          >
            <RotateCcw className="size-2.5" />
            Clear
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div id="reconstruction-record-filters" className="mt-1.5 grid grid-cols-2 gap-1.5">
        <div>
          <FilterLabel htmlFor="reconstruction-source-filter">Source</FilterLabel>
          <select
            id="reconstruction-source-filter"
            className={SELECT_CLASS}
            value={filters.sourceFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const next: ReconstructionSourceFilter =
                value === "codex" || value === "provider" ? value : null;
              update({ sourceFilter: next });
            }}
          >
            <option value="all">All sources</option>
            <option value="codex">Codex</option>
            <option value="provider">Provider</option>
          </select>
        </div>
        <ValueSelect
          id="reconstruction-model-filter"
          label="Model"
          value={filters.modelFilter}
          values={options.models}
          allLabel="All models"
          hasUnknown={options.hasUnknownModel}
          onChange={(value) => update({ modelFilter: value })}
        />
        <ValueSelect
          id="reconstruction-scale-filter"
          label="Scale"
          value={filters.scaleSourceFilter}
          values={options.scaleSources}
          allLabel="Any scale"
          onChange={(value) => update({ scaleSourceFilter: value })}
        />
        {options.batches.length > 0 ? (
          <ValueSelect
            id="reconstruction-batch-filter"
            label="Batch"
            value={filters.batchFilter}
            values={options.batches}
            allLabel="Any batch"
            hasUnknown
            onChange={(value) => update({ batchFilter: value })}
          />
        ) : null}
        <div>
          <FilterLabel htmlFor="reconstruction-rating-filter">Rating</FilterLabel>
          <select
            id="reconstruction-rating-filter"
            className={SELECT_CLASS}
            value={filters.ratingFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const parsed = Number(value);
              const next: ReconstructionRatingFilter =
                value === "unrated"
                  ? "unrated"
                  : Number.isInteger(parsed) && parsed >= 1 && parsed <= 5
                    ? (parsed as 1 | 2 | 3 | 4 | 5)
                    : null;
              update({ ratingFilter: next });
            }}
          >
            <option value="all">Any rating</option>
            <option value="unrated">Unrated</option>
            {[1, 2, 3, 4, 5].map((rating) => (
              <option key={rating} value={rating}>
                {rating} star{rating === 1 ? "" : "s"}
              </option>
            ))}
          </select>
        </div>
        <div>
          <FilterLabel htmlFor="reconstruction-recall-filter">AXIAL recall</FilterLabel>
          <select
            id="reconstruction-recall-filter"
            className={SELECT_CLASS}
            value={filters.recallFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const next: ReconstructionRecallFilter =
                value === "high" || value === "mid" || value === "low" || value === "none"
                  ? value
                  : null;
              update({ recallFilter: next });
            }}
          >
            <option value="all">Any recall</option>
            <option value="high">≥ 0.90</option>
            <option value="mid">0.70 – 0.90</option>
            <option value="low">&lt; 0.70</option>
            <option value="none">No diagnostic</option>
          </select>
        </div>
        <div>
          <FilterLabel htmlFor="reconstruction-sort">Sort</FilterLabel>
          <select
            id="reconstruction-sort"
            className={SELECT_CLASS}
            value={filters.sort}
            onChange={(event) => {
              const value = event.target.value as ReconstructionSort;
              update({
                sort:
                  value === "recall_desc" ||
                  value === "recall_asc" ||
                  value === "rating_desc"
                    ? value
                    : "updated",
              });
            }}
          >
            <option value="updated">Newest first</option>
            <option value="recall_desc">Recall high → low</option>
            <option value="recall_asc">Recall low → high</option>
            <option value="rating_desc">Rating high → low</option>
          </select>
        </div>
        </div>
      ) : null}
    </div>
  );
}
