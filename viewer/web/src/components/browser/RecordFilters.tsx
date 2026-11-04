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

import { UNKNOWN_MODEL_FILTER } from "@/lib/workbench-filters";
import { useViewer, useViewerDispatch } from "@/lib/viewer-context";
import type { InputKindFilter, RatingFilter, SourceFilter } from "@/lib/types";

type RecordFiltersProps = {
  models: string[];
  hasUnknownModel: boolean;
};

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

export function RecordFilters({ models, hasUnknownModel }: RecordFiltersProps): JSX.Element {
  const { modelFilter, inputKindFilter, ratingFilter, sourceFilter } = useViewer();
  const dispatch = useViewerDispatch();
  const [expanded, setExpanded] = useState(false);
  const activeFilterCount = [sourceFilter, modelFilter, inputKindFilter, ratingFilter].filter(
    (value) => value !== null,
  ).length;
  const hasActiveFilter =
    sourceFilter !== null ||
    modelFilter !== null ||
    inputKindFilter !== null ||
    ratingFilter !== null;

  return (
    <div className="mt-2 border-t border-[var(--border-subtle)] pt-1.5">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          className="flex h-7 min-w-0 flex-1 items-center gap-1.5 rounded-md px-1.5 text-[9.5px] font-medium text-[var(--text-tertiary)] outline-none transition hover:bg-[var(--surface-1)] hover:text-[var(--text-secondary)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]"
          aria-expanded={expanded}
          aria-controls="generation-record-filters"
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
        {hasActiveFilter ? (
          <button
            type="button"
            onClick={() => {
              dispatch({ type: "SET_SOURCE_FILTER", payload: null });
              dispatch({ type: "SET_MODEL_FILTER", payload: null });
              dispatch({ type: "SET_INPUT_KIND_FILTER", payload: null });
              dispatch({ type: "SET_RATING_FILTER", payload: null });
            }}
            className="flex items-center gap-1 rounded px-1 py-0.5 text-[9.5px] text-[var(--text-tertiary)] hover:bg-[var(--surface-2)] hover:text-[var(--text-secondary)]"
          >
            <RotateCcw className="size-2.5" />
            Clear
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div id="generation-record-filters" className="mt-1.5 grid grid-cols-2 gap-1.5">
        <div className="col-span-2">
          <FilterLabel htmlFor="workbench-source-filter">Source</FilterLabel>
          <select
            id="workbench-source-filter"
            className={SELECT_CLASS}
            value={sourceFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const nextSource: SourceFilter =
                value === "codex" || value === "provider" ? value : null;
              dispatch({ type: "SET_SOURCE_FILTER", payload: nextSource });
            }}
          >
            <option value="all">All sources</option>
            <option value="codex">Codex</option>
            <option value="provider">Provider</option>
          </select>
        </div>

        <div className="col-span-2">
          <FilterLabel htmlFor="workbench-model-filter">Model</FilterLabel>
          <select
            id="workbench-model-filter"
            className={SELECT_CLASS}
            value={modelFilter ?? "all"}
            onChange={(event) =>
              dispatch({
                type: "SET_MODEL_FILTER",
                payload: event.target.value === "all" ? null : event.target.value,
              })
            }
          >
            <option value="all">All models</option>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
            {hasUnknownModel ? <option value={UNKNOWN_MODEL_FILTER}>Unknown model</option> : null}
          </select>
        </div>

        <div>
          <FilterLabel htmlFor="workbench-input-filter">Input</FilterLabel>
          <select
            id="workbench-input-filter"
            className={SELECT_CLASS}
            value={inputKindFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const inputFilter: InputKindFilter =
                value === "text" || value === "text_image" ? value : null;
              dispatch({ type: "SET_INPUT_KIND_FILTER", payload: inputFilter });
            }}
          >
            <option value="all">Any input</option>
            <option value="text">Text only</option>
            <option value="text_image">Text + image</option>
          </select>
        </div>

        <div>
          <FilterLabel htmlFor="workbench-rating-filter">Rating</FilterLabel>
          <select
            id="workbench-rating-filter"
            className={SELECT_CLASS}
            value={ratingFilter ?? "all"}
            onChange={(event) => {
              const value = event.target.value;
              const parsed = Number(value);
              const nextRating: RatingFilter =
                value === "unrated"
                  ? "unrated"
                  : Number.isInteger(parsed) && parsed >= 1 && parsed <= 5
                    ? (parsed as 1 | 2 | 3 | 4 | 5)
                    : null;
              dispatch({ type: "SET_RATING_FILTER", payload: nextRating });
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
        </div>
      ) : null}
    </div>
  );
}
