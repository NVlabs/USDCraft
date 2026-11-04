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

import { describe, expect, it } from "vitest";

import type { RecordSummary, WorkbenchEntry } from "@/lib/types";

import {
  UNKNOWN_MODEL_FILTER,
  availableWorkbenchModels,
  filterWorkbenchEntries,
  reconcileWorkbenchSelection,
} from "./workbench-filters";

function summary(
  recordId: string,
  overrides: Partial<RecordSummary> = {},
): RecordSummary {
  return {
    record_id: recordId,
    pipeline: "generation",
    status: "generation_ready",
    input_kind: "text",
    rating: null,
    title: recordId,
    prompt_preview: `Prompt for ${recordId}`,
    created_at: null,
    updated_at: null,
    viewer_asset_updated_at: null,
    provider: "codex-cli",
    model_id: "gpt-a",
    agent_harness: "codex",
    has_traces: false,
    thinking_level: "high",
    turn_count: null,
    input_tokens: null,
    output_tokens: null,
    total_cost_usd: null,
    active_revision_id: "rev_000001",
    revision_count: 1,
    materialization_status: "available",
    materialization_displayable: true,
    materialization_validated: true,
    materialization_issues: [],
    has_compile_report: true,
    has_provenance: true,
    has_cost: false,
    ...overrides,
  };
}

function entry(record: RecordSummary, archived = false): WorkbenchEntry {
  return {
    record_id: record.record_id,
    added_at: "2026-07-13T00:00:00Z",
    label: null,
    tags: [],
    archived,
    record,
  };
}

const ENTRIES = [
  entry(summary("rec_text", { title: "Desk lamp", rating: 5 })),
  entry(
    summary("rec_image", {
      agent_harness: "provider",
      provider: "nvidia",
      input_kind: "text_image",
      model_id: "gpt-b",
      rating: 4,
      title: "Fire extinguisher",
    }),
  ),
  entry(
    summary("rec_unknown", {
      input_kind: "text_image",
      model_id: null,
      rating: null,
      title: "Unknown model asset",
    }),
  ),
  entry(summary("rec_archived", { model_id: "gpt-c", rating: 1 }), true),
];

describe("workbench filters", () => {
  it("combines search, model, input kind, and exact rating", () => {
    expect(
      filterWorkbenchEntries(ENTRIES, {
        searchQuery: "extinguisher",
        sourceFilter: "provider",
        modelFilter: "gpt-b",
        inputKindFilter: "text_image",
        ratingFilter: 4,
      }).map((item) => item.record_id),
    ).toEqual(["rec_image"]);
  });

  it("supports unknown-model and unrated records", () => {
    expect(
      filterWorkbenchEntries(ENTRIES, {
        searchQuery: "",
        sourceFilter: null,
        modelFilter: UNKNOWN_MODEL_FILTER,
        inputKindFilter: null,
        ratingFilter: "unrated",
      }).map((item) => item.record_id),
    ).toEqual(["rec_unknown"]);
  });

  it("derives stable model options without archived records", () => {
    expect(availableWorkbenchModels(ENTRIES)).toEqual({
      models: ["gpt-a", "gpt-b"],
      hasUnknown: true,
    });
  });

  it("keeps a visible selection and otherwise selects the first combined-filter result", () => {
    const filters = {
      searchQuery: "",
      sourceFilter: "provider" as const,
      modelFilter: "gpt-b",
      inputKindFilter: "text_image" as const,
      ratingFilter: 4 as const,
    };

    expect(reconcileWorkbenchSelection(ENTRIES, filters, "rec_image")).toBe("rec_image");
    expect(reconcileWorkbenchSelection(ENTRIES, filters, "rec_text")).toBe("rec_image");
  });

  it("clears selection when no entry satisfies the combined filters", () => {
    expect(
      reconcileWorkbenchSelection(
        ENTRIES,
        {
          searchQuery: "does-not-exist",
          sourceFilter: null,
          modelFilter: null,
          inputKindFilter: null,
          ratingFilter: null,
        },
        "rec_text",
      ),
    ).toBeNull();
  });

  it("filters Codex and Provider records by authoring source", () => {
    expect(
      filterWorkbenchEntries(ENTRIES, {
        searchQuery: "",
        sourceFilter: "codex",
        modelFilter: null,
        inputKindFilter: null,
        ratingFilter: null,
      }).map((item) => item.record_id),
    ).toEqual(["rec_text", "rec_unknown"]);

    expect(
      filterWorkbenchEntries(ENTRIES, {
        searchQuery: "",
        sourceFilter: "provider",
        modelFilter: null,
        inputKindFilter: null,
        ratingFilter: null,
      }).map((item) => item.record_id),
    ).toEqual(["rec_image"]);
  });
});
