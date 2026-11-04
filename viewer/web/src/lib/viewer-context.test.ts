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

import type { RecordSummary, ViewerState, WorkbenchEntry } from "@/lib/types";

import { needsSelectedRecordSummary, viewerReducer } from "./viewer-context";

function summary(recordId: string, rating: number | null = null): RecordSummary {
  return {
    record_id: recordId,
    pipeline: "generation",
    status: "generation_ready",
    input_kind: "text",
    rating,
    title: recordId,
    prompt_preview: recordId,
    created_at: null,
    updated_at: null,
    viewer_asset_updated_at: null,
    provider: "codex-cli",
    model_id: "gpt-test",
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
  };
}

function entry(record: RecordSummary): WorkbenchEntry {
  return {
    record_id: record.record_id,
    added_at: "2026-07-13T00:00:00Z",
    label: null,
    tags: [],
    archived: false,
    record,
  };
}

function state(records: RecordSummary[], selectedRecordId: string): ViewerState {
  const recordCache = Object.fromEntries(records.map((record) => [record.record_id, record]));
  return {
    bootstrap: {
      repo_root: "/repo",
      generated_at: "2026-07-13T00:00:00Z",
      workbench_entries: records.map(entry),
    },
    recordCache,
    selectedRecordId,
    selectedRecordSummary: recordCache[selectedRecordId] ?? null,
    selectedInspectorTab: "inspect",
    inspectorOpen: true,
    loading: false,
    error: null,
    searchQuery: "",
    sourceFilter: null,
    modelFilter: null,
    inputKindFilter: null,
    ratingFilter: null,
    workbenchNotice: null,
  };
}

describe("viewer reducer record curation", () => {
  it("projects an updated rating into bootstrap, cache, and selection", () => {
    const original = summary("rec_one");
    const updated = { ...original, rating: 4, updated_at: "2026-07-13T01:00:00Z" };

    const next = viewerReducer(state([original], original.record_id), {
      type: "UPSERT_RECORDS",
      payload: [updated],
    });

    expect(next.bootstrap?.workbench_entries[0]?.record?.rating).toBe(4);
    expect(next.recordCache[original.record_id]?.rating).toBe(4);
    expect(next.selectedRecordSummary?.rating).toBe(4);
  });

  it("removes a deleted record and selects the requested next record", () => {
    const first = summary("rec_first");
    const second = summary("rec_second");
    const third = summary("rec_third");

    const next = viewerReducer(state([first, second, third], first.record_id), {
      type: "DELETE_RECORD_LOCAL",
      payload: { recordId: first.record_id, nextRecordId: second.record_id },
    });

    expect(next.bootstrap?.workbench_entries.map((item) => item.record_id)).toEqual([
      "rec_second",
      "rec_third",
    ]);
    expect(next.recordCache.rec_first).toBeUndefined();
    expect(next.selectedRecordId).toBe("rec_second");
    expect(next.selectedRecordSummary?.record_id).toBe("rec_second");
  });

  it("does not fall back to a hidden record when deleting the final visible result", () => {
    const visible = summary("rec_visible");
    const hidden = summary("rec_hidden");

    const next = viewerReducer(state([visible, hidden], visible.record_id), {
      type: "DELETE_RECORD_LOCAL",
      payload: { recordId: visible.record_id, nextRecordId: null },
    });

    expect(next.selectedRecordId).toBeNull();
    expect(next.selectedRecordSummary).toBeNull();
    expect(next.bootstrap?.workbench_entries.map((item) => item.record_id)).toEqual([
      "rec_hidden",
    ]);
  });

  it("keeps the current selection when a different record is deleted", () => {
    const first = summary("rec_first");
    const second = summary("rec_second");

    const next = viewerReducer(state([first, second], first.record_id), {
      type: "DELETE_RECORD_LOCAL",
      payload: { recordId: second.record_id, nextRecordId: null },
    });

    expect(next.selectedRecordId).toBe("rec_first");
    expect(next.selectedRecordSummary?.record_id).toBe("rec_first");
  });

  it("ignores a stale summary response for a previously selected record", () => {
    const first = summary("rec_first");
    const second = summary("rec_second");
    const current = state([first, second], second.record_id);

    const next = viewerReducer(current, {
      type: "SET_SELECTED_RECORD_SUMMARY",
      payload: { ...first, rating: 2 },
    });

    expect(next).toBe(current);
    expect(next.selectedRecordSummary?.record_id).toBe("rec_second");
  });

  it("stores and dismisses a persistent Workbench notice", () => {
    const record = summary("rec_one");
    const original = state([record], record.record_id);
    const warned = viewerReducer(original, {
      type: "SET_WORKBENCH_NOTICE",
      payload: "Cleanup remains pending.",
    });

    expect(warned.workbenchNotice).toBe("Cleanup remains pending.");
    expect(
      viewerReducer(warned, { type: "SET_WORKBENCH_NOTICE", payload: null })
        .workbenchNotice,
    ).toBeNull();
  });

  it("stores the local authoring-source filter", () => {
    const record = summary("rec_one");
    const original = state([record], record.record_id);

    const next = viewerReducer(original, {
      type: "SET_SOURCE_FILTER",
      payload: "provider",
    });

    expect(next.sourceFilter).toBe("provider");
  });
});

describe("needsSelectedRecordSummary", () => {
  it("fetches the full summary for an uncached or lazily listed selection only", () => {
    const validated = summary("rec_validated");
    const lazy = { ...summary("rec_lazy"), materialization_status: "not_validated" as const };
    const recordCache = { rec_validated: validated, rec_lazy: lazy };

    expect(needsSelectedRecordSummary({ selectedRecordId: null, recordCache })).toBe(false);
    expect(needsSelectedRecordSummary({ selectedRecordId: "rec_missing", recordCache })).toBe(true);
    expect(needsSelectedRecordSummary({ selectedRecordId: "rec_lazy", recordCache })).toBe(true);
    expect(needsSelectedRecordSummary({ selectedRecordId: "rec_validated", recordCache })).toBe(false);
  });
});
