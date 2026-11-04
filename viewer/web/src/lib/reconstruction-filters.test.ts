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

import {
  DEFAULT_RECONSTRUCTION_FILTERS,
  UNKNOWN_VALUE_FILTER,
  filterReconstructionRecords,
  recallBand,
  reconstructionFilterOptions,
  scaleSourceLabel,
  sortReconstructionRecords,
} from "./reconstruction-filters";
import type { AxialDiagnostic, ReconstructionRecordSummary } from "./types";

function diagnostic(recall: number | null): AxialDiagnostic {
  return {
    calls: 1,
    revision: 1,
    first_recall: recall,
    recall,
    input_cells: null,
    matched_cells: null,
    missing_cells: null,
    candidate_only_cells: null,
    exterior_excess_cells: null,
    exterior_excess_fraction: 0.01,
    surface_fit_p50_mm: null,
    surface_fit_p90_mm: null,
    surface_faces_off_plane: null,
    candidate_openings: null,
    unmodelled_source_openings: null,
  };
}

function record(overrides: Partial<ReconstructionRecordSummary>): ReconstructionRecordSummary {
  return {
    record_id: "rec",
    pipeline: "reconstruction",
    status: "reconstruction_ready",
    input_kind: "mesh_image",
    title: "Toaster",
    prompt_preview: "",
    rating: null,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    viewer_asset_updated_at: null,
    provider: "nvidia",
    model_id: "m",
    agent_harness: "provider",
    reasoning_effort: "high",
    thinking_level: "high",
    turn_count: null,
    input_tokens: null,
    output_tokens: null,
    total_cost_usd: null,
    active_revision_id: "rev_000001",
    revision_count: 1,
    input_count: 4,
    source_available: true,
    candidate_available: true,
    materialization_status: "available",
    materialization_displayable: true,
    materialization_validated: true,
    materialization_issues: [],
    has_compile_report: true,
    has_provenance: true,
    has_cost: false,
    has_traces: true,
    ...overrides,
  };
}

const HIGH = record({
  record_id: "high",
  rating: 5,
  scale_source: "estimated_from_semantic_image",
  axial_diagnostic: diagnostic(0.95),
  updated_at: "2026-09-03T00:00:00Z",
});
const MID = record({
  record_id: "mid",
  rating: 3,
  scale_source: "declared",
  axial_diagnostic: diagnostic(0.8),
  updated_at: "2026-09-02T00:00:00Z",
});
const NONE = record({
  record_id: "none",
  agent_harness: "codex",
  model_id: null,
  axial_diagnostic: null,
  updated_at: "2026-09-04T00:00:00Z",
});

describe("reconstruction filters", () => {
  it("bands recall and treats a missing diagnostic as none", () => {
    expect(recallBand(HIGH)).toBe("high");
    expect(recallBand(MID)).toBe("mid");
    expect(recallBand(record({ axial_diagnostic: diagnostic(0.4) }))).toBe("low");
    expect(recallBand(NONE)).toBe("none");
  });

  it("filters by every facet and by the unknown-model sentinel", () => {
    const all = [HIGH, MID, NONE];
    const ids = (records: ReconstructionRecordSummary[]) => records.map((r) => r.record_id);
    expect(ids(filterReconstructionRecords(all, DEFAULT_RECONSTRUCTION_FILTERS))).toEqual([
      "none",
      "high",
      "mid",
    ]);
    expect(
      ids(
        filterReconstructionRecords(all, {
          ...DEFAULT_RECONSTRUCTION_FILTERS,
          scaleSourceFilter: "estimated_from_semantic_image",
        }),
      ),
    ).toEqual(["high"]);
    expect(
      ids(filterReconstructionRecords(all, { ...DEFAULT_RECONSTRUCTION_FILTERS, recallFilter: "none" })),
    ).toEqual(["none"]);
    expect(
      ids(
        filterReconstructionRecords(all, {
          ...DEFAULT_RECONSTRUCTION_FILTERS,
          modelFilter: UNKNOWN_VALUE_FILTER,
        }),
      ),
    ).toEqual(["none"]);
    expect(
      ids(filterReconstructionRecords(all, { ...DEFAULT_RECONSTRUCTION_FILTERS, ratingFilter: "unrated" })),
    ).toEqual(["none"]);
    expect(
      ids(filterReconstructionRecords(all, { ...DEFAULT_RECONSTRUCTION_FILTERS, sourceFilter: "codex" })),
    ).toEqual(["none"]);
  });

  it("sorts by recall with undiagnosed records last, and by rating", () => {
    const all = [MID, NONE, HIGH];
    expect(sortReconstructionRecords(all, "recall_desc").map((r) => r.record_id)).toEqual([
      "high",
      "mid",
      "none",
    ]);
    expect(sortReconstructionRecords(all, "recall_asc").map((r) => r.record_id)).toEqual([
      "mid",
      "high",
      "none",
    ]);
    expect(sortReconstructionRecords(all, "rating_desc").map((r) => r.record_id)).toEqual([
      "high",
      "mid",
      "none",
    ]);
  });

  it("collects filter options and labels scale provenance", () => {
    const options = reconstructionFilterOptions([HIGH, MID, NONE]);
    expect(options.models).toEqual(["m"]);
    expect(options.hasUnknownModel).toBe(true);
    expect(options.scaleSources).toEqual(["declared", "estimated_from_semantic_image"]);
    expect(options.batches).toEqual([]);
    expect(
      reconstructionFilterOptions([
        record({ batch_id: "round1_20260905T091800Z" }),
        record({ batch_id: "round2_20260906T010000Z" }),
      ]).batches,
    ).toEqual(["round2_20260906T010000Z", "round1_20260905T091800Z"]);
    expect(scaleSourceLabel("estimated_from_semantic_image")).toBe("scale: estimated");
    expect(scaleSourceLabel("format_contract")).toBe("scale: assumed metres");
    expect(scaleSourceLabel(null)).toBeNull();
  });
});
