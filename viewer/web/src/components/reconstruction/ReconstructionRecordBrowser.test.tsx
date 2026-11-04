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
  reconstructionRecordListMetadata,
  reconstructionRecordQuality,
  replaceReconstructionRecord,
  withoutReconstructionRecord,
} from "./ReconstructionRecordBrowser";
import type { AxialDiagnostic, ReconstructionRecordSummary } from "@/lib/types";

function record(
  overrides: Partial<ReconstructionRecordSummary> = {},
): ReconstructionRecordSummary {
  return {
    record_id: "rec_reconstruction_001",
    pipeline: "reconstruction",
    status: "reconstruction_ready",
    input_kind: "mesh_image",
    title: "Drawer",
    prompt_preview: "",
    rating: null,
    created_at: "2026-08-21T08:00:00Z",
    updated_at: null,
    viewer_asset_updated_at: "2026-08-21T08:00:00Z",
    provider: "codex-cli",
    model_id: "test-model",
    agent_harness: "codex",
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

describe("Reconstruction record list presentation", () => {
  it("keeps only authoring details in the compact metadata line", () => {
    expect(reconstructionRecordListMetadata(record())).toEqual([
      "Codex",
      "test-model",
      "high reasoning",
      expect.any(String),
    ]);
  });

  it("labels a Provider-arm record as Provider", () => {
    expect(
      reconstructionRecordListMetadata(record({ agent_harness: "provider", provider: "nvidia" }))[0],
    ).toBe("Provider");
  });

  it("presents recall around 60% and exterior excess above the 2% threshold distinctly", () => {
    const diagnostic: AxialDiagnostic = {
      calls: 1,
      revision: 1,
      first_recall: 0.59,
      recall: 0.59,
      input_cells: null,
      matched_cells: null,
      missing_cells: null,
      candidate_only_cells: null,
      exterior_excess_cells: 1,
      exterior_excess_fraction: 0.001,
      surface_fit_p50_mm: null,
      surface_fit_p90_mm: null,
      surface_faces_off_plane: null,
      candidate_openings: null,
      unmodelled_source_openings: null,
    };
    const metrics = reconstructionRecordQuality(record({ axial_diagnostic: diagnostic }));
    expect(metrics).toEqual([
      {
        key: "recall",
        label: "AXIAL recall",
        value: "59%",
        detail: "Below 60%",
        tone: "warn",
      },
      {
        key: "excess",
        label: "Exterior excess",
        value: "0.1%",
        detail: "No excess",
        tone: "pass",
      },
    ]);
    expect(
      reconstructionRecordQuality(
        record({ axial_diagnostic: { ...diagnostic, recall: 0.6 } }),
      )[0],
    ).toMatchObject({ value: "60%", detail: "60%+", tone: "pass" });
    expect(
      reconstructionRecordQuality(
        record({
          axial_diagnostic: { ...diagnostic, exterior_excess_fraction: 0.05 },
        }),
      )[1],
    ).toMatchObject({ value: "5.0%", detail: "Outside source", tone: "warn" });
  });
});

describe("Reconstruction record list mutations", () => {
  it("replaces a rated record in the cached list and appends an unknown one", () => {
    const first = record({ record_id: "rec_a" });
    const second = record({ record_id: "rec_b" });
    const rated = { ...second, rating: 4 };
    expect(replaceReconstructionRecord([first, second], rated)).toEqual([first, rated]);
    expect(replaceReconstructionRecord([first], rated)).toEqual([first, rated]);
    expect(replaceReconstructionRecord(undefined, rated)).toEqual([rated]);
  });

  it("drops a deleted record from the cached list", () => {
    const first = record({ record_id: "rec_a" });
    const second = record({ record_id: "rec_b" });
    expect(withoutReconstructionRecord([first, second], "rec_a")).toEqual([second]);
    expect(withoutReconstructionRecord(undefined, "rec_a")).toEqual([]);
  });
});
