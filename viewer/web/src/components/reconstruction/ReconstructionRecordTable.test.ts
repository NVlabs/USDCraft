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

import { RECONSTRUCTION_TABLE_COLUMNS, reconstructionTableCells } from "./ReconstructionRecordTable";
import type { ReconstructionRecordSummary } from "@/lib/types";

const base = {
  record_id: "rec",
  pipeline: "reconstruction",
  status: "reconstruction_ready",
  input_kind: "mesh_image",
  title: "Toaster",
  prompt_preview: "",
  rating: 4,
  created_at: "2026-09-05T10:00:00Z",
  updated_at: "2026-09-05T10:00:00Z",
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
  scale_source: "format_contract",
  materialization_status: "available",
  materialization_displayable: true,
  materialization_issues: [],
  has_compile_report: true,
  has_provenance: true,
  has_cost: false,
  has_traces: true,
} as unknown as ReconstructionRecordSummary;

describe("reconstruction dense table", () => {
  it("has one cell per non-title column and flags weak readings", () => {
    const cells = reconstructionTableCells({
      ...base,
      axial_diagnostic: {
        calls: 1,
        revision: 1,
        first_recall: 0.55,
        recall: 0.55,
        input_cells: null,
        matched_cells: null,
        missing_cells: null,
        candidate_only_cells: null,
        exterior_excess_cells: 10,
        exterior_excess_fraction: 0.12,
        surface_fit_p50_mm: null,
        surface_fit_p90_mm: null,
        surface_faces_off_plane: null,
        candidate_openings: null,
        unmodelled_source_openings: null,
      },
    });
    expect(cells.map((cell) => cell.key)).toEqual(
      RECONSTRUCTION_TABLE_COLUMNS.slice(1).map((column) => column.key),
    );
    const byKey = Object.fromEntries(cells.map((cell) => [cell.key, cell]));
    expect(byKey.scale).toEqual({ key: "scale", value: "assumed metres", tone: "warn" });
    expect(byKey.recall).toEqual({ key: "recall", value: "55%", tone: "warn" });
    expect(byKey.excess).toEqual({ key: "excess", value: "12.0%", tone: "warn" });
    expect(byKey.controls).toBeUndefined();
    expect(byKey.rating.value).toBe("4★");
  });

  it("mutes missing readings instead of inventing them", () => {
    const byKey = Object.fromEntries(
      reconstructionTableCells({ ...base, axial_diagnostic: null, rating: null }).map((cell) => [
        cell.key,
        cell,
      ]),
    );
    expect(byKey.recall).toEqual({ key: "recall", value: "--", tone: "muted" });
    expect(byKey.rating).toEqual({ key: "rating", value: "--", tone: "muted" });
  });
});
