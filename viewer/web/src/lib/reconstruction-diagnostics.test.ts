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

import { axialDiagnosticRows, diagnosticSections } from "./reconstruction-diagnostics";
import type { ReconstructionRecordDetail } from "./types";

const detail = {
  summary: {
    record_id: "rec",
    pipeline: "reconstruction",
    status: "reconstruction_ready",
    input_kind: "mesh_image",
    title: "Toaster",
    prompt_preview: "",
    rating: null,
    created_at: null,
    updated_at: null,
    viewer_asset_updated_at: null,
    provider: "nvidia",
    model_id: "m",
    agent_harness: "provider",
    reasoning_effort: "high",
    thinking_level: "high",
    turn_count: 26,
    input_tokens: null,
    output_tokens: null,
    total_cost_usd: null,
    active_revision_id: "rev_000001",
    revision_count: 1,
    input_count: 4,
    source_available: true,
    candidate_available: true,
    scale_source: "estimated_from_semantic_image",
    scale_meters_per_unit: 0.2506,
    scale_estimate_category: "microwave oven",
    scale_estimate_longest_m: 0.5,
    batch_id: "round2_20260905T140000Z",
    batch_row_id: "microwave_oven",
    axial_diagnostic: {
      calls: 2,
      revision: 5,
      first_recall: 0.61,
      recall: 0.74,
      input_cells: 1000,
      matched_cells: 740,
      missing_cells: 260,
      candidate_only_cells: 120,
      exterior_excess_cells: 90,
      exterior_excess_fraction: 0.081,
      surface_fit_p50_mm: 4.2,
      surface_fit_p90_mm: 15.1,
      surface_faces_off_plane: 1,
      candidate_openings: 2,
      unmodelled_source_openings: 1,
    },
    materialization_status: "available",
    materialization_displayable: true,
    materialization_issues: [],
    has_compile_report: true,
    has_provenance: true,
    has_cost: false,
    has_traces: true,
  },
  source: { available: true, label: "src" },
  candidate: { available: true, label: "cand" },
  inputs: [],
  record: {},
  reconstruction_manifest: null,
  evidence_manifest: null,
  request_text: null,
  provenance: {
    runtime: {
      turn_count: 26,
      tool_call_count: 26,
      elapsed_seconds: 488.4,
      stage_timings: [
        { stage: "freeze", duration_seconds: 0.4, status: "ok" },
        { stage: "author", duration_seconds: 480.1, status: "ok" },
        { stage: "host_gate", duration_seconds: 63.2, status: "failed" },
      ],
      scale: {
        source: "estimated_from_semantic_image",
        meters_per_unit: 0.2506,
        inspection: { longest_extent_units: 1.995, suspected_unit_normalized: true },
        estimate: {
          category: "microwave oven",
          longest_dimension_m: 0.5,
          confidence: "high",
          rationale: "typical countertop width",
        },
      },
    },
  },
  compile_report: {
    warnings: ["Material 'glass' texture was not found: x.png"],
    notes: ["Newton/MuJoCo has no Isaac Sim 6 MJC mapping for joint velocity limits: a, b"],
  },
} as unknown as ReconstructionRecordDetail;

describe("reconstruction diagnostics presentation", () => {
  it("renders the AXIAL card with call history and flagged tones", () => {
    const rows = axialDiagnosticRows(detail.summary.axial_diagnostic!);
    const byLabel = Object.fromEntries(rows.map((row) => [row.label, row]));
    expect(byLabel["check_axial calls"].value).toBe("2 (recall 0.610 → 0.740)");
    expect(byLabel["Exterior excess"]).toEqual({
      label: "Exterior excess",
      value: "90 cells (8.1%)",
      tone: "warn",
    });
    expect(byLabel["Controls"]).toBeUndefined();
    expect(byLabel["Source of these numbers"]).toBeUndefined();
  });

  it("builds scale, run, warning and note sections from the detail", () => {
    const sections = Object.fromEntries(
      diagnosticSections(detail).map((section) => [section.key, section]),
    );
    expect(sections.scale.rows.map((row) => row.label)).toEqual([
      "Metres decided by",
      "meters_per_unit",
      "Source longest side",
      "Estimate",
      "Rationale",
    ]);
    expect(sections.scale.rows[2].value).toBe("1.995 units (suspected unit-normalized)");
    expect(sections.scale.rows[3].value).toBe("microwave oven: 0.50 m (high confidence)");
    const run = sections.run.rows.map((row) => [row.label, row.value]);
    expect(run).toContainEqual(["Author elapsed", "8 min 8 s"]);
    expect(run).toContainEqual(["stage host_gate", "63.2 s · failed"]);
    expect(run).toContainEqual(["Tool calls", "26"]);
    expect(sections.warnings.rows).toHaveLength(1);
    expect(sections.notes.rows[0].tone).toBe("muted");
  });

  it("explains an absent diagnostic instead of rendering empty rows", () => {
    const bare = {
      ...detail,
      summary: { ...detail.summary, axial_diagnostic: null, scale_source: null },
      provenance: null,
      compile_report: null,
    } as unknown as ReconstructionRecordDetail;
    const sections = Object.fromEntries(
      diagnosticSections(bare).map((section) => [section.key, section]),
    );
    expect(sections.axial.rows).toEqual([]);
    expect(sections.axial.note).toContain("No check_axial call");
    expect(sections.warnings.note).toContain("No asset-specific");
  });
});

it("distinguishes a stale measurement from a final geometry match", () => {
  const diagnostic = detail.summary.axial_diagnostic!;
  const stale = axialDiagnosticRows({ ...diagnostic, binding_status: "stale" });
  expect(stale.find((row) => row.label === "Measurement binding")).toMatchObject({ tone: "warn" });
  expect(stale.find((row) => row.label === "Measurement binding")?.value).toContain("Stale");
  const current = axialDiagnosticRows({ ...diagnostic, binding_status: "current" });
  expect(current.find((row) => row.label === "Measurement binding")?.value).toContain("Matches final");
});

it("leaves run settings to the Runtime tab instead of repeating them as Run rows", () => {
  const run = diagnosticSections(detail).find((section) => section.key === "run")!;
  const labels = run.rows.map((row) => row.label);
  for (const moved of ["mesh views", "image observation", "max turns", "Author turns", "Batch"]) {
    expect(labels).not.toContain(moved);
  }
});
