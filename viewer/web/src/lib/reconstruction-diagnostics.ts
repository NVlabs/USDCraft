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

import type { AxialDiagnostic, ReconstructionRecordDetail } from "@/lib/types";

export type DiagnosticRow = { label: string; value: string; tone?: "warn" | "ok" | "muted" };
export type DiagnosticSection = { key: string; title: string; note?: string; rows: DiagnosticRow[] };

function asRecord(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function pct(value: number | null): string {
  return value == null ? "--" : `${(value * 100).toFixed(1)}%`;
}

function fixed(value: number | null, digits: number, suffix = ""): string {
  return value == null ? "--" : `${value.toFixed(digits)}${suffix}`;
}

function seconds(value: number | null): string {
  if (value == null) return "--";
  if (value < 90) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  return `${minutes} min ${Math.round(value - minutes * 60)} s`;
}

/** The AXIAL diagnostic card: last `check_axial` readings, plus how many calls and the first recall. */
export function axialDiagnosticRows(diagnostic: AxialDiagnostic): DiagnosticRow[] {
  const rows: DiagnosticRow[] = [
    {
      label: "Measurement binding",
      value: diagnostic.binding_status === "current"
        ? "Matches final source and candidate"
        : diagnostic.binding_status === "stale"
          ? "Stale artifact binding; geometry equivalence unverified"
          : "Final artifact binding unverified",
      tone: diagnostic.binding_status === "stale" ? "warn" : "muted",
    },
    {
      label: "check_axial calls",
      value:
        diagnostic.first_recall != null && diagnostic.calls > 1
          ? `${diagnostic.calls} (recall ${diagnostic.first_recall.toFixed(3)} → ${fixed(diagnostic.recall, 3)})`
          : `${diagnostic.calls}`,
    },
    { label: "Recall (last call)", value: fixed(diagnostic.recall, 3) },
    {
      label: "Matched / missing source cells",
      value:
        diagnostic.matched_cells != null && diagnostic.input_cells != null
          ? `${diagnostic.matched_cells} / ${diagnostic.input_cells} matched, ${diagnostic.missing_cells ?? "--"} missing`
          : "--",
    },
    { label: "Candidate-only cells", value: diagnostic.candidate_only_cells?.toString() ?? "--" },
    {
      label: "Exterior excess",
      value:
        diagnostic.exterior_excess_cells != null
          ? `${diagnostic.exterior_excess_cells} cells (${pct(diagnostic.exterior_excess_fraction)})`
          : "--",
      tone:
        diagnostic.exterior_excess_fraction != null && diagnostic.exterior_excess_fraction > 0.05
          ? "warn"
          : undefined,
    },
    {
      label: "Surface fit candidate → source",
      value:
        diagnostic.surface_fit_p50_mm != null
          ? `p50 ${fixed(diagnostic.surface_fit_p50_mm, 1, " mm")}, p90 ${fixed(diagnostic.surface_fit_p90_mm, 1, " mm")}`
          : "--",
    },
    {
      label: "Faces off plane",
      value: diagnostic.surface_faces_off_plane?.toString() ?? "--",
      tone: diagnostic.surface_faces_off_plane ? "warn" : undefined,
    },
    {
      label: "Openings",
      value:
        diagnostic.candidate_openings != null
          ? `${diagnostic.candidate_openings} candidate, ${diagnostic.unmodelled_source_openings ?? 0} source openings unmodelled`
          : "--",
    },
  ];
  return rows;
}

export function diagnosticSections(detail: ReconstructionRecordDetail): DiagnosticSection[] {
  const summary = detail.summary;
  const runtime = asRecord(asRecord(detail.provenance)?.runtime) ?? {};
  const sections: DiagnosticSection[] = [];

  const diagnostic = summary.axial_diagnostic ?? null;
  sections.push({
    key: "axial",
    title: "AXIAL diagnostic",
    note: diagnostic
      ? "Mechanical readings of the Author's own check_axial calls against the source mesh. Never a gate."
      : "No check_axial call is recorded for this record.",
    rows: diagnostic ? axialDiagnosticRows(diagnostic) : [],
  });

  const scale = asRecord(runtime.scale);
  const estimate = asRecord(scale?.estimate);
  const inspection = asRecord(scale?.inspection);
  const scaleRows: DiagnosticRow[] = [];
  if (summary.scale_source) {
    scaleRows.push({
      label: "Metres decided by",
      value:
        summary.scale_source === "estimated_from_semantic_image"
          ? "category estimate from the semantic image"
          : summary.scale_source === "declared"
            ? "user declaration (--meters-per-unit)"
            : summary.scale_source,
      tone: summary.scale_source === "format_contract" ? "warn" : undefined,
    });
  }
  if (summary.scale_meters_per_unit != null) {
    scaleRows.push({ label: "meters_per_unit", value: summary.scale_meters_per_unit.toFixed(4) });
  }
  const longestUnits = asNumber(inspection?.longest_extent_units);
  if (longestUnits != null) {
    scaleRows.push({
      label: "Source longest side",
      value: `${longestUnits.toFixed(3)} units${inspection?.suspected_unit_normalized ? " (suspected unit-normalized)" : ""}`,
    });
  }
  if (estimate) {
    scaleRows.push({
      label: "Estimate",
      value: `${String(estimate.category ?? "--")}: ${fixed(asNumber(estimate.longest_dimension_m), 2, " m")} (${String(estimate.confidence ?? "--")} confidence)`,
    });
    if (typeof estimate.rationale === "string" && estimate.rationale) {
      scaleRows.push({ label: "Rationale", value: estimate.rationale, tone: "muted" });
    }
  }
  sections.push({
    key: "scale",
    title: "Scale",
    rows: scaleRows,
  });

  // Run settings (arm, model, image observation, turns against the budget, batch) live in the
  // Runtime tab's Run settings section; this section keeps the run's measured cost in time.
  const runRows: DiagnosticRow[] = [];
  const toolCalls = asNumber(runtime.tool_call_count);
  if (toolCalls != null) runRows.push({ label: "Tool calls", value: `${toolCalls}` });
  const elapsed = asNumber(runtime.elapsed_seconds);
  if (elapsed != null) runRows.push({ label: "Author elapsed", value: seconds(elapsed) });
  const stageTimings = Array.isArray(runtime.stage_timings) ? runtime.stage_timings : [];
  for (const item of stageTimings) {
    const stage = asRecord(item);
    if (!stage || typeof stage.stage !== "string") continue;
    const detailText = typeof stage.detail === "string" && stage.detail ? ` (${stage.detail})` : "";
    runRows.push({
      label: `stage ${stage.stage}${detailText}`,
      value: `${seconds(asNumber(stage.duration_seconds))}${stage.status === "failed" ? " · failed" : ""}`,
      tone: stage.status === "failed" ? "warn" : "muted",
    });
  }
  sections.push({ key: "run", title: "Run", rows: runRows });

  const compileReport = asRecord(detail.compile_report) ?? {};
  const warnings = asStringList(compileReport.warnings);
  const notes = asStringList(compileReport.notes);
  sections.push({
    key: "warnings",
    title: "Compile warnings",
    note: warnings.length ? undefined : "No asset-specific compile warnings.",
    rows: warnings.map((warning, index) => ({ label: `${index + 1}`, value: warning, tone: "warn" })),
  });
  sections.push({
    key: "notes",
    title: "Backend notes",
    note: notes.length
      ? "Fixed PhysX/Newton capability boundaries recorded for every conforming joint; informational."
      : "No backend notes.",
    rows: notes.map((note, index) => ({ label: `${index + 1}`, value: note, tone: "muted" })),
  });
  return sections;
}
