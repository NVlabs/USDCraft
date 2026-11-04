// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
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

export type AgentHarness = "codex" | "provider";
export type SourceFilter = AgentHarness | null;

export type InputKind = "text" | "text_image";
export type InputKindFilter = InputKind | null;
export type RatingFilter = 1 | 2 | 3 | 4 | 5 | "unrated" | null;

export type AppMode = "generation" | "reconstruction";

export type AuthorTimelineEvent = {
  turn: number;
  tool: string;
  ok: boolean;
  revision: number | null;
  tool_elapsed_seconds: number | null;
  response_elapsed_seconds: number | null;
  recall: number | null;
};

export type AuthorTimeline = {
  record_id: string;
  source: "provider_events";
  events: AuthorTimelineEvent[];
};

export type AxialExcessSlice = {
  stack_slice: number;
  side: string;
  excess_cells: number;
  bounds_min_xyz_m: number[];
  bounds_max_xyz_m: number[];
};

export type AxialGeometry = {
  record_id: string;
  source: "provider_events";
  grid_bounds_min_xyz_m: number[];
  grid_bounds_max_xyz_m: number[];
  stack_pitch_m: number | null;
  excess_slices: AxialExcessSlice[];
};

export type AxialDiagnostic = {
  binding_status?: "current" | "stale" | "unverified";
  calls: number;
  revision: number | null;
  first_recall: number | null;
  recall: number | null;
  input_cells: number | null;
  matched_cells: number | null;
  missing_cells: number | null;
  candidate_only_cells: number | null;
  exterior_excess_cells: number | null;
  exterior_excess_fraction: number | null;
  surface_fit_p50_mm: number | null;
  surface_fit_p90_mm: number | null;
  surface_faces_off_plane: number | null;
  candidate_openings: number | null;
  unmodelled_source_openings: number | null;
};

export type ProviderApiStyle = "responses" | "chat-completions";

export type ReconstructionRecordSummary = {
  record_id: string;
  pipeline: "reconstruction";
  status: "reconstruction_ready" | null;
  input_kind: "mesh_image" | "mesh_image_text" | null;
  title: string;
  prompt_preview: string;
  rating: number | null;
  created_at: string | null;
  updated_at: string | null;
  viewer_asset_updated_at: string | null;
  provider: string | null;
  model_id: string | null;
  agent_harness: AgentHarness | null;
  reasoning_effort: string | null;
  thinking_level: string | null;
  turn_count: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_cost_usd: number | null;
  active_revision_id: string | null;
  revision_count: number;
  input_count: number;
  source_available: boolean;
  candidate_available: boolean;
  duration_seconds?: number | null;
  runtime_source?: string | null;
  // Provider wire protocol; null on the Codex arm.
  api_style?: ProviderApiStyle | null;
  image_observation?: string | null;
  // The enforced Provider turn budget; the Codex arm has no turn limit and records none.
  max_turns?: number | null;
  mesh_cleanup?: "run" | "skipped" | null;
  scale_source?: string | null;
  scale_meters_per_unit?: number | null;
  scale_estimate_category?: string | null;
  scale_estimate_longest_m?: number | null;
  batch_id?: string | null;
  batch_row_id?: string | null;
  axial_diagnostic?: AxialDiagnostic | null;
  materialization_status: ListMaterializationStatus;
  materialization_displayable: boolean;
  materialization_validated: boolean;
  materialization_issues: MaterializationIssue[];
  has_compile_report: boolean;
  has_provenance: boolean;
  has_cost: boolean;
  has_traces: boolean;
};

export type ReconstructionRecordEvidence = {
  role: string;
  name: string;
  label: string;
  relative_path: string;
  file_url: string;
  media_kind: "image" | "mesh" | "text" | "json" | "binary";
  byte_count: number;
  sha256: string | null;
  metadata: Record<string, unknown>;
};

export type ReconstructionRecordAsset = {
  available: boolean;
  label: string;
  file_url: string | null;
  base_file_url: string | null;
  format: string | null;
  revision_key: string | null;
};

export type ReconstructionRecordDetail = {
  summary: ReconstructionRecordSummary;
  source: ReconstructionRecordAsset;
  candidate: ReconstructionRecordAsset;
  inputs: ReconstructionRecordEvidence[];
  record: Record<string, unknown>;
  reconstruction_manifest: Record<string, unknown> | null;
  evidence_manifest: Record<string, unknown> | null;
  request_text: string | null;
  // "default" when request_text is the Host's default Reconstruction request.
  request_prompt?: "default" | "custom" | null;
  // User guidance recorded in `reconstruction.json` `input.guidance`.
  guidance_status?: "present" | "absent" | "unrecorded";
  guidance?: string | null;
  provenance: Record<string, unknown> | null;
  compile_report: Record<string, unknown> | null;
};


export type MaterializationStatus =
  | "available"
  | "missing"
  | "compile_failed"
  | "contract_invalid";

// The Viewer record list may defer the full canonical contract check for a row; this marks that
// placeholder. It is never a canonical classification and never means "available" — see
// `viewer/README.md` ("Lazy canonical validation").
export type ListMaterializationStatus = MaterializationStatus | "not_validated";

export type MaterializationIssue = {
  path: string;
  message: string;
};

export type RecordSummary = {
  record_id: string;
  pipeline: "generation";
  status: "generation_ready" | null;
  input_kind: InputKind | null;
  rating: number | null;
  title: string;
  prompt_preview: string;
  created_at: string | null;
  updated_at: string | null;
  viewer_asset_updated_at: string | null;
  provider: string | null;
  model_id: string | null;
  agent_harness: AgentHarness | null;
  runtime_source?: string | null;
  api_style?: ProviderApiStyle | null;
  image_observation?: string | null;
  has_traces: boolean;
  thinking_level: string | null;
  batch_id?: string | null;
  batch_row_id?: string | null;
  turn_count: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_cost_usd: number | null;
  active_revision_id: string | null;
  revision_count: number;
  materialization_status: ListMaterializationStatus;
  materialization_displayable: boolean;
  materialization_validated: boolean;
  materialization_issues: MaterializationIssue[];
  has_compile_report: boolean;
  has_provenance: boolean;
  has_cost: boolean;
};

export type RecordInputImage = {
  path: string | null;
};

export type DeleteRecordResult = {
  status: "deleted";
  record_id: string;
  materialization_deleted: boolean;
  cleanup_pending: boolean;
};

export type WorkbenchEntry = {
  record_id: string;
  added_at: string;
  label: string | null;
  tags: string[];
  archived: boolean;
  record: RecordSummary | null;
};

export type ViewerStatus = {
  status: "ok";
  repo_root: string | null;
  /** True when the server runs with `--read-only`: rating and deletion are refused and hidden. */
  read_only: boolean;
};

export type ViewerBootstrap = {
  repo_root: string;
  generated_at: string;
  workbench_entries: WorkbenchEntry[];
};

export type RecordTextFile = {
  record_id: string;
  file_path: string;
  content: string;
  truncated: boolean;
  byte_count: number;
  preview_byte_limit: number | null;
};

export type InspectorTab = "inspect" | "render" | "code" | "metadata";
export type ReconstructionInspectorTab = InspectorTab | "inputs" | "diagnostics";

export type ViewerState = {
  bootstrap: ViewerBootstrap | null;
  recordCache: Record<string, RecordSummary>;
  selectedRecordId: string | null;
  selectedRecordSummary: RecordSummary | null;
  selectedInspectorTab: InspectorTab;
  inspectorOpen: boolean;
  loading: boolean;
  error: string | null;
  searchQuery: string;
  sourceFilter: SourceFilter;
  modelFilter: string | null;
  inputKindFilter: InputKindFilter;
  ratingFilter: RatingFilter;
  workbenchNotice: string | null;
};

export type ViewerAction =
  | { type: "SET_BOOTSTRAP"; payload: ViewerBootstrap }
  | { type: "UPSERT_RECORDS"; payload: RecordSummary[] }
  | { type: "SET_SELECTED_RECORD_SUMMARY"; payload: RecordSummary | null }
  | { type: "SELECT_RECORD"; payload: string | null }
  | { type: "SET_INSPECTOR_TAB"; payload: InspectorTab }
  | { type: "TOGGLE_INSPECTOR" }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_SEARCH"; payload: string }
  | { type: "SET_SOURCE_FILTER"; payload: SourceFilter }
  | { type: "SET_MODEL_FILTER"; payload: string | null }
  | { type: "SET_INPUT_KIND_FILTER"; payload: InputKindFilter }
  | { type: "SET_RATING_FILTER"; payload: RatingFilter }
  | { type: "SET_WORKBENCH_NOTICE"; payload: string | null }
  | {
      type: "DELETE_RECORD_LOCAL";
      payload: { recordId: string; nextRecordId?: string | null };
    };
