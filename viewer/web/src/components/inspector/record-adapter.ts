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

import type { ReactNode } from "react";

import { commonRunSettings, type RunSettingRow } from "@/lib/run-settings";
import type { ProviderApiStyle, RecordTextFile } from "@/lib/types";

export const BASE_INSPECTOR_TABS = ["inspect", "render", "code", "metadata"] as const;

export type BaseInspectorTab = (typeof BASE_INSPECTOR_TABS)[number];
export type InspectorPanelTab = BaseInspectorTab | "inputs" | "diagnostics";

export type InspectorRecordSummary = {
  record_id: string;
  pipeline: "generation" | "reconstruction";
  status: string | null;
  input_kind?: string | null;
  title: string;
  prompt_preview?: string;
  created_at?: string | null;
  updated_at?: string | null;
  provider?: string | null;
  model_id?: string | null;
  agent_harness?: "codex" | "provider" | null;
  runtime_source?: string | null;
  api_style?: ProviderApiStyle | null;
  image_observation?: string | null;
  thinking_level?: string | null;
  reasoning_effort?: string | null;
  turn_count?: number | null;
  max_turns?: number | null;
  batch_id?: string | null;
  batch_row_id?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_cost_usd?: number | null;
  active_revision_id?: string | null;
  materialization_status?: string | null;
  has_compile_report?: boolean;
  has_provenance?: boolean;
  has_cost?: boolean;
  has_traces?: boolean;
};

export type InspectorTextLoaderOptions = {
  previewBytes?: number;
  full?: boolean;
  signal?: AbortSignal;
};

export type InspectorTextLoader = (
  recordId: string,
  filePath: string,
  options?: InspectorTextLoaderOptions,
) => Promise<RecordTextFile>;

export type InspectorFileUrlBuilder = (recordId: string, filePath: string) => string;

export type InspectorCodeFilePaths = Readonly<{
  "model.py": string;
  "model.urdf": string;
}>;

export const DEFAULT_INSPECTOR_CODE_FILE_PATHS: InspectorCodeFilePaths = {
  "model.py": "model.py",
  "model.urdf": "model.urdf",
};

export type InspectorMetadataArtifact = Readonly<{
  filePath: string;
  label: string;
  available: boolean;
}>;

export type InspectorMetadataPresentation = Readonly<{
  recordId: string;
  pipeline: "generation" | "reconstruction";
  status: string | null;
  inputLabel: string;
  activeRevisionId: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  runtimeHeading: string;
  /** Rows of the Run settings section (arm, protocol, model, effort, turns, ...). */
  runSettings: readonly RunSettingRow[];
  provider: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  totalCostUsd: number | null;
  materializationStatus: string | null;
  artifacts: readonly InspectorMetadataArtifact[];
  tracesAvailable: boolean;
}>;

export type InspectorMetadataPresentationOptions = Readonly<{
  inputLabel?: string;
  runtimeHeading?: string;
  runSettings?: readonly RunSettingRow[];
  artifacts?: readonly InspectorMetadataArtifact[];
}>;

export type InspectorRecordAdapter = Readonly<{
  recordId: string | null;
  summary: InspectorRecordSummary | null;
  requestText?: string | null;
  requestFilePath?: string | null;
  textLoader: InspectorTextLoader;
  fileUrl?: InspectorFileUrlBuilder;
  /** URL of one file of the compiled materialization (`isaac/model.usdc`, `model.usdz`). */
  materializedFileUrl?: InspectorFileUrlBuilder;
  codeFilePaths?: Partial<InspectorCodeFilePaths>;
  metadataPresentation?: InspectorMetadataPresentation | null;
}>;

export type RenderOptionsExtraSection = Readonly<{
  key: string;
  label: string;
  content: ReactNode;
}>;

export function inspectorTabs(
  includeInputs: boolean,
  includeDiagnostics = false,
): readonly InspectorPanelTab[] {
  if (!includeInputs && !includeDiagnostics) return BASE_INSPECTOR_TABS;
  return [
    "inspect",
    "render",
    "code",
    ...(includeInputs ? (["inputs"] as const) : []),
    ...(includeDiagnostics ? (["diagnostics"] as const) : []),
    "metadata",
  ];
}

export function inspectorInputLabel(
  inputKind: string | null | undefined,
  pipeline: InspectorRecordSummary["pipeline"] = "generation",
): string {
  switch (inputKind) {
    case "text_image":
      return "text + image";
    case "mesh_image_text":
      return "mesh + image + text";
    case "mesh_image":
      return "mesh + image";
    case "text":
      return "text";
    default:
      return inputKind?.trim() || (pipeline === "generation" ? "text" : "--");
  }
}

function pipelineRuntimeHeading(pipeline: InspectorRecordSummary["pipeline"]): string {
  return pipeline === "reconstruction" ? "Reconstruction Runtime" : "Generation Runtime";
}

function defaultArtifacts(summary: InspectorRecordSummary): readonly InspectorMetadataArtifact[] {
  return [
    {
      filePath: "provenance.json",
      label: "provenance.json",
      available: summary.has_provenance === true,
    },
    {
      filePath: "cost.json",
      label: "cost.json",
      available: summary.has_cost === true,
    },
    {
      filePath: "compile_report.json",
      label: "compile_report.json",
      available: summary.has_compile_report === true,
    },
  ];
}

export function buildInspectorMetadataPresentation(
  summary: InspectorRecordSummary,
  options: InspectorMetadataPresentationOptions = {},
): InspectorMetadataPresentation {
  return {
    recordId: summary.record_id,
    pipeline: summary.pipeline,
    status: summary.status,
    inputLabel: options.inputLabel ?? inspectorInputLabel(summary.input_kind, summary.pipeline),
    activeRevisionId: summary.active_revision_id ?? null,
    createdAt: summary.created_at ?? null,
    updatedAt: summary.updated_at ?? null,
    runtimeHeading: options.runtimeHeading ?? pipelineRuntimeHeading(summary.pipeline),
    runSettings: options.runSettings ?? commonRunSettings(summary),
    provider:
      summary.provider ?? (summary.agent_harness === "codex" ? "codex-cli" : null),
    inputTokens: summary.input_tokens ?? null,
    outputTokens: summary.output_tokens ?? null,
    totalCostUsd: summary.total_cost_usd ?? null,
    materializationStatus: summary.materialization_status ?? null,
    artifacts: options.artifacts ?? defaultArtifacts(summary),
    tracesAvailable: summary.has_traces === true,
  };
}
