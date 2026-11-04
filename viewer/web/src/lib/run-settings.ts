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

import type { ProviderApiStyle, ReconstructionRecordDetail } from "@/lib/types";
import { agentHarnessLabel } from "@/lib/utils";

/** One compact Run settings row; `expand` carries long text shown behind a disclosure. */
export type RunSettingRow = Readonly<{
  key: string;
  label: string;
  value: string;
  /** Secondary reading shown muted after the value (e.g. the raw runtime source). */
  hint?: string;
  tone?: "muted" | "warn";
  mono?: boolean;
  expand?: Readonly<{ label: string; text: string }>;
}>;

/** The run facts both products record; every field is optional so older shapes degrade to "--". */
export type RunSettingsSource = Readonly<{
  agent_harness?: "codex" | "provider" | null;
  runtime_source?: string | null;
  api_style?: ProviderApiStyle | null;
  model_id?: string | null;
  thinking_level?: string | null;
  reasoning_effort?: string | null;
  image_observation?: string | null;
  turn_count?: number | null;
  max_turns?: number | null;
  batch_id?: string | null;
  batch_row_id?: string | null;
}>;

const API_STYLE_LABELS: Record<ProviderApiStyle, string> = {
  responses: "Responses API",
  "chat-completions": "Chat Completions API",
};

const IMAGE_OBSERVATION_LABELS: Record<string, string> = {
  external_inventory: "external inventory",
  author_only: "author-only",
  disabled: "disabled",
  no_image: "no image",
};

const SCALE_SOURCE_LABELS: Record<string, string> = {
  declared: "declared",
  format_contract: "format contract",
  embedded_metadata: "embedded",
  estimated_from_semantic_image: "estimated from image",
};

function protocolRow(source: RunSettingsSource): RunSettingRow {
  const runtimeSource = source.runtime_source ?? undefined;
  if (source.agent_harness === "codex") {
    return { key: "protocol", label: "Wire protocol", value: "Codex CLI", hint: runtimeSource };
  }
  if (source.agent_harness === "provider") {
    const style = source.api_style ?? "responses";
    return {
      key: "protocol",
      label: "Wire protocol",
      value: API_STYLE_LABELS[style] ?? style,
      hint: runtimeSource,
    };
  }
  return { key: "protocol", label: "Wire protocol", value: runtimeSource ?? "--" };
}

function turnsRow(source: RunSettingsSource): RunSettingRow {
  const turns = source.turn_count;
  const used = turns != null ? `${turns}` : "--";
  if (source.agent_harness === "codex") {
    return { key: "turns", label: "Turns", value: `${used} · no turn limit` };
  }
  if (source.agent_harness === "provider" && source.max_turns != null) {
    const exhausted = turns != null && turns >= source.max_turns;
    return {
      key: "turns",
      label: "Turns / budget",
      value: `${used} / ${source.max_turns}`,
      tone: exhausted ? "warn" : undefined,
    };
  }
  return { key: "turns", label: "Turns", value: used };
}

/** Run settings every record carries: arm, protocol, model, effort, image reading, turns, batch. */
export function commonRunSettings(source: RunSettingsSource): RunSettingRow[] {
  const rows: RunSettingRow[] = [
    { key: "arm", label: "Arm", value: agentHarnessLabel(source.agent_harness) ?? "--" },
    protocolRow(source),
    { key: "model", label: "Model", value: source.model_id ?? "--", mono: true },
    {
      key: "effort",
      label: "Reasoning effort",
      value: source.thinking_level ?? source.reasoning_effort ?? "--",
    },
    {
      key: "image-observation",
      label: "Image observation",
      value: source.image_observation
        ? IMAGE_OBSERVATION_LABELS[source.image_observation] ?? source.image_observation
        : "--",
      tone: source.image_observation === "author_only" ? "muted" : undefined,
    },
    turnsRow(source),
  ];
  if (source.batch_id) {
    rows.push({
      key: "batch",
      label: "Batch",
      value: source.batch_row_id ? `row ${source.batch_row_id}` : source.batch_id,
      hint: source.batch_row_id ? source.batch_id : undefined,
      mono: true,
    });
  }
  return rows;
}

function guidanceRow(detail: ReconstructionRecordDetail): RunSettingRow {
  if (detail.guidance_status === "present" && detail.guidance) {
    return {
      key: "guidance",
      label: "Guidance",
      value: "provided",
      expand: { label: "Show guidance", text: detail.guidance },
    };
  }
  if (detail.guidance_status === "absent") {
    return { key: "guidance", label: "Guidance", value: "none", tone: "muted" };
  }
  return { key: "guidance", label: "Guidance", value: "not recorded", tone: "muted" };
}

function requestRow(detail: ReconstructionRecordDetail): RunSettingRow {
  const text = detail.request_text?.trim() ?? "";
  if (detail.request_prompt === "default") {
    return { key: "request", label: "Request prompt", value: "default", tone: "muted" };
  }
  if (detail.request_prompt === "custom" && text) {
    return {
      key: "request",
      label: "Request prompt",
      value: "custom",
      expand: { label: "Show request", text },
    };
  }
  return { key: "request", label: "Request prompt", value: "--" };
}

function scaleRow(detail: ReconstructionRecordDetail): RunSettingRow {
  const { scale_source: source, scale_meters_per_unit: metersPerUnit } = detail.summary;
  const sourceLabel = source ? SCALE_SOURCE_LABELS[source] ?? source : null;
  const unit = metersPerUnit != null ? `${metersPerUnit.toFixed(4)} m/unit` : null;
  return {
    key: "scale",
    label: "Scale",
    value: [sourceLabel, unit].filter(Boolean).join(" · ") || "--",
    tone: source === "format_contract" ? "warn" : undefined,
  };
}

/** Reconstruction adds guidance, request prompt, scale and mesh cleanup to the common rows. */
export function reconstructionRunSettings(detail: ReconstructionRecordDetail): RunSettingRow[] {
  const summary = detail.summary;
  const rows = commonRunSettings(summary);
  const batchIndex = rows.findIndex((row) => row.key === "batch");
  const reconstructionRows: RunSettingRow[] = [
    guidanceRow(detail),
    requestRow(detail),
    scaleRow(detail),
    {
      key: "mesh-cleanup",
      label: "Mesh cleanup",
      value: summary.mesh_cleanup ?? "--",
      tone: summary.mesh_cleanup === "skipped" ? "muted" : undefined,
    },
  ];
  // Batch membership stays last, after the Reconstruction settings.
  rows.splice(batchIndex < 0 ? rows.length : batchIndex, 0, ...reconstructionRows);
  return rows;
}
