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

import { commonRunSettings, reconstructionRunSettings, type RunSettingRow } from "./run-settings";
import type { ReconstructionRecordDetail } from "./types";

function byKey(rows: readonly RunSettingRow[]): Record<string, RunSettingRow> {
  return Object.fromEntries(rows.map((row) => [row.key, row]));
}

function reconstructionDetail(
  summary: Record<string, unknown> = {},
  detail: Record<string, unknown> = {},
): ReconstructionRecordDetail {
  return {
    summary: {
      record_id: "rec",
      pipeline: "reconstruction",
      agent_harness: "provider",
      runtime_source: "nvidia-chat-completions-api",
      api_style: "chat-completions",
      model_id: "azure/openai/gpt-6-astra",
      reasoning_effort: "low",
      thinking_level: "low",
      image_observation: "external_inventory",
      turn_count: 19,
      max_turns: 80,
      mesh_cleanup: "skipped",
      scale_source: "estimated_from_semantic_image",
      scale_meters_per_unit: 0.24566,
      batch_id: "reconstruction_smoke5_20260927T073025Z",
      batch_row_id: "case_0031",
      ...summary,
    },
    request_text: "Reconstruct the primary subject object shown in the semantic image.",
    request_prompt: "default",
    guidance_status: "absent",
    guidance: null,
    ...detail,
  } as unknown as ReconstructionRecordDetail;
}

describe("run settings", () => {
  it("shows a Provider run's protocol, budget and batch row", () => {
    const rows = byKey(reconstructionRunSettings(reconstructionDetail()));
    expect(rows.arm.value).toBe("Provider");
    expect(rows.protocol).toMatchObject({
      value: "Chat Completions API",
      hint: "nvidia-chat-completions-api",
    });
    expect(rows.model.value).toBe("azure/openai/gpt-6-astra");
    expect(rows.effort.value).toBe("low");
    expect(rows["image-observation"].value).toBe("external inventory");
    expect(rows.turns).toMatchObject({ label: "Turns / budget", value: "19 / 80" });
    expect(rows.turns.tone).toBeUndefined();
    expect(rows.scale.value).toBe("estimated from image · 0.2457 m/unit");
    expect(rows["mesh-cleanup"].value).toBe("skipped");
    expect(rows.batch).toMatchObject({
      value: "row case_0031",
      hint: "reconstruction_smoke5_20260927T073025Z",
    });
  });

  it("keeps the Reconstruction rows before the batch row", () => {
    const keys = reconstructionRunSettings(reconstructionDetail()).map((row) => row.key);
    expect(keys).toEqual([
      "arm",
      "protocol",
      "model",
      "effort",
      "image-observation",
      "turns",
      "guidance",
      "request",
      "scale",
      "mesh-cleanup",
      "batch",
    ]);
  });

  it("marks an exhausted turn budget", () => {
    const rows = byKey(reconstructionRunSettings(reconstructionDetail({ turn_count: 80 })));
    expect(rows.turns.tone).toBe("warn");
  });

  it("says the Codex arm has no turn limit and no API style", () => {
    const rows = byKey(
      reconstructionRunSettings(
        reconstructionDetail({
          agent_harness: "codex",
          runtime_source: "codex_root_thread_state",
          api_style: null,
          max_turns: null,
          turn_count: null,
          mesh_cleanup: "run",
          batch_id: null,
          batch_row_id: null,
        }),
      ),
    );
    expect(rows.arm.value).toBe("Codex");
    expect(rows.protocol).toMatchObject({ value: "Codex CLI", hint: "codex_root_thread_state" });
    expect(rows.turns).toMatchObject({ label: "Turns", value: "-- · no turn limit" });
    expect(rows["mesh-cleanup"].value).toBe("run");
    expect(rows.batch).toBeUndefined();
  });

  it("expands present guidance and a custom request, and flags unrecorded guidance", () => {
    const custom = byKey(
      reconstructionRunSettings(
        reconstructionDetail(
          {},
          {
            request_text: "Rebuild only the drawer cabinet.\n",
            request_prompt: "custom",
            guidance_status: "present",
            guidance: "Treat the handles as fixed.",
          },
        ),
      ),
    );
    expect(custom.guidance).toMatchObject({
      value: "provided",
      expand: { label: "Show guidance", text: "Treat the handles as fixed." },
    });
    expect(custom.request).toMatchObject({
      value: "custom",
      expand: { label: "Show request", text: "Rebuild only the drawer cabinet." },
    });

    const defaults = byKey(reconstructionRunSettings(reconstructionDetail()));
    expect(defaults.guidance).toMatchObject({ value: "none", tone: "muted" });
    expect(defaults.guidance.expand).toBeUndefined();
    expect(defaults.request).toMatchObject({ value: "default" });
    expect(defaults.request.expand).toBeUndefined();

    const unrecorded = byKey(
      reconstructionRunSettings(reconstructionDetail({}, { guidance_status: "unrecorded" })),
    );
    expect(unrecorded.guidance.value).toBe("not recorded");
  });

  it("gives Generation the common rows only", () => {
    const rows = commonRunSettings({
      agent_harness: "provider",
      runtime_source: "nvidia-responses-api",
      api_style: "responses",
      model_id: "gpt-test",
      thinking_level: "high",
      image_observation: "no_image",
      turn_count: 18,
    });
    const keyed = byKey(rows);
    expect(rows.map((row) => row.key)).toEqual([
      "arm",
      "protocol",
      "model",
      "effort",
      "image-observation",
      "turns",
    ]);
    expect(keyed.protocol.value).toBe("Responses API");
    expect(keyed["image-observation"].value).toBe("no image");
    expect(keyed.turns).toMatchObject({ label: "Turns", value: "18" });
  });
});
