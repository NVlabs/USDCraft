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

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { MetadataPanelContent } from "@/components/inspector/MetadataPanel";
import {
  BASE_INSPECTOR_TABS,
  DEFAULT_INSPECTOR_CODE_FILE_PATHS,
  buildInspectorMetadataPresentation,
  inspectorTabs,
  type InspectorRecordSummary,
} from "@/components/inspector/record-adapter";

function summary(
  overrides: Partial<InspectorRecordSummary> = {},
): InspectorRecordSummary {
  return {
    record_id: "rec_generation_001",
    pipeline: "generation",
    status: "generation_ready",
    input_kind: "text_image",
    title: "Workbench asset",
    prompt_preview: "A useful asset",
    created_at: "2026-08-21T08:00:00Z",
    updated_at: "2026-08-21T08:01:00Z",
    provider: "codex-cli",
    model_id: "gpt-test",
    agent_harness: "codex",
    thinking_level: "high",
    turn_count: 3,
    input_tokens: 120,
    output_tokens: 34,
    total_cost_usd: 0.125,
    active_revision_id: "rev_000001",
    materialization_status: "available",
    has_compile_report: true,
    has_provenance: true,
    has_cost: true,
    has_traces: true,
    ...overrides,
  };
}

describe("inspector record adapters", () => {
  it("keeps the Generation base tabs and appends the optional Inputs slot", () => {
    expect(inspectorTabs(false)).toBe(BASE_INSPECTOR_TABS);
    expect(inspectorTabs(true)).toEqual([
      "inspect",
      "render",
      "code",
      "inputs",
      "metadata",
    ]);
  });

  it("keeps final model.py as the default code source beside model.urdf", () => {
    expect(DEFAULT_INSPECTOR_CODE_FILE_PATHS).toEqual({
      "model.py": "model.py",
      "model.urdf": "model.urdf",
    });
  });

  it("projects Generation metadata without changing its established labels", () => {
    const presentation = buildInspectorMetadataPresentation(summary());

    expect(presentation.pipeline).toBe("generation");
    expect(presentation.inputLabel).toBe("text + image");
    expect(presentation.runtimeHeading).toBe("Generation Runtime");
    const settings = Object.fromEntries(
      presentation.runSettings.map((row) => [row.key, row.value]),
    );
    expect(settings.arm).toBe("Codex");
    expect(settings.effort).toBe("high");
    expect(settings.turns).toBe("3 · no turn limit");
    expect(presentation.artifacts.map((item) => item.label)).toEqual([
      "provenance.json",
      "cost.json",
      "compile_report.json",
    ]);
  });

  it("renders Reconstruction metadata in the same structured presentation with missing values", () => {
    const presentation = buildInspectorMetadataPresentation(
      summary({
        record_id: "rec_reconstruction_001",
        pipeline: "reconstruction",
        status: "reconstruction_ready",
        input_kind: "mesh_image",
        provider: null,
        model_id: null,
        agent_harness: null,
        thinking_level: null,
        reasoning_effort: null,
        turn_count: null,
        input_tokens: null,
        output_tokens: null,
        total_cost_usd: null,
        materialization_status: "available",
        has_cost: false,
      }),
    );
    const html = renderToStaticMarkup(
      <MetadataPanelContent presentation={presentation} textLoader={vi.fn()} />,
    );

    expect(html).toContain("Identity");
    expect(html).toContain("reconstruction_ready");
    expect(html).toContain("mesh + image");
    expect(html).toContain("Reconstruction Runtime");
    expect(html).toContain("Read-only Artifacts");
    expect(html).toContain("--");
    expect(html).not.toContain("Generation Runtime");
    expect(html).not.toContain("evidence_manifest");
  });

  it("renders the Run settings section once, above the usage rows", () => {
    const presentation = buildInspectorMetadataPresentation(
      summary({
        agent_harness: "provider",
        provider: "nvidia",
        runtime_source: "nvidia-responses-api",
        api_style: "responses",
        image_observation: "external_inventory",
        batch_id: "generation_smoke4_20260927T073023Z",
        batch_row_id: "text_desk_lamp",
      }),
    );
    const html = renderToStaticMarkup(
      <MetadataPanelContent presentation={presentation} textLoader={vi.fn()} />,
    );

    expect(html).toContain("Run settings");
    expect(html).toContain("Responses API");
    expect(html).toContain("external inventory");
    expect(html).toContain("row text_desk_lamp");
    expect(html.match(/gpt-test/g)).toHaveLength(1);
    expect(html.indexOf("Run settings")).toBeLessThan(html.indexOf("Generation Runtime"));
    expect(html).not.toContain(">Source<");
    expect(html).not.toContain("Reasoning effort</span><span class=\"prop-value");
  });

  it("uses Reconstruction run settings and discloses guidance text", () => {
    const presentation = buildInspectorMetadataPresentation(
      summary({ pipeline: "reconstruction", record_id: "rec_reconstruction_001" }),
      {
        runSettings: [
          {
            key: "guidance",
            label: "Guidance",
            value: "provided",
            expand: { label: "Show guidance", text: "Keep the lid closed." },
          },
        ],
      },
    );
    const html = renderToStaticMarkup(
      <MetadataPanelContent presentation={presentation} textLoader={vi.fn()} />,
    );

    expect(html).toContain('data-run-setting="guidance"');
    expect(html).toContain("<details");
    expect(html).toContain("Show guidance");
    expect(html).toContain("Keep the lid closed.");
    expect(html).not.toContain('data-run-setting="arm"');
  });
});
