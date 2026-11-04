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
import { describe, expect, it } from "vitest";

import { ReconstructionMaterializationStatus } from "./ReconstructionMaterializationStatus";
import type { ReconstructionRecordSummary } from "@/lib/types";

function record(
  overrides: Partial<ReconstructionRecordSummary>,
): ReconstructionRecordSummary {
  return {
    record_id: "rec_reconstruction_001",
    pipeline: "reconstruction",
    status: "reconstruction_ready",
    input_kind: "mesh_image",
    title: "Drawer",
    prompt_preview: "",
    rating: null,
    created_at: null,
    updated_at: null,
    viewer_asset_updated_at: null,
    provider: "codex-cli",
    model_id: null,
    agent_harness: "codex",
    reasoning_effort: null,
    thinking_level: null,
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
    has_traces: false,
    ...overrides,
  };
}

describe("ReconstructionMaterializationStatus", () => {
  it("shows a structured warning for a displayable contract-invalid candidate", () => {
    const html = renderToStaticMarkup(
      <ReconstructionMaterializationStatus
        record={record({
          materialization_status: "contract_invalid",
          materialization_issues: [
            { path: "reconstruction.json", message: "input manifest hash does not match" },
          ],
        })}
      />,
    );

    expect(html).toContain("contract_invalid");
    expect(html).toContain("Artifact contract is invalid");
    expect(html).toContain("reconstruction.json");
    expect(html).toContain("input manifest hash does not match");
  });

  it("reports compile failure as a blocked candidate preview", () => {
    const html = renderToStaticMarkup(
      <ReconstructionMaterializationStatus
        record={record({
          candidate_available: false,
          materialization_status: "compile_failed",
          materialization_displayable: false,
          materialization_issues: [
            { path: "compile_report.json", message: "status must be 'success'" },
          ],
        })}
      />,
    );

    expect(html).toContain("compile_failed");
    expect(html).toContain("candidate preview is blocked");
    expect(html).toContain("compile_report.json");
  });
});
