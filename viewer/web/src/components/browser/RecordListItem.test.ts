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

import type { RecordSummary } from "@/lib/types";

import {
  recordListMetadata,
  recordListPrimaryText,
  recordRatingLabel,
} from "./RecordListItem";

function summary(overrides: Partial<RecordSummary> = {}): RecordSummary {
  return {
    record_id: "rec_generation_ready",
    pipeline: "generation",
    status: "generation_ready",
    input_kind: "text_image",
    rating: null,
    title: "Backyard playground set",
    prompt_preview: "Create a backyard wooden playground set based on this reference image.",
    created_at: null,
    updated_at: null,
    viewer_asset_updated_at: null,
    provider: null,
    model_id: "test-model-b",
    agent_harness: "codex",
    has_traces: false,
    thinking_level: "ultra",
    turn_count: null,
    input_tokens: null,
    output_tokens: null,
    total_cost_usd: null,
    active_revision_id: "rev_000001",
    revision_count: 1,
    materialization_status: "available",
    materialization_displayable: true,
    materialization_validated: true,
    materialization_issues: [],
    has_compile_report: true,
    has_provenance: true,
    has_cost: false,
    ...overrides,
  };
}

describe("record list display", () => {
  it("uses the persisted Generation title ahead of the prompt", () => {
    const record = summary();

    expect(recordListPrimaryText(record, record.record_id)).toBe(
      "Backyard playground set",
    );
  });

  it("falls back to the prompt and then record id when no title exists", () => {
    expect(recordListPrimaryText(summary({ title: "" }), "rec_fallback")).toBe(
      "Create a backyard wooden playground set based on this reference image.",
    );
    expect(
      recordListPrimaryText(summary({ title: "", prompt_preview: "" }), "rec_fallback"),
    ).toBe("rec_fallback");
  });

  it("keeps only the model in list metadata", () => {
    expect(recordListMetadata(summary())).toEqual({ model: "test-model-b" });
  });

  it("does not invent a missing model", () => {
    expect(recordListMetadata(summary({ model_id: null }))).toEqual({ model: null });
  });

  it("labels persisted and missing ratings explicitly", () => {
    expect(recordRatingLabel(null)).toBe("Unrated");
    expect(recordRatingLabel(4)).toBe("4 / 5");
  });
});
