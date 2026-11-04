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

import { formatSeconds, groupTimelineTurns, summarizeTimeline } from "./author-timeline";
import type { AuthorTimeline } from "./types";

const timeline: AuthorTimeline = {
  record_id: "rec",
  source: "provider_events",
  events: [
    { turn: 1, tool: "read_file", ok: true, revision: 0, tool_elapsed_seconds: 0.01, response_elapsed_seconds: 12.5, recall: null },
    { turn: 2, tool: "apply_patch", ok: true, revision: 1, tool_elapsed_seconds: 0.02, response_elapsed_seconds: 40, recall: null },
    { turn: 3, tool: "compile_model", ok: false, revision: 1, tool_elapsed_seconds: 8, response_elapsed_seconds: 9, recall: null },
    { turn: 4, tool: "compile_model", ok: true, revision: 2, tool_elapsed_seconds: 7, response_elapsed_seconds: 10, recall: null },
    { turn: 5, tool: "check_axial", ok: true, revision: 2, tool_elapsed_seconds: 3, response_elapsed_seconds: 11, recall: 0.61 },
    { turn: 7, tool: "check_axial", ok: true, revision: 3, tool_elapsed_seconds: 3, response_elapsed_seconds: 12, recall: 0.74 },
  ],
};

describe("author timeline presentation", () => {
  it("groups events by turn and keeps the failed flag and recall", () => {
    const turns = groupTimelineTurns(timeline.events);
    expect(turns.map((turn) => turn.turn)).toEqual([1, 2, 3, 4, 5, 7]);
    expect(turns[2]).toMatchObject({ tools: ["compile_model"], failed: true, revision: 1 });
    expect(turns[4].recall).toBe(0.61);
  });

  it("summarizes calls, time and the recall series", () => {
    const summary = summarizeTimeline(timeline);
    expect(summary).toMatchObject({
      turns: 6,
      toolCalls: 6,
      patches: 1,
      compiles: 2,
      checkAxialCalls: 2,
      failedCalls: 1,
      recallSeries: [0.61, 0.74],
    });
    expect(summary.responseSeconds).toBeCloseTo(94.5);
    expect(summary.toolSeconds).toBeCloseTo(21.03);
    expect(formatSeconds(94.5)).toBe("1 min 35 s");
    expect(formatSeconds(12.34)).toBe("12.3 s");
  });
});
