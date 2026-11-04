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

import type { AuthorTimeline, AuthorTimelineEvent } from "@/lib/types";

export type TimelineTurn = {
  turn: number;
  tools: string[];
  failed: boolean;
  revision: number | null;
  responseSeconds: number | null;
  toolSeconds: number;
  recall: number | null;
};

/** Group tool events by Author turn; one row per turn keeps the list readable for 30–80 turns. */
export function groupTimelineTurns(events: readonly AuthorTimelineEvent[]): TimelineTurn[] {
  const byTurn = new Map<number, TimelineTurn>();
  for (const event of events) {
    const row = byTurn.get(event.turn) ?? {
      turn: event.turn,
      tools: [],
      failed: false,
      revision: null,
      responseSeconds: null,
      toolSeconds: 0,
      recall: null,
    };
    row.tools.push(event.tool);
    row.failed = row.failed || !event.ok;
    row.revision = event.revision ?? row.revision;
    row.responseSeconds = event.response_elapsed_seconds ?? row.responseSeconds;
    row.toolSeconds += event.tool_elapsed_seconds ?? 0;
    if (event.recall != null) row.recall = event.recall;
    byTurn.set(event.turn, row);
  }
  return [...byTurn.values()].sort((a, b) => a.turn - b.turn);
}

export type TimelineSummary = {
  turns: number;
  toolCalls: number;
  patches: number;
  compiles: number;
  checkAxialCalls: number;
  failedCalls: number;
  responseSeconds: number;
  toolSeconds: number;
  recallSeries: number[];
};

export function summarizeTimeline(timeline: AuthorTimeline): TimelineSummary {
  const turns = groupTimelineTurns(timeline.events);
  return {
    turns: turns.length,
    toolCalls: timeline.events.length,
    patches: timeline.events.filter((event) => event.tool === "apply_patch").length,
    compiles: timeline.events.filter((event) => event.tool === "compile_model").length,
    checkAxialCalls: timeline.events.filter((event) => event.tool === "check_axial").length,
    failedCalls: timeline.events.filter((event) => !event.ok).length,
    responseSeconds: turns.reduce((total, turn) => total + (turn.responseSeconds ?? 0), 0),
    toolSeconds: turns.reduce((total, turn) => total + turn.toolSeconds, 0),
    recallSeries: timeline.events
      .map((event) => event.recall)
      .filter((value): value is number => value != null),
  };
}

export function formatSeconds(value: number | null): string {
  if (value == null) return "--";
  if (value < 90) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  return `${minutes} min ${Math.round(value - minutes * 60)} s`;
}
