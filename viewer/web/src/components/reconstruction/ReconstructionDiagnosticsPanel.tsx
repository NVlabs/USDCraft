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

import { useQuery } from "@tanstack/react-query";
import { Columns2 } from "lucide-react";
import type { JSX, ReactNode } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import {
  formatSeconds,
  groupTimelineTurns,
  summarizeTimeline,
  type TimelineTurn,
} from "@/lib/author-timeline";
import { diagnosticSections, type DiagnosticRow } from "@/lib/reconstruction-diagnostics";
import type { ReconstructionRecordDetail, ReconstructionRecordSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { reconstructionAuthorTimelineQueryOptions } from "@/lib/viewer-queries";

const SELECT_CLASS =
  "h-7 w-full rounded-md border border-[var(--border-subtle)] bg-[var(--surface-1)] px-2 text-[10.5px] text-[var(--text-secondary)] outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-soft)]";

function SectionLabel({ children }: { children: ReactNode }): JSX.Element {
  return (
    <div className="flex items-center gap-2 pb-2">
      <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-[var(--text-tertiary)]">
        {children}
      </span>
      <div className="h-px flex-1 bg-[var(--border-subtle)]" />
    </div>
  );
}

function Row({ row }: { row: DiagnosticRow }): JSX.Element {
  return (
    <div className="grid grid-cols-[minmax(0,11rem)_1fr] gap-x-3 gap-y-0.5 py-1">
      <span className="text-[10px] text-[var(--text-tertiary)]">{row.label}</span>
      <span
        className={cn(
          "break-words font-mono text-[10px] leading-[1.5]",
          row.tone === "warn"
            ? "text-[var(--warning)]"
            : row.tone === "muted"
              ? "text-[var(--text-quaternary)]"
              : "text-[var(--text-primary)]",
        )}
      >
        {row.value}
      </span>
    </div>
  );
}

/** Short tool names for the per-turn row; unknown tools keep their name. */
const TOOL_SHORT: Record<string, string> = {
  apply_patch: "patch",
  compile_model: "compile",
  check_axial: "check_axial",
  read_file: "read",
  list_files: "ls",
  report_subject_scale: "scale",
  render_views: "render",
  view_image: "view",
};

function TurnRow({ turn }: { turn: TimelineTurn }): JSX.Element {
  return (
    <div
      className={cn(
        "grid grid-cols-[2.2rem_minmax(0,1fr)_3.2rem_3.6rem_3rem] items-baseline gap-x-2 py-[3px] font-mono text-[9.5px]",
        turn.failed ? "text-[var(--warning)]" : "text-[var(--text-secondary)]",
      )}
      data-timeline-turn={turn.turn}
    >
      <span className="text-[var(--text-quaternary)]">t{turn.turn}</span>
      <span className="truncate">
        {turn.tools.map((tool) => TOOL_SHORT[tool] ?? tool).join(" · ")}
        {turn.failed ? " ✕" : ""}
      </span>
      <span className="text-right tabular-nums text-[var(--text-tertiary)]">
        {turn.revision != null ? `r${turn.revision}` : ""}
      </span>
      <span className="text-right tabular-nums text-[var(--text-tertiary)]">
        {turn.responseSeconds != null ? `${turn.responseSeconds.toFixed(0)}s` : ""}
      </span>
      <span className="text-right tabular-nums">
        {turn.recall != null ? turn.recall.toFixed(2) : ""}
      </span>
    </div>
  );
}

function AuthorTimelineSection({ detail }: { detail: ReconstructionRecordDetail }): JSX.Element {
  const provider = detail.summary.agent_harness === "provider";
  const query = useQuery({
    ...reconstructionAuthorTimelineQueryOptions(detail.summary.record_id),
    enabled: provider,
  });
  let body: ReactNode;
  if (!provider) {
    body = (
      <p className="text-[9.5px] leading-4 text-[var(--text-quaternary)]">
        Codex records carry no per-turn tool trace.
      </p>
    );
  } else if (query.isPending) {
    body = <p className="text-[9.5px] text-[var(--text-quaternary)]">Loading provider_events…</p>;
  } else if (query.error) {
    body = (
      <p className="text-[9.5px] text-[var(--warning)]">
        {query.error instanceof Error ? query.error.message : "Timeline unavailable"}
      </p>
    );
  } else if (!query.data || query.data.events.length === 0) {
    body = (
      <p className="text-[9.5px] leading-4 text-[var(--text-quaternary)]">
        No tool calls were recorded in provider_events.jsonl.
      </p>
    );
  } else {
    const summary = summarizeTimeline(query.data);
    const turns = groupTimelineTurns(query.data.events);
    const recallText =
      summary.recallSeries.length > 0
        ? summary.recallSeries.map((value) => value.toFixed(2)).join(" → ")
        : "--";
    body = (
      <div className="space-y-2">
        <div className="divide-y divide-[var(--border-subtle)]">
          <Row
            row={{
              label: "Turns / tool calls",
              value: `${summary.turns} turns · ${summary.toolCalls} calls · ${summary.patches} patches · ${summary.compiles} compiles · ${summary.checkAxialCalls} check_axial`,
            }}
          />
          <Row
            row={{
              label: "Failed tool calls",
              value: String(summary.failedCalls),
              tone: summary.failedCalls > 0 ? "warn" : undefined,
            }}
          />
          <Row
            row={{
              label: "Model / tool time",
              value: `${formatSeconds(summary.responseSeconds)} · ${formatSeconds(summary.toolSeconds)}`,
            }}
          />
          <Row row={{ label: "Recall series", value: recallText }} />
        </div>
        <div className="rounded-lg border border-[var(--border-default)] bg-[var(--surface-1)] px-2 py-1">
          <div className="grid grid-cols-[2.2rem_minmax(0,1fr)_3.2rem_3.6rem_3rem] gap-x-2 border-b border-[var(--border-subtle)] pb-1 text-[8.5px] uppercase tracking-[0.06em] text-[var(--text-quaternary)]">
            <span>turn</span>
            <span>tools</span>
            <span className="text-right">rev</span>
            <span className="text-right">model</span>
            <span className="text-right">recall</span>
          </div>
          <div className="max-h-72 overflow-y-auto">
            {turns.map((turn) => (
              <TurnRow key={turn.turn} turn={turn} />
            ))}
          </div>
        </div>
      </div>
    );
  }
  return (
    <section data-author-timeline>
      <SectionLabel>Author timeline</SectionLabel>
      {body}
    </section>
  );
}

function CompareSection({
  detail,
  records,
  compareId,
  onCompareChange,
}: {
  detail: ReconstructionRecordDetail;
  records: readonly ReconstructionRecordSummary[];
  compareId: string | null;
  onCompareChange: (recordId: string | null) => void;
}): JSX.Element {
  const others = records.filter((record) => record.record_id !== detail.summary.record_id);
  const sameSource = others.filter(
    (record) =>
      record.batch_row_id != null && record.batch_row_id === detail.summary.batch_row_id,
  );
  const ordered = [...sameSource, ...others.filter((record) => !sameSource.includes(record))];
  return (
    <section data-reconstruction-compare-picker>
      <SectionLabel>Compare</SectionLabel>
      <p className="mb-1.5 text-[9.5px] leading-4 text-[var(--text-quaternary)]">
        Show another record's candidate beside this one at q0. Records sharing this batch row are
        listed first.
      </p>
      <div className="flex items-center gap-2">
        <Columns2 className="size-3 shrink-0 text-[var(--text-tertiary)]" />
        <select
          aria-label="Compare with record"
          className={SELECT_CLASS}
          value={compareId ?? ""}
          disabled={ordered.length === 0}
          onChange={(event) => onCompareChange(event.target.value || null)}
        >
          <option value="">{ordered.length === 0 ? "No other records" : "Not comparing"}</option>
          {ordered.map((record) => (
            <option key={record.record_id} value={record.record_id}>
              {(record.title || record.record_id).slice(0, 48)}
              {record.model_id ? ` · ${record.model_id}` : ""}
              {record.axial_diagnostic?.recall != null
                ? ` · recall ${record.axial_diagnostic.recall.toFixed(2)}`
                : ""}
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}

/** Diagnostics tab of the Recon Viewer: AXIAL readings, scale provenance, run stages, warnings. */
export function ReconstructionDiagnosticsPanel({
  detail,
  records = [],
  compareId = null,
  onCompareChange,
}: {
  detail: ReconstructionRecordDetail;
  records?: readonly ReconstructionRecordSummary[];
  compareId?: string | null;
  onCompareChange?: (recordId: string | null) => void;
}): JSX.Element {
  const sections = diagnosticSections(detail);
  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 px-3 pb-4 pt-3" data-reconstruction-diagnostics>
        {onCompareChange ? (
          <CompareSection
            detail={detail}
            records={records}
            compareId={compareId}
            onCompareChange={onCompareChange}
          />
        ) : null}
        {sections.map((section) => (
          <section key={section.key}>
            <SectionLabel>{section.title}</SectionLabel>
            {section.note ? (
              <p className="mb-1 text-[9.5px] leading-4 text-[var(--text-quaternary)]">
                {section.note}
              </p>
            ) : null}
            {section.rows.length > 0 ? (
              <div className="divide-y divide-[var(--border-subtle)]">
                {section.rows.map((row) => (
                  <Row key={`${section.key}-${row.label}`} row={row} />
                ))}
              </div>
            ) : null}
          </section>
        ))}
        <AuthorTimelineSection detail={detail} />
      </div>
    </ScrollArea>
  );
}
