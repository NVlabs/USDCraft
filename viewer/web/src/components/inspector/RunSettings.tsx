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

import type { JSX, ReactNode } from "react";
import { ChevronRight } from "lucide-react";

import type { RunSettingRow } from "@/lib/run-settings";
import { cn } from "@/lib/utils";

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

function SettingRow({ row }: { row: RunSettingRow }): JSX.Element {
  return (
    <div className="py-[3px]" data-run-setting={row.key}>
      <div className="grid grid-cols-[minmax(0,8.5rem)_1fr] items-baseline gap-x-3">
        <span className="text-[10px] text-[var(--text-tertiary)]">{row.label}</span>
        <span
          className={cn(
            "min-w-0 break-words text-[10.5px] leading-[1.5]",
            row.mono && "font-mono text-[10px]",
            row.tone === "warn"
              ? "text-[var(--warning)]"
              : row.tone === "muted"
                ? "text-[var(--text-tertiary)]"
                : "text-[var(--text-primary)]",
          )}
        >
          {row.value}
          {row.hint ? (
            <span className="font-mono text-[9.5px] text-[var(--text-quaternary)]">
              {" · "}
              {row.hint}
            </span>
          ) : null}
        </span>
      </div>
      {row.expand ? (
        <details className="group mt-1 rounded-md border border-[var(--border-subtle)] bg-[var(--surface-1)]">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2 py-1 text-[10px] text-[var(--text-secondary)]">
            <ChevronRight className="size-3 text-[var(--text-quaternary)] transition-transform group-open:rotate-90" />
            {row.expand.label}
          </summary>
          <p className="max-h-48 overflow-auto whitespace-pre-wrap border-t border-[var(--border-subtle)] px-2 py-1.5 text-[10.5px] leading-[1.6] text-[var(--text-secondary)]">
            {row.expand.text}
          </p>
        </details>
      ) : null}
    </div>
  );
}

/** Compact, read-only "Run settings" section of the Runtime tab (both products). */
export function RunSettings({ rows }: { rows: readonly RunSettingRow[] }): JSX.Element | null {
  if (rows.length === 0) return null;
  return (
    <section data-run-settings>
      <SectionLabel>Run settings</SectionLabel>
      <div className="divide-y divide-[var(--border-subtle)]">
        {rows.map((row) => (
          <SettingRow key={row.key} row={row} />
        ))}
      </div>
    </section>
  );
}
