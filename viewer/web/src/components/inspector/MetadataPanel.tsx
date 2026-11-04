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

import { useEffect, useState, type JSX, type ReactNode } from "react";
import { FileJson } from "lucide-react";

import {
  buildInspectorMetadataPresentation,
  type InspectorMetadataPresentation,
  type InspectorRecordAdapter,
  type InspectorTextLoader,
} from "@/components/inspector/record-adapter";
import { RunSettings } from "@/components/inspector/RunSettings";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fetchRecordTextFile } from "@/lib/api";
import { useViewer } from "@/lib/viewer-context";

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

function Property({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
}): JSX.Element {
  return (
    <div className="prop-row-stacked">
      <span className="prop-label">{label}</span>
      <span className={`prop-value break-words ${mono ? "font-mono text-[10px]" : ""}`}>
        {value}
      </span>
    </div>
  );
}

function formatDate(value: string | null): string {
  if (!value) return "--";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatCost(value: number | null): string {
  return value == null ? "--" : `$${value.toFixed(4)}`;
}

function formatTokens(value: number | null): string {
  return value == null ? "--" : value.toLocaleString();
}

function RawArtifact({
  recordId,
  filePath,
  label,
  enabled,
  textLoader,
}: {
  recordId: string;
  filePath: string;
  label: string;
  enabled: boolean;
  textLoader: InspectorTextLoader;
}): JSX.Element | null {
  const [content, setContent] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setContent(null);
      return;
    }
    const controller = new AbortController();
    void textLoader(recordId, filePath, { previewBytes: 65536, signal: controller.signal })
      .then((payload) => setContent(payload.content))
      .catch(() => setContent(null));
    return () => controller.abort();
  }, [enabled, filePath, recordId, textLoader]);

  if (!enabled) return null;
  return (
    <details className="rounded-md border border-[var(--border-subtle)] bg-[var(--surface-1)]">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-2 text-[10.5px] text-[var(--text-secondary)]">
        <FileJson className="size-3 text-[var(--text-quaternary)]" />
        {label}
      </summary>
      <pre className="max-h-64 overflow-auto border-t border-[var(--border-subtle)] p-2.5 font-mono text-[9.5px] leading-4 text-[var(--text-tertiary)]">
        {content ?? "Unavailable"}
      </pre>
    </details>
  );
}

export function MetadataPanelContent({
  presentation,
  textLoader,
}: {
  presentation: InspectorMetadataPresentation;
  textLoader: InspectorTextLoader;
}): JSX.Element {
  const hasAuxiliaryMetadata =
    presentation.artifacts.some((artifact) => artifact.available)
    || presentation.tracesAvailable;

  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 pb-3">
        <section>
          <SectionLabel>Identity</SectionLabel>
          <div className="mb-2 flex flex-wrap gap-1.5">
            <Badge variant="secondary">{presentation.pipeline}</Badge>
            {presentation.status ? <Badge variant="success">{presentation.status}</Badge> : null}
            <Badge variant="default">{presentation.inputLabel}</Badge>
          </div>
          <Property label="Record ID" value={presentation.recordId} mono />
          <Property label="Active revision" value={presentation.activeRevisionId ?? "--"} mono />
          <Property label="Created" value={formatDate(presentation.createdAt)} />
          <Property label="Updated" value={formatDate(presentation.updatedAt)} />
        </section>

        <RunSettings rows={presentation.runSettings} />

        <section>
          <SectionLabel>{presentation.runtimeHeading}</SectionLabel>
          <Property label="Provider" value={presentation.provider ?? "--"} />
          <Property label="Input tokens" value={formatTokens(presentation.inputTokens)} />
          <Property label="Output tokens" value={formatTokens(presentation.outputTokens)} />
          <Property label="Cost" value={formatCost(presentation.totalCostUsd)} mono />
        </section>

        <section>
          <SectionLabel>Read-only Artifacts</SectionLabel>
          <Property
            label="Materialization"
            value={
              presentation.materializationStatus
              ?? (presentation.pipeline === "generation" ? "missing" : "--")
            }
          />
          <div className="mt-2 space-y-2">
            {presentation.artifacts.map((artifact) => (
              <RawArtifact
                key={artifact.filePath}
                recordId={presentation.recordId}
                filePath={artifact.filePath}
                label={artifact.label}
                enabled={artifact.available}
                textLoader={textLoader}
              />
            ))}
            <Property
              label="Authoring traces"
              value={presentation.tracesAvailable ? "available" : "unavailable"}
            />
            {!hasAuxiliaryMetadata ? (
              <p className="flex items-center gap-2 text-[10.5px] text-[var(--text-quaternary)]">
                <FileJson className="size-3" /> No auxiliary metadata files
              </p>
            ) : null}
          </div>
        </section>
      </div>
    </ScrollArea>
  );
}

export type MetadataPanelProps = {
  recordAdapter?: InspectorRecordAdapter;
};

export function MetadataPanel({ recordAdapter }: MetadataPanelProps = {}): JSX.Element {
  const {
    selectedRecordId: generationRecordId,
    selectedRecordSummary: generationRecordSummary,
  } = useViewer();
  const selectedRecordId = recordAdapter?.recordId ?? generationRecordId;
  const record = recordAdapter?.summary ?? generationRecordSummary;

  if (!selectedRecordId || !record) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[var(--text-quaternary)]">Select a record</p>
      </div>
    );
  }

  const generatedPresentation = buildInspectorMetadataPresentation(record);
  const presentation =
    recordAdapter != null && "metadataPresentation" in recordAdapter
      ? recordAdapter.metadataPresentation
      : generatedPresentation;

  if (!presentation) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[var(--text-quaternary)]">Select a record</p>
      </div>
    );
  }

  return (
    <MetadataPanelContent
      presentation={presentation}
      textLoader={recordAdapter?.textLoader ?? fetchRecordTextFile}
    />
  );
}
