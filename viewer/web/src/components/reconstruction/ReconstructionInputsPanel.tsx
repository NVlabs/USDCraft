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

import { Box, FileText, Image } from "lucide-react";
import type { JSX } from "react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ReconstructionRecordEvidence } from "@/lib/types";

const PRIMARY_INPUT_GROUPS = [
  {
    id: "source-mesh",
    label: "Source Mesh",
    keys: ["original_source"],
  },
  {
    id: "semantic-input-image",
    label: "Semantic Input Image",
    keys: ["semantic_image"],
  },
  {
    id: "axial",
    label: "AXIAL",
    keys: ["axial"],
  },
] as const;

const PRIMARY_INPUT_KEYS = new Set<string>(
  PRIMARY_INPUT_GROUPS.flatMap((group) => group.keys),
);

function bytesLabel(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

export function ReconstructionEvidenceCard({
  evidence,
}: {
  evidence: ReconstructionRecordEvidence;
}): JSX.Element {
  return (
    <article className="space-y-2 rounded-md border border-[var(--border-default)] bg-[var(--surface-0)] p-2.5">
      {evidence.media_kind === "image" ? (
        <a href={evidence.file_url} target="_blank" rel="noreferrer">
          <img
            src={evidence.file_url}
            alt={evidence.label}
            loading="lazy"
            className="max-h-64 w-full rounded object-contain"
          />
        </a>
      ) : null}
      <div className="flex items-start gap-2">
        {evidence.media_kind === "image" ? (
          <Image className="mt-0.5 size-3.5 shrink-0 text-[var(--accent)]" />
        ) : evidence.media_kind === "mesh" ? (
          <Box className="mt-0.5 size-3.5 shrink-0 text-[var(--text-tertiary)]" />
        ) : (
          <FileText className="mt-0.5 size-3.5 shrink-0 text-[var(--text-tertiary)]" />
        )}
        <div className="min-w-0 flex-1">
          <a
            href={evidence.file_url}
            target="_blank"
            rel="noreferrer"
            className="text-[10.5px] font-medium text-[var(--text-primary)] hover:underline"
          >
            {evidence.label}
          </a>
          <p className="mt-0.5 break-all font-mono text-[8.5px] leading-4 text-[var(--text-quaternary)]">
            {evidence.relative_path}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary">{evidence.role}</Badge>
        <Badge variant="default">{evidence.media_kind}</Badge>
        <Badge variant="default">{bytesLabel(evidence.byte_count)}</Badge>
        {typeof evidence.metadata.author_input === "boolean" ? (
          <Badge variant={evidence.metadata.author_input ? "secondary" : "default"}>
            {evidence.metadata.author_input ? "Author input" : "Host input"}
          </Badge>
        ) : null}
        {typeof evidence.metadata.authority === "string" ? (
          <Badge variant="default">{evidence.metadata.authority}</Badge>
        ) : null}
      </div>
    </article>
  );
}

function SectionLabel({ children }: { children: string }): JSX.Element {
  return (
    <div className="flex items-center gap-2 pb-2">
      <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-[var(--text-tertiary)]">
        {children}
      </span>
      <div className="h-px flex-1 bg-[var(--border-subtle)]" />
    </div>
  );
}

export function ReconstructionInputsPanel({
  inputs,
}: {
  inputs: readonly ReconstructionRecordEvidence[];
}): JSX.Element {
  const evidenceByKey = new Map<string, ReconstructionRecordEvidence>();
  for (const evidence of inputs) {
    const key = evidence.metadata.key;
    if (typeof key === "string" && PRIMARY_INPUT_KEYS.has(key) && !evidenceByKey.has(key)) {
      evidenceByKey.set(key, evidence);
    }
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 px-3 pb-4 pt-3">
        <p className="text-[9.5px] leading-4 text-[var(--text-quaternary)]">
          Three primary Reconstruction input groups are shown here. Internal preprocessing and
          audit artifacts remain in the record provenance, not in this view.
        </p>
        {PRIMARY_INPUT_GROUPS.map((group) => {
          const evidence = group.keys.flatMap((key) => {
            const item = evidenceByKey.get(key);
            return item ? [item] : [];
          });
          return (
            <section key={group.id} data-input-group={group.id}>
              <SectionLabel>{group.label}</SectionLabel>
              {evidence.length > 0 ? (
                <div className="space-y-3">
                  {evidence.map((item) => (
                    <ReconstructionEvidenceCard
                      key={item.relative_path}
                      evidence={item}
                    />
                  ))}
                </div>
              ) : (
                <p className="text-[10px] leading-4 text-[var(--text-quaternary)]">
                  Not available in this record.
                </p>
              )}
            </section>
          );
        })}
      </div>
    </ScrollArea>
  );
}
