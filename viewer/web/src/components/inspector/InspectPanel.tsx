// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
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

import { useEffect, useState, type JSX } from "react";

import { Download } from "lucide-react";

import { fetchRecordTextFile, recordFileUrl } from "@/lib/api";
import { useViewer } from "@/lib/viewer-context";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { JointPanel } from "@/components/inspector/JointPanel";
import { modelDownloadLinks } from "@/components/inspector/model-downloads";
import {
  inspectorInputLabel,
  type InspectorRecordAdapter,
} from "@/components/inspector/record-adapter";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";

export type InspectPanelProps = {
  urdfSpec: UrdfSpec | null;
  jointValues: Map<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetAll: () => void;
  recordAdapter?: InspectorRecordAdapter;
};

function SectionLabel({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="flex items-center gap-2 pb-2">
      <span className="text-[10px] font-medium uppercase tracking-[0.05em] text-[var(--text-tertiary)]">
        {children}
      </span>
      <div className="h-px flex-1 bg-[var(--border-subtle)]" />
    </div>
  );
}

export function InspectPanel({
  urdfSpec,
  jointValues,
  onJointChange,
  onResetAll,
  recordAdapter,
}: InspectPanelProps): JSX.Element {
  const {
    selectedRecordId: generationRecordId,
    selectedRecordSummary: generationRecordSummary,
  } = useViewer();
  const [prompt, setPrompt] = useState<string | null>(null);
  const selectedRecordId = recordAdapter?.recordId ?? generationRecordId;
  const selectedRecordSummary = recordAdapter?.summary ?? generationRecordSummary;
  const hasDeclaredRequestText = recordAdapter != null && "requestText" in recordAdapter;
  const hasDeclaredRequestFile = recordAdapter != null && "requestFilePath" in recordAdapter;
  const requestFilePath = hasDeclaredRequestFile
    ? recordAdapter.requestFilePath ?? null
    : selectedRecordSummary?.pipeline === "reconstruction"
      ? "request.txt"
      : "prompt.txt";
  const textLoader = recordAdapter?.textLoader ?? fetchRecordTextFile;

  useEffect(() => {
    if (hasDeclaredRequestText || !selectedRecordId || !requestFilePath) {
      setPrompt(null);
      return;
    }
    const controller = new AbortController();
    setPrompt(null);
    void textLoader(selectedRecordId, requestFilePath, { signal: controller.signal })
      .then((payload) => setPrompt(payload.content.trim()))
      .catch(() => setPrompt(null));
    return () => controller.abort();
  }, [hasDeclaredRequestText, requestFilePath, selectedRecordId, textLoader]);

  if (!selectedRecordId || !selectedRecordSummary) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[var(--text-quaternary)]">Select a record</p>
      </div>
    );
  }

  const promptText = hasDeclaredRequestText
    ? recordAdapter.requestText?.trim() || ""
    : prompt || selectedRecordSummary.prompt_preview || "";
  const pipelineLabel =
    selectedRecordSummary.pipeline === "reconstruction" ? "Reconstruction" : "Generation";
  const downloads = modelDownloadLinks(
    selectedRecordId,
    selectedRecordSummary.materialization_status,
    recordAdapter?.materializedFileUrl ?? recordFileUrl,
  );
  const [primaryDownload, ...otherDownloads] = downloads;

  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 pb-3">
        <section>
          <SectionLabel>{pipelineLabel}</SectionLabel>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="default">
              {inspectorInputLabel(
                selectedRecordSummary.input_kind,
                selectedRecordSummary.pipeline,
              )}
            </Badge>
          </div>
          <h2 className="mt-3 text-[14px] font-semibold leading-5 text-[var(--text-primary)]">
            {selectedRecordSummary.title}
          </h2>
          {promptText ? (
            <p className="mt-2 whitespace-pre-wrap text-[11px] leading-[1.65] text-[var(--text-secondary)]">
              {promptText}
            </p>
          ) : null}
        </section>

        <section data-model-files>
          <SectionLabel>Model files</SectionLabel>
          {primaryDownload ? (
            <div className="space-y-2">
              <a
                href={primaryDownload.url}
                download={primaryDownload.fileName}
                className="inline-flex h-8 items-center gap-1.5 rounded-md border border-[var(--border-default)] bg-[var(--surface-0)] px-3 text-[11px] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--surface-1)]"
              >
                <Download className="size-3.5" />
                Download {primaryDownload.label}
              </a>
              <p className="text-[9.5px] leading-4 text-[var(--text-quaternary)]">
                {otherDownloads.map((link, index) => (
                  <span key={link.key}>
                    {index > 0 ? " · " : "Also: "}
                    <a
                      href={link.url}
                      download={link.fileName}
                      className="underline hover:text-[var(--text-secondary)]"
                    >
                      {link.label}
                    </a>
                  </span>
                ))}
              </p>
            </div>
          ) : (
            <p className="text-[10px] leading-4 text-[var(--text-quaternary)]">
              No downloadable compiled model: the materialization is not available for this record.
            </p>
          )}
        </section>

        <section>
          <SectionLabel>Motion</SectionLabel>
          <div className="h-[360px] min-h-32">
            <JointPanel
              urdfSpec={urdfSpec}
              jointValues={jointValues}
              onJointChange={onJointChange}
              onResetAll={onResetAll}
            />
          </div>
        </section>
      </div>
    </ScrollArea>
  );
}
