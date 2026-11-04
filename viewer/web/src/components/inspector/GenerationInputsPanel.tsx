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
import { Image as ImageIcon } from "lucide-react";
import type { JSX, ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { recordFileUrl } from "@/lib/api";
import { recordInputImageQueryOptions } from "@/lib/viewer-queries";

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

export type GenerationInputsPanelContentProps = {
  recordId: string | null;
  /** `inputs/<file>` relative to the record file endpoint, null for a text-only request. */
  imagePath: string | null;
  loading?: boolean;
};

/** The Generation Evidence tab: the one declared inspiration image, or the text-only statement. */
export function GenerationInputsPanelContent({
  recordId,
  imagePath,
  loading = false,
}: GenerationInputsPanelContentProps): JSX.Element {
  if (!recordId) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[var(--text-quaternary)]">Select a record</p>
      </div>
    );
  }
  const imageUrl = imagePath ? recordFileUrl(recordId, imagePath) : null;
  return (
    <ScrollArea className="h-full">
      <div className="space-y-5 px-3 pb-4 pt-3">
        <p className="text-[9.5px] leading-4 text-[var(--text-quaternary)]">
          Generation evidence is the user's request: its text is shown under Model, and the one
          inspiration image, when supplied, is shown here.
        </p>
        <section data-input-group="inspiration-image">
          <SectionLabel>Inspiration Image</SectionLabel>
          {loading ? (
            <p className="text-[10px] leading-4 text-[var(--text-quaternary)]">Loading...</p>
          ) : imageUrl && imagePath ? (
            <article className="space-y-2 rounded-md border border-[var(--border-default)] bg-[var(--surface-0)] p-2.5">
              <a href={imageUrl} target="_blank" rel="noreferrer">
                <img
                  src={imageUrl}
                  alt="Generation inspiration image"
                  loading="lazy"
                  className="max-h-64 w-full rounded object-contain"
                />
              </a>
              <div className="flex items-start gap-2">
                <ImageIcon className="mt-0.5 size-3.5 shrink-0 text-[var(--accent)]" />
                <div className="min-w-0 flex-1">
                  <a
                    href={imageUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10.5px] font-medium text-[var(--text-primary)] hover:underline"
                  >
                    Inspiration image
                  </a>
                  <p className="mt-0.5 break-all font-mono text-[8.5px] leading-4 text-[var(--text-quaternary)]">
                    {imagePath}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="secondary">user input</Badge>
                <Badge variant="default">image</Badge>
              </div>
            </article>
          ) : (
            <p className="text-[10px] leading-4 text-[var(--text-quaternary)]">
              Text-only request: no image was supplied for this record.
            </p>
          )}
        </section>
      </div>
    </ScrollArea>
  );
}

export function GenerationInputsPanel({ recordId }: { recordId: string | null }): JSX.Element {
  const query = useQuery({
    ...recordInputImageQueryOptions(recordId ?? ""),
    enabled: Boolean(recordId),
  });
  return (
    <GenerationInputsPanelContent
      recordId={recordId}
      imagePath={query.data?.path ?? null}
      loading={Boolean(recordId) && query.isPending}
    />
  );
}
