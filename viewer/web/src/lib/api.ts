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

import type {
  AuthorTimeline,
  AxialGeometry,
  DeleteRecordResult,
  ReconstructionRecordDetail,
  ReconstructionRecordSummary,
  RecordInputImage,
  RecordSummary,
  RecordTextFile,
  ViewerBootstrap,
  ViewerStatus,
} from "@/lib/types";

export class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string" && payload.detail) {
        message = payload.detail;
      } else if (
        payload.detail &&
        typeof payload.detail === "object" &&
        "message" in payload.detail &&
        typeof payload.detail.message === "string"
      ) {
        message = payload.detail.message;
      }
    } catch {
      // Keep the status-based message for non-JSON responses.
    }
    throw new HttpError(response.status, message);
  }
  return (await response.json()) as T;
}

export function fetchViewerStatus(signal?: AbortSignal): Promise<ViewerStatus> {
  return fetchJson<ViewerStatus>("/api/status", { signal });
}

export function fetchBootstrap(signal?: AbortSignal): Promise<ViewerBootstrap> {
  return fetchJson<ViewerBootstrap>("/api/bootstrap", { signal });
}

export function fetchRecordSummary(recordId: string, signal?: AbortSignal): Promise<RecordSummary> {
  return fetchJson<RecordSummary>(
    `/api/records/${encodeURIComponent(recordId)}/summary`,
    { signal },
  );
}

export function saveRecordRating(
  recordId: string,
  rating: number | null,
): Promise<RecordSummary> {
  return fetchJson<RecordSummary>(
    `/api/records/${encodeURIComponent(recordId)}/rating`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    },
  );
}

export function deleteRecord(recordId: string): Promise<DeleteRecordResult> {
  return fetchJson<DeleteRecordResult>(`/api/records/${encodeURIComponent(recordId)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm_record_id: recordId }),
  });
}

export function fetchReconstructionAuthorTimeline(
  recordId: string,
  signal?: AbortSignal,
): Promise<AuthorTimeline> {
  return fetchJson<AuthorTimeline>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}/author-timeline`,
    { signal },
  );
}

export function fetchReconstructionAxialGeometry(
  recordId: string,
  signal?: AbortSignal,
): Promise<AxialGeometry> {
  return fetchJson<AxialGeometry>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}/axial-geometry`,
    { signal },
  );
}

export function saveReconstructionRecordRating(
  recordId: string,
  rating: number | null,
): Promise<ReconstructionRecordSummary> {
  return fetchJson<ReconstructionRecordSummary>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}/rating`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    },
  );
}

export function deleteReconstructionRecord(recordId: string): Promise<DeleteRecordResult> {
  return fetchJson<DeleteRecordResult>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_record_id: recordId }),
    },
  );
}

export function fetchRecordInputImage(
  recordId: string,
  signal?: AbortSignal,
): Promise<RecordInputImage> {
  return fetchJson<RecordInputImage>(
    `/api/records/${encodeURIComponent(recordId)}/input-image`,
    { signal },
  );
}

export function fetchRecordTextFile(
  recordId: string,
  filePath: string,
  options: { previewBytes?: number; full?: boolean; signal?: AbortSignal } = {},
): Promise<RecordTextFile> {
  const params = new URLSearchParams();
  if (options.previewBytes != null) params.set("preview_bytes", String(options.previewBytes));
  if (options.full) params.set("full", "true");
  const query = params.size > 0 ? `?${params.toString()}` : "";
  return fetchJson<RecordTextFile>(
    `/api/records/${encodeURIComponent(recordId)}/text/${filePath}${query}`,
    { signal: options.signal },
  );
}

export function recordFileUrl(recordId: string, filePath: string): string {
  return `/api/records/${encodeURIComponent(recordId)}/files/${filePath}`;
}



export function fetchReconstructionRecords(
  signal?: AbortSignal,
): Promise<ReconstructionRecordSummary[]> {
  return fetchJson<ReconstructionRecordSummary[]>("/api/reconstruction-records", { signal });
}

export function fetchReconstructionRecord(
  recordId: string,
  signal?: AbortSignal,
): Promise<ReconstructionRecordDetail> {
  return fetchJson<ReconstructionRecordDetail>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}`,
    { signal },
  );
}

export function fetchReconstructionRecordTextFile(
  recordId: string,
  filePath: string,
  options: { previewBytes?: number; full?: boolean; signal?: AbortSignal } = {},
): Promise<RecordTextFile> {
  const params = new URLSearchParams();
  if (options.previewBytes != null) params.set("preview_bytes", String(options.previewBytes));
  if (options.full) params.set("full", "true");
  const query = params.size > 0 ? `?${params.toString()}` : "";
  return fetchJson<RecordTextFile>(
    `/api/reconstruction-records/${encodeURIComponent(recordId)}/text/${filePath}${query}`,
    { signal: options.signal },
  );
}
