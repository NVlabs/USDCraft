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

import { queryOptions } from "@tanstack/react-query";

import {
  fetchBootstrap,
  fetchReconstructionAuthorTimeline,
  fetchReconstructionAxialGeometry,
  fetchReconstructionRecord,
  fetchReconstructionRecords,
  fetchRecordInputImage,
  fetchRecordSummary,
  fetchViewerStatus,
} from "@/lib/api";

export const viewerQueryKeys = {
  root: () => ["viewer"] as const,
  bootstrap: () => [...viewerQueryKeys.root(), "bootstrap"] as const,
  status: () => [...viewerQueryKeys.root(), "status"] as const,
  recordSummary: (recordId: string) => [...viewerQueryKeys.root(), "record", recordId] as const,
  recordInputImage: (recordId: string) =>
    [...viewerQueryKeys.root(), "record", recordId, "input-image"] as const,
  reconstructionRecord: (recordId: string) =>
    [...viewerQueryKeys.root(), "reconstruction-record", recordId] as const,
  reconstructionRecords: () =>
    [...viewerQueryKeys.root(), "reconstruction-records"] as const,
  reconstructionAuthorTimeline: (recordId: string) =>
    [...viewerQueryKeys.reconstructionRecord(recordId), "author-timeline"] as const,
  reconstructionAxialGeometry: (recordId: string) =>
    [...viewerQueryKeys.reconstructionRecord(recordId), "axial-geometry"] as const,
};

export function reconstructionAuthorTimelineQueryOptions(recordId: string) {
  return queryOptions({
    queryKey: viewerQueryKeys.reconstructionAuthorTimeline(recordId),
    queryFn: ({ signal }) => fetchReconstructionAuthorTimeline(recordId, signal),
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}

export function reconstructionAxialGeometryQueryOptions(recordId: string) {
  return queryOptions({
    queryKey: viewerQueryKeys.reconstructionAxialGeometry(recordId),
    queryFn: ({ signal }) => fetchReconstructionAxialGeometry(recordId, signal),
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}

export function viewerStatusQueryOptions() {
  return queryOptions({
    queryKey: viewerQueryKeys.status(),
    queryFn: ({ signal }) => fetchViewerStatus(signal),
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function bootstrapQueryOptions() {
  return queryOptions({
    queryKey: viewerQueryKeys.bootstrap(),
    queryFn: ({ signal }) => fetchBootstrap(signal),
    staleTime: 5_000,
  });
}

export function recordSummaryQueryOptions(recordId: string) {
  return queryOptions({
    queryKey: viewerQueryKeys.recordSummary(recordId),
    queryFn: ({ signal }) => fetchRecordSummary(recordId, signal),
    staleTime: 5_000,
  });
}

export function recordInputImageQueryOptions(recordId: string) {
  return queryOptions({
    queryKey: viewerQueryKeys.recordInputImage(recordId),
    queryFn: ({ signal }) => fetchRecordInputImage(recordId, signal),
    staleTime: 30_000,
  });
}





export function reconstructionRecordQueryOptions(recordId: string) {
  return queryOptions({
    queryKey: viewerQueryKeys.reconstructionRecord(recordId),
    queryFn: ({ signal }) => fetchReconstructionRecord(recordId, signal),
    staleTime: 30_000,
  });
}

export function reconstructionRecordsQueryOptions() {
  return queryOptions({
    queryKey: viewerQueryKeys.reconstructionRecords(),
    queryFn: ({ signal }) => fetchReconstructionRecords(signal),
    staleTime: 5_000,
  });
}
