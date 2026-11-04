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

import type { InspectorFileUrlBuilder } from "@/components/inspector/record-adapter";

/** One downloadable compiled-model file; the first entry is the default (PhysX USD). */
export type ModelDownloadLink = Readonly<{
  key: "physx" | "usdz";
  label: string;
  filePath: string;
  fileName: string;
  url: string;
}>;

const MODEL_DOWNLOADS: readonly Omit<ModelDownloadLink, "url">[] = [
  { key: "physx", label: "USD (PhysX)", filePath: "isaac/model.usdc", fileName: "model.usdc" },
  { key: "usdz", label: "USDZ", filePath: "model.usdz", fileName: "model.usdz" },
];

/**
 * Download links for a record's compiled model, or an empty list when the materialization is not
 * `available` (the Viewer never builds or repairs files, so nothing else is offered). Only the two
 * files every materialization carries are offered: the PhysX USD (default) and the shared USDZ;
 * the Newton USD is absent from records compiled before it existed.
 */
export function modelDownloadLinks(
  recordId: string | null,
  materializationStatus: string | null | undefined,
  fileUrl: InspectorFileUrlBuilder,
): readonly ModelDownloadLink[] {
  if (!recordId || materializationStatus !== "available") return [];
  return MODEL_DOWNLOADS.map((item) => ({ ...item, url: fileUrl(recordId, item.filePath) }));
}
