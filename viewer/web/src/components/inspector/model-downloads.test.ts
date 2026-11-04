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

import { describe, expect, it } from "vitest";

import { modelDownloadLinks } from "@/components/inspector/model-downloads";

const url = (recordId: string, filePath: string): string => `/api/records/${recordId}/files/${filePath}`;

describe("model download links", () => {
  it("offers the PhysX USD first, then the USDZ, for an available materialization", () => {
    const links = modelDownloadLinks("rec_a", "available", url);
    expect(links.map((link) => [link.key, link.url, link.fileName])).toEqual([
      ["physx", "/api/records/rec_a/files/isaac/model.usdc", "model.usdc"],
      ["usdz", "/api/records/rec_a/files/model.usdz", "model.usdz"],
    ]);
  });

  it("offers nothing without a record or an available materialization", () => {
    expect(modelDownloadLinks(null, "available", url)).toEqual([]);
    expect(modelDownloadLinks("rec_a", "contract_invalid", url)).toEqual([]);
    expect(modelDownloadLinks("rec_a", "not_validated", url)).toEqual([]);
    expect(modelDownloadLinks("rec_a", null, url)).toEqual([]);
  });
});
