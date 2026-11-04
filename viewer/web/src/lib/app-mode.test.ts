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

import { appModeFromSearch, searchWithAppMode } from "@/lib/app-mode";

describe("app mode URL state", () => {
  it("defaults unknown and absent values to the Generation viewer", () => {
    expect(appModeFromSearch("?record=rec_1")).toBe("generation");
    expect(appModeFromSearch("?mode=unknown")).toBe("generation");
  });

  it("selects the canonical Recon Viewer", () => {
    expect(appModeFromSearch("?mode=reconstruction")).toBe("reconstruction");
  });

  it("changes only mode and preserves record browsing state", () => {
    const original = "?record=rec_1&q=lamp&pose=%7B%7D";
    const reconstruction = searchWithAppMode(original, "reconstruction");
    expect(reconstruction).toContain("record=rec_1");
    expect(reconstruction).toContain("q=lamp");
    expect(reconstruction).toContain("pose=%7B%7D");
    expect(reconstruction).toContain("mode=reconstruction");

    expect(searchWithAppMode(reconstruction, "generation")).toBe(original);
  });

  it("clears mode-specific state when changing products", () => {
    expect(
      searchWithAppMode("?mode=reconstruction&recon-record=rec_reconstruction_001&record=rec_1", "generation"),
    ).toBe("?record=rec_1");
  });
});
