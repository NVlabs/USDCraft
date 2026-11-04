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

import {
  DEFAULT_RECONSTRUCTION_INSPECTOR_TAB,
  resolveReconstructionViewport,
} from "@/lib/reconstruction-viewport";
import type { ReconstructionInspectorTab } from "@/lib/types";

describe("Recon Viewer viewport selection", () => {
  it("opens on Inspect", () => {
    expect(DEFAULT_RECONSTRUCTION_INSPECTOR_TAB).toBe("inspect");
  });

  it.each<ReconstructionInspectorTab>(["inspect", "code", "inputs", "metadata"])(
    "keeps the compiled candidate visible on the %s tab",
    (activeTab) => {
      expect(
        resolveReconstructionViewport({
          activeTab,
          overlayEnabled: true,
          sourceAvailable: true,
          candidateAvailable: true,
        }),
      ).toBe("candidate");
    },
  );

  it("uses overlay only when the Render option is enabled and both layers are available", () => {
    expect(
      resolveReconstructionViewport({
        activeTab: "render",
        overlayEnabled: true,
        sourceAvailable: true,
        candidateAvailable: true,
      }),
    ).toBe("overlay");

    for (const unavailable of ["source", "candidate"] as const) {
      expect(
        resolveReconstructionViewport({
          activeTab: "render",
          overlayEnabled: true,
          sourceAvailable: unavailable !== "source",
          candidateAvailable: unavailable !== "candidate",
        }),
      ).toBe("candidate");
    }
  });

  it("defaults Render to the compiled candidate", () => {
    expect(
      resolveReconstructionViewport({
        activeTab: "render",
        overlayEnabled: false,
        sourceAvailable: true,
        candidateAvailable: true,
      }),
    ).toBe("candidate");
  });
});
