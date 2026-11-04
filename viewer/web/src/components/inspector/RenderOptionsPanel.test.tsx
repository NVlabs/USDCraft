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

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RenderOptionsPanel } from "@/components/inspector/RenderOptionsPanel";
import { defaultRenderOptions } from "@/components/viewer3d/useRenderOptions";

function renderPanel(withComparison: boolean): string {
  return renderToStaticMarkup(
    <RenderOptionsPanel
      options={defaultRenderOptions}
      onOptionChange={() => {}}
      extraSections={
        withComparison
          ? [
              {
                key: "reconstruction-overlay",
                label: "Comparison",
                content: <div>Source / candidate overlay</div>,
              },
            ]
          : undefined
      }
    />,
  );
}

describe("RenderOptionsPanel extension sections", () => {
  it("keeps the Generation controls unchanged when no extension is declared", () => {
    const html = renderPanel(false);

    expect(html).toContain("Graphics");
    expect(html).toContain("Edges");
    expect(html).toContain("Collision Geometry");
    expect(html).toContain("Preview motion");
    expect(html).toContain("Snapshot");
    expect(html).not.toContain("Source / candidate overlay");
  });

  it("renders a Reconstruction-only section without adding a shared render option", () => {
    const html = renderPanel(true);

    expect(html).toContain("Comparison");
    expect(html).toContain("Source / candidate overlay");
    expect(Object.keys(defaultRenderOptions)).not.toContain("overlay");
  });
});
