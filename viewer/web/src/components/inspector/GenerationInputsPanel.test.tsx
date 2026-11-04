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

import { GenerationInputsPanelContent } from "@/components/inspector/GenerationInputsPanel";

describe("GenerationInputsPanelContent", () => {
  it("shows the inspiration image as evidence with its record file link", () => {
    const html = renderToStaticMarkup(
      <GenerationInputsPanelContent recordId="rec_a" imagePath="inputs/inspiration.png" />,
    );
    expect(html).toContain("Inspiration Image");
    expect(html).toContain('src="/api/records/rec_a/files/inputs/inspiration.png"');
    expect(html).toContain("inputs/inspiration.png");
    expect(html).not.toContain("Text-only request");
  });

  it("states a text-only request instead of an empty card", () => {
    const html = renderToStaticMarkup(<GenerationInputsPanelContent recordId="rec_a" imagePath={null} />);
    expect(html).toContain("Text-only request");
    expect(html).not.toContain("<img");
  });

  it("asks for a record when none is selected", () => {
    const html = renderToStaticMarkup(<GenerationInputsPanelContent recordId={null} imagePath={null} />);
    expect(html).toContain("Select a record");
  });
});
