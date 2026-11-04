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

import { ReconstructionOverlayControls } from "./ReconstructionOverlayControls";

const settings = {
  sourceVisible: true,
  sourceOpacity: 0.42,
  candidateVisible: true,
  candidateOpacity: 0.72,
  diagnosticsVisible: false,
};

describe("ReconstructionOverlayControls", () => {
  it("keeps comparison compact while overlay is off", () => {
    const html = renderToStaticMarkup(
      <ReconstructionOverlayControls
        enabled={false}
        settings={settings}
        sourceAvailable
        candidateAvailable
        onEnabledChange={() => {}}
        onSettingsChange={() => {}}
      />,
    );

    expect(html).toContain("Source overlay");
    expect(html).not.toContain("Normalized source<!-- --> ");
    expect(html).not.toContain("Comparison is shown only");
  });

  it("shows both q0 layer controls only when enabled", () => {
    const html = renderToStaticMarkup(
      <ReconstructionOverlayControls
        enabled
        settings={settings}
        sourceAvailable
        candidateAvailable
        onEnabledChange={() => {}}
        onSettingsChange={() => {}}
      />,
    );

    expect(html).toContain("Normalized source");
    expect(html).toContain("Reconstructed candidate");
    expect(html).toContain("42%");
    expect(html).toContain("72%");
    expect(html).toContain("Only Provider records with a check_axial call carry readings");
  });

  it("offers the check_axial geometry toggle when the record carries readings", () => {
    const html = renderToStaticMarkup(
      <ReconstructionOverlayControls
        enabled
        settings={settings}
        sourceAvailable
        candidateAvailable
        diagnosticsAvailable
        onEnabledChange={() => {}}
        onSettingsChange={() => {}}
      />,
    );

    expect(html).toContain("Exterior-excess slices as amber boxes");
    expect(html).not.toContain("Only Provider records");
  });
});
