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

import { OverlayLegend } from "./OverlayLegend";

describe("OverlayLegend", () => {
  it("shows only the free-body explanation when only floating bodies are enabled", () => {
    const html = renderToStaticMarkup(
      <OverlayLegend
        onClose={() => undefined}
        showJointOverlay={false}
        showFloatingBodies
      />,
    );

    expect(html).toContain("Free bodies");
    expect(html).toContain("FREE");
    expect(html).toContain("Unconstrained 6DOF · no physical joint");
    expect(html).toContain(
      "Select a FREE part + Preview motion for one-shot lift / wiggle / return",
    );
    expect(html).toContain("Kinematic review gesture · not dynamics");
    expect(html).not.toContain("Constrained joints");
    expect(html).not.toContain("REV");
  });

  it("shows constrained joints and free bodies when both overlays are enabled", () => {
    const html = renderToStaticMarkup(
      <OverlayLegend onClose={() => undefined} showJointOverlay showFloatingBodies />,
    );

    expect(html).toContain("Constrained joints");
    expect(html).toContain("Free bodies");
  });

  it("shows only the constrained-joint presentation by default", () => {
    const html = renderToStaticMarkup(<OverlayLegend onClose={() => undefined} />);

    expect(html).toContain("Constrained joints");
    expect(html).not.toContain("Free bodies");
  });
});
