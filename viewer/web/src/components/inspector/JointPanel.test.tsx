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
import { describe, expect, it, vi } from "vitest";

import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";

import { JointPanel } from "./JointPanel";

const spec: UrdfSpec = {
  name: "floating-demo",
  links: [
    { name: "base", visuals: [], collisions: [] },
    { name: "free_body", visuals: [], collisions: [] },
    { name: "free_body_lid", visuals: [], collisions: [] },
  ],
  joints: [
    {
      name: "free_placement",
      type: "floating",
      parent: "base",
      child: "free_body",
    },
    {
      name: "lid_hinge",
      type: "revolute",
      parent: "free_body",
      child: "free_body_lid",
      limit: { lower: 0, upper: 1.2 },
    },
  ],
};

describe("JointPanel motion grouping", () => {
  it("separates constrained joints from free-body islands", () => {
    const html = renderToStaticMarkup(
      <JointPanel
        urdfSpec={spec}
        jointValues={new Map()}
        onJointChange={vi.fn()}
        onResetAll={vi.fn()}
      />,
    );

    expect(html).toContain("Constrained joints");
    expect(html).toContain("Free bodies");
    expect(html).toContain("FREE · 6DOF");
    expect(html).toContain("Unconstrained placement · no physical joint");
    expect(html).toContain("free_body");
    expect(html).toContain("base");
    expect(html).toContain("2 parts in island");
  });
});
