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
  describeBodyModel,
  describeFloatingMembership,
  shouldIncludePartInLegend,
} from "./PartLegend";

describe("PartLegend body model badges", () => {
  it("labels a flexible body as a static rest-pose presentation", () => {
    expect(describeBodyModel("flexible_1d", "static_rest_pose")).toEqual({
      label: "flex 1D",
      title: "flexible_1d",
      showStaticRest: true,
    });
  });

  it("does not mark a native rigid body as static", () => {
    expect(describeBodyModel("rigid", "native").showStaticRest).toBe(false);
  });
});

describe("PartLegend floating-body badges", () => {
  it("labels the direct floating child as the free island root", () => {
    expect(describeFloatingMembership("root", "removable_lid")).toEqual({
      label: "FREE",
      title: "Unconstrained 6DOF free-body island root",
    });
  });

  it("identifies constrained descendants as members of the same island", () => {
    expect(describeFloatingMembership("member", "free_carriage")).toEqual({
      label: "free member",
      title: "Member of unconstrained island rooted at free_carriage",
    });
  });

  it("keeps collision-only or logical floating parts selectable", () => {
    expect(shouldIncludePartInLegend(0, { role: "root", rootPartName: "payload" })).toBe(true);
    expect(shouldIncludePartInLegend(0, null)).toBe(false);
  });
});
