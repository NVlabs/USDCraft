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

import { excessBoxes, faceUvToWorld } from "./axial-geometry";
import type { AxialGeometry } from "./types";

const geometry: AxialGeometry = {
  record_id: "rec",
  source: "provider_events",
  grid_bounds_min_xyz_m: [-1, -0.5, 0],
  grid_bounds_max_xyz_m: [1, 0.5, 0.8],
  stack_pitch_m: 0.05,
  excess_slices: [
    { stack_slice: 4, side: "+X", excess_cells: 9, bounds_min_xyz_m: [0.9, -0.2, 0.2], bounds_max_xyz_m: [1.0, 0.2, 0.25] },
  ],
};

describe("axial geometry for the viewport overlay", () => {
  it("lifts face readings onto the envelope plane of that face", () => {
    expect(faceUvToWorld("-Y", [0.2, 0.3], geometry.grid_bounds_min_xyz_m, geometry.grid_bounds_max_xyz_m)).toEqual([0.2, -0.5, 0.3]);
    expect(faceUvToWorld("+Z", [0.0, 0.1], geometry.grid_bounds_min_xyz_m, geometry.grid_bounds_max_xyz_m)).toEqual([0.0, 0.1, 0.8]);
    expect(faceUvToWorld("+X", [-0.1, 0.4], geometry.grid_bounds_min_xyz_m, geometry.grid_bounds_max_xyz_m)).toEqual([1, -0.1, 0.4]);
    expect(faceUvToWorld("??", [0, 0], geometry.grid_bounds_min_xyz_m, geometry.grid_bounds_max_xyz_m)).toBeNull();
  });

  it("builds excess boxes", () => {
    const boxes = excessBoxes(geometry);
    expect(boxes[0].center).toEqual([0.95, 0, 0.225]);
    expect(boxes[0].size[0]).toBeCloseTo(0.1);
    expect(boxes[0].label).toBe("slice 4 · +X · 9 cells");
  });
});
