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
import * as THREE from "three";

import { computeFit } from "@/components/viewer3d/camera-utils";
import { positionGroundHelpers } from "@/components/viewer3d/lighting";

function boundedGroup(centerX: number): THREE.Group {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(2, 2, 2),
    new THREE.MeshBasicMaterial(),
  );
  mesh.position.x = centerX;
  group.add(mesh);
  return group;
}

describe("multi-object camera and ground fitting", () => {
  it("fits the union rather than clipping an outlying comparison layer", () => {
    const source = boundedGroup(0);
    const rebuild = boundedGroup(9);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    camera.position.set(3, 3, 3);

    const fit = computeFit(camera, [source, rebuild]);

    expect(fit.target.x).toBeCloseTo(4.5);
    expect(fit.target.y).toBeCloseTo(0);
    expect(fit.target.z).toBeCloseTo(0);
    expect(fit.far).toBeGreaterThan(fit.near);

    const grid = new THREE.Group();
    const axes = new THREE.Group();
    positionGroundHelpers(grid, axes, [source, rebuild]);
    expect(grid.scale.x).toBeCloseTo(3.3);
    expect(axes.scale.x).toBeCloseTo(4.4);
  });
});
