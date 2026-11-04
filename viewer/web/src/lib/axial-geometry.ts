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

import type { AxialGeometry } from "@/lib/types";

/** Face name → (normal axis index, sign, u axis index, v axis index) in recon_world (Z up). */
const FACE_AXES: Record<string, [number, number, number, number]> = {
  "-Y": [1, -1, 0, 2],
  "+Y": [1, +1, 0, 2],
  "-X": [0, -1, 1, 2],
  "+X": [0, +1, 1, 2],
  "+Z": [2, +1, 0, 1],
  "-Z": [2, -1, 0, 1],
};

export type Point3 = [number, number, number];

/** Lift a face (u, v) reading onto the grid envelope plane of that face, in recon_world metres. */
export function faceUvToWorld(
  face: string,
  uv: readonly [number, number] | readonly number[],
  boundsMin: readonly number[],
  boundsMax: readonly number[],
): Point3 | null {
  const axes = FACE_AXES[face];
  if (!axes || uv.length < 2) return null;
  const [normal, sign, uAxis, vAxis] = axes;
  const point: Point3 = [0, 0, 0];
  point[normal] = sign > 0 ? boundsMax[normal] : boundsMin[normal];
  point[uAxis] = uv[0];
  point[vAxis] = uv[1];
  return point;
}

export type ExcessBox3 = { center: Point3; size: Point3; label: string };

export function excessBoxes(geometry: AxialGeometry): ExcessBox3[] {
  return geometry.excess_slices.map((slice) => {
    const lo = slice.bounds_min_xyz_m;
    const hi = slice.bounds_max_xyz_m;
    return {
      center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
      size: [
        Math.max(hi[0] - lo[0], 0.002),
        Math.max(hi[1] - lo[1], 0.002),
        Math.max(hi[2] - lo[2], 0.002),
      ],
      label: `slice ${slice.stack_slice} · ${slice.side} · ${slice.excess_cells} cells`,
    };
  });
}
