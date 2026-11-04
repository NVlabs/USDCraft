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

import { describe, expect, it } from 'vitest';

import { resolveVisualMaterialSpec } from './materials';
import type { UrdfVisual } from './urdf-parser';

describe('visual material color contract', () => {
  it('decodes display sRGB color while preserving linear emissive RGB', () => {
    const visual: UrdfVisual = {
      geometry: { type: 'box', size: [1, 1, 1] },
      material: {
        name: 'authored_finish',
        color: { rgba: [0.5, 0.25, 0.1, 0.8] },
        pbr: {
          emissive: [0.5, 0.25, 0.1],
          clearcoat: 0.4,
        },
      },
    };

    const spec = resolveVisualMaterialSpec(visual);

    expect(spec.color.r).toBeCloseTo(0.21404114);
    expect(spec.color.g).toBeCloseTo(0.05087609);
    expect(spec.color.b).toBeCloseTo(0.01002283);
    expect(spec.opacity).toBeCloseTo(0.8);
    expect(spec.emissive?.r).toBeCloseTo(0.5);
    expect(spec.emissive?.g).toBeCloseTo(0.25);
    expect(spec.emissive?.b).toBeCloseTo(0.1);
    // A URDF may omit this value; the fallback matches the portable default the
    // SDK serializes explicitly.
    expect(spec.clearcoatRoughness).toBeCloseTo(0.2);
  });
});
