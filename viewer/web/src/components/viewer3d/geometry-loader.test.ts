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
import * as THREE from 'three';
import { applyLoadedMeshPresentation } from './geometry-loader';
import { resolveVisualMaterialSpec } from './materials';

describe('applyLoadedMeshPresentation', () => {
  it('preserves every native GLB material slot when the URDF omits material', () => {
    const albedo = new THREE.Texture();
    const textured = new THREE.MeshStandardMaterial({
      map: albedo,
      metalness: 0.72,
      roughness: 0.28,
    });
    const vertexColored = new THREE.MeshStandardMaterial({
      metalness: 0.19,
      roughness: 0.81,
      vertexColors: true,
    });
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      [textured, vertexColored],
    );

    applyLoadedMeshPresentation(mesh, { kind: 'visual', depthBias: 2.5 });

    const materials = mesh.material as THREE.Material[];
    expect(materials[0]).toBe(textured);
    expect(materials[1]).toBe(vertexColored);
    expect((materials[0] as THREE.MeshStandardMaterial).map).toBe(albedo);
    expect((materials[0] as THREE.MeshStandardMaterial).metalness).toBeCloseTo(0.72);
    expect((materials[1] as THREE.MeshStandardMaterial).vertexColors).toBe(true);
    expect((materials[1] as THREE.MeshStandardMaterial).roughness).toBeCloseTo(0.81);
    expect(materials[0].polygonOffsetUnits).toBeCloseTo(2.5);
    expect(materials[1].polygonOffsetUnits).toBeCloseTo(2.5);
  });

  it('applies explicit URDF material per slot while retaining textured slots', () => {
    const albedo = new THREE.Texture();
    const textured = new THREE.MeshStandardMaterial({ map: albedo });
    const plain = new THREE.MeshStandardMaterial({ color: 0x112233 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), [textured, plain]);
    const materialSpec = resolveVisualMaterialSpec({
      geometry: { type: 'mesh', filename: 'source.glb' },
      material: {
        name: 'explicit_metal',
        color: { rgba: [0.8, 0.7, 0.6, 1] },
        pbr: { metallic: 0.91, roughness: 0.13 },
      },
    });

    applyLoadedMeshPresentation(mesh, { kind: 'visual', materialSpec });

    const materials = mesh.material as THREE.Material[];
    expect(materials[0]).toBe(textured);
    expect((materials[0] as THREE.MeshStandardMaterial).map).toBe(albedo);
    expect(materials[1]).toBeInstanceOf(THREE.MeshPhysicalMaterial);
    expect((materials[1] as THREE.MeshPhysicalMaterial).metalness).toBeCloseTo(0.91);
    expect((materials[1] as THREE.MeshPhysicalMaterial).roughness).toBeCloseTo(0.13);
  });
});

describe('preserved-material opacity normalization', () => {
  it('restores depth writing for fully opaque materials mislabeled as blended', () => {
    const mislabeled = new THREE.MeshStandardMaterial({ transparent: true, opacity: 1.0 });
    mislabeled.depthWrite = false;
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mislabeled);

    applyLoadedMeshPresentation(mesh, { kind: 'visual' });

    expect(mislabeled.transparent).toBe(false);
    expect(mislabeled.depthWrite).toBe(true);
    expect(mislabeled.alphaTest).toBeCloseTo(0.5);
  });

  it('keeps genuinely transparent materials transparent', () => {
    const glass = new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.35 });
    glass.depthWrite = false;
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), glass);

    applyLoadedMeshPresentation(mesh, { kind: 'visual' });

    expect(glass.transparent).toBe(true);
    expect(glass.depthWrite).toBe(false);
    expect(glass.opacity).toBeCloseTo(0.35);
  });
});
