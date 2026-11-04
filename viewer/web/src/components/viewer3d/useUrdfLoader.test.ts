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
import {
  containsOwnLinkVisual,
  matchUsdVisualToAuthored,
  rebindUsdVisualMaterial,
} from './useUrdfLoader';
import type { UrdfVisual } from './urdf-parser';

function taggedVisual(linkName: string): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
  mesh.userData.usdcraftVisual = true;
  mesh.userData.usdcraftLinkName = linkName;
  return mesh;
}

function urdfVisual(material: UrdfVisual['material']): UrdfVisual {
  return {
    geometry: { type: 'box', size: [1, 1, 1] } as UrdfVisual['geometry'],
    material,
  };
}

describe('rebindUsdVisualMaterial', () => {
  it('rebinds authored usdcraft_pbr (glass transmission) onto USD meshes', () => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0x777777, opacity: 0.35, transparent: true }),
    );
    rebindUsdVisualMaterial(
      mesh,
      urdfVisual({
        name: 'smoky_glass',
        color: { rgba: [0.45, 0.45, 0.45, 0.35] },
        pbr: { transmission: 0.7, ior: 1.5 },
      }),
    );

    const material = mesh.material as THREE.MeshPhysicalMaterial;
    expect(material).toBeInstanceOf(THREE.MeshPhysicalMaterial);
    // The USD export approximates transmission as opacity; the rebound material
    // restores the authored transmission semantics.
    expect(material.transmission).toBeCloseTo(0.7);
    expect(material.ior).toBeCloseTo(1.5);
    expect(material.transparent).toBe(true);
  });

  it('rebinds metallic parameters for polished metal', () => {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: 0xcccccc }),
    );
    rebindUsdVisualMaterial(
      mesh,
      urdfVisual({
        name: 'polished_stainless',
        color: { rgba: [0.82, 0.82, 0.82, 1.0] },
        pbr: { metallic: 0.98, roughness: 0.12, clearcoat: 0.35 },
      }),
    );

    const material = mesh.material as THREE.MeshPhysicalMaterial;
    expect(material.metalness).toBeCloseTo(0.98);
    expect(material.roughness).toBeCloseTo(0.12);
    expect(material.clearcoat).toBeCloseTo(0.35);
  });

  it('keeps textured USD materials (decals carry their albedo in the texture)', () => {
    const texture = new THREE.Texture();
    const decalMaterial = new THREE.MeshStandardMaterial({ map: texture });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), decalMaterial);
    rebindUsdVisualMaterial(
      mesh,
      urdfVisual({ name: 'label', color: { rgba: [1, 0, 0, 1] } }),
    );

    expect(mesh.material).toBe(decalMaterial);
    expect((mesh.material as THREE.MeshStandardMaterial).map).toBe(texture);
  });

  it('preserves textured slots and rebinds untextured slots on a multi-material mesh', () => {
    const texture = new THREE.Texture();
    const decal = new THREE.MeshStandardMaterial({ map: texture });
    const plain = new THREE.MeshStandardMaterial({ color: 0x777777 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), [decal, plain]);

    rebindUsdVisualMaterial(
      mesh,
      urdfVisual({
        name: 'mixed',
        color: { rgba: [0.8, 0.8, 0.8, 1] },
        pbr: { metallic: 0.9, roughness: 0.2 },
      }),
    );

    const materials = mesh.material as THREE.Material[];
    expect(materials[0]).toBe(decal);
    expect((materials[0] as THREE.MeshStandardMaterial).map).toBe(texture);
    expect((materials[1] as THREE.MeshPhysicalMaterial).metalness).toBeCloseTo(0.9);
    expect((materials[1] as THREE.MeshPhysicalMaterial).roughness).toBeCloseTo(0.2);
  });

  it('leaves the USD material untouched when no descriptor is available', () => {
    const original = new THREE.MeshStandardMaterial({ color: 0x123456 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), original);
    rebindUsdVisualMaterial(mesh, undefined);
    expect(mesh.material).toBe(original);
  });

  it('preserves native USD PBR and vertex colors when the URDF omits material', () => {
    const original = new THREE.MeshStandardMaterial({
      color: 0x123456,
      metalness: 0.83,
      roughness: 0.17,
      vertexColors: true,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), original);

    rebindUsdVisualMaterial(mesh, urdfVisual(undefined));

    expect(mesh.material).toBe(original);
    expect((mesh.material as THREE.MeshStandardMaterial).metalness).toBeCloseTo(0.83);
    expect((mesh.material as THREE.MeshStandardMaterial).roughness).toBeCloseTo(0.17);
    expect((mesh.material as THREE.MeshStandardMaterial).vertexColors).toBe(true);
  });
});

describe('matchUsdVisualToAuthored', () => {
  const left = { ...urdfVisual({ name: 'left_mat' }), name: 'left panel' };
  const right = { ...urdfVisual({ name: 'right_mat' }), name: 'right-panel' };

  it('matches the uniquely mangled authored visual name', () => {
    expect(matchUsdVisualToAuthored('right_panel', [left, right])).toBe(right);
  });

  it('does not fall back to index/order when multiple names miss', () => {
    expect(matchUsdVisualToAuthored('exporter_renamed_prim', [left, right])).toBeUndefined();
  });

  it('uses the sole authored visual when no ambiguity exists', () => {
    expect(matchUsdVisualToAuthored('exporter_renamed_prim', [left])).toBe(left);
  });
});

describe('containsOwnLinkVisual', () => {
  it('matches a direct visual owned by the current link', () => {
    expect(containsOwnLinkVisual(taggedVisual('base'), 'base')).toBe(true);
  });

  it('inherits link ownership from a visual container', () => {
    const visualContainer = new THREE.Group();
    visualContainer.userData.usdcraftLinkName = 'base';
    const nestedVisual = taggedVisual('');
    delete nestedVisual.userData.usdcraftLinkName;
    visualContainer.add(nestedVisual);

    expect(containsOwnLinkVisual(visualContainer, 'base')).toBe(true);
  });

  it('does not treat an articulated child subtree as a visual of the parent link', () => {
    const jointFrame = new THREE.Group();
    jointFrame.name = 'joint-frame:wheel_axle';
    const jointMotion = new THREE.Group();
    const childLink = new THREE.Group();
    childLink.name = 'link:wheel';
    childLink.add(taggedVisual('wheel'));
    jointMotion.add(childLink);
    jointFrame.add(jointMotion);

    expect(containsOwnLinkVisual(jointFrame, 'base')).toBe(false);

    const baseLink = new THREE.Group();
    baseLink.name = 'link:base';
    baseLink.add(taggedVisual('base'), jointFrame);
    for (const child of [...baseLink.children]) {
      if (containsOwnLinkVisual(child, 'base')) {
        child.removeFromParent();
      }
    }

    expect(baseLink.getObjectByName('joint-frame:wheel_axle')).toBe(jointFrame);
    expect(baseLink.getObjectByName('link:wheel')).toBe(childLink);
  });
});
