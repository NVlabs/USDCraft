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

import {
  parsePbr,
  parseUrdfVendorCollisionElement,
  resolveUrdfBodySemantics,
  resolveUrdfCollisionSemantics,
  resolveUrdfJointType,
  summarizeUrdfCollisionGeometry,
} from './urdf-parser';

function attributes(values: Record<string, string>): Element {
  return {
    getAttribute(name: string): string | null {
      return values[name] ?? null;
    },
  } as unknown as Element;
}

function mockElement(
  values: Record<string, string> = {},
  children: Record<string, Element | null> = {},
): Element {
  return {
    getAttribute(name: string): string | null {
      return values[name] ?? null;
    },
    querySelector(selector: string): Element | null {
      return children[selector] ?? null;
    },
  } as unknown as Element;
}

describe('USDCraft URDF extensions', () => {
  it('parses scalar PBR values emitted by usdcraft_pbr', () => {
    expect(parsePbr(attributes({
      metallic: '0.92',
      roughness: '0.18',
      emissive_rgb: '0.1 0.2 0.3',
      transmission: '0.7',
      ior: '1.52',
      clearcoat: '0.4',
      clearcoat_roughness: '0.12',
    }))).toEqual({
      metallic: 0.92,
      roughness: 0.18,
      emissive: [0.1, 0.2, 0.3],
      transmission: 0.7,
      ior: 1.52,
      clearcoat: 0.4,
      clearcoatRoughness: 0.12,
    });
  });

  it('restores a spherical joint from portable floating URDF metadata', () => {
    expect(resolveUrdfJointType('floating', 'spherical')).toBe('spherical');
    expect(resolveUrdfJointType('revolute', null)).toBe('revolute');
  });

  it('reads typed non-rigid body semantics and keeps untyped links rigid', () => {
    expect(resolveUrdfBodySemantics('flexible_1d', 'static_rest_pose')).toEqual({
      bodyModel: 'flexible_1d',
      bodyRealization: 'static_rest_pose',
    });
    expect(resolveUrdfBodySemantics(null, null)).toEqual({
      bodyModel: 'rigid',
      bodyRealization: 'native',
    });
  });

  it('classifies vendor collisions as deformable rest-shape geometry', () => {
    expect(resolveUrdfCollisionSemantics(
      'deformable_contact',
      'rest_shape',
      'flexible_1d',
      'flexible_1d',
    )).toEqual({
      collisionRole: 'deformable_contact',
      collisionRealization: 'rest_shape',
      bodyModel: 'flexible_1d',
    });

    expect(resolveUrdfCollisionSemantics(
      'deformable_contact',
      'rest_shape',
      null,
      'surface_deformable',
    )).toEqual({
      collisionRole: 'deformable_contact',
      collisionRealization: 'rest_shape',
      bodyModel: 'surface_deformable',
    });
  });

  it('rejects malformed or contradictory vendor collision semantics', () => {
    expect(resolveUrdfCollisionSemantics(
      null,
      'rest_shape',
      null,
      'flexible_1d',
    )).toBeNull();
    expect(resolveUrdfCollisionSemantics(
      'deformable_contact',
      null,
      null,
      'flexible_1d',
    )).toBeNull();
    expect(resolveUrdfCollisionSemantics(
      'deformable_contact',
      'rest_shape',
      'surface_deformable',
      'flexible_1d',
    )).toBeNull();
    expect(resolveUrdfCollisionSemantics(
      'deformable_contact',
      'rest_shape',
      null,
      'rigid',
    )).toBeNull();
  });

  it('parses a realistic deformable collision element with origin and geometry', () => {
    const origin = mockElement({ xyz: '0.1 0.2 0.3', rpy: '0 1.57 0' });
    const box = mockElement({ size: '0.4 0.05 0.05' });
    const geometry = mockElement({}, { box });
    const vendorCollision = mockElement(
      {
        name: 'hose_rest_shape',
        role: 'deformable_contact',
        realization: 'rest_shape',
        body_model: 'flexible_1d',
      },
      { origin, geometry },
    );

    expect(parseUrdfVendorCollisionElement(vendorCollision, 'flexible_1d')).toEqual({
      name: 'hose_rest_shape',
      origin: {
        xyz: [0.1, 0.2, 0.3],
        rpy: [0, 1.57, 0],
      },
      geometry: {
        type: 'box',
        size: [0.4, 0.05, 0.05],
      },
      material: undefined,
      collisionRole: 'deformable_contact',
      collisionRealization: 'rest_shape',
      bodyModel: 'flexible_1d',
    });
    expect(parseUrdfVendorCollisionElement(vendorCollision, 'surface_deformable')).toBeNull();
  });

  it('counts rigid runtime and deformable rest-shape collisions separately', () => {
    expect(summarizeUrdfCollisionGeometry({
      name: 'collision-semantics',
      joints: [],
      links: [{
        name: 'part',
        visuals: [],
        collisions: [
          {
            geometry: { type: 'box', size: [1, 1, 1] },
            collisionRole: 'rigid_runtime',
            collisionRealization: 'runtime',
            bodyModel: 'rigid',
          },
          {
            geometry: { type: 'mesh', filename: 'hose.glb' },
            collisionRole: 'deformable_contact',
            collisionRealization: 'rest_shape',
            bodyModel: 'flexible_1d',
          },
        ],
      }],
    })).toEqual({
      declaredCount: 2,
      rigidRuntimeCount: 1,
      deformableRestShapeCount: 1,
      missingDeformableRestShapePartCount: 0,
      missingDeformableRestShapePartNames: [],
      primitiveCount: 1,
      meshCount: 1,
    });
  });

  it('reports a flexible visual part missing rest-shape collision amid rigid coverage', () => {
    expect(summarizeUrdfCollisionGeometry({
      name: 'partial-collision-coverage',
      joints: [],
      links: [
        {
          name: 'rigid_body',
          bodyModel: 'rigid',
          bodyRealization: 'native',
          visuals: [{ geometry: { type: 'box', size: [1, 1, 1] } }],
          collisions: [{
            geometry: { type: 'box', size: [1, 1, 1] },
            collisionRole: 'rigid_runtime',
            collisionRealization: 'runtime',
            bodyModel: 'rigid',
          }],
        },
        {
          name: 'hose',
          bodyModel: 'flexible_1d',
          bodyRealization: 'static_rest_pose',
          visuals: [{ geometry: { type: 'cylinder', radius: 0.02, length: 0.5 } }],
          collisions: [],
        },
      ],
    })).toEqual({
      declaredCount: 1,
      rigidRuntimeCount: 1,
      deformableRestShapeCount: 0,
      missingDeformableRestShapePartCount: 1,
      missingDeformableRestShapePartNames: ['hose'],
      primitiveCount: 1,
      meshCount: 0,
    });
  });
});
