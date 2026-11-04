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

import { applyJointValue } from './useJointController';
import type { UrdfJoint } from './urdf-parser';

function joint(type: UrdfJoint['type']): UrdfJoint {
  return {
    name: `${type}_joint`,
    type,
    parent: 'base',
    child: 'moving',
  };
}

describe('joint controller transforms', () => {
  it.each(['revolute', 'continuous'] as const)('applies %s rotation around its axis', (type) => {
    const node = new THREE.Object3D();
    const axis = new THREE.Vector3(0, 0, 1);
    applyJointValue(node, joint(type), axis, Math.PI / 2);

    const rotated = new THREE.Vector3(1, 0, 0).applyQuaternion(node.quaternion);
    expect(rotated.x).toBeCloseTo(0);
    expect(rotated.y).toBeCloseTo(1);
  });

  it('applies prismatic travel along its axis', () => {
    const node = new THREE.Object3D();
    applyJointValue(node, joint('prismatic'), new THREE.Vector3(1, 0, 0), 0.25);
    expect(node.position.toArray()).toEqual([0.25, 0, 0]);
  });

  it('keeps spherical joints static until a 3-axis controller exists', () => {
    const node = new THREE.Object3D();
    applyJointValue(node, joint('spherical'), new THREE.Vector3(0, 0, 1), 0.8);
    expect(node.position.toArray()).toEqual([0, 0, 0]);
    expect(node.quaternion.toArray()).toEqual([0, 0, 0, 1]);
  });
});
