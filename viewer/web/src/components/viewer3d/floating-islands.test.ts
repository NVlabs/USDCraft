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

import { classifyFloatingIslands } from './floating-islands';
import type { UrdfJoint, UrdfSpec } from './urdf-parser';

function joint(
  name: string,
  type: UrdfJoint['type'],
  parent: string,
  child: string,
): UrdfJoint {
  return { name, type, parent, child };
}

function spec(linkNames: string[], joints: UrdfJoint[]): UrdfSpec {
  return {
    name: 'floating-island-test',
    links: linkNames.map((name) => ({ name, visuals: [], collisions: [] })),
    joints,
  };
}

describe('floating island classification', () => {
  it('collects fixed and constrained descendants under a floating root', () => {
    const result = classifyFloatingIslands(spec(
      ['world', 'free_root', 'fixed_child', 'hinged_child', 'slider_child'],
      [
        joint('free-placement', 'floating', 'world', 'free_root'),
        joint('weld', 'fixed', 'free_root', 'fixed_child'),
        joint('hinge', 'revolute', 'fixed_child', 'hinged_child'),
        joint('slider', 'prismatic', 'hinged_child', 'slider_child'),
      ],
    ));

    expect(result.islands).toHaveLength(1);
    expect(result.islands[0]).toMatchObject({
      id: 'free-placement',
      parentPartName: 'world',
      rootPartName: 'free_root',
      memberPartNames: ['free_root', 'fixed_child', 'hinged_child', 'slider_child'],
    });
    expect(result.membershipByPartName.get('free_root')).toEqual({
      islandId: 'free-placement',
      rootPartName: 'free_root',
      role: 'root',
    });
    expect(result.membershipByPartName.get('hinged_child')).toEqual({
      islandId: 'free-placement',
      rootPartName: 'free_root',
      role: 'member',
    });
    expect(result.membershipByPartName.has('world')).toBe(false);
  });

  it('cuts traversal at nested floating edges and creates a separate island', () => {
    const result = classifyFloatingIslands(spec(
      ['world', 'outer', 'outer_tip', 'inner', 'inner_tip'],
      [
        joint('outer-free', 'floating', 'world', 'outer'),
        joint('outer-hinge', 'continuous', 'outer', 'outer_tip'),
        joint('inner-free', 'floating', 'outer_tip', 'inner'),
        joint('inner-ball', 'spherical', 'inner', 'inner_tip'),
      ],
    ));

    expect(result.islands.map((island) => ({
      id: island.id,
      members: island.memberPartNames,
    }))).toEqual([
      { id: 'outer-free', members: ['outer', 'outer_tip'] },
      { id: 'inner-free', members: ['inner', 'inner_tip'] },
    ]);
    expect(result.membershipByPartName.get('inner')).toMatchObject({
      islandId: 'inner-free',
      role: 'root',
    });
  });

  it('does not treat a parsed portable spherical joint as floating', () => {
    const result = classifyFloatingIslands(spec(
      ['base', 'ball', 'tip'],
      [
        // resolveUrdfJointType has already converted the portable URDF fallback to spherical.
        joint('portable-ball', 'spherical', 'base', 'ball'),
        joint('tip-weld', 'fixed', 'ball', 'tip'),
      ],
    ));

    expect(result.islands).toEqual([]);
    expect(result.membershipByPartName.size).toBe(0);
  });

  it('ignores joints that reference absent links without creating phantom members', () => {
    const result = classifyFloatingIslands(spec(
      ['world', 'free_root', 'known_child'],
      [
        joint('missing-parent', 'floating', 'ghost_parent', 'free_root'),
        joint('missing-root', 'floating', 'world', 'ghost_root'),
        joint('valid-free', 'floating', 'world', 'free_root'),
        joint('known-child', 'fixed', 'free_root', 'known_child'),
        joint('missing-child', 'revolute', 'free_root', 'ghost_child'),
      ],
    ));

    expect(result.islands).toHaveLength(1);
    expect(result.islands[0].id).toBe('valid-free');
    expect(result.islands[0].memberPartNames).toEqual(['free_root', 'known_child']);
    expect(result.membershipByPartName.has('ghost_root')).toBe(false);
    expect(result.membershipByPartName.has('ghost_child')).toBe(false);
  });

  it('terminates safely when malformed constrained edges contain a cycle', () => {
    const result = classifyFloatingIslands(spec(
      ['world', 'free_root', 'child'],
      [
        joint('free', 'floating', 'world', 'free_root'),
        joint('down', 'fixed', 'free_root', 'child'),
        joint('cycle', 'fixed', 'child', 'free_root'),
      ],
    ));

    expect(result.islands[0].memberPartNames).toEqual(['free_root', 'child']);
  });
});
