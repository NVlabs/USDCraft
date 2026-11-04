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

import { buildRobotSceneGraph } from './scene-graph-builder';
import type { UrdfJoint, UrdfSpec } from './urdf-parser';
import { originToMatrix4 } from './urdf-parser';

function link(name: string): UrdfSpec['links'][number] {
  return { name, visuals: [], collisions: [] };
}

function joint(
  name: string,
  type: UrdfJoint['type'],
  parent: string,
  child: string,
  xyz: [number, number, number],
  rpy: [number, number, number] = [0, 0, 0],
): UrdfJoint {
  return { name, type, parent, child, origin: { xyz, rpy }, axis: [0, 0, 1] };
}

function worldMatrix(object: THREE.Object3D): THREE.Matrix4 {
  object.updateWorldMatrix(true, false);
  return object.matrixWorld.clone();
}

function expectMatrixClose(actual: THREE.Matrix4, expected: THREE.Matrix4): void {
  actual.elements.forEach((value, index) => {
    expect(value).toBeCloseTo(expected.elements[index], 9);
  });
}

describe('floating preview scene hierarchy', () => {
  it('preserves authored rest placement and isolates nested floating islands from ancestors', () => {
    const arm = joint('arm-hinge', 'revolute', 'base', 'arm', [0.8, -0.1, 0.3], [0.1, 0, 0.2]);
    const mount = joint('mount-fixed', 'fixed', 'arm', 'mount', [0.2, 0.4, 0.1], [0, -0.15, 0]);
    const outer = joint('outer-free', 'floating', 'mount', 'outer', [-0.1, 0.5, 0.25], [0.2, 0.1, -0.1]);
    const slide = joint('outer-slide', 'prismatic', 'outer', 'carriage', [0.35, 0, 0.2]);
    slide.axis = [1, 0, 0];
    const inner = joint('inner-free', 'floating', 'carriage', 'inner', [0.15, -0.2, 0.4], [0, 0.25, 0.1]);
    const tip = joint('inner-tip', 'revolute', 'inner', 'tip', [0, 0.1, 0.3]);

    const spec: UrdfSpec = {
      name: 'nested-floating',
      links: ['base', 'arm', 'mount', 'outer', 'carriage', 'inner', 'tip'].map(link),
      // Deliberately child-first: hierarchy semantics must not depend on joint order.
      joints: [tip, inner, slide, outer, mount, arm],
    };
    const graph = buildRobotSceneGraph(spec);

    const armRest = originToMatrix4(arm.origin);
    const mountRest = armRest.clone().multiply(originToMatrix4(mount.origin));
    const outerRest = mountRest.clone().multiply(originToMatrix4(outer.origin));
    const carriageRest = outerRest.clone().multiply(originToMatrix4(slide.origin));
    const innerRest = carriageRest.clone().multiply(originToMatrix4(inner.origin));
    const tipRest = innerRest.clone().multiply(originToMatrix4(tip.origin));

    expectMatrixClose(worldMatrix(graph.linkNodes.get('outer')!), outerRest);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('carriage')!), carriageRest);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('inner')!), innerRest);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('tip')!), tipRest);
    expect(graph.jointFrames.get('outer-free')?.parent).toBe(graph.root);
    expect(graph.jointFrames.get('inner-free')?.parent).toBe(graph.root);
    expect(graph.floatingNodes.get('outer-free')).toBe(graph.jointNodes.get('outer-free'));
    expect(graph.floatingNodes.get('inner-free')).toBe(graph.jointNodes.get('inner-free'));

    const outerWorldBefore = worldMatrix(graph.linkNodes.get('outer')!);
    const innerWorldBefore = worldMatrix(graph.linkNodes.get('inner')!);
    graph.jointNodes.get('arm-hinge')!.rotation.z = 0.7;
    graph.root.updateWorldMatrix(true, true);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('outer')!), outerWorldBefore);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('inner')!), innerWorldBefore);

    graph.jointNodes.get('outer-slide')!.position.x = 0.22;
    graph.root.updateWorldMatrix(true, true);
    expect(worldMatrix(graph.linkNodes.get('carriage')!).elements).not.toEqual(carriageRest.elements);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('inner')!), innerWorldBefore);

    const carriageAfterSlide = worldMatrix(graph.linkNodes.get('carriage')!);
    graph.floatingNodes.get('outer-free')!.position.z = 0.18;
    graph.root.updateWorldMatrix(true, true);
    expect(worldMatrix(graph.linkNodes.get('carriage')!).elements).not.toEqual(
      carriageAfterSlide.elements,
    );
    expectMatrixClose(worldMatrix(graph.linkNodes.get('inner')!), innerWorldBefore);
  });

  it('keeps the whole asset wrapper transform while leaving non-floating joints attached', () => {
    const ball = joint('portable-ball', 'spherical', 'base', 'ball', [0.3, 0.1, 0.5], [0.1, 0.2, 0]);
    const free = joint('free', 'floating', 'ball', 'payload', [0.2, -0.1, 0.4]);
    const spec: UrdfSpec = {
      name: 'wrapped-floating',
      links: ['base', 'ball', 'payload'].map(link),
      joints: [free, ball],
    };
    const graph = buildRobotSceneGraph(spec);
    const payloadRest = originToMatrix4(ball.origin).multiply(originToMatrix4(free.origin));

    expect(graph.jointFrames.get('portable-ball')?.parent).toBe(graph.linkNodes.get('base'));
    expect(graph.floatingNodes.has('portable-ball')).toBe(false);

    const wrapper = new THREE.Group();
    wrapper.position.set(2, -1, 3);
    wrapper.rotation.set(0.2, -0.3, 0.4);
    wrapper.add(graph.root);
    wrapper.updateWorldMatrix(true, true);

    const expectedWorld = wrapper.matrixWorld.clone().multiply(payloadRest);
    expectMatrixClose(worldMatrix(graph.linkNodes.get('payload')!), expectedWorld);
  });
});
