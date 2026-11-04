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

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import type { FloatingIsland } from './floating-islands';
import { attachFloatingBodyOverlay, measureFloatingGeometryExtent } from './floating-overlay';

function island(): FloatingIsland {
  return {
    id: 'free-placement',
    joint: {
      name: 'free-placement',
      type: 'floating',
      parent: 'base',
      child: 'free-body',
    },
    parentPartName: 'base',
    rootPartName: 'free-body',
    memberPartNames: ['free-body'],
  };
}

function sceneFixture(): {
  scene: THREE.Scene;
  robot: THREE.Group;
  jointFrames: Map<string, THREE.Group>;
} {
  const scene = new THREE.Scene();
  const robot = new THREE.Group();
  const link = new THREE.Group();
  link.name = 'link:free-body';
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.4, 0.2, 0.1),
    new THREE.MeshBasicMaterial(),
  );
  mesh.userData.usdcraftVisual = true;
  mesh.userData.usdcraftLinkName = 'free-body';
  link.add(mesh);
  robot.add(link);
  scene.add(robot);
  return {
    scene,
    robot,
    jointFrames: new Map([['free-placement', new THREE.Group()]]),
  };
}

function stubCanvasDocument(): void {
  const context = {
    font: '',
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    textBaseline: 'alphabetic',
    measureText: () => ({ width: 80 }),
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    arc: () => {},
    closePath: () => {},
    fill: () => {},
    stroke: () => {},
    fillText: () => {},
  };
  vi.stubGlobal('document', {
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => context,
    }),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('floating-body overlay', () => {
  it('measures gesture extent from asset geometry without counting debug overlays', () => {
    const { robot } = sceneFixture();
    const debugOverlay = new THREE.Mesh(
      new THREE.BoxGeometry(20, 20, 20),
      new THREE.MeshBasicMaterial(),
    );
    debugOverlay.userData.usdcraftOverlay = true;
    robot.add(debugOverlay);

    expect(measureFloatingGeometryExtent(robot)).toBeCloseTo(0.4, 7);

    debugOverlay.geometry.dispose();
    (debugOverlay.material as THREE.Material).dispose();
  });

  it('uses one low-cost bounds helper per island when nothing is selected', () => {
    const { scene, robot, jointFrames } = sceneFixture();
    const overlay = attachFloatingBodyOverlay(scene, robot, jointFrames, [island()], null);

    expect(overlay.objects).toHaveLength(1);
    expect(overlay.objects[0]).toBeInstanceOf(THREE.Box3Helper);
    expect(overlay.objects[0].userData.usdcraftFloatingIslandId).toBe('free-placement');

    overlay.dispose();
    expect(scene.children).toEqual([robot]);
  });

  it('adds a six-axis translation and three-ring rotation glyph only for the selected island', () => {
    stubCanvasDocument();
    const { scene, robot, jointFrames } = sceneFixture();
    const overlay = attachFloatingBodyOverlay(
      scene,
      robot,
      jointFrames,
      [island()],
      'free-placement',
    );
    const glyph = overlay.objects.find((object) => object.name === 'floating-overlay:freedom-glyph');

    expect(glyph).toBeDefined();
    expect(glyph?.children.filter((child) => child instanceof THREE.ArrowHelper)).toHaveLength(6);
    expect(
      glyph?.children.filter(
        (child) => child instanceof THREE.Mesh && child.geometry instanceof THREE.TorusGeometry,
      ),
    ).toHaveLength(3);
    expect(glyph?.children.some((child) => child instanceof THREE.Sprite)).toBe(true);

    overlay.dispose();
    expect(scene.children).toEqual([robot]);
  });

  it('refreshes bounds when collision-only geometry is attached asynchronously', () => {
    stubCanvasDocument();
    const scene = new THREE.Scene();
    const robot = new THREE.Group();
    const link = new THREE.Group();
    robot.add(link);
    scene.add(robot);
    const jointFrames = new Map([['free-placement', new THREE.Group()]]);
    const overlay = attachFloatingBodyOverlay(
      scene,
      robot,
      jointFrames,
      [island()],
      'free-placement',
    );
    const helper = overlay.objects[0] as THREE.Box3Helper;
    const fallbackSize = helper.box.getSize(new THREE.Vector3());
    const fallbackGlyph = overlay.objects.find(
      (object) => object.name === 'floating-overlay:freedom-glyph',
    );

    const collision = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 0.8, 0.4),
      new THREE.MeshBasicMaterial(),
    );
    collision.userData.usdcraftCollision = true;
    collision.userData.usdcraftLinkName = 'free-body';
    link.add(collision);
    overlay.refreshGeometry();

    const refreshedSize = helper.box.getSize(new THREE.Vector3());
    const refreshedGlyph = overlay.objects.find(
      (object) => object.name === 'floating-overlay:freedom-glyph',
    );
    expect(refreshedSize.x).toBeGreaterThan(fallbackSize.x);
    expect(refreshedSize.x).toBeGreaterThan(1.2);
    expect(refreshedGlyph).toBeDefined();
    expect(refreshedGlyph).not.toBe(fallbackGlyph);
    expect(scene.children).not.toContain(fallbackGlyph);

    overlay.dispose();
    collision.geometry.dispose();
    (collision.material as THREE.Material).dispose();
  });

  it('moves an empty-island fallback with its floating preview node', () => {
    const scene = new THREE.Scene();
    const robot = new THREE.Group();
    scene.add(robot);
    const previewNode = new THREE.Group();
    robot.add(previewNode);
    const overlay = attachFloatingBodyOverlay(
      scene,
      robot,
      new Map([['free-placement', previewNode]]),
      [island()],
      null,
    );
    const helper = overlay.objects[0] as THREE.Box3Helper;
    const centerBefore = helper.box.getCenter(new THREE.Vector3());

    previewNode.position.set(0.3, -0.2, 0.4);
    overlay.update();

    expect(helper.box.getCenter(new THREE.Vector3())).toEqual(
      centerBefore.add(new THREE.Vector3(0.3, -0.2, 0.4)),
    );
    overlay.dispose();
  });
});
