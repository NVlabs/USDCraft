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

import * as THREE from 'three';

import type { FloatingIsland } from './floating-islands';

const FLOATING_COLOR = 0xc06020;
const LABEL_SCREEN_PX = 22;

type IslandOverlayEntry = {
  island: FloatingIsland;
  box: THREE.Box3;
  helper: THREE.Box3Helper;
  meshes: THREE.Mesh[];
  fallback: THREE.Object3D | null;
};

export type FloatingBodyOverlay = {
  objects: THREE.Object3D[];
  update: () => void;
  refreshGeometry: () => void;
  dispose: () => void;
};

function collectLinkMeshes(
  robot: THREE.Object3D,
): {
  visuals: Map<string, THREE.Mesh[]>;
  collisions: Map<string, THREE.Mesh[]>;
} {
  const visuals = new Map<string, THREE.Mesh[]>();
  const collisions = new Map<string, THREE.Mesh[]>();

  robot.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }
    const linkName = object.userData.usdcraftLinkName;
    if (typeof linkName !== 'string' || linkName.length === 0) {
      return;
    }

    const target = object.userData.usdcraftVisual === true
      ? visuals
      : object.userData.usdcraftCollision === true
        ? collisions
        : null;
    if (!target) {
      return;
    }

    const meshes = target.get(linkName) ?? [];
    meshes.push(object);
    target.set(linkName, meshes);
  });

  return { visuals, collisions };
}

function islandMeshes(
  island: FloatingIsland,
  visualMeshes: Map<string, THREE.Mesh[]>,
  collisionMeshes: Map<string, THREE.Mesh[]>,
): THREE.Mesh[] {
  return island.memberPartNames.flatMap((partName) => {
    const visuals = visualMeshes.get(partName) ?? [];
    return visuals.length > 0 ? visuals : collisionMeshes.get(partName) ?? [];
  });
}

function expandBoxByMeshes(box: THREE.Box3, meshes: THREE.Mesh[]): void {
  const meshBox = new THREE.Box3();
  for (const mesh of meshes) {
    if (!mesh.geometry.boundingBox) {
      mesh.geometry.computeBoundingBox();
    }
    if (!mesh.geometry.boundingBox) {
      continue;
    }
    mesh.updateWorldMatrix(true, false);
    meshBox.copy(mesh.geometry.boundingBox).applyMatrix4(mesh.matrixWorld);
    box.union(meshBox);
  }
}

export function measureFloatingGeometryExtent(
  root: THREE.Object3D,
  fallbackExtent = 0.25,
): number {
  const { visuals, collisions } = collectLinkMeshes(root);
  const linkNames = new Set([...visuals.keys(), ...collisions.keys()]);
  const meshes = [...linkNames].flatMap((linkName) => {
    const linkVisuals = visuals.get(linkName) ?? [];
    return linkVisuals.length > 0 ? linkVisuals : collisions.get(linkName) ?? [];
  });
  const box = new THREE.Box3();
  expandBoxByMeshes(box, meshes);
  if (box.isEmpty()) {
    return fallbackExtent;
  }
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  return Number.isFinite(extent) && extent > 0 ? extent : fallbackExtent;
}

function updateIslandBox(entry: IslandOverlayEntry): void {
  entry.box.makeEmpty();
  expandBoxByMeshes(entry.box, entry.meshes);

  if (entry.box.isEmpty() && entry.fallback) {
    const center = entry.fallback.getWorldPosition(new THREE.Vector3());
    entry.box.setFromCenterAndSize(center, new THREE.Vector3(0.06, 0.06, 0.06));
  }

  if (!entry.box.isEmpty()) {
    const size = entry.box.getSize(new THREE.Vector3());
    const padding = THREE.MathUtils.clamp(Math.max(size.x, size.y, size.z) * 0.035, 0.004, 0.04);
    entry.box.expandByScalar(padding);
  }
  entry.helper.updateMatrixWorld(true);
}

function configureOverlayMaterial(material: THREE.Material, opacity: number): void {
  material.transparent = true;
  material.opacity = opacity;
  material.depthTest = false;
  material.depthWrite = false;
  material.toneMapped = false;
  material.needsUpdate = true;
}

function createIslandHelper(box: THREE.Box3, selected: boolean): THREE.Box3Helper {
  const helper = new THREE.Box3Helper(box, FLOATING_COLOR);
  helper.name = 'floating-overlay:island-bounds';
  helper.renderOrder = selected ? 34 : 28;
  helper.userData.usdcraftOverlay = true;
  helper.userData.usdcraftFloatingOverlay = true;
  const materials = Array.isArray(helper.material) ? helper.material : [helper.material];
  for (const material of materials) {
    configureOverlayMaterial(material, selected ? 0.96 : 0.34);
  }
  return helper;
}

function overlayMaterial(opacity: number): THREE.MeshBasicMaterial {
  const material = new THREE.MeshBasicMaterial({ color: FLOATING_COLOR });
  configureOverlayMaterial(material, opacity);
  return material;
}

function createFreedomGlyph(unit: number): THREE.Group {
  const group = new THREE.Group();
  group.name = 'floating-overlay:freedom-glyph';
  group.renderOrder = 36;
  group.userData.usdcraftOverlay = true;
  group.userData.usdcraftFloatingOverlay = true;

  const directions = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0, -1),
  ];
  for (const direction of directions) {
    const arrow = new THREE.ArrowHelper(
      direction,
      new THREE.Vector3(),
      unit,
      FLOATING_COLOR,
      unit * 0.24,
      unit * 0.11,
    );
    arrow.renderOrder = 36;
    arrow.userData.usdcraftOverlay = true;
    arrow.traverse((object) => {
      if ('material' in object) {
        const material = (object as { material?: THREE.Material | THREE.Material[] }).material;
        if (Array.isArray(material)) {
          material.forEach((entry) => configureOverlayMaterial(entry, 0.88));
        } else if (material) {
          configureOverlayMaterial(material, 0.88);
        }
      }
    });
    group.add(arrow);
  }

  const ringGeometry = new THREE.TorusGeometry(unit * 0.6, unit * 0.018, 6, 40);
  const ringRotations = [
    new THREE.Euler(0, Math.PI / 2, 0),
    new THREE.Euler(Math.PI / 2, 0, 0),
    new THREE.Euler(0, 0, 0),
  ];
  for (const rotation of ringRotations) {
    const ring = new THREE.Mesh(ringGeometry.clone(), overlayMaterial(0.52));
    ring.rotation.copy(rotation);
    ring.renderOrder = 35;
    ring.userData.usdcraftOverlay = true;
    group.add(ring);
  }
  ringGeometry.dispose();

  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(unit * 0.07, 12, 8),
    overlayMaterial(0.98),
  );
  dot.renderOrder = 37;
  dot.userData.usdcraftOverlay = true;
  group.add(dot);

  const label = createLabel('FREE · 6DOF', FLOATING_COLOR);
  label.position.set(0, 0, unit * 1.28);
  group.add(label);
  return group;
}

function createLabel(text: string, tint: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Failed to create canvas context for floating-body overlay label.');
  }

  const dpr = 2;
  const fontSize = 11 * dpr;
  const horizontalPadding = 8 * dpr;
  const verticalPadding = 5 * dpr;
  const dotRadius = 3 * dpr;
  const dotGap = 5 * dpr;
  context.font = `600 ${fontSize}px "DM Sans", system-ui, sans-serif`;
  const textWidth = context.measureText(text).width;
  const width = Math.ceil(
    horizontalPadding + dotRadius * 2 + dotGap + textWidth + horizontalPadding,
  );
  const height = Math.ceil(verticalPadding + fontSize + verticalPadding);
  canvas.width = width;
  canvas.height = height;

  context.fillStyle = 'rgba(8, 8, 8, 0.78)';
  roundedRect(context, 0, 0, width, height);
  context.fill();
  context.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  context.lineWidth = dpr;
  roundedRect(context, dpr * 0.5, dpr * 0.5, width - dpr, height - dpr);
  context.stroke();

  context.beginPath();
  context.arc(horizontalPadding + dotRadius, height / 2, dotRadius, 0, Math.PI * 2);
  context.fillStyle = `#${tint.toString(16).padStart(6, '0')}`;
  context.fill();
  context.font = `600 ${fontSize}px "DM Sans", system-ui, sans-serif`;
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(255, 255, 255, 0.94)';
  context.fillText(
    text,
    horizontalPadding + dotRadius * 2 + dotGap,
    height / 2 + dpr * 0.5,
  );

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 40;
  sprite.userData.usdcraftOverlay = true;
  sprite.userData.usdcraftOverlayTexture = texture;

  const aspect = width / height;
  const worldPosition = new THREE.Vector3();
  const cameraPosition = new THREE.Vector3();
  sprite.onBeforeRender = (renderer, _scene, camera) => {
    if (!(camera instanceof THREE.PerspectiveCamera)) {
      return;
    }
    const screenHeight = renderer.domElement.clientHeight;
    if (screenHeight === 0) {
      return;
    }
    sprite.getWorldPosition(worldPosition);
    camera.getWorldPosition(cameraPosition);
    const distance = worldPosition.distanceTo(cameraPosition);
    const worldPerPixel = (
      2 * distance * Math.tan(camera.fov * THREE.MathUtils.DEG2RAD * 0.5)
    ) / screenHeight;
    const spriteHeight = LABEL_SCREEN_PX * worldPerPixel;
    sprite.scale.set(spriteHeight * aspect, spriteHeight, 1);
    sprite.updateMatrixWorld(true);
  };
  return sprite;
}

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const radius = height / 2;
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.arc(x + width - radius, y + radius, radius, -Math.PI / 2, Math.PI / 2);
  context.lineTo(x + radius, y + height);
  context.arc(x + radius, y + radius, radius, Math.PI / 2, -Math.PI / 2);
  context.closePath();
}

function disposeObject(root: THREE.Object3D): void {
  root.parent?.remove(root);
  root.traverse((object) => {
    if (object instanceof THREE.Sprite) {
      object.onBeforeRender = () => {};
    }
    if ('geometry' in object) {
      (object as { geometry?: THREE.BufferGeometry }).geometry?.dispose();
    }
    if ('material' in object) {
      const material = (object as { material?: THREE.Material | THREE.Material[] }).material;
      const materials = Array.isArray(material) ? material : material ? [material] : [];
      for (const entry of materials) {
        if (entry instanceof THREE.SpriteMaterial) {
          entry.map?.dispose();
        }
        entry.dispose();
      }
    }
  });
}

export function attachFloatingBodyOverlay(
  scene: THREE.Scene,
  robot: THREE.Object3D,
  fallbackNodes: Map<string, THREE.Object3D>,
  islands: FloatingIsland[],
  selectedIslandId: string | null,
): FloatingBodyOverlay {
  const entries = islands.map((island) => {
    const box = new THREE.Box3();
    const helper = createIslandHelper(box, island.id === selectedIslandId);
    helper.userData.usdcraftFloatingIslandId = island.id;
    scene.add(helper);
    return {
      island,
      box,
      helper,
      meshes: [] as THREE.Mesh[],
      fallback: fallbackNodes.get(island.joint.name) ?? null,
    } satisfies IslandOverlayEntry;
  });

  const selectedEntry = entries.find((entry) => entry.island.id === selectedIslandId) ?? null;
  let glyph: THREE.Group | null = null;
  let glyphUnit: number | null = null;
  const robotOrientation = new THREE.Quaternion();
  const objects: THREE.Object3D[] = entries.map((entry) => entry.helper);

  const updateOverlay = (resizeGlyph: boolean): void => {
    robot.updateWorldMatrix(true, true);
    for (const entry of entries) {
      updateIslandBox(entry);
    }

    if (!selectedEntry || selectedEntry.box.isEmpty()) {
      if (glyph) {
        glyph.visible = false;
      }
      return;
    }

    const size = selectedEntry.box.getSize(new THREE.Vector3());
    const unit = THREE.MathUtils.clamp(Math.max(size.x, size.y, size.z) * 0.34, 0.07, 0.42);
    if (
      !glyph
      || glyphUnit == null
      || (resizeGlyph && Math.abs(unit - glyphUnit) > 1e-6)
    ) {
      if (glyph) {
        const previousIndex = objects.indexOf(glyph);
        if (previousIndex >= 0) {
          objects.splice(previousIndex, 1);
        }
        disposeObject(glyph);
      }
      glyph = createFreedomGlyph(unit);
      glyphUnit = unit;
      scene.add(glyph);
      objects.push(glyph);
    }
    glyph.visible = true;
    selectedEntry.box.getCenter(glyph.position);
    robot.getWorldQuaternion(robotOrientation);
    glyph.quaternion.copy(robotOrientation);
    glyph.updateMatrixWorld(true);
  };

  const update = (): void => updateOverlay(false);

  const refreshGeometry = (): void => {
    const { visuals, collisions } = collectLinkMeshes(robot);
    for (const entry of entries) {
      entry.meshes = islandMeshes(entry.island, visuals, collisions);
    }
    updateOverlay(true);
  };

  refreshGeometry();
  return {
    objects,
    update,
    refreshGeometry,
    dispose: () => {
      for (const object of objects) {
        disposeObject(object);
      }
    },
  };
}
