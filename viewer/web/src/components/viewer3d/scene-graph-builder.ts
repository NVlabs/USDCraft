// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
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
import type { UrdfCollisionRole, UrdfSpec, UrdfLink } from './urdf-parser';
import { describeLinkVisuals, findRootLink, originToMatrix4 } from './urdf-parser';
import { depthBiasForOrdinal, resolveVisualMaterialSpec } from './materials';
import { buildPrimitiveMesh } from './geometry-loader';

export interface SceneGraphOptions {
  /** Whether to show visual geometries (default: true) */
  showVisuals?: boolean;
  /** Whether to show collision geometries (default: false) */
  showCollisions?: boolean;
  /** Override material opacity for all links */
  opacity?: number;
}

export interface RobotSceneGraph {
  root: THREE.Group;
  jointNodes: Map<string, THREE.Object3D>;
  jointFrames: Map<string, THREE.Group>;
  floatingNodes: Map<string, THREE.Object3D>;
  linkNodes: Map<string, THREE.Group>;
}

const VISUAL_USER_DATA_KEY = "usdcraftVisual";
const COLLISION_USER_DATA_KEY = "usdcraftCollision";
const RIGID_COLLISION_DEBUG_PALETTE = [
  "#ff5a36",
  "#ffd400",
  "#ff2f92",
  "#ff8a00",
  "#ff6b6b",
] as const;
const DEFORMABLE_COLLISION_DEBUG_PALETTE = [
  "#00b3ff",
  "#00c2a8",
  "#7c5cff",
  "#16c47f",
  "#38bdf8",
] as const;

export function collisionColorForIndex(
  index: number,
  role: UrdfCollisionRole = 'rigid_runtime',
): THREE.Color {
  const palette = role === 'deformable_contact'
    ? DEFORMABLE_COLLISION_DEBUG_PALETTE
    : RIGID_COLLISION_DEBUG_PALETTE;
  return new THREE.Color(palette[index % palette.length]);
}

function createCollisionMaterial(
  color: THREE.Color,
  depthBias: number,
): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.88,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 0,
    polygonOffsetUnits: depthBias,
  });
}

function createCollisionEdges(geometry: THREE.BufferGeometry, color: THREE.Color): THREE.LineSegments {
  const edgeGeometry = new THREE.EdgesGeometry(geometry, 20);
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: color.clone().offsetHSL(0, 0, -0.32),
    transparent: true,
    opacity: 1,
  });
  return new THREE.LineSegments(edgeGeometry, edgeMaterial);
}

function isDescendantOf(object: THREE.Object3D, ancestor: THREE.Object3D): boolean {
  let current: THREE.Object3D | null = object;
  while (current) {
    if (current === ancestor) {
      return true;
    }
    current = current.parent;
  }
  return false;
}

function detachFloatingFramesAtRest(
  root: THREE.Group,
  urdfSpec: UrdfSpec,
  jointFrames: Map<string, THREE.Group>,
  jointNodes: Map<string, THREE.Object3D>,
): Map<string, THREE.Object3D> {
  root.updateWorldMatrix(true, true);
  const rootWorldInverse = root.matrixWorld.clone().invert();
  const restPlacements: Array<{
    name: string;
    frame: THREE.Group;
    motion: THREE.Object3D;
    matrix: THREE.Matrix4;
  }> = [];

  for (const joint of urdfSpec.joints) {
    if (joint.type !== 'floating') {
      continue;
    }
    const frame = jointFrames.get(joint.name);
    const motion = jointNodes.get(joint.name);
    if (!frame || !motion || !isDescendantOf(frame, root)) {
      continue;
    }
    frame.updateWorldMatrix(true, false);
    restPlacements.push({
      name: joint.name,
      frame,
      motion,
      matrix: rootWorldInverse.clone().multiply(frame.matrixWorld),
    });
  }

  const floatingNodes = new Map<string, THREE.Object3D>();
  for (const placement of restPlacements) {
    root.add(placement.frame);
    placement.matrix.decompose(
      placement.frame.position,
      placement.frame.quaternion,
      placement.frame.scale,
    );
    placement.frame.updateMatrix();
    placement.frame.userData.usdcraftFloatingPreviewFrame = true;
    placement.motion.userData.usdcraftFloatingPreviewNode = true;
    floatingNodes.set(placement.name, placement.motion);
  }
  root.updateWorldMatrix(true, true);
  return floatingNodes;
}

/**
 * Build a Three.js scene graph from a parsed URDF spec.
 *
 * Creates a hierarchy of Groups:
 *   root
 *     └─ linkGroup (root link)
 *          ├─ visual meshes
 *          └─ jointGroup (joint transform)
 *               └─ linkGroup (child link)
 *                    ├─ visual meshes
 *                    └─ ...
 */
export function buildRobotSceneGraph(
  urdfSpec: UrdfSpec,
  options: SceneGraphOptions = {},
): RobotSceneGraph {
  const { showVisuals = true, showCollisions = false, opacity } = options;

  const linkNodes = new Map<string, THREE.Group>();
  const jointNodes = new Map<string, THREE.Object3D>();
  const jointFrames = new Map<string, THREE.Group>();

  // Index links by name for quick lookup
  const linkByName = new Map<string, UrdfLink>();
  for (const link of urdfSpec.links) {
    linkByName.set(link.name, link);
  }

  // Create a group for each link and populate with visual meshes
  let collisionIndex = 0;
  let visualIndex = 0;
  for (const link of urdfSpec.links) {
    const linkGroup = new THREE.Group();
    linkGroup.name = `link:${link.name}`;
    const visualDescriptors = describeLinkVisuals(link);

    if (showVisuals) {
      for (const [visualIndexWithinLink, visual] of link.visuals.entries()) {
        const matSpec = resolveVisualMaterialSpec(visual);
        if (opacity !== undefined) {
          matSpec.opacity = opacity;
        }

        const visualDescriptor = visualDescriptors[visualIndexWithinLink];
        const mesh = buildPrimitiveMesh(visual.geometry, matSpec, {
          depthBias: depthBiasForOrdinal(visualIndex),
        });
        visualIndex += 1;
        if (mesh) {
          // Apply visual-level origin transform
          if (visual.origin) {
            mesh.applyMatrix4(originToMatrix4(visual.origin));
          }
          mesh.userData[VISUAL_USER_DATA_KEY] = true;
          mesh.userData.usdcraftLinkName = link.name;
          mesh.userData.usdcraftVisualKey = visualDescriptor.key;
          mesh.userData.usdcraftVisualLabel = visualDescriptor.label;
          mesh.name = visual.name ?? `visual:${link.name}`;
          linkGroup.add(mesh);
        }
      }
    }

    if (showCollisions) {
      for (const collision of link.collisions) {
        const depthBias = depthBiasForOrdinal(collisionIndex);
        const debugColor = collisionColorForIndex(collisionIndex, collision.collisionRole);
        collisionIndex += 1;
        const mesh = buildPrimitiveMesh(collision.geometry, {
          name: 'collision_debug',
          color: debugColor,
          opacity: 0.88,
          metalness: 0,
          roughness: 1,
          transmission: 0,
          thickness: 0,
          ior: 1.45,
          clearcoat: 0,
          clearcoatRoughness: 0.2,
          envMapIntensity: 0.8,
        }, {
          depthBias,
        });
        if (mesh) {
          if (collision.origin) {
            mesh.applyMatrix4(originToMatrix4(collision.origin));
          }
          const mat = createCollisionMaterial(debugColor, depthBias);
          mesh.material = mat;
          mesh.visible = false;
          mesh.userData[COLLISION_USER_DATA_KEY] = true;
          mesh.userData.usdcraftCollisionRole = collision.collisionRole;
          mesh.userData.usdcraftCollisionRealization = collision.collisionRealization;
          mesh.userData.usdcraftCollisionBodyModel = collision.bodyModel;
          mesh.userData.usdcraftLinkName = link.name;
          mesh.renderOrder = 10;
          mesh.add(createCollisionEdges(mesh.geometry, debugColor));
          mesh.name = collision.name ?? `collision:${link.name}`;
          linkGroup.add(mesh);
        }
      }
    }

    linkNodes.set(link.name, linkGroup);
  }

  // Wire up joint hierarchy: parent link → joint group → child link
  for (const joint of urdfSpec.joints) {
    const parentGroup = linkNodes.get(joint.parent);
    const childGroup = linkNodes.get(joint.child);
    if (!parentGroup || !childGroup) continue;

    const jointFrame = new THREE.Group();
    jointFrame.name = `joint-frame:${joint.name}`;

    // Apply joint origin transform
    if (joint.origin) {
      jointFrame.applyMatrix4(originToMatrix4(joint.origin));
    }

    const motionGroup = new THREE.Group();
    motionGroup.name = `joint-motion:${joint.name}`;

    motionGroup.add(childGroup);
    jointFrame.add(motionGroup);
    parentGroup.add(jointFrame);
    jointFrames.set(joint.name, jointFrame);
    jointNodes.set(joint.name, motionGroup);
  }

  // Find root link and wrap in a top-level group
  const rootLinkName = findRootLink(urdfSpec);
  const root = new THREE.Group();
  root.name = `robot:${urdfSpec.name}`;

  if (rootLinkName) {
    const rootLinkGroup = linkNodes.get(rootLinkName);
    if (rootLinkGroup) {
      root.add(rootLinkGroup);
    }
  }

  // FLOATING is an unconstrained placement edge rather than a physical joint.
  // Preserve its authored q=0 placement, then detach its frame from articulation
  // ancestors so scalar preview motion cannot carry the free-body island.
  const floatingNodes = detachFloatingFramesAtRest(root, urdfSpec, jointFrames, jointNodes);

  return { root, jointNodes, jointFrames, floatingNodes, linkNodes };
}
