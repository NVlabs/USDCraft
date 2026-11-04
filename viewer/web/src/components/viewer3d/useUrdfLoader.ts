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

import { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { USDLoader } from 'three/addons/loaders/USDLoader.js';
import {
  describeLinkVisuals,
  originToMatrix4,
  parseUrdf,
  rewriteAbsoluteMeshFilenames,
  summarizeUrdfCollisionGeometry,
} from './urdf-parser';
import type { UrdfSpec, UrdfVisual } from './urdf-parser';
import { buildRobotSceneGraph, collisionColorForIndex } from './scene-graph-builder';
import { computeFit, preserveViewAcrossModelSwitch, updateCameraClipping } from './camera-utils';
import { positionGroundHelpers } from './lighting';
import { loadGeometryObject } from './geometry-loader';
import { createMaterial, depthBiasForOrdinal, resolveVisualMaterialSpec } from './materials';

/** Sentinel name attached to the robot group so we can find and remove it later. */
export const ROBOT_GROUP_NAME = '__usdcraft_robot__';

function buildClippingTargets(
  robotGroup: THREE.Object3D,
  gridGroup: THREE.Group | null,
  axisGroup: THREE.Group | null,
): THREE.Object3D[] {
  const targets: THREE.Object3D[] = [robotGroup];
  if (gridGroup) {
    targets.push(gridGroup);
  }
  if (axisGroup) {
    targets.push(axisGroup);
  }
  return targets;
}

export interface UrdfLoaderState {
  urdfSpec: UrdfSpec | null;
  jointNodes: Map<string, THREE.Object3D> | null;
  jointFrames: Map<string, THREE.Group> | null;
  floatingNodes: Map<string, THREE.Object3D> | null;
  /** The complete, source-frame-preserving group currently attached to the scene. */
  robotGroup: THREE.Group | null;
  /** URL+revision whose geometry has actually replaced the prior scene. */
  loadedAssetKey: string | null;
  loading: boolean;
  error: string | null;
  collisionGeometryState: CollisionGeometryState;
}

export interface CollisionGeometryState {
  declaredCount: number;
  rigidRuntimeCount: number;
  deformableRestShapeCount: number;
  missingDeformableRestShapePartCount: number;
  missingDeformableRestShapePartNames: string[];
  primitiveCount: number;
  meshCount: number;
  loadedMeshCount: number;
  failedMeshCount: number;
  loading: boolean;
}

const EMPTY_COLLISION_GEOMETRY_STATE: CollisionGeometryState = {
  declaredCount: 0,
  rigidRuntimeCount: 0,
  deformableRestShapeCount: 0,
  missingDeformableRestShapePartCount: 0,
  missingDeformableRestShapePartNames: [],
  primitiveCount: 0,
  meshCount: 0,
  loadedMeshCount: 0,
  failedMeshCount: 0,
  loading: false,
};

function syncLoadedMeshVisibility(root: THREE.Object3D, showCollisionMeshes: boolean): void {
  root.traverse((child) => {
    if (child.userData.usdcraftVisual === true) {
      child.visible = !showCollisionMeshes;
    }
    if (child.userData.usdcraftCollision === true) {
      child.visible = showCollisionMeshes;
    }
  });
}

/**
 * Hook that loads the USDZ render asset while retaining URDF articulation metadata.
 *
 * When baseFileUrl changes the hook:
 * 1. Fetches the URDF from {baseFileUrl}/model.urdf
 * 2. Parses with parseUrdf, rewrites absolute mesh filenames
 * 3. Builds the articulation graph and prefers model.usdz for visual geometry
 * 4. Removes the previous robot group from the scene and adds the new one
 * 5. Fits the camera to the new model with computeFit
 * 6. Preserves the authored asset frame and scales only the ground helpers
 */
export function useUrdfLoader(
  baseFileUrl: string | null,
  assetRevisionKey: string | null,
  jointPoseSignal: Map<string, number>,
  showCollisionMeshes: boolean,
  scene: THREE.Scene | null,
  camera: THREE.PerspectiveCamera | null,
  controls: OrbitControls | null,
  gridGroup: THREE.Group | null,
  axisGroup: THREE.Group | null,
  invalidate: () => void,
  localUsdFile: File | null = null,
): UrdfLoaderState {
  const [urdfSpec, setUrdfSpec] = useState<UrdfSpec | null>(null);
  const [jointNodes, setJointNodes] = useState<Map<string, THREE.Object3D> | null>(null);
  const [jointFrames, setJointFrames] = useState<Map<string, THREE.Group> | null>(null);
  const [floatingNodes, setFloatingNodes] = useState<Map<string, THREE.Object3D> | null>(null);
  const [robotGroup, setRobotGroup] = useState<THREE.Group | null>(null);
  const [loadedAssetKey, setLoadedAssetKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collisionGeometryState, setCollisionGeometryState] = useState<CollisionGeometryState>(
    EMPTY_COLLISION_GEOMETRY_STATE,
  );

  // Track the current robot group so we can remove it when swapping models.
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const linkNodesRef = useRef<Map<string, THREE.Group> | null>(null);
  const specRef = useRef<UrdfSpec | null>(null);
  const collisionMeshesLoadedKeyRef = useRef<string | null>(null);
  const collisionMeshesLoadingKeyRef = useRef<string | null>(null);

  useEffect(() => {
    // Reset when no base URL is available.
    if ((!baseFileUrl && !localUsdFile) || !scene) {
      if (robotGroupRef.current && scene) {
        scene.remove(robotGroupRef.current);
        disposeGroup(robotGroupRef.current);
      }
      robotGroupRef.current = null;
      linkNodesRef.current = null;
      specRef.current = null;
      setUrdfSpec(null);
      setJointNodes(null);
      setJointFrames(null);
      setFloatingNodes(null);
      setRobotGroup(null);
      setLoadedAssetKey(null);
      setLoading(false);
      setError(null);
      setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      setLoadedAssetKey(null);
      linkNodesRef.current = null;
      specRef.current = null;
      collisionMeshesLoadedKeyRef.current = null;
      collisionMeshesLoadingKeyRef.current = null;
      setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);

      try {
        if (localUsdFile) {
          const parsed = new USDLoader().parse(await localUsdFile.arrayBuffer());
          if (cancelled) return;
          const local = buildLocalUsdScene(parsed, localUsdFile.name);
          const hadPreviousRobot = robotGroupRef.current !== null;
          if (robotGroupRef.current) {
            scene.remove(robotGroupRef.current);
            disposeGroup(robotGroupRef.current);
          }
          const robotGroup = new THREE.Group();
          robotGroup.name = ROBOT_GROUP_NAME;
          robotGroup.rotation.x = -Math.PI / 2;
          robotGroup.add(local.root);
          scene.add(robotGroup);
          robotGroupRef.current = robotGroup;
          setRobotGroup(robotGroup);
          if (camera && controls) {
            if (hadPreviousRobot) {
              preserveViewAcrossModelSwitch(camera, controls, robotGroup);
            } else {
              const fit = computeFit(camera, robotGroup);
              camera.position.copy(fit.position);
              controls.target.copy(fit.target);
              camera.near = fit.near;
              camera.far = fit.far;
              camera.updateProjectionMatrix();
              controls.update();
            }
          }
          if (gridGroup && axisGroup) {
            positionGroundHelpers(gridGroup, axisGroup, robotGroup);
          }
          if (camera) {
            updateCameraClipping(camera, buildClippingTargets(robotGroup, gridGroup, axisGroup));
          }
          invalidate();
          if (!cancelled) {
            specRef.current = local.spec;
            linkNodesRef.current = local.linkNodes;
            setUrdfSpec(local.spec);
            setJointNodes(new Map());
            setJointFrames(new Map());
            setFloatingNodes(new Map());
            setLoadedAssetKey(`local:${localUsdFile.name}`);
          }
          return;
        }
        if (!baseFileUrl) {
          throw new Error('No record URL or local USD file was provided');
        }
        const recordBaseFileUrl = baseFileUrl;

        // 1. Fetch URDF text.
        const urdfUrl = new URL(`${recordBaseFileUrl}/model.urdf`, window.location.origin);
        if (assetRevisionKey) {
          urdfUrl.searchParams.set('rev', assetRevisionKey);
        }

        const response = await fetch(urdfUrl.toString(), {
          cache: assetRevisionKey ? 'no-store' : 'default',
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch URDF: ${response.status} ${response.statusText}`);
        }
        const urdfXml = await response.text();
        if (cancelled) return;

        // 2. Parse and rewrite mesh filenames.
        const rawSpec = parseUrdf(urdfXml);
        const spec = rewriteAbsoluteMeshFilenames(rawSpec);
        const collisionSummary = summarizeUrdfCollisionGeometry(spec);
        if (!cancelled) {
          setCollisionGeometryState({
            ...collisionSummary,
            loadedMeshCount: 0,
            failedMeshCount: 0,
            loading: false,
          });
        }

        // 3. Build scene graph.
        const {
          root,
          jointNodes: joints,
          jointFrames: frames,
          floatingNodes: floating,
          linkNodes,
        } = buildRobotSceneGraph(spec, { showCollisions: true });
        const usdLinkNames = await attachUsdVisualGeometry(
          spec,
          linkNodes,
          recordBaseFileUrl,
          assetRevisionKey,
        );
        const fallbackLinkNames = new Set(
          spec.links
            .filter((link) => !usdLinkNames.has(link.name))
            .map((link) => link.name),
        );
        if (fallbackLinkNames.size > 0) {
          await attachVisualMeshGeometry(
            spec,
            linkNodes,
            recordBaseFileUrl,
            assetRevisionKey,
            fallbackLinkNames,
          );
        }
        assertVisualGeometryCoverage(spec, root, linkNodes);
        if (cancelled) return;

        // 4. Swap robot group in scene.
        const hadPreviousRobot = robotGroupRef.current !== null;
        if (robotGroupRef.current) {
          scene.remove(robotGroupRef.current);
          disposeGroup(robotGroupRef.current);
        }

        const robotGroup = new THREE.Group();
        robotGroup.name = ROBOT_GROUP_NAME;
        robotGroup.rotation.x = -Math.PI / 2;
        robotGroup.add(root);
        scene.add(robotGroup);
        robotGroupRef.current = robotGroup;
        setRobotGroup(robotGroup);

        // 5. Fit camera on first load, otherwise preserve the current view direction
        // while re-centering and scaling to the new object.
        if (camera && controls) {
          if (hadPreviousRobot) {
            preserveViewAcrossModelSwitch(camera, controls, robotGroup);
          } else {
            const fit = computeFit(camera, robotGroup);
            camera.position.copy(fit.position);
            controls.target.copy(fit.target);
            camera.near = fit.near;
            camera.far = fit.far;
            camera.updateProjectionMatrix();
            controls.update();
          }
        }

        // 6. Position grid and axis under the model.
        if (gridGroup && axisGroup) {
          positionGroundHelpers(gridGroup, axisGroup, robotGroup);
        }
        if (camera) {
          updateCameraClipping(camera, buildClippingTargets(robotGroup, gridGroup, axisGroup));
        }
        invalidate();

        if (!cancelled) {
          specRef.current = spec;
          linkNodesRef.current = linkNodes;
          setUrdfSpec(spec);
          setJointNodes(joints);
          setJointFrames(frames);
          setFloatingNodes(floating);
          setLoadedAssetKey(`${recordBaseFileUrl}|${assetRevisionKey ?? ''}`);
        }
      } catch (err) {
        if (!cancelled) {
          if (robotGroupRef.current) {
            scene.remove(robotGroupRef.current);
            disposeGroup(robotGroupRef.current);
            robotGroupRef.current = null;
          }
          linkNodesRef.current = null;
          specRef.current = null;
          setError(err instanceof Error ? err.message : 'Unknown error loading model');
          setUrdfSpec(null);
          setJointNodes(null);
          setJointFrames(null);
          setFloatingNodes(null);
          setRobotGroup(null);
          setLoadedAssetKey(null);
          setCollisionGeometryState(EMPTY_COLLISION_GEOMETRY_STATE);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();

    return () => {
      cancelled = true;
    };
  }, [baseFileUrl, assetRevisionKey, scene, camera, controls, gridGroup, axisGroup, invalidate, localUsdFile]);

  useEffect(() => {
    const assetKey = baseFileUrl ? `${baseFileUrl}|${assetRevisionKey ?? ''}` : null;
    if (!assetKey) {
      return;
    }
    const baseUrl = baseFileUrl;

    const spec = specRef.current;
    const linkNodes = linkNodesRef.current;
    if (!spec || !linkNodes || !baseUrl) {
      return;
    }
    const meshCollisionCount = spec.links.reduce(
      (total, link) =>
        total + link.collisions.filter((collision) => collision.geometry.type === 'mesh').length,
      0,
    );
    if (meshCollisionCount === 0) {
      return;
    }
    if (!showCollisionMeshes) {
      return;
    }
    if (collisionMeshesLoadedKeyRef.current === assetKey || collisionMeshesLoadingKeyRef.current === assetKey) {
      return;
    }

    let cancelled = false;
    collisionMeshesLoadingKeyRef.current = assetKey;
    setCollisionGeometryState((current) => ({
      ...current,
      loading: true,
      failedMeshCount: 0,
    }));

    void attachCollisionMeshGeometry(spec, linkNodes, baseUrl, assetRevisionKey)
      .then(({ loadedCount, failedCount }) => {
        if (cancelled || collisionMeshesLoadingKeyRef.current !== assetKey) {
          return;
        }
        for (const linkGroup of linkNodes.values()) {
          syncLoadedMeshVisibility(linkGroup, showCollisionMeshes);
        }
        collisionMeshesLoadingKeyRef.current = null;
        collisionMeshesLoadedKeyRef.current = assetKey;
        setCollisionGeometryState((current) => ({
          ...current,
          loadedMeshCount: loadedCount,
          failedMeshCount: failedCount,
          loading: false,
        }));
        invalidate();
      })
      .catch(() => {
        if (cancelled || collisionMeshesLoadingKeyRef.current !== assetKey) {
          return;
        }
        collisionMeshesLoadingKeyRef.current = null;
        collisionMeshesLoadedKeyRef.current = assetKey;
        setCollisionGeometryState((current) => ({
          ...current,
          loadedMeshCount: 0,
          failedMeshCount: current.meshCount,
          loading: false,
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [assetRevisionKey, baseFileUrl, invalidate, showCollisionMeshes, urdfSpec]);

  useEffect(() => {
    if (!camera || !controls || !urdfSpec || !robotGroupRef.current) {
      return;
    }

    const robotGroup = robotGroupRef.current;
    const refreshClipping = () => {
      updateCameraClipping(camera, buildClippingTargets(robotGroup, gridGroup, axisGroup));
      invalidate();
    };

    refreshClipping();
    controls.addEventListener('end', refreshClipping);

    return () => {
      controls.removeEventListener('end', refreshClipping);
    };
  }, [camera, controls, urdfSpec, gridGroup, axisGroup, invalidate]);

  useEffect(() => {
    if (!camera || !urdfSpec || !robotGroupRef.current) {
      return;
    }

    updateCameraClipping(
      camera,
      buildClippingTargets(robotGroupRef.current, gridGroup, axisGroup),
    );
    invalidate();
  }, [camera, urdfSpec, gridGroup, axisGroup, jointPoseSignal, invalidate]);

  return {
    urdfSpec,
    jointNodes,
    jointFrames,
    floatingNodes,
    robotGroup,
    loadedAssetKey,
    loading,
    error,
    collisionGeometryState,
  };
}

function buildLocalUsdScene(
  usdRoot: THREE.Group,
  fileName: string,
): { root: THREE.Group; spec: UrdfSpec; linkNodes: Map<string, THREE.Group> } {
  const root = new THREE.Group();
  root.name = `robot:${fileName}`;
  const linkNodes = new Map<string, THREE.Group>();
  const links = findNamedDescendant(usdRoot, 'Links');
  const specs: UrdfSpec['links'] = [];

  if (links && links.children.length > 0) {
    for (const usdLink of [...links.children]) {
      const name = usdLink.name || `part_${specs.length + 1}`;
      const linkGroup = new THREE.Group();
      linkGroup.name = `link:${name}`;
      usdLink.updateMatrix();
      linkGroup.applyMatrix4(usdLink.matrix);
      const visuals = findNamedDescendant(usdLink, 'Visuals');
      const visualSpecs: UrdfSpec['links'][number]['visuals'] = [];
      if (visuals) {
        for (const [index, visualObject] of [...visuals.children].entries()) {
          visualObject.removeFromParent();
          const label = visualObject.name || `visual_${index + 1}`;
          visualObject.traverse((object) => {
            if (!(object instanceof THREE.Mesh)) return;
            object.userData.usdcraftVisual = true;
            object.userData.usdcraftLinkName = name;
            object.userData.usdcraftVisualKey = `${name}::visual:${index}`;
            object.userData.usdcraftVisualLabel = label;
            object.castShadow = true;
            object.receiveShadow = true;
          });
          linkGroup.add(visualObject);
          visualSpecs.push({ name: label, geometry: { type: 'mesh' } });
        }
      }
      if (visualSpecs.length > 0) {
        root.add(linkGroup);
        linkNodes.set(name, linkGroup);
        specs.push({ name, visuals: visualSpecs, collisions: [] });
      }
    }
  }

  if (specs.length === 0) {
    const name = 'USD';
    const linkGroup = new THREE.Group();
    linkGroup.name = `link:${name}`;
    const visuals: UrdfSpec['links'][number]['visuals'] = [];
    let index = 0;
    usdRoot.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const label = object.name || `visual_${index + 1}`;
      object.userData.usdcraftVisual = true;
      object.userData.usdcraftLinkName = name;
      object.userData.usdcraftVisualKey = `${name}::visual:${index}`;
      object.userData.usdcraftVisualLabel = label;
      object.castShadow = true;
      object.receiveShadow = true;
      visuals.push({ name: label, geometry: { type: 'mesh' } });
      index += 1;
    });
    linkGroup.add(usdRoot);
    root.add(linkGroup);
    linkNodes.set(name, linkGroup);
    specs.push({ name, visuals, collisions: [] });
  } else {
    disposeGroup(usdRoot);
  }

  return {
    root,
    linkNodes,
    spec: { name: fileName, links: specs, joints: [] },
  };
}

/**
 * Rebind an attached USD visual's materials to the URDF-authored PBR descriptor.
 *
 * The USD render layer approximates transmission as opacity at export time, so
 * glass/metal semantics degrade when the USDLoader material is used as-is. The
 * URDF `<material>` + `usdcraft_pbr` extension is the authoring ground truth;
 * resolving it through the SAME MaterialSpec pipeline as the URDF fallback path
 * gives both render layers (interactive viewer and the headless review
 * renderer) one material semantics. Textured meshes (decals/labels) keep their
 * loaded USD material — the baked texture carries the albedo.
 */
export function rebindUsdVisualMaterial(
  visualObject: THREE.Object3D,
  visual: UrdfVisual | undefined,
): void {
  // An absent URDF material is intentional for source carriers: the mesh file's
  // embedded GLB/USD material is the appearance authority in that case. Only an
  // explicitly authored descriptor is allowed to replace untextured slots.
  if (!visual?.material) {
    return;
  }
  const spec = resolveVisualMaterialSpec(visual);
  visualObject.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    const rebound = materials.map((material) => {
      // Preserve each textured slot independently. A multi-material mesh may mix
      // a baked decal slot with ordinary untextured surfaces; checking only slot
      // zero either discarded the decal or prevented PBR rebinding everywhere.
      if (material && (material as THREE.MeshStandardMaterial).map) {
        return material;
      }
      material?.dispose();
      return createMaterial(spec);
    });
    object.material = Array.isArray(object.material) ? rebound : rebound[0];
  });
}

export function matchUsdVisualToAuthored(
  usdVisualName: string,
  authoredVisuals: UrdfVisual[],
): UrdfVisual | undefined {
  const named = authoredVisuals.filter(
    (visual) => visual.name && makeUsdIdentifier(visual.name) === usdVisualName,
  );
  if (named.length === 1) {
    return named[0];
  }
  // A single descriptor is unambiguous even if an exporter changed its prim
  // name. With multiple descriptors, never guess by child order: USDLoader does
  // not promise to preserve that order, so keeping the loaded material is safer.
  return authoredVisuals.length === 1 ? authoredVisuals[0] : undefined;
}

async function attachUsdVisualGeometry(
  spec: UrdfSpec,
  linkNodes: Map<string, THREE.Group>,
  baseUrl: string,
  assetRevisionKey: string | null,
): Promise<Set<string>> {
  const url = new URL(`${baseUrl}/model.usdz`, window.location.origin);
  if (assetRevisionKey) {
    url.searchParams.set('rev', assetRevisionKey);
  }

  let usdRoot: THREE.Group;
  try {
    usdRoot = await new USDLoader().loadAsync(url.toString());
  } catch (error) {
    // The USDZ render layer is optional, but a real
    // load/parse failure is indistinguishable here from "no asset". Surface it so a
    // 500/corrupt-asset does not silently degrade the whole model to URDF primitives.
    console.warn(`USDZ visual layer unavailable for ${baseUrl}; falling back to URDF.`, error);
    return new Set();
  }

  const asset = findNamedDescendant(usdRoot, 'Asset');
  const links = asset ? findNamedDescendant(asset, 'Links') : null;
  if (!links) {
    disposeGroup(usdRoot);
    return new Set();
  }

  // The exporter mangles part names to valid USD identifiers and de-duplicates
  // collisions with numeric suffixes; this guesses the identifier from the link
  // name only. If two link names mangle to the same identifier we cannot tell which
  // USD link belongs to which, so we skip USD attach for those (URDF fallback) rather
  // than risk binding the wrong geometry.
  const identifierCounts = new Map<string, number>();
  for (const link of spec.links) {
    const id = makeUsdIdentifier(link.name);
    identifierCounts.set(id, (identifierCounts.get(id) ?? 0) + 1);
  }

  const loadedLinkNames = new Set<string>();
  const skipped: string[] = [];
  for (const link of spec.links) {
    const linkNode = linkNodes.get(link.name);
    const identifier = makeUsdIdentifier(link.name);
    if ((identifierCounts.get(identifier) ?? 0) > 1) {
      skipped.push(link.name);
      continue;
    }
    const usdLink = links.children.find((child) => child.name === identifier);
    const visuals = usdLink ? findNamedDescendant(usdLink, 'Visuals') : null;
    if (!linkNode || !visuals) {
      if (linkNode) {
        skipped.push(link.name);
      }
      continue;
    }

    const visualObjects = [...visuals.children];
    const usdMeshCount = visualObjects.reduce((total, visualObject) => {
      let count = 0;
      visualObject.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          count += 1;
        }
      });
      return total + count;
    }, 0);
    // Use the USD render geometry whenever the link carries any mesh. Do NOT require
    // the USD child count to match the URDF visual count: the exporter may merge
    // multiple URDF visuals into one USD mesh prim, and dropping the whole link in
    // that case would silently fall back to gray URDF primitives.
    if (usdMeshCount === 0) {
      skipped.push(link.name);
      continue;
    }

    for (const child of [...linkNode.children]) {
      if (containsOwnLinkVisual(child, link.name)) {
        linkNode.remove(child);
        disposeGroup(child);
      }
    }

    const descriptors = describeLinkVisuals(link);
    for (const [index, visualObject] of visualObjects.entries()) {
      visualObject.removeFromParent();
      const descriptor = descriptors[index];
      visualObject.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) {
          return;
        }
        object.userData.usdcraftVisual = true;
        object.userData.usdcraftLinkName = link.name;
        object.userData.usdcraftVisualKey = descriptor?.key ?? `usd:${link.name}:${index}`;
        object.userData.usdcraftVisualLabel = descriptor?.label ?? visualObject.name;
        object.castShadow = true;
        object.receiveShadow = true;
      });
      // Match by the authored prim name. USDLoader does not preserve child order,
      // so a multi-visual name miss keeps the loaded USD material rather than
      // borrowing an unrelated descriptor by index.
      const authoredVisual = matchUsdVisualToAuthored(visualObject.name, link.visuals);
      rebindUsdVisualMaterial(visualObject, authoredVisual);
      linkNode.add(visualObject);
    }
    loadedLinkNames.add(link.name);
  }

  if (skipped.length > 0) {
    console.warn(
      `USDZ render geometry missing for ${skipped.length} link(s); using URDF fallback: ` +
        skipped.join(', '),
    );
  }

  disposeGroup(usdRoot);
  return loadedLinkNames;
}

function findNamedDescendant(root: THREE.Object3D, name: string): THREE.Object3D | null {
  let match: THREE.Object3D | null = null;
  root.traverse((object) => {
    if (!match && object.name === name) {
      match = object;
    }
  });
  return match;
}

export function containsOwnLinkVisual(root: THREE.Object3D, linkName: string): boolean {
  const stack: Array<{ object: THREE.Object3D; inheritedLinkName: string | null }> = [
    { object: root, inheritedLinkName: null },
  ];

  while (stack.length > 0) {
    const entry = stack.pop();
    if (!entry) {
      continue;
    }

    const { object, inheritedLinkName } = entry;
    if (object !== root && object.name.startsWith('link:')) {
      continue;
    }

    const explicitLinkName = object.userData.usdcraftLinkName;
    const effectiveLinkName =
      typeof explicitLinkName === 'string' ? explicitLinkName : inheritedLinkName;
    if (object.userData.usdcraftVisual === true && effectiveLinkName === linkName) {
      return true;
    }

    for (const child of object.children) {
      stack.push({ object: child, inheritedLinkName: effectiveLinkName });
    }
  }

  return false;
}

function makeUsdIdentifier(value: string): string {
  const replaced = value.replace(/[^A-Za-z0-9_]/g, '_');
  if (!replaced) {
    return 'Part';
  }
  return /^[A-Za-z_]/.test(replaced) ? replaced : `_${replaced}`;
}

async function attachVisualMeshGeometry(
  spec: UrdfSpec,
  linkNodes: Map<string, THREE.Group>,
  baseUrl: string,
  assetRevisionKey: string | null,
  linkNames: ReadonlySet<string> | null = null,
): Promise<void> {
  const pending: Array<Promise<void>> = [];
  let visualIndex = 0;

  for (const link of spec.links) {
    if (linkNames && !linkNames.has(link.name)) {
      continue;
    }
    const linkGroup = linkNodes.get(link.name);
    if (!linkGroup) {
      continue;
    }
    const visualDescriptors = describeLinkVisuals(link);

    for (const [visualIndexWithinLink, visual] of link.visuals.entries()) {
      const depthBias = depthBiasForOrdinal(visualIndex);
      visualIndex += 1;
      if (visual.geometry.type !== 'mesh') {
        continue;
      }

      const visualDescriptor = visualDescriptors[visualIndexWithinLink];
      pending.push(
        loadGeometryObject(visual.geometry, baseUrl, {
          kind: 'visual',
          assetRevisionKey,
          depthBias,
          materialSpec: visual.material ? resolveVisualMaterialSpec(visual) : undefined,
        }).then((meshGroup) => {
          if (visual.origin) {
            meshGroup.applyMatrix4(originToMatrix4(visual.origin));
          }
          meshGroup.userData.usdcraftLinkName = link.name;
          meshGroup.userData.usdcraftVisualKey = visualDescriptor.key;
          meshGroup.userData.usdcraftVisualLabel = visualDescriptor.label;
          meshGroup.name = visual.name ?? `visual:${link.name}`;
          linkGroup.add(meshGroup);
        }),
      );
    }

  }

  await Promise.all(pending);
}

function countOwnVisualMeshes(linkGroup: THREE.Object3D): number {
  let count = 0;
  const stack = [...linkGroup.children];

  while (stack.length > 0) {
    const object = stack.pop();
    if (!object) {
      continue;
    }
    if (object.name.startsWith('link:')) {
      continue;
    }
    if (object instanceof THREE.Mesh && object.userData.usdcraftVisual === true) {
      count += 1;
    }
    stack.push(...object.children);
  }

  return count;
}

function assertVisualGeometryCoverage(
  spec: UrdfSpec,
  root: THREE.Object3D,
  linkNodes: Map<string, THREE.Group>,
): void {
  const missingLinkNames = spec.links
    .filter((link) => link.visuals.length > 0)
    .filter((link) => {
      const linkNode = linkNodes.get(link.name);
      if (!linkNode || countOwnVisualMeshes(linkNode) === 0) {
        return true;
      }
      let ancestor: THREE.Object3D | null = linkNode;
      while (ancestor && ancestor !== root) {
        ancestor = ancestor.parent;
      }
      return ancestor !== root;
    })
    .map((link) => link.name);

  if (missingLinkNames.length > 0) {
    throw new Error(
      `Missing visual geometry for articulated links: ${missingLinkNames.join(', ')}`,
    );
  }
}

async function attachCollisionMeshGeometry(
  spec: UrdfSpec,
  linkNodes: Map<string, THREE.Group>,
  baseUrl: string,
  assetRevisionKey: string | null,
): Promise<{ loadedCount: number; failedCount: number }> {
  const pending: Array<Promise<void>> = [];
  let collisionIndex = 0;

  for (const link of spec.links) {
    const linkGroup = linkNodes.get(link.name);
    if (!linkGroup) {
      continue;
    }

    for (const collision of link.collisions) {
      const depthBias = depthBiasForOrdinal(collisionIndex);
      const color = collisionColorForIndex(collisionIndex, collision.collisionRole);
      collisionIndex += 1;

      if (collision.geometry.type !== 'mesh') {
        continue;
      }

      pending.push(
        loadGeometryObject(collision.geometry, baseUrl, {
          kind: 'collision',
          assetRevisionKey,
          depthBias,
          materialSpec: {
            name: 'collision_debug',
            color,
            opacity: 0.88,
            metalness: 0,
            roughness: 1,
            transmission: 0,
            thickness: 0,
            ior: 1.45,
            clearcoat: 0,
            clearcoatRoughness: 0.2,
            envMapIntensity: 0.8,
          },
        }).then((meshGroup) => {
          if (collision.origin) {
            meshGroup.applyMatrix4(originToMatrix4(collision.origin));
          }
          meshGroup.traverse((child) => {
            if (child.userData.usdcraftCollision !== true) {
              return;
            }
            child.userData.usdcraftCollisionRole = collision.collisionRole;
            child.userData.usdcraftCollisionRealization = collision.collisionRealization;
            child.userData.usdcraftCollisionBodyModel = collision.bodyModel;
            child.userData.usdcraftLinkName = link.name;
          });
          meshGroup.name = collision.name ?? `collision:${link.name}`;
          linkGroup.add(meshGroup);
        }),
      );
    }
  }

  const settled = await Promise.allSettled(pending);
  const loadedCount = settled.filter((result) => result.status === 'fulfilled').length;
  return {
    loadedCount,
    failedCount: settled.length - loadedCount,
  };
}

/**
 * Recursively dispose all geometry and materials within a group.
 */
function disposeGroup(group: THREE.Object3D): void {
  group.traverse((object) => {
    if ("geometry" in object) {
      const geometry = (object as { geometry?: THREE.BufferGeometry }).geometry;
      geometry?.dispose();
    }

    if ("material" in object) {
      const material = (object as { material?: THREE.Material | THREE.Material[] }).material;
      if (Array.isArray(material)) {
        material.forEach((entry) => {
          if (entry instanceof THREE.SpriteMaterial) {
            entry.map?.dispose();
          }
          entry.dispose();
        });
      } else if (material) {
        if (material instanceof THREE.SpriteMaterial) {
          material.map?.dispose();
        }
        material.dispose();
      }
    }
  });
}
