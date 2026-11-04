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

/**
 * Headless review renderer entry (A2+).
 *
 * Mounts the SAME scene/loader/material stack the interactive viewer uses
 * (useThreeScene + useUrdfLoader: USDZ-first with URDF fallback, authored-PBR
 * rebinding, PMREM environment, studio lighting rig) without any UI chrome,
 * and exposes a controlled API on `window.usdcraftRenderer` for the Python
 * render backend.
 *
 * Rendering state is pinned by a named review profile (src/render/
 * review-profiles.ts); the camera convention is shared with the numpy
 * renderer through src/render/review-camera.ts. SceneCanvas's studio-shading
 * override is intentionally NOT applied: review-pbr shows authored PBR,
 * review-shape overrides everything with matte clay.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { useThreeScene } from './components/viewer3d/useThreeScene';
import { useJointController } from './components/viewer3d/useJointController';
import { ROBOT_GROUP_NAME, useUrdfLoader } from './components/viewer3d/useUrdfLoader';
import {
  applyRendererJointPose,
  normalizeRendererJointPose,
} from './render/headless-joint-pose';
import {
  resolveReviewCamera,
  type ResolvedReviewCamera,
  type ReviewCameraPose,
} from './render/review-camera';
import {
  CLAY_MATERIAL_PARAMS,
  partsColourHexForIndex,
  REVIEW_BACKGROUND,
  resolveReviewProfile,
  type ReviewProfile,
} from './render/review-profiles';

interface RendererDiagnostics {
  webglVendor: string | null;
  webglRenderer: string | null;
  bundleMode: string;
  threeRevision: string;
  devicePixelRatio: number;
  profile: string;
  loaderState: string;
  loadError: string | null;
  robotPresent: boolean;
  boundingSphere: { center: number[]; radius: number } | null;
  assetRevisionKey: string | null;
}

interface UsdcraftRenderer {
  load(baseFileUrl: string, assetRevisionKey?: string | null): Promise<void>;
  setProfile(profileId: string): string;
  partsLegend(): Record<string, string> | null;
  setJointPose(pose: Record<string, number>): void;
  setCamera(pose: ReviewCameraPose): ResolvedReviewCamera;
  renderOnce(): void;
  capturePng(): string;
  probeWebgl2(): { available: boolean; vendor: string | null; renderer: string | null };
  diagnostics(): RendererDiagnostics;
  readyState(): string;
}

declare global {
  interface Window {
    usdcraftRenderer?: UsdcraftRenderer;
    usdcraftRendererReady?: boolean;
  }
}

function computeModelBoundingSphere(scene: THREE.Scene): THREE.Sphere | null {
  const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
  if (!robot) {
    return null;
  }
  const box = new THREE.Box3().setFromObject(robot);
  if (box.isEmpty()) {
    return null;
  }
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  return sphere;
}

const CLAY_MATERIAL = new THREE.MeshPhysicalMaterial({
  color: CLAY_MATERIAL_PARAMS.color,
  metalness: CLAY_MATERIAL_PARAMS.metalness,
  roughness: CLAY_MATERIAL_PARAMS.roughness,
  clearcoat: CLAY_MATERIAL_PARAMS.clearcoat,
  envMapIntensity: CLAY_MATERIAL_PARAMS.envMapIntensity,
});

/**
 * Colour legend of the active review-parts override (link name -> hex),
 * null under every other profile. One renderer per page, so module state is safe.
 */
let partsLegendCurrent: Record<string, string> | null = null;

/**
 * Apply the active profile's material override (clay, parts colours, or none).
 * Authored materials are always restored first and stashed on userData, so
 * switching between profiles never loses or mis-stashes authored PBR.
 */
function applyReviewMaterialOverride(scene: THREE.Scene, profile: ReviewProfile): void {
  const robot = scene.getObjectByName(ROBOT_GROUP_NAME);
  partsLegendCurrent = null;
  if (!robot) {
    return;
  }
  const partsMaterials = new Map<string, THREE.MeshBasicMaterial>();
  if (profile.partsColourOverride) {
    const linkNames = new Set<string>();
    robot.traverse((object) => {
      if (object instanceof THREE.Mesh && object.userData.usdcraftVisual === true) {
        const link = object.userData.usdcraftLinkName;
        if (typeof link === 'string' && link.length > 0) {
          linkNames.add(link);
        }
      }
    });
    const legend: Record<string, string> = {};
    [...linkNames].sort().forEach((name, index) => {
      const hexText = partsColourHexForIndex(index);
      legend[name] = hexText;
      partsMaterials.set(name, new THREE.MeshBasicMaterial({ color: new THREE.Color(hexText) }));
    });
    partsLegendCurrent = legend;
  }
  robot.traverse((object) => {
    if (!(object instanceof THREE.Mesh) || object.userData.usdcraftVisual !== true) {
      return;
    }
    if (object.userData.usdcraftAuthoredMaterial !== undefined) {
      object.material = object.userData.usdcraftAuthoredMaterial;
      delete object.userData.usdcraftAuthoredMaterial;
    }
    let override: THREE.Material | undefined;
    if (profile.clayOverride) {
      override = CLAY_MATERIAL;
    } else if (profile.partsColourOverride) {
      const link = object.userData.usdcraftLinkName;
      override = typeof link === 'string' ? partsMaterials.get(link) : undefined;
    }
    if (override !== undefined) {
      object.userData.usdcraftAuthoredMaterial = object.material;
      object.material = override;
    }
  });
}

function RenderHarness() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [baseFileUrl, setBaseFileUrl] = useState<string | null>(null);
  const [assetRevisionKey, setAssetRevisionKey] = useState<string | null>(null);
  const [jointPose, setJointPose] = useState<Map<string, number>>(new Map());
  const profileRef = useRef<ReviewProfile>(resolveReviewProfile('review-pbr'));
  const requestedAssetRef = useRef<{ baseUrl: string; revision: string | null } | null>(null);

  const sceneState = useThreeScene(containerRef, {
    fancyGraphics: true,
    continuousRender: false,
    maxPixelRatio: 1,
    background: REVIEW_BACKGROUND,
  });
  const { scene, camera, controls, gridGroup, axisGroup, invalidate } = sceneState;

  const loaderState = useUrdfLoader(
    baseFileUrl,
    assetRevisionKey,
    jointPose,
    false,
    scene,
    camera,
    controls,
    gridGroup,
    axisGroup,
    invalidate,
    null,
  );
  const jointController = useJointController(loaderState.jointNodes, loaderState.urdfSpec);
  const jointControllerRef = useRef(jointController);
  jointControllerRef.current = jointController;

  useEffect(() => {
    applyRendererJointPose(
      jointPose,
      { applyJointValues: jointController.applyJointValues },
      invalidate,
    );
  }, [invalidate, jointController.applyJointValues, jointPose, loaderState.loadedAssetKey]);

  // Review profiles never show helpers: the object alone on the backdrop.
  useEffect(() => {
    if (gridGroup) gridGroup.visible = false;
    if (axisGroup) axisGroup.visible = false;
    invalidate();
  }, [gridGroup, axisGroup, invalidate]);

  const loaderRef = useRef(loaderState);
  loaderRef.current = loaderState;
  const sceneRef = useRef(sceneState);
  sceneRef.current = sceneState;
  // One bounding sphere per loaded model, shared by every orbit view so the
  // object's on-screen scale cannot drift between images (A2+ point 8).
  const sphereRef = useRef<THREE.Sphere | null>(null);

  const api = useMemo<UsdcraftRenderer>(() => {
    const readyState = () => {
      const loader = loaderRef.current;
      const state = sceneRef.current;
      const requested = requestedAssetRef.current;
      if (!state.sceneReady) return 'scene_initializing';
      if (!requested) return 'idle';
      if (loader.error) return `error:${loader.error}`;
      if (loader.loading) return 'loading';
      const requestedKey = `${requested.baseUrl}|${requested.revision ?? ''}`;
      if (loader.loadedAssetKey !== requestedKey) return 'loading_requested_revision';
      if (!state.scene?.getObjectByName(ROBOT_GROUP_NAME)) return 'awaiting_robot';
      return 'ready';
    };

    return {
      async load(nextBaseFileUrl: string, nextRevisionKey: string | null = null) {
        sphereRef.current = null;
        requestedAssetRef.current = {
          baseUrl: nextBaseFileUrl,
          revision: nextRevisionKey,
        };
        setJointPose(new Map());
        setAssetRevisionKey(nextRevisionKey);
        setBaseFileUrl(nextBaseFileUrl);
      },
      setProfile(profileId: string) {
        const state = sceneRef.current;
        const profile = resolveReviewProfile(profileId);
        profileRef.current = profile;
        if (state.scene) {
          applyReviewMaterialOverride(state.scene, profile);
          state.invalidate();
        }
        return profile.id;
      },
      partsLegend() {
        return partsLegendCurrent;
      },
      setJointPose(pose: Record<string, number>) {
        const nextPose = normalizeRendererJointPose(pose);
        sphereRef.current = null;
        setJointPose(nextPose);
        applyRendererJointPose(
          nextPose,
          jointControllerRef.current,
          sceneRef.current.invalidate,
        );
      },
      setCamera(pose: ReviewCameraPose) {
        const state = sceneRef.current;
        if (!state.scene || !state.camera || !state.controls) {
          throw new Error('scene not ready');
        }
        // Profile materials must be in place before the sphere is computed the
        // first time (material overrides do not change bounds, but a late
        // profile switch after capture would).
        applyReviewMaterialOverride(state.scene, profileRef.current);
        if (!sphereRef.current) {
          sphereRef.current = computeModelBoundingSphere(state.scene);
        }
        const sphere = sphereRef.current;
        if (!sphere) {
          throw new Error('no robot geometry to frame');
        }
        const resolved = resolveReviewCamera(
          { ...pose, fovDeg: pose.fovDeg ?? profileRef.current.fovDeg },
          sphere,
        );
        state.camera.fov = resolved.fovDeg;
        state.camera.position.fromArray(resolved.eye);
        state.camera.near = Math.max(resolved.distance / 1000, 0.001);
        state.camera.far = resolved.distance * 20;
        state.camera.updateProjectionMatrix();
        state.controls.target.fromArray(resolved.center);
        state.controls.update();
        state.invalidate();
        return resolved;
      },
      renderOnce() {
        const state = sceneRef.current;
        if (!state.renderer || !state.scene || !state.camera) {
          throw new Error('scene not ready');
        }
        state.renderer.render(state.scene, state.camera);
      },
      capturePng() {
        const state = sceneRef.current;
        if (!state.renderer || !state.scene || !state.camera) {
          throw new Error('scene not ready');
        }
        // Synchronous render + readback in the same task so the drawing buffer
        // is still valid without preserveDrawingBuffer.
        state.renderer.render(state.scene, state.camera);
        return state.renderer.domElement.toDataURL('image/png');
      },
      probeWebgl2() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
          return { available: false, vendor: null, renderer: null };
        }
        const info = gl.getExtension('WEBGL_debug_renderer_info');
        const vendor = info ? String(gl.getParameter(info.UNMASKED_VENDOR_WEBGL)) : null;
        const rendererName = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : null;
        gl.getExtension('WEBGL_lose_context')?.loseContext();
        return { available: true, vendor, renderer: rendererName };
      },
      diagnostics() {
        const state = sceneRef.current;
        const gl = state.renderer?.getContext() ?? null;
        let vendor: string | null = null;
        let rendererName: string | null = null;
        if (gl) {
          const info = gl.getExtension('WEBGL_debug_renderer_info');
          if (info) {
            vendor = String(gl.getParameter(info.UNMASKED_VENDOR_WEBGL));
            rendererName = String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL));
          }
        }
        const sphere = sphereRef.current;
        return {
          webglVendor: vendor,
          webglRenderer: rendererName,
          bundleMode: import.meta.env.MODE,
          threeRevision: THREE.REVISION,
          devicePixelRatio: window.devicePixelRatio,
          profile: profileRef.current.id,
          loaderState: readyState(),
          loadError: loaderRef.current.error ?? null,
          robotPresent: Boolean(state.scene?.getObjectByName(ROBOT_GROUP_NAME)),
          boundingSphere: sphere
            ? { center: sphere.center.toArray(), radius: sphere.radius }
            : null,
          assetRevisionKey: requestedAssetRef.current?.revision ?? null,
        };
      },
      readyState,
    };
  }, []);

  useEffect(() => {
    window.usdcraftRenderer = api;
    window.usdcraftRendererReady = true;
    return () => {
      window.usdcraftRendererReady = false;
      delete window.usdcraftRenderer;
    };
  }, [api]);

  return <div id="render-container" ref={containerRef} style={{ position: 'fixed', inset: 0 }} />;
}

createRoot(document.getElementById('render-root')!).render(<RenderHarness />);
