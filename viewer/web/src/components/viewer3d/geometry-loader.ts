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
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import type { UrdfVisualGeometry } from './urdf-parser';
import type { MaterialSpec } from './materials';
import { createMaterial, createEdgeLines } from './materials';

export interface LoadGeometryOptions {
  kind?: 'visual' | 'collision';
  materialSpec?: MaterialSpec;
  doubleSided?: boolean;
  depthBias?: number;
  assetRevisionKey?: string | null;
}

const geometryTemplateCache = new Map<string, Promise<THREE.Object3D>>();

function appendRevisionParam(url: string, assetRevisionKey: string | null | undefined): string {
  if (!assetRevisionKey) {
    return url;
  }

  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}rev=${encodeURIComponent(assetRevisionKey)}`;
}

function cloneCachedObject(root: THREE.Object3D): THREE.Object3D {
  const clone = root.clone(true);

  clone.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) {
      return;
    }

    const material = child.material;
    if (Array.isArray(material)) {
      child.material = material.map((entry) => entry.clone());
      return;
    }

    if (material) {
      child.material = material.clone();
    }
  });

  return clone;
}

function loadGeometryTemplate(
  resolvedUrl: string,
  extension: string | undefined,
): Promise<THREE.Object3D> {
  const cached = geometryTemplateCache.get(resolvedUrl);
  if (cached) {
    return cached;
  }

  const pending = (async () => {
    if (extension === 'glb' || extension === 'gltf') {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(resolvedUrl);
      return gltf.scene;
    }

    if (extension === 'obj') {
      const loader = new OBJLoader();
      return loader.loadAsync(resolvedUrl);
    }

    throw new Error(`Unsupported mesh file format for ${resolvedUrl}`);
  })();

  geometryTemplateCache.set(resolvedUrl, pending);
  void pending.catch(() => {
    if (geometryTemplateCache.get(resolvedUrl) === pending) {
      geometryTemplateCache.delete(resolvedUrl);
    }
  });

  return pending;
}

/**
 * Build a Three.js mesh from a URDF primitive geometry spec (box, cylinder, sphere).
 * Returns null for mesh-type geometries (use loadGeometryObject instead).
 */
export function buildPrimitiveMesh(
  geometry: UrdfVisualGeometry,
  materialSpec: MaterialSpec,
  options: Pick<LoadGeometryOptions, 'doubleSided' | 'depthBias'> = {},
): THREE.Mesh | null {
  let bufferGeometry: THREE.BufferGeometry;

  switch (geometry.type) {
    case 'box': {
      const [x, y, z] = geometry.size ?? [1, 1, 1];
      bufferGeometry = new THREE.BoxGeometry(x, y, z);
      break;
    }
    case 'cylinder': {
      const radius = geometry.radius ?? 1;
      const length = geometry.length ?? 1;
      bufferGeometry = new THREE.CylinderGeometry(radius, radius, length, 32);
      // URDF cylinders are aligned to the Z axis; Three.js uses Y.
      bufferGeometry.rotateX(Math.PI / 2);
      break;
    }
    case 'sphere': {
      const radius = geometry.radius ?? 1;
      bufferGeometry = new THREE.SphereGeometry(radius, 32, 16);
      break;
    }
    default:
      return null;
  }

  const material = createMaterial(materialSpec, {
    side: options.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
    depthBias: options.depthBias,
  });
  const mesh = new THREE.Mesh(bufferGeometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/**
 * Load an external mesh file (GLB/GLTF or OBJ) referenced by a URDF mesh geometry.
 * Returns a promise that resolves to a THREE.Group containing the loaded model.
 */
export async function loadGeometryObject(
  geometry: UrdfVisualGeometry,
  baseUrl: string,
  options: LoadGeometryOptions = {},
): Promise<THREE.Group> {
  if (geometry.type !== 'mesh' || !geometry.filename) {
    throw new Error(`loadGeometryObject requires a mesh geometry with a filename`);
  }

  const filename = geometry.filename;
  const url = baseUrl.endsWith('/')
    ? `${baseUrl}${filename}`
    : `${baseUrl}/${filename}`;
  const resolvedUrl = appendRevisionParam(url, options.assetRevisionKey);

  const extension = filename.split('.').pop()?.toLowerCase();
  const scale = geometry.scale ?? [1, 1, 1];
  if (extension !== 'glb' && extension !== 'gltf' && extension !== 'obj') {
    throw new Error(`Unsupported mesh file format: .${extension} (${filename})`);
  }

  const template = await loadGeometryTemplate(resolvedUrl, extension);
  const group = new THREE.Group();
  group.add(cloneCachedObject(template));
  group.scale.set(scale[0], scale[1], scale[2]);
  applyLoadedMeshPresentation(group, options);
  return group;
}

function hasBaseColorTexture(material: THREE.Material): boolean {
  return Boolean((material as THREE.MeshStandardMaterial).map);
}

function configurePreservedMaterial(
  material: THREE.Material,
  options: Pick<LoadGeometryOptions, 'doubleSided' | 'depthBias'>,
): void {
  // Blender-exported GLBs routinely tag fully opaque PBR materials as alphaMode BLEND. GLTFLoader
  // then disables depth writing, so shells stop occluding their interiors and evidence renders
  // become see-through. A preserved material with full opacity carries no real blend signal:
  // restore depth-correct rendering and keep any texture cut-outs through alphaTest. Materials
  // with a genuine partial opacity stay transparent.
  if (material.transparent && material.opacity >= 0.999) {
    material.transparent = false;
    material.depthWrite = true;
    material.alphaTest = Math.max(material.alphaTest, 0.5);
  }
  if (options.doubleSided !== undefined) {
    material.side = options.doubleSided ? THREE.DoubleSide : THREE.FrontSide;
  }
  if (options.depthBias) {
    material.polygonOffset = true;
    material.polygonOffsetFactor = 0;
    material.polygonOffsetUnits = options.depthBias;
  }
  material.needsUpdate = true;
}

export function applyLoadedMeshPresentation(
  root: THREE.Object3D,
  options: LoadGeometryOptions,
): void {
  const kind = options.kind ?? 'visual';
  const materialSpec = options.materialSpec;
  const side = options.doubleSided ? THREE.DoubleSide : THREE.FrontSide;
  const depthBias = options.depthBias;

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) {
      return;
    }

    if (!child.geometry.attributes.normal) {
      child.geometry.computeVertexNormals();
    }

    const loadedMaterials = Array.isArray(child.material)
      ? child.material
      : [child.material];

    if (!materialSpec) {
      // Source carriers intentionally omit an explicit URDF material. Preserve
      // every embedded GLB material slot, including untextured PBR parameters and
      // vertex-color enablement; a base-color map is not required for authority.
      for (const material of loadedMaterials) {
        configurePreservedMaterial(material, options);
      }
    } else {
      const presentedMaterials = loadedMaterials.map((loadedMaterial) => {
        // A textured GLB (e.g. a decal/label) carries its own baseColorTexture;
        // preserve each such slot independently instead of flattening a mixed
        // material array based on slot zero.
        if (hasBaseColorTexture(loadedMaterial)) {
          configurePreservedMaterial(loadedMaterial, options);
          return loadedMaterial;
        }

        loadedMaterial.dispose();
        return kind === 'collision'
          ? createMaterial(materialSpec, {
              side,
              transparent: true,
              depthBias,
            })
          : createMaterial(materialSpec, {
              side,
              depthBias,
            });
      });
      child.material = Array.isArray(child.material)
        ? presentedMaterials
        : presentedMaterials[0];
    }

    child.castShadow = kind === 'visual';
    child.receiveShadow = kind === 'visual';
    child.userData.usdcraftVisual = kind === 'visual';
    child.userData.usdcraftCollision = kind === 'collision';
    child.visible = kind === 'visual';

    if (kind === 'collision') {
      child.renderOrder = 10;
      child.add(createEdgeLines(child.geometry, materialSpec?.color.getHex() ?? 0x000000));
    }
  });
}
