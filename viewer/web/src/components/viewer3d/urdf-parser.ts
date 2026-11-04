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

export interface UrdfJoint {
  name: string;
  type: 'fixed' | 'revolute' | 'continuous' | 'prismatic' | 'spherical' | 'floating' | 'planar';
  parent: string;
  child: string;
  origin?: {
    xyz?: [number, number, number];
    rpy?: [number, number, number];
  };
  axis?: [number, number, number];
  limit?: {
    lower?: number;
    upper?: number;
    effort?: number;
    velocity?: number;
  };
  mimic?: {
    joint: string;
    multiplier: number;
    offset: number;
  };
  coneDeg?: number;
}

export interface UrdfVisualGeometry {
  type: 'box' | 'cylinder' | 'sphere' | 'mesh';
  size?: [number, number, number]; // box
  radius?: number; // cylinder, sphere
  length?: number; // cylinder
  filename?: string; // mesh
  scale?: [number, number, number]; // mesh
}

export interface UrdfMaterialPbr {
  metallic?: number;
  roughness?: number;
  emissive?: [number, number, number];
  transmission?: number;
  ior?: number;
  clearcoat?: number;
  clearcoatRoughness?: number;
}

export interface UrdfVisual {
  name?: string;
  origin?: {
    xyz?: [number, number, number];
    rpy?: [number, number, number];
  };
  geometry: UrdfVisualGeometry;
  material?: {
    name?: string;
    color?: { rgba: [number, number, number, number] };
    texture?: { filename: string };
    pbr?: UrdfMaterialPbr;
  };
}

export type UrdfCollisionRole = 'rigid_runtime' | 'deformable_contact';
export type UrdfCollisionRealization = 'runtime' | 'rest_shape';

export interface UrdfCollision extends UrdfVisual {
  collisionRole: UrdfCollisionRole;
  collisionRealization: UrdfCollisionRealization;
  bodyModel: UrdfBodyModel;
}

type UrdfMaterial = NonNullable<UrdfVisual['material']>;

export type UrdfBodyModel =
  | 'rigid'
  | 'flexible_1d'
  | 'surface_deformable'
  | 'volume_deformable';

export type UrdfBodyRealization = 'native' | 'static_rest_pose';

export interface UrdfBodySemantics {
  bodyModel: UrdfBodyModel;
  bodyRealization: UrdfBodyRealization;
}

export interface UrdfLink {
  name: string;
  visuals: UrdfVisual[];
  collisions: UrdfCollision[];
  bodyModel?: UrdfBodyModel;
  bodyRealization?: UrdfBodyRealization;
}

export interface UrdfSpec {
  name: string;
  links: UrdfLink[];
  joints: UrdfJoint[];
}

export interface UrdfCollisionGeometrySummary {
  declaredCount: number;
  rigidRuntimeCount: number;
  deformableRestShapeCount: number;
  missingDeformableRestShapePartCount: number;
  missingDeformableRestShapePartNames: string[];
  primitiveCount: number;
  meshCount: number;
}

export interface UrdfVisualDescriptor {
  key: string;
  label: string;
  index: number;
}

/**
 * Parse a vec3 string like "1 2 3" or "1.0 2.0 3.0"
 */
export function parseVec3(str: string | undefined): [number, number, number] {
  if (!str) return [0, 0, 0];
  const parts = str.trim().split(/\s+/).map(parseFloat);
  return [
    Number.isFinite(parts[0]) ? parts[0] : 0,
    Number.isFinite(parts[1]) ? parts[1] : 0,
    Number.isFinite(parts[2]) ? parts[2] : 0,
  ];
}

/**
 * Parse a vec4 string like "1 0 0 1" for RGBA
 */
export function parseVec4(str: string | undefined): [number, number, number, number] {
  if (!str) return [1, 1, 1, 1];
  const parts = str.trim().split(/\s+/).map(parseFloat);
  return [
    Number.isFinite(parts[0]) ? parts[0] : 1,
    Number.isFinite(parts[1]) ? parts[1] : 1,
    Number.isFinite(parts[2]) ? parts[2] : 1,
    Number.isFinite(parts[3]) ? parts[3] : 1,
  ];
}

export function buildUrdfVisualKey(linkName: string, visualIndex: number): string {
  return `${linkName}::visual:${visualIndex}`;
}

function meshFilenameStem(filename: string | undefined): string | null {
  if (!filename) {
    return null;
  }

  const normalized = filename.split(/[?#]/, 1)[0] ?? '';
  const basename = normalized.split('/').at(-1)?.trim() ?? '';
  if (!basename) {
    return null;
  }

  const stem = basename.replace(/\.[^.]+$/, '');
  return stem || null;
}

function baseVisualLabel(visual: UrdfVisual, index: number): string {
  const explicitName = visual.name?.trim();
  if (explicitName) {
    return explicitName;
  }

  if (visual.geometry.type === 'mesh') {
    const stem = meshFilenameStem(visual.geometry.filename);
    if (stem) {
      return stem;
    }
  }

  return `${visual.geometry.type} ${index + 1}`;
}

export function describeLinkVisuals(link: UrdfLink): UrdfVisualDescriptor[] {
  const baseLabels = link.visuals.map((visual, index) => baseVisualLabel(visual, index));
  const labelTotals = new Map<string, number>();
  for (const label of baseLabels) {
    labelTotals.set(label, (labelTotals.get(label) ?? 0) + 1);
  }

  const seenLabels = new Map<string, number>();
  return baseLabels.map((baseLabel, index) => {
    const nextCount = (seenLabels.get(baseLabel) ?? 0) + 1;
    seenLabels.set(baseLabel, nextCount);

    return {
      key: buildUrdfVisualKey(link.name, index),
      label: (labelTotals.get(baseLabel) ?? 0) > 1 ? `${baseLabel} ${nextCount}` : baseLabel,
      index,
    };
  });
}

export function summarizeUrdfCollisionGeometry(
  urdfSpec: UrdfSpec,
): UrdfCollisionGeometrySummary {
  let declaredCount = 0;
  let rigidRuntimeCount = 0;
  let deformableRestShapeCount = 0;
  let meshCount = 0;
  const missingDeformableRestShapePartNames: string[] = [];

  for (const link of urdfSpec.links) {
    let linkHasDeformableRestShape = false;
    for (const collision of link.collisions) {
      declaredCount += 1;
      if (collision.geometry.type === 'mesh') {
        meshCount += 1;
      }
      if (collision.collisionRole === 'rigid_runtime') {
        rigidRuntimeCount += 1;
      }
      if (
        collision.collisionRole === 'deformable_contact'
        && collision.collisionRealization === 'rest_shape'
      ) {
        deformableRestShapeCount += 1;
        linkHasDeformableRestShape = true;
      }
    }
    if (
      (link.bodyModel ?? 'rigid') !== 'rigid'
      && link.visuals.length > 0
      && !linkHasDeformableRestShape
    ) {
      missingDeformableRestShapePartNames.push(link.name);
    }
  }

  return {
    declaredCount,
    rigidRuntimeCount,
    deformableRestShapeCount,
    missingDeformableRestShapePartCount: missingDeformableRestShapePartNames.length,
    missingDeformableRestShapePartNames,
    primitiveCount: declaredCount - meshCount,
    meshCount,
  };
}

/**
 * Convert URDF fixed-axis RPY (roll, pitch, yaw) to THREE.Matrix4.
 * URDF convention: Rz(yaw) * Ry(pitch) * Rx(roll).
 */
export function rpyToMatrix4(rpy: [number, number, number]): THREE.Matrix4 {
  const [roll, pitch, yaw] = rpy;
  const mx = new THREE.Matrix4().makeRotationX(roll);
  const my = new THREE.Matrix4().makeRotationY(pitch);
  const mz = new THREE.Matrix4().makeRotationZ(yaw);
  return new THREE.Matrix4().multiplyMatrices(mz, new THREE.Matrix4().multiplyMatrices(my, mx));
}

/**
 * Convert URDF origin (xyz + rpy) to THREE.Matrix4
 */
export function originToMatrix4(origin?: { xyz?: [number, number, number]; rpy?: [number, number, number] }): THREE.Matrix4 {
  const xyz = origin?.xyz ?? [0, 0, 0];
  const rpy = origin?.rpy ?? [0, 0, 0];
  const translation = new THREE.Matrix4().makeTranslation(xyz[0], xyz[1], xyz[2]);
  const rotation = rpyToMatrix4(rpy);
  return new THREE.Matrix4().multiplyMatrices(translation, rotation);
}

/**
 * Parse <origin> element
 */
function parseOrigin(originEl: Element | null): { xyz?: [number, number, number]; rpy?: [number, number, number] } | undefined {
  if (!originEl) return undefined;

  const xyzStr = originEl.getAttribute('xyz') ?? undefined;
  const rpyStr = originEl.getAttribute('rpy') ?? undefined;

  const result: { xyz?: [number, number, number]; rpy?: [number, number, number] } = {};

  if (xyzStr) result.xyz = parseVec3(xyzStr);
  if (rpyStr) result.rpy = parseVec3(rpyStr);

  return Object.keys(result).length > 0 ? result : undefined;
}

/**
 * Parse <geometry> element
 */
function parseGeometry(geomEl: Element | null): UrdfVisualGeometry | null {
  if (!geomEl) return null;

  const boxEl = geomEl.querySelector('box');
  if (boxEl) {
    const sizeStr = boxEl.getAttribute('size') ?? undefined;
    return {
      type: 'box',
      size: parseVec3(sizeStr),
    };
  }

  const cylinderEl = geomEl.querySelector('cylinder');
  if (cylinderEl) {
    return {
      type: 'cylinder',
      radius: parseFloat(cylinderEl.getAttribute('radius') || '1'),
      length: parseFloat(cylinderEl.getAttribute('length') || '1'),
    };
  }

  const sphereEl = geomEl.querySelector('sphere');
  if (sphereEl) {
    return {
      type: 'sphere',
      radius: parseFloat(sphereEl.getAttribute('radius') || '1'),
    };
  }

  const meshEl = geomEl.querySelector('mesh');
  if (meshEl) {
    const scaleStr = meshEl.getAttribute('scale') ?? undefined;
    return {
      type: 'mesh',
      filename: meshEl.getAttribute('filename') || '',
      scale: scaleStr ? parseVec3(scaleStr) : undefined,
    };
  }

  return null;
}

/**
 * Parse <material> element
 */
function parseMaterial(matEl: Element | null): UrdfVisual['material'] | undefined {
  if (!matEl) return undefined;

  const name = matEl.getAttribute('name') || undefined;
  const colorEl = matEl.querySelector('color');
  const textureEl = matEl.querySelector('texture');
  const pbrEl = matEl.querySelector('usdcraft_pbr');

  const material: UrdfVisual['material'] = { name };

  if (colorEl) {
    const rgbaStr = colorEl.getAttribute('rgba') ?? undefined;
    material.color = { rgba: parseVec4(rgbaStr) };
  }

  if (textureEl) {
    material.texture = { filename: textureEl.getAttribute('filename') || '' };
  }

  if (pbrEl) {
    const pbr = parsePbr(pbrEl);
    if (pbr) {
      material.pbr = pbr;
    }
  }

  return material;
}

export function parsePbr(pbrEl: Element): UrdfMaterialPbr | undefined {
  const pbr: UrdfMaterialPbr = {};
  const scalar = (attr: string): number | undefined => {
    const raw = pbrEl.getAttribute(attr);
    if (raw === null) return undefined;
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? value : undefined;
  };

  pbr.metallic = scalar('metallic');
  pbr.roughness = scalar('roughness');
  pbr.transmission = scalar('transmission');
  pbr.ior = scalar('ior');
  pbr.clearcoat = scalar('clearcoat');
  pbr.clearcoatRoughness = scalar('clearcoat_roughness');

  const emissiveRaw = pbrEl.getAttribute('emissive_rgb');
  if (emissiveRaw) {
    const parts = emissiveRaw.trim().split(/\s+/).map(Number.parseFloat);
    if (parts.length === 3 && parts.every(Number.isFinite)) {
      pbr.emissive = [parts[0], parts[1], parts[2]];
    }
  }

  const hasAnyField = Object.values(pbr).some((value) => value !== undefined);
  return hasAnyField ? pbr : undefined;
}

function resolveMaterial(
  material: UrdfVisual['material'],
  materialLibrary: Map<string, UrdfMaterial>,
): UrdfVisual['material'] {
  if (!material?.name) {
    return material;
  }

  const libraryMaterial = materialLibrary.get(material.name);
  if (!libraryMaterial) {
    return material;
  }

  return {
    ...libraryMaterial,
    ...material,
    color: material.color ?? libraryMaterial.color,
    texture: material.texture ?? libraryMaterial.texture,
    pbr: material.pbr ?? libraryMaterial.pbr,
  };
}

/**
 * Parse <visual> or <collision> element
 */
function parseVisual(visualEl: Element, materialLibrary: Map<string, UrdfMaterial>): UrdfVisual {
  const name = visualEl.getAttribute('name') || undefined;
  const originEl = visualEl.querySelector('origin');
  const geomEl = visualEl.querySelector('geometry');
  const matEl = visualEl.querySelector('material');

  const geometry = parseGeometry(geomEl);
  if (!geometry) {
    throw new Error('Visual element missing valid geometry');
  }

  return {
    name,
    origin: parseOrigin(originEl),
    geometry,
    material: resolveMaterial(parseMaterial(matEl), materialLibrary),
  };
}

export function parseUrdfVendorCollisionElement(
  collisionEl: Element,
  parentBodyModel: UrdfBodyModel,
): UrdfCollision | null {
  const collisionSemantics = resolveUrdfCollisionSemantics(
    collisionEl.getAttribute('role'),
    collisionEl.getAttribute('realization'),
    collisionEl.getAttribute('body_model'),
    parentBodyModel,
  );
  if (!collisionSemantics) {
    return null;
  }

  try {
    return {
      ...parseVisual(collisionEl, new Map()),
      ...collisionSemantics,
    };
  } catch {
    return null;
  }
}

/**
 * Parse <link> element
 */
function parseLink(linkEl: Element, materialLibrary: Map<string, UrdfMaterial>): UrdfLink {
  const name = linkEl.getAttribute('name') || '';
  const bodySemantics = resolveUrdfBodySemantics(
    linkEl.getAttribute('usdcraft_body_model'),
    linkEl.getAttribute('usdcraft_body_realization'),
  );

  const visuals: UrdfVisual[] = [];
  for (const visualEl of linkEl.querySelectorAll(':scope > visual')) {
    visuals.push(parseVisual(visualEl, materialLibrary));
  }

  const collisions: UrdfCollision[] = [];
  for (const collisionEl of linkEl.querySelectorAll(':scope > collision')) {
    collisions.push({
      ...parseVisual(collisionEl, materialLibrary),
      collisionRole: 'rigid_runtime',
      collisionRealization: 'runtime',
      bodyModel: 'rigid',
    });
  }

  for (const collisionEl of linkEl.querySelectorAll(':scope > usdcraft_collision')) {
    const collision = parseUrdfVendorCollisionElement(collisionEl, bodySemantics.bodyModel);
    if (collision) {
      collisions.push(collision);
    } else {
      console.warn(
        `Ignoring invalid usdcraft_collision on link "${name || '<unnamed>'}".`,
      );
    }
  }

  return { name, visuals, collisions, ...bodySemantics };
}

export function resolveUrdfCollisionSemantics(
  roleValue: string | null,
  realizationValue: string | null,
  bodyModelValue: string | null,
  parentBodyModel: UrdfBodyModel,
): Pick<UrdfCollision, 'collisionRole' | 'collisionRealization' | 'bodyModel'> | null {
  if (
    roleValue !== 'deformable_contact'
    || realizationValue !== 'rest_shape'
    || parentBodyModel === 'rigid'
  ) {
    return null;
  }

  const resolvedBodyModel = bodyModelValue === null ? parentBodyModel : bodyModelValue;
  if (resolvedBodyModel !== parentBodyModel) {
    return null;
  }

  return {
    collisionRole: 'deformable_contact',
    collisionRealization: 'rest_shape',
    bodyModel: parentBodyModel,
  };
}

export function resolveUrdfBodySemantics(
  bodyModelValue: string | null,
  bodyRealizationValue: string | null,
): UrdfBodySemantics {
  const bodyModels: UrdfBodyModel[] = [
    'rigid',
    'flexible_1d',
    'surface_deformable',
    'volume_deformable',
  ];
  const bodyModel = bodyModels.includes(bodyModelValue as UrdfBodyModel)
    ? bodyModelValue as UrdfBodyModel
    : 'rigid';
  const defaultRealization: UrdfBodyRealization = bodyModel === 'rigid'
    ? 'native'
    : 'static_rest_pose';
  const bodyRealization = bodyRealizationValue === 'native'
    || bodyRealizationValue === 'static_rest_pose'
    ? bodyRealizationValue
    : defaultRealization;

  return { bodyModel, bodyRealization };
}

/**
 * Parse <joint> element
 */
export function resolveUrdfJointType(
  urdfType: string | null,
  usdcraftType: string | null,
): UrdfJoint['type'] {
  if (usdcraftType === 'spherical') return 'spherical';
  return (urdfType || 'fixed') as UrdfJoint['type'];
}

function parseJoint(jointEl: Element): UrdfJoint {
  const name = jointEl.getAttribute('name') || '';
  const type = resolveUrdfJointType(
    jointEl.getAttribute('type'),
    jointEl.getAttribute('usdcraft_type'),
  );

  const parentEl = jointEl.querySelector('parent');
  const childEl = jointEl.querySelector('child');

  const parent = parentEl?.getAttribute('link') || '';
  const child = childEl?.getAttribute('link') || '';

  const originEl = jointEl.querySelector('origin');
  const axisEl = jointEl.querySelector('axis');
  const limitEl = jointEl.querySelector('limit');
  const mimicEl = jointEl.querySelector('mimic');

  const joint: UrdfJoint = {
    name,
    type,
    parent,
    child,
  };

  if (originEl) {
    joint.origin = parseOrigin(originEl);
  }

  const sphericalAxis = type === 'spherical'
    ? jointEl.getAttribute('usdcraft_axis') ?? undefined
    : undefined;
  if (axisEl || sphericalAxis) {
    const xyzStr = sphericalAxis ?? axisEl?.getAttribute('xyz') ?? undefined;
    joint.axis = parseVec3(xyzStr);
  }

  if (type === 'spherical') {
    const coneDeg = Number.parseFloat(jointEl.getAttribute('usdcraft_cone_deg') ?? '');
    if (Number.isFinite(coneDeg)) joint.coneDeg = coneDeg;
  }

  if (limitEl) {
    joint.limit = {
      lower: parseFloat(limitEl.getAttribute('lower') || '0'),
      upper: parseFloat(limitEl.getAttribute('upper') || '0'),
      effort: parseFloat(limitEl.getAttribute('effort') || '0'),
      velocity: parseFloat(limitEl.getAttribute('velocity') || '0'),
    };
  }

  if (mimicEl) {
    const mimicJoint = mimicEl.getAttribute('joint') || '';
    if (mimicJoint) {
      joint.mimic = {
        joint: mimicJoint,
        multiplier: parseFloat(mimicEl.getAttribute('multiplier') || '1'),
        offset: parseFloat(mimicEl.getAttribute('offset') || '0'),
      };
    }
  }

  return joint;
}

/**
 * Rewrite mesh filenames from absolute package:// paths to repo-relative paths
 * e.g., "package://robot/assets/meshes/link.dae" -> "assets/meshes/link.dae"
 */
export function rewriteAbsoluteMeshFilenames(urdfSpec: UrdfSpec): UrdfSpec {
  const rewrite = (filename: string): string => {
    if (filename.startsWith('package://')) {
      // Extract everything after the first package name
      const match = filename.match(/^package:\/\/[^/]+\/(.+)$/);
      return match ? match[1] : filename;
    }
    return filename;
  };

  const rewriteVisuals = <T extends UrdfVisual>(visuals: T[]): T[] => {
    return visuals.map(v => {
      if (v.geometry.type === 'mesh' && v.geometry.filename) {
        return {
          ...v,
          geometry: {
            ...v.geometry,
            filename: rewrite(v.geometry.filename),
          },
        };
      }
      return v;
    });
  };

  return {
    ...urdfSpec,
    links: urdfSpec.links.map(link => ({
      ...link,
      visuals: rewriteVisuals(link.visuals),
      collisions: rewriteVisuals(link.collisions),
    })),
  };
}

/**
 * Find the root link (link that is not a child of any joint)
 */
export function findRootLink(urdfSpec: UrdfSpec): string | null {
  const childLinks = new Set(urdfSpec.joints.map(j => j.child));
  const rootLink = urdfSpec.links.find(link => !childLinks.has(link.name));
  return rootLink?.name || null;
}

/**
 * Parse URDF XML string into structured spec
 */
export function parseUrdf(urdfXml: string): UrdfSpec {
  const parser = new DOMParser();
  const doc = parser.parseFromString(urdfXml, 'application/xml');

  const robotEl = doc.querySelector('robot');
  if (!robotEl) {
    throw new Error('URDF missing <robot> root element');
  }

  const name = robotEl.getAttribute('name') || 'robot';
  const materialLibrary = new Map<string, UrdfMaterial>();

  for (const materialEl of robotEl.querySelectorAll(':scope > material')) {
    const material = parseMaterial(materialEl);
    if (!material?.name) {
      continue;
    }
    materialLibrary.set(material.name, material);
  }

  const links: UrdfLink[] = [];
  for (const linkEl of robotEl.querySelectorAll(':scope > link')) {
    links.push(parseLink(linkEl, materialLibrary));
  }

  const joints: UrdfJoint[] = [];
  for (const jointEl of robotEl.querySelectorAll(':scope > joint')) {
    joints.push(parseJoint(jointEl));
  }

  return { name, links, joints };
}
