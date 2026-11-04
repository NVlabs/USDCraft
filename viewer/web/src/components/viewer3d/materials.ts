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
import type { UrdfMaterialPbr, UrdfVisual } from './urdf-parser';

export interface MaterialSpec {
  name?: string;
  color: THREE.Color;
  opacity: number;
  metalness: number;
  roughness: number;
  transmission: number;
  thickness: number;
  ior: number;
  clearcoat: number;
  clearcoatRoughness: number;
  envMapIntensity: number;
  emissive?: THREE.Color;
}

const DEFAULT_CLEARCOAT_ROUGHNESS = 0.2;

export function depthBiasForOrdinal(ordinal: number): number {
  return 1 + ordinal * 0.25;
}

function srgbColor(r: number, g: number, b: number): THREE.Color {
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

function linearColor(r: number, g: number, b: number): THREE.Color {
  return new THREE.Color().setRGB(r, g, b, THREE.LinearSRGBColorSpace);
}

function softenBrightDiffuse(spec: MaterialSpec, explicit?: UrdfMaterialPbr): void {
  const maxChannel = Math.max(spec.color.r, spec.color.g, spec.color.b);
  const minChannel = Math.min(spec.color.r, spec.color.g, spec.color.b);
  const isBrightNeutral = maxChannel > 0.78 && maxChannel - minChannel < 0.08;

  if (!isBrightNeutral || spec.metalness > 0.2 || spec.transmission > 0.01) {
    return;
  }

  // The studio lighting rig blows out bright diffuse albedo, so the color is
  // always toned down. Authored PBR scalars are respected: only fields the
  // model did not set explicitly get clamped.
  spec.color.lerp(new THREE.Color(0.82, 0.82, 0.78), 0.38);
  if (explicit?.roughness === undefined) {
    spec.roughness = Math.max(spec.roughness, 0.62);
  }
  if (explicit?.clearcoat === undefined) {
    spec.clearcoat = Math.min(spec.clearcoat, 0.08);
  }
  spec.envMapIntensity = Math.min(spec.envMapIntensity, 0.55);
}

/**
 * Resolve visual material specification from URDF data
 * Defaults to a neutral gray if no material specified
 */
export function resolveVisualMaterialSpec(visual: UrdfVisual): MaterialSpec {
  const defaultSpec: MaterialSpec = {
    name: undefined,
    color: srgbColor(0.74, 0.74, 0.72),
    opacity: 1.0,
    metalness: 0.05,
    roughness: 0.48,
    transmission: 0,
    thickness: 0,
    ior: 1.45,
    clearcoat: 0,
    // Matches the SDK normalization used whenever clearcoat is authored
    // without an explicit roughness. Do not inherit renderer-specific defaults.
    clearcoatRoughness: DEFAULT_CLEARCOAT_ROUGHNESS,
    envMapIntensity: 0.8,
  };

  if (!visual.material) {
    return defaultSpec;
  }

  const spec = { ...defaultSpec };
  spec.name = visual.material.name;

  if (visual.material.color) {
    const [r, g, b, a] = visual.material.color.rgba;
    spec.color = srgbColor(r, g, b);
    spec.opacity = a;
  }

  // Name-based presets provide the baseline; explicitly authored PBR fields
  // override per field. The brightness-softening pass still runs afterwards
  // (the lighting rig depends on it) but leaves authored scalars alone.
  const pbr = visual.material.pbr;
  const tuned = applyNamedMaterialPreset(spec);
  applyExplicitPbr(tuned, pbr);
  softenBrightDiffuse(tuned, pbr);
  return tuned;
}

function applyExplicitPbr(spec: MaterialSpec, pbr: UrdfMaterialPbr | undefined): void {
  if (!pbr) return;

  if (pbr.metallic !== undefined) spec.metalness = pbr.metallic;
  if (pbr.roughness !== undefined) spec.roughness = pbr.roughness;
  if (pbr.transmission !== undefined) {
    spec.transmission = pbr.transmission;
    if (pbr.transmission > 0.01) {
      spec.thickness = Math.max(spec.thickness, 0.006);
    }
  }
  if (pbr.ior !== undefined) spec.ior = pbr.ior;
  if (pbr.clearcoat !== undefined) spec.clearcoat = pbr.clearcoat;
  if (pbr.clearcoatRoughness !== undefined) spec.clearcoatRoughness = pbr.clearcoatRoughness;
  if (pbr.emissive !== undefined) {
    // The SDK emissive tuple is already scene-linear (unlike rgba/color).
    spec.emissive = linearColor(pbr.emissive[0], pbr.emissive[1], pbr.emissive[2]);
  }
}

function applyNamedMaterialPreset(spec: MaterialSpec): MaterialSpec {
  const name = spec.name?.toLowerCase() ?? '';
  const tuned = { ...spec };
  const tokens = name.split(/[^a-z0-9]+/).filter(Boolean);
  const hasToken = (token: string): boolean => tokens.includes(token);
  const hasAnyToken = (...candidates: string[]): boolean => candidates.some(hasToken);
  const mentionsTransparentDescriptor = hasAnyToken(
    'clear',
    'glass',
    'transparent',
    'translucent',
    'smoked',
    'frosted',
  );
  const mentionsTransparentPlasticFamily = hasAnyToken(
    'polycarbonate',
    'acrylic',
    'plexi',
    'plexiglass',
    'lexan',
  );
  const isClearRigidMaterial = (
    mentionsTransparentDescriptor
    || (mentionsTransparentPlasticFamily && tuned.opacity < 0.999)
  );

  if (isClearRigidMaterial) {
    tuned.metalness = 0.0;
    tuned.roughness = 0.08;
    tuned.transmission = Math.max(0.45, 1.0 - tuned.opacity * 0.55);
    tuned.thickness = 0.006;
    tuned.ior = 1.45;
    tuned.clearcoat = 0.9;
    tuned.clearcoatRoughness = 0.06;
    tuned.envMapIntensity = 1.25;
    return tuned;
  }

  if (name.includes('reflector')) {
    tuned.metalness = 0.9;
    tuned.roughness = 0.18;
    tuned.opacity = Math.max(tuned.opacity, 0.78);
    tuned.clearcoat = 0.18;
    tuned.clearcoatRoughness = 0.12;
    tuned.envMapIntensity = 1.4;
    return tuned;
  }

  if (name.includes('brass') || name.includes('metal')) {
    tuned.metalness = 0.88;
    tuned.roughness = 0.22;
    tuned.clearcoat = 0.16;
    tuned.clearcoatRoughness = 0.18;
    tuned.envMapIntensity = 1.3;
    return tuned;
  }

  if (name.includes('bakelite')) {
    tuned.metalness = 0.02;
    tuned.roughness = 0.38;
    tuned.clearcoat = 0.24;
    tuned.clearcoatRoughness = 0.2;
    tuned.envMapIntensity = 0.7;
    return tuned;
  }

  if (name.includes('felt')) {
    tuned.metalness = 0.0;
    tuned.roughness = 0.92;
    tuned.clearcoat = 0.0;
    tuned.envMapIntensity = 0.25;
  }

  return tuned;
}

/**
 * Create a THREE.js material from a material spec
 */
export function createMaterial(spec: MaterialSpec, options?: {
  wireframe?: boolean;
  transparent?: boolean;
  side?: THREE.Side;
  depthBias?: number;
}): THREE.MeshPhysicalMaterial {
  const transparent = options?.transparent ?? (spec.opacity < 0.999 || spec.transmission > 0.01);
  const side = options?.side ?? THREE.FrontSide;
  const depthBias = options?.depthBias ?? 0;

  return new THREE.MeshPhysicalMaterial({
    color: spec.color,
    opacity: spec.opacity,
    emissive: spec.emissive ?? new THREE.Color(0x000000),
    metalness: spec.metalness,
    roughness: spec.roughness,
    transmission: spec.transmission,
    thickness: spec.thickness,
    ior: spec.ior,
    clearcoat: spec.clearcoat,
    clearcoatRoughness: spec.clearcoatRoughness,
    envMapIntensity: spec.envMapIntensity,
    transparent,
    depthWrite: !transparent,
    forceSinglePass: transparent,
    wireframe: options?.wireframe ?? false,
    side,
    polygonOffset: depthBias !== 0,
    polygonOffsetFactor: 0,
    polygonOffsetUnits: depthBias,
  });
}

/**
 * Create edge lines for a geometry
 */
export function createEdgeLines(geometry: THREE.BufferGeometry, color: number = 0x000000): THREE.LineSegments {
  const edges = new THREE.EdgesGeometry(geometry, 15); // 15 degree threshold
  const lineMaterial = new THREE.LineBasicMaterial({ color, opacity: 0.3, transparent: true });
  return new THREE.LineSegments(edges, lineMaterial);
}
