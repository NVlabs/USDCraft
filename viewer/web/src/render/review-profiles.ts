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
 * Canonical review render profiles (A2+ point 3).
 *
 * A profile pins every renderer state that could make "what the reviewer saw"
 * differ from "what the user sees": lighting quality, PMREM environment,
 * shadows, whether authored PBR reaches the screen, background, camera FOV,
 * and pixel ratio. Review renders MUST name a profile; the profile id is
 * hash-bound into the render manifest.
 *
 * - `review-pbr`: appearance evidence. PMREM + shadows + the fancy lighting
 *   rig with AUTHORED PBR untouched — the interactive viewer's studio-shading
 *   material override (which clamps roughness/clearcoat for readability) is
 *   deliberately NOT applied, so polished metal and glass keep their authored
 *   parameters.
 * - `review-shape`: geometry evidence. The same lighting/camera, but every
 *   visual is overridden with one matte neutral clay material so silhouettes,
 *   part boundaries, and negative space stay readable regardless of authored
 *   reflectance (a mirror finish can hide geometry).
 */

export type ReviewProfileId = 'review-pbr' | 'review-shape' | 'review-parts';

export interface ReviewProfile {
  id: ReviewProfileId;
  /** Authored PBR materials reach the screen unmodified. */
  authoredPbr: boolean;
  /** Override every visual with the neutral clay material. */
  clayOverride: boolean;
  /** Override every visual with one flat unlit colour per URDF link (part identification). */
  partsColourOverride: boolean;
  /** useThreeScene lighting quality (fancy = studio rig + shadows). */
  fancyGraphics: boolean;
  /** PMREM environment map (the only reflection source for metals). */
  environment: boolean;
  shadows: boolean;
  /** Fixed vertical field of view for every review camera. */
  fovDeg: number;
  /** Renderer pixel ratio is pinned to 1 (no DPR-dependent output). */
  pixelRatio: 1;
  /** Grid/axis/edges/overlays are always disabled in review renders. */
  helpers: false;
}

export const REVIEW_PBR: ReviewProfile = {
  id: 'review-pbr',
  authoredPbr: true,
  clayOverride: false,
  partsColourOverride: false,
  fancyGraphics: true,
  environment: true,
  shadows: true,
  fovDeg: 35,
  pixelRatio: 1,
  helpers: false,
};

export const REVIEW_SHAPE: ReviewProfile = {
  id: 'review-shape',
  authoredPbr: false,
  clayOverride: true,
  partsColourOverride: false,
  fancyGraphics: true,
  environment: true,
  shadows: true,
  fovDeg: 35,
  pixelRatio: 1,
  helpers: false,
};

/**
 * review-parts: part-identification evidence. Every visual mesh of one URDF
 * link is painted a single flat unlit colour from a deterministic palette
 * (golden-angle hue spacing over the sorted link names), so a reviewer can
 * name a part by its colour without any lighting or material ambiguity. The
 * colour legend (link name -> hex) is exposed by the render page and recorded
 * in the render manifest.
 */
export const REVIEW_PARTS: ReviewProfile = {
  id: 'review-parts',
  authoredPbr: false,
  clayOverride: false,
  partsColourOverride: true,
  fancyGraphics: true,
  environment: true,
  shadows: false,
  fovDeg: 35,
  pixelRatio: 1,
  helpers: false,
};

/**
 * Backdrop of every review render (hex for Three). Pinned independently of the interactive
 * Viewer's light viewport so review evidence images keep their established dark backdrop.
 */
export const REVIEW_BACKGROUND = 0x0b0e0b;

export const REVIEW_PROFILES: Record<ReviewProfileId, ReviewProfile> = {
  'review-pbr': REVIEW_PBR,
  'review-shape': REVIEW_SHAPE,
  'review-parts': REVIEW_PARTS,
};

/**
 * Deterministic flat colour for the i-th link of a sorted link-name list
 * (golden-angle hue spacing; saturation/lightness fixed for legibility).
 */
export function partsColourHexForIndex(index: number): string {
  const hue = (index * 137.508) % 360;
  const s = 0.72;
  const l = 0.52;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = hue / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let rgb: [number, number, number];
  if (hp < 1) rgb = [c, x, 0];
  else if (hp < 2) rgb = [x, c, 0];
  else if (hp < 3) rgb = [0, c, x];
  else if (hp < 4) rgb = [0, x, c];
  else if (hp < 5) rgb = [x, 0, c];
  else rgb = [c, 0, x];
  const m = l - c / 2;
  const to255 = (v: number) => Math.round((v + m) * 255);
  const hex = (v: number) => to255(v).toString(16).padStart(2, '0');
  return `#${hex(rgb[0])}${hex(rgb[1])}${hex(rgb[2])}`;
}

/** Matte neutral clay used by review-shape (geometry readability). */
export const CLAY_MATERIAL_PARAMS = {
  color: 0xb8b6b0,
  metalness: 0.0,
  roughness: 0.85,
  clearcoat: 0.0,
  envMapIntensity: 0.35,
} as const;

export function resolveReviewProfile(id: string): ReviewProfile {
  const profile = REVIEW_PROFILES[id as ReviewProfileId];
  if (!profile) {
    throw new Error(
      `unknown review profile ${JSON.stringify(id)}; known: ${Object.keys(REVIEW_PROFILES).join(', ')}`,
    );
  }
  return profile;
}
