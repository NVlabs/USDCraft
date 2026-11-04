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

import * as THREE from "three";

export type OverlayAppearance = {
  color: THREE.ColorRepresentation;
  opacity: number;
  renderOrder: number;
};

type ColorMaterial = THREE.Material & {
  color: THREE.Color;
};

type MapMaterial = THREE.Material & {
  map: THREE.Texture | null;
};

type MaterialSnapshot = {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
  color: THREE.Color | null;
  map: THREE.Texture | null | undefined;
};

function hasColor(material: THREE.Material): material is ColorMaterial {
  return "color" in material && (material as Partial<ColorMaterial>).color instanceof THREE.Color;
}

function hasMap(material: THREE.Material): material is MapMaterial {
  return "map" in material;
}

export function normalizedOverlayOpacity(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return THREE.MathUtils.clamp(value, 0, 1);
}

/**
 * Temporarily turns every mesh material below a layer into a flat comparison tint.
 *
 * Materials are changed in place so geometry ownership stays with the ordinary
 * Viewer loader. The returned cleanup restores every property, including texture
 * maps, and does not dispose loader-owned resources.
 */
export function applyOverlayAppearance(
  root: THREE.Object3D,
  appearance: OverlayAppearance,
): () => void {
  const opacity = normalizedOverlayOpacity(appearance.opacity);
  const materialSnapshots = new Map<THREE.Material, MaterialSnapshot>();
  const meshSnapshots = new Map<THREE.Mesh, { renderOrder: number; castShadow: boolean }>();

  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;

    meshSnapshots.set(object, {
      renderOrder: object.renderOrder,
      castShadow: object.castShadow,
    });
    object.renderOrder = appearance.renderOrder;
    // Transparent diagnostic layers must never leave an opaque silhouette in
    // the lighting pass, including at 0% opacity.
    object.castShadow = false;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!materialSnapshots.has(material)) {
        materialSnapshots.set(material, {
          opacity: material.opacity,
          transparent: material.transparent,
          depthWrite: material.depthWrite,
          color: hasColor(material) ? material.color.clone() : null,
          map: hasMap(material) ? material.map : undefined,
        });
      }
      material.opacity = opacity;
      material.transparent = opacity < 1;
      // Translucent comparison layers must not hide layers rendered after them.
      material.depthWrite = opacity >= 1;
      if (hasColor(material)) material.color.set(appearance.color);
      if (hasMap(material)) material.map = null;
      material.needsUpdate = true;
    }
  });

  return () => {
    for (const [mesh, snapshot] of meshSnapshots) {
      mesh.renderOrder = snapshot.renderOrder;
      mesh.castShadow = snapshot.castShadow;
    }
    for (const [material, snapshot] of materialSnapshots) {
      material.opacity = snapshot.opacity;
      material.transparent = snapshot.transparent;
      material.depthWrite = snapshot.depthWrite;
      if (snapshot.color && hasColor(material)) material.color.copy(snapshot.color);
      if (snapshot.map !== undefined && hasMap(material)) material.map = snapshot.map;
      material.needsUpdate = true;
    }
  };
}
