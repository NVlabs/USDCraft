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

import { describe, expect, it } from "vitest";
import * as THREE from "three";

import {
  applyOverlayAppearance,
  normalizedOverlayOpacity,
} from "@/components/reconstruction/overlay-appearance";

describe("overlay comparison appearance", () => {
  it("clamps opacity to the supported layer range", () => {
    expect(normalizedOverlayOpacity(-2)).toBe(0);
    expect(normalizedOverlayOpacity(0.45)).toBe(0.45);
    expect(normalizedOverlayOpacity(3)).toBe(1);
    expect(normalizedOverlayOpacity(Number.NaN)).toBe(1);
  });

  it("tints a layer without taking ownership of loader materials", () => {
    const texture = new THREE.Texture();
    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, map: texture });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    mesh.renderOrder = 4;
    mesh.castShadow = true;
    const group = new THREE.Group();
    group.add(mesh);

    const restore = applyOverlayAppearance(group, {
      color: 0x0c8ce9,
      opacity: 0.35,
      renderOrder: 20,
    });

    expect(material.color.getHex()).toBe(0x0c8ce9);
    expect(material.map).toBeNull();
    expect(material.opacity).toBe(0.35);
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(mesh.renderOrder).toBe(20);
    expect(mesh.castShadow).toBe(false);

    restore();
    expect(material.color.getHex()).toBe(0xffffff);
    expect(material.map).toBe(texture);
    expect(material.opacity).toBe(1);
    expect(material.transparent).toBe(false);
    expect(material.depthWrite).toBe(true);
    expect(mesh.renderOrder).toBe(4);
    expect(mesh.castShadow).toBe(true);

    mesh.geometry.dispose();
    material.dispose();
    texture.dispose();
  });
});
