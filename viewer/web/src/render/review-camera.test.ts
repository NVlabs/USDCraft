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

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  fitDistance,
  orbitDirectionRobot,
  orbitDirectionWorld,
  resolveReviewCamera,
} from './review-camera';
import { resolveReviewProfile, REVIEW_PBR, REVIEW_SHAPE } from './review-profiles';

describe('headless camera convention', () => {
  it('matches the numpy renderer eye formula in the robot frame', () => {
    // azimuth 0, elevation 0: the numpy renderer looks from -Y ("front").
    const front = orbitDirectionRobot(0, 0);
    expect(front.x).toBeCloseTo(0);
    expect(front.y).toBeCloseTo(-1);
    expect(front.z).toBeCloseTo(0);

    // azimuth 90: from +X (side); elevation 90: straight above (+Z).
    const side = orbitDirectionRobot(90, 0);
    expect(side.x).toBeCloseTo(1);
    expect(side.y).toBeCloseTo(0);
    const top = orbitDirectionRobot(0, 90);
    expect(top.z).toBeCloseTo(1);
  });

  it('maps the Z-up robot frame into the Y-up viewer world (rotation.x = -90deg)', () => {
    // robot (x, y, z) -> world (x, z, -y): front (-Y robot) becomes +Z world,
    // and "above" (+Z robot) becomes +Y world.
    const front = orbitDirectionWorld(0, 0);
    expect(front.x).toBeCloseTo(0);
    expect(front.y).toBeCloseTo(0);
    expect(front.z).toBeCloseTo(1);
    const top = orbitDirectionWorld(0, 90);
    expect(top.y).toBeCloseTo(1);
  });

  it('fits the bounding sphere at the requested FOV', () => {
    // At fov=90 the half-angle is 45deg: distance = r / sin(45deg).
    expect(fitDistance(1, 90)).toBeCloseTo(Math.SQRT2);
  });

  it('shares one center/radius across every orbit view', () => {
    const sphere = new THREE.Sphere(new THREE.Vector3(1, 2, 3), 0.5);
    const a = resolveReviewCamera({ azimuthDeg: 0, elevationDeg: 8 }, sphere);
    const b = resolveReviewCamera({ azimuthDeg: 135, elevationDeg: 35 }, sphere);
    expect(a.center).toEqual(b.center);
    expect(a.radius).toBeCloseTo(b.radius);
    expect(a.distance).toBeCloseTo(b.distance); // same fov + factor => same distance
    expect(a.eye).not.toEqual(b.eye);
  });
});

describe('review profiles', () => {
  it('review-pbr keeps authored PBR; review-shape overrides with clay', () => {
    expect(REVIEW_PBR.authoredPbr).toBe(true);
    expect(REVIEW_PBR.clayOverride).toBe(false);
    expect(REVIEW_SHAPE.authoredPbr).toBe(false);
    expect(REVIEW_SHAPE.clayOverride).toBe(true);
  });

  it('both profiles pin environment, shadows, FOV, DPR, and no helpers', () => {
    for (const profile of [REVIEW_PBR, REVIEW_SHAPE]) {
      expect(profile.environment).toBe(true);
      expect(profile.shadows).toBe(true);
      expect(profile.fovDeg).toBe(35);
      expect(profile.pixelRatio).toBe(1);
      expect(profile.helpers).toBe(false);
    }
  });

  it('rejects unknown profile ids', () => {
    expect(() => resolveReviewProfile('review-unknown')).toThrow(/unknown review profile/);
    expect(resolveReviewProfile('review-shape').id).toBe('review-shape');
  });
});
