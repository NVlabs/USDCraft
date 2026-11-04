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
 * Shared review camera definition (A2+ point 8).
 *
 * Review views are specified in the SAME convention as the numpy renderer
 * a Z-up robot frame where the eye sits at
 *   (sin(az)cos(el), -cos(az)cos(el), sin(el))
 * relative to the model center. The viewer parents the robot under a -90deg
 * X-rotation (Z-up robot -> Y-up three.js world), mapping robot (x, y, z) to
 * world (x, z, -y). This module is the single place that conversion lives;
 * both the render page and its tests import it so the framing cannot drift
 * from the Python definition silently.
 */

import * as THREE from 'three';

export interface ReviewCameraPose {
  /** Robot-frame (Z-up) orbit azimuth, degrees; 0 looks from -Y (front). */
  azimuthDeg: number;
  /** Robot-frame elevation above the horizon, degrees. */
  elevationDeg: number;
  /** Distance as a multiple of the fitted bounding-sphere distance. */
  distanceFactor?: number;
  fovDeg?: number;
}

export interface ResolvedReviewCamera {
  azimuthDeg: number;
  elevationDeg: number;
  fovDeg: number;
  distance: number;
  center: [number, number, number];
  radius: number;
  eye: [number, number, number];
}

/** Robot-frame (Z-up) orbit direction for the numpy-renderer convention. */
export function orbitDirectionRobot(azimuthDeg: number, elevationDeg: number): THREE.Vector3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  return new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    -Math.cos(az) * Math.cos(el),
    Math.sin(el),
  );
}

/** Convert a robot-frame (Z-up) vector to three.js world space (Y-up). */
export function robotToWorld(vector: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3(vector.x, vector.z, -vector.y);
}

export function orbitDirectionWorld(azimuthDeg: number, elevationDeg: number): THREE.Vector3 {
  return robotToWorld(orbitDirectionRobot(azimuthDeg, elevationDeg));
}

/**
 * Fit distance for a bounding sphere at the given FOV, matching the framing
 * discipline: ONE sphere per model, shared by every orbit view.
 */
export function fitDistance(radius: number, fovDeg: number): number {
  return radius / Math.sin(((fovDeg / 2) * Math.PI) / 180);
}

/**
 * Resolve a review camera pose against a model bounding sphere (world space).
 * Returns both the eye position and the manifest entry describing the shot.
 */
export function resolveReviewCamera(
  pose: ReviewCameraPose,
  sphere: THREE.Sphere,
): ResolvedReviewCamera {
  const fovDeg = pose.fovDeg ?? 35;
  const distance = fitDistance(sphere.radius, fovDeg) * (pose.distanceFactor ?? 1.15);
  const eye = sphere.center
    .clone()
    .addScaledVector(orbitDirectionWorld(pose.azimuthDeg, pose.elevationDeg), distance);
  return {
    azimuthDeg: pose.azimuthDeg,
    elevationDeg: pose.elevationDeg,
    fovDeg,
    distance,
    center: sphere.center.toArray() as [number, number, number],
    radius: sphere.radius,
    eye: eye.toArray() as [number, number, number],
  };
}
