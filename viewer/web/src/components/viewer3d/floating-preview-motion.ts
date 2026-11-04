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

import * as THREE from 'three';

export const FLOATING_PREVIEW_DURATION_MS = 2700;
export const FLOATING_PREVIEW_LIFT_END_MS = 500;
export const FLOATING_PREVIEW_WIGGLE_END_MS = 2100;

export type FloatingPreviewPhase = 'lift' | 'wiggle' | 'return' | 'complete';

export interface FloatingPreviewSample {
  x: number;
  y: number;
  z: number;
  rx: number;
  ry: number;
  rz: number;
  phase: FloatingPreviewPhase;
}

export interface FloatingPreviewScheduler {
  requestFrame: (callback: (timeMs: number) => void) => number;
  cancelFrame: (handle: number) => void;
}

export interface FloatingPreviewController {
  cancel: () => void;
  pause: () => void;
  resume: () => void;
  isActive: () => boolean;
  isPaused: () => boolean;
}

export interface FloatingPreviewOptions {
  node: THREE.Object3D;
  referenceRoot: THREE.Object3D;
  extent: number;
  scheduler?: FloatingPreviewScheduler;
  onUpdate?: () => void;
  onFinish?: (reason: 'completed' | 'cancelled') => void;
}

const COMPLETE_SAMPLE: FloatingPreviewSample = {
  x: 0,
  y: 0,
  z: 0,
  rx: 0,
  ry: 0,
  rz: 0,
  phase: 'complete',
};

function clamp01(value: number): number {
  return THREE.MathUtils.clamp(value, 0, 1);
}

function smoothStep(value: number): number {
  const t = clamp01(value);
  return t * t * (3 - 2 * t);
}

function resetSample(target: FloatingPreviewSample, phase: FloatingPreviewPhase): void {
  target.x = 0;
  target.y = 0;
  target.z = 0;
  target.rx = 0;
  target.ry = 0;
  target.rz = 0;
  target.phase = phase;
}

export function sampleFloatingPreviewMotion(
  elapsedMs: number,
  extent: number,
  target: FloatingPreviewSample = { ...COMPLETE_SAMPLE },
): FloatingPreviewSample {
  const elapsed = Number.isFinite(elapsedMs)
    ? THREE.MathUtils.clamp(elapsedMs, 0, FLOATING_PREVIEW_DURATION_MS)
    : 0;
  const safeExtent = Number.isFinite(extent) && extent > 0 ? extent : 0.25;
  const lift = THREE.MathUtils.clamp(safeExtent * 0.18, 0.035, 0.2);
  const lateral = THREE.MathUtils.clamp(safeExtent * 0.05, 0.01, 0.06);
  const angle = THREE.MathUtils.degToRad(9);

  if (elapsed >= FLOATING_PREVIEW_DURATION_MS) {
    resetSample(target, 'complete');
    return target;
  }

  if (elapsed < FLOATING_PREVIEW_LIFT_END_MS) {
    resetSample(target, 'lift');
    target.z = lift * smoothStep(elapsed / FLOATING_PREVIEW_LIFT_END_MS);
    return target;
  }

  if (elapsed < FLOATING_PREVIEW_WIGGLE_END_MS) {
    resetSample(target, 'wiggle');
    const progress = (
      elapsed - FLOATING_PREVIEW_LIFT_END_MS
    ) / (
      FLOATING_PREVIEW_WIGGLE_END_MS - FLOATING_PREVIEW_LIFT_END_MS
    );
    const envelope = Math.sin(Math.PI * progress);
    target.x = lateral * Math.sin(Math.PI * 2 * progress) * envelope;
    target.y = lateral * 0.65 * Math.sin(Math.PI * 4 * progress) * envelope;
    target.z = lift * (1 + 0.04 * Math.sin(Math.PI * 2 * progress) * envelope);
    target.rx = angle * 0.65 * Math.sin(Math.PI * 2 * progress) * envelope;
    target.ry = angle * Math.sin(Math.PI * 4 * progress) * envelope;
    target.rz = angle * 0.55 * Math.sin(Math.PI * 6 * progress) * envelope;
    return target;
  }

  resetSample(target, 'return');
  const returnProgress = (
    elapsed - FLOATING_PREVIEW_WIGGLE_END_MS
  ) / (
    FLOATING_PREVIEW_DURATION_MS - FLOATING_PREVIEW_WIGGLE_END_MS
  );
  target.z = lift * (1 - smoothStep(returnProgress));
  return target;
}

function browserScheduler(): FloatingPreviewScheduler {
  return {
    requestFrame: (callback) => window.requestAnimationFrame(callback),
    cancelFrame: (handle) => window.cancelAnimationFrame(handle),
  };
}

export function startFloatingPreviewMotion(
  options: FloatingPreviewOptions,
): FloatingPreviewController {
  const { node, referenceRoot, extent, onUpdate, onFinish } = options;
  const parent = node.parent;
  if (!parent) {
    throw new Error('Floating preview node must have a parent frame.');
  }
  const scheduler = options.scheduler ?? browserScheduler();
  const restPosition = node.position.clone();
  const restQuaternion = node.quaternion.clone();
  const restScale = node.scale.clone();
  const sample: FloatingPreviewSample = { ...COMPLETE_SAMPLE };
  const offset = new THREE.Vector3();
  const rotationEuler = new THREE.Euler();
  const deltaInReference = new THREE.Quaternion();
  const deltaInParent = new THREE.Quaternion();

  referenceRoot.updateWorldMatrix(true, false);
  parent.updateWorldMatrix(true, false);
  const referenceWorldQuaternion = referenceRoot.getWorldQuaternion(new THREE.Quaternion());
  const parentWorldQuaternion = parent.getWorldQuaternion(new THREE.Quaternion());
  const parentInReference = referenceWorldQuaternion
    .clone()
    .invert()
    .multiply(parentWorldQuaternion)
    .normalize();
  const referenceToParent = parentInReference.clone().invert();

  let active = true;
  let paused = false;
  let frameHandle: number | null = null;
  let startedAt: number | null = null;
  let elapsedMs = 0;

  const commitTransform = (): void => {
    node.updateMatrix();
    node.updateWorldMatrix(true, true);
    onUpdate?.();
  };

  const restore = (): void => {
    node.position.copy(restPosition);
    node.quaternion.copy(restQuaternion);
    node.scale.copy(restScale);
    commitTransform();
  };

  const stop = (reason: 'completed' | 'cancelled'): void => {
    if (!active) {
      return;
    }
    active = false;
    paused = false;
    if (frameHandle != null) {
      scheduler.cancelFrame(frameHandle);
      frameHandle = null;
    }
    restore();
    onFinish?.(reason);
  };

  const schedule = (): void => {
    if (active && !paused && frameHandle == null) {
      frameHandle = scheduler.requestFrame(tick);
    }
  };

  const tick = (now: number): void => {
    frameHandle = null;
    if (!active || paused) {
      return;
    }
    if (startedAt == null) {
      startedAt = now - elapsedMs;
    }
    elapsedMs = Math.max(0, now - startedAt);
    if (elapsedMs >= FLOATING_PREVIEW_DURATION_MS) {
      stop('completed');
      return;
    }

    sampleFloatingPreviewMotion(elapsedMs, extent, sample);
    offset.set(sample.x, sample.y, sample.z).applyQuaternion(referenceToParent);
    rotationEuler.set(sample.rx, sample.ry, sample.rz, 'XYZ');
    deltaInReference.setFromEuler(rotationEuler);
    deltaInParent
      .copy(referenceToParent)
      .multiply(deltaInReference)
      .multiply(parentInReference)
      .normalize();
    node.position.copy(restPosition).add(offset);
    node.quaternion.copy(restQuaternion).premultiply(deltaInParent).normalize();
    node.scale.copy(restScale);
    commitTransform();
    schedule();
  };

  const controller: FloatingPreviewController = {
    cancel: () => stop('cancelled'),
    pause: () => {
      if (!active || paused) {
        return;
      }
      paused = true;
      if (frameHandle != null) {
        scheduler.cancelFrame(frameHandle);
        frameHandle = null;
      }
    },
    resume: () => {
      if (!active || !paused) {
        return;
      }
      paused = false;
      startedAt = null;
      schedule();
    },
    isActive: () => active,
    isPaused: () => paused,
  };
  schedule();
  return controller;
}
