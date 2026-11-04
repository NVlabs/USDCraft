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

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import {
  FLOATING_PREVIEW_DURATION_MS,
  FLOATING_PREVIEW_LIFT_END_MS,
  FLOATING_PREVIEW_WIGGLE_END_MS,
  sampleFloatingPreviewMotion,
  startFloatingPreviewMotion,
  type FloatingPreviewScheduler,
} from './floating-preview-motion';

class ManualScheduler implements FloatingPreviewScheduler {
  private nextHandle = 1;
  private callbacks = new Map<number, (timeMs: number) => void>();
  lastCallback: ((timeMs: number) => void) | null = null;

  requestFrame = (callback: (timeMs: number) => void): number => {
    const handle = this.nextHandle;
    this.nextHandle += 1;
    this.callbacks.set(handle, callback);
    this.lastCallback = callback;
    return handle;
  };

  cancelFrame = (handle: number): void => {
    this.callbacks.delete(handle);
  };

  step(timeMs: number): void {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    callbacks.forEach((callback) => callback(timeMs));
  }

  get pendingCount(): number {
    return this.callbacks.size;
  }
}

function expectVectorClose(actual: THREE.Vector3, expected: THREE.Vector3): void {
  expect(actual.x).toBeCloseTo(expected.x, 10);
  expect(actual.y).toBeCloseTo(expected.y, 10);
  expect(actual.z).toBeCloseTo(expected.z, 10);
}

function expectQuaternionClose(actual: THREE.Quaternion, expected: THREE.Quaternion): void {
  expect(Math.abs(actual.dot(expected))).toBeCloseTo(1, 10);
}

describe('floating preview motion sampling', () => {
  it('is continuous across lift, wiggle, and return and ends at exact rest', () => {
    const start = sampleFloatingPreviewMotion(0, 1);
    const liftEnd = sampleFloatingPreviewMotion(FLOATING_PREVIEW_LIFT_END_MS, 1);
    const wiggle = sampleFloatingPreviewMotion(900, 1);
    const wiggleEnd = sampleFloatingPreviewMotion(FLOATING_PREVIEW_WIGGLE_END_MS, 1);
    const complete = sampleFloatingPreviewMotion(FLOATING_PREVIEW_DURATION_MS, 1);

    expect(start).toMatchObject({ x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0 });
    expect(liftEnd.z).toBeCloseTo(0.18, 10);
    expect(liftEnd.x).toBeCloseTo(0, 10);
    expect(Math.abs(wiggle.x) + Math.abs(wiggle.rx)).toBeGreaterThan(0.01);
    expect(wiggleEnd).toMatchObject({ x: 0, y: 0, rx: 0, ry: 0, rz: 0 });
    expect(wiggleEnd.z).toBeCloseTo(liftEnd.z, 10);
    expect(complete).toEqual({
      x: 0,
      y: 0,
      z: 0,
      rx: 0,
      ry: 0,
      rz: 0,
      phase: 'complete',
    });
  });

  it('keeps invalid and extreme extents finite and clamped', () => {
    for (const extent of [Number.NaN, Number.POSITIVE_INFINITY, -4, 1e9]) {
      const sample = sampleFloatingPreviewMotion(800, extent);
      expect(Object.values(sample).filter((value) => typeof value === 'number').every(Number.isFinite)).toBe(true);
      expect(sample.z).toBeLessThanOrEqual(0.208);
    }
  });
});

describe('floating preview motion controller', () => {
  it('lifts along authored model Z even when the viewer and joint frame are rotated', () => {
    const scheduler = new ManualScheduler();
    const referenceRoot = new THREE.Group();
    referenceRoot.rotation.x = -Math.PI / 2;
    const frame = new THREE.Group();
    frame.rotation.set(0.3, -0.2, 0.4);
    const node = new THREE.Group();
    referenceRoot.add(frame);
    frame.add(node);
    referenceRoot.updateWorldMatrix(true, true);
    const restWorld = node.getWorldPosition(new THREE.Vector3());
    const restWorldQuaternion = node.getWorldQuaternion(new THREE.Quaternion());
    const referenceWorldQuaternion = referenceRoot.getWorldQuaternion(new THREE.Quaternion());
    const modelUpWorld = new THREE.Vector3(0, 0, 1)
      .applyQuaternion(referenceRoot.getWorldQuaternion(new THREE.Quaternion()))
      .normalize();

    const controller = startFloatingPreviewMotion({
      node,
      referenceRoot,
      extent: 1,
      scheduler,
    });
    scheduler.step(0);
    scheduler.step(FLOATING_PREVIEW_LIFT_END_MS);

    const worldOffset = node.getWorldPosition(new THREE.Vector3()).sub(restWorld).normalize();
    expect(worldOffset.dot(modelUpWorld)).toBeCloseTo(1, 9);
    scheduler.step(900);
    const wiggle = sampleFloatingPreviewMotion(900, 1);
    const deltaInReference = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(wiggle.rx, wiggle.ry, wiggle.rz, 'XYZ'),
    );
    const deltaInWorld = referenceWorldQuaternion
      .clone()
      .multiply(deltaInReference)
      .multiply(referenceWorldQuaternion.clone().invert());
    const expectedWorldQuaternion = deltaInWorld.multiply(restWorldQuaternion);
    expectQuaternionClose(
      node.getWorldQuaternion(new THREE.Quaternion()),
      expectedWorldQuaternion,
    );
    controller.cancel();
  });

  it('restores an exact non-identity baseline on cancel and ignores stale callbacks', () => {
    const scheduler = new ManualScheduler();
    const referenceRoot = new THREE.Group();
    const frame = new THREE.Group();
    const node = new THREE.Group();
    const constrainedDescendant = new THREE.Group();
    referenceRoot.add(frame);
    frame.add(node);
    node.add(constrainedDescendant);
    node.position.set(0.2, -0.3, 0.4);
    node.quaternion.setFromEuler(new THREE.Euler(0.1, 0.2, -0.15));
    node.scale.set(1.1, 0.9, 1.05);
    const restPosition = node.position.clone();
    const restQuaternion = node.quaternion.clone();
    const restScale = node.scale.clone();
    constrainedDescendant.rotation.z = 0.45;
    const onFinish = vi.fn();

    const controller = startFloatingPreviewMotion({
      node,
      referenceRoot,
      extent: 0.8,
      scheduler,
      onFinish,
    });
    scheduler.step(0);
    scheduler.step(1000);
    expect(node.position.equals(restPosition)).toBe(false);
    const staleCallback = scheduler.lastCallback;

    controller.cancel();
    expectVectorClose(node.position, restPosition);
    expectQuaternionClose(node.quaternion, restQuaternion);
    expectVectorClose(node.scale, restScale);
    expect(constrainedDescendant.rotation.z).toBeCloseTo(0.45, 10);
    expect(scheduler.pendingCount).toBe(0);
    expect(onFinish).toHaveBeenCalledWith('cancelled');

    staleCallback?.(1500);
    expectVectorClose(node.position, restPosition);
    expectQuaternionClose(node.quaternion, restQuaternion);
  });

  it('pauses in place, resumes from the same elapsed pose, and completes without drift', () => {
    const scheduler = new ManualScheduler();
    const referenceRoot = new THREE.Group();
    const frame = new THREE.Group();
    const node = new THREE.Group();
    referenceRoot.add(frame);
    frame.add(node);
    const restPosition = node.position.clone();
    const restQuaternion = node.quaternion.clone();
    const onFinish = vi.fn();

    const controller = startFloatingPreviewMotion({
      node,
      referenceRoot,
      extent: 0.6,
      scheduler,
      onFinish,
    });
    scheduler.step(100);
    scheduler.step(900);
    const pausedPosition = node.position.clone();
    const pausedQuaternion = node.quaternion.clone();
    controller.pause();
    expect(scheduler.pendingCount).toBe(0);
    scheduler.step(5000);
    expectVectorClose(node.position, pausedPosition);
    expectQuaternionClose(node.quaternion, pausedQuaternion);

    controller.resume();
    scheduler.step(6000);
    expectVectorClose(node.position, pausedPosition);
    expectQuaternionClose(node.quaternion, pausedQuaternion);
    scheduler.step(6000 + FLOATING_PREVIEW_DURATION_MS);

    expect(controller.isActive()).toBe(false);
    expectVectorClose(node.position, restPosition);
    expectQuaternionClose(node.quaternion, restQuaternion);
    expect(onFinish).toHaveBeenCalledWith('completed');
    expect(scheduler.pendingCount).toBe(0);
  });
});
