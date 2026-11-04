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

import { applyRendererJointPose, normalizeRendererJointPose } from './headless-joint-pose';

describe('headless renderer joint pose bridge', () => {
  it('forwards finite poses to the actual joint controller and redraws', () => {
    const applyJointValues = vi.fn();
    const invalidate = vi.fn();
    const pose = normalizeRendererJointPose({
      hinge: Math.PI / 3,
      slide: 0.12,
      invalid: Number.NaN,
    });

    applyRendererJointPose(pose, { applyJointValues }, invalidate);

    expect([...pose.entries()]).toEqual([
      ['hinge', Math.PI / 3],
      ['slide', 0.12],
    ]);
    expect(applyJointValues).toHaveBeenCalledOnce();
    expect(applyJointValues).toHaveBeenCalledWith(pose, { commit: true });
    expect(invalidate).toHaveBeenCalledOnce();
  });
});
