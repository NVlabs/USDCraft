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

import type { JointControllerState } from '@/components/viewer3d/useJointController';

type JointPoseApplier = Pick<JointControllerState, 'applyJointValues'>;

export function normalizeRendererJointPose(pose: Record<string, number>): Map<string, number> {
  return new Map(
    Object.entries(pose).filter(
      (entry): entry is [string, number] =>
        entry[0].trim().length > 0 && Number.isFinite(entry[1]),
    ),
  );
}

export function applyRendererJointPose(
  pose: Map<string, number>,
  controller: JointPoseApplier,
  invalidate: () => void,
): void {
  controller.applyJointValues(pose, { commit: true });
  invalidate();
}
