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

import type { UrdfJoint, UrdfSpec } from './urdf-parser';

export type FloatingIslandPartRole = 'root' | 'member';

/** A free-body island introduced by one parsed, genuinely floating joint. */
export interface FloatingIsland {
  /** Stable identifier inherited from the floating joint. */
  id: string;
  joint: UrdfJoint;
  parentPartName: string;
  rootPartName: string;
  /** The root followed by every reachable part across non-floating joints. */
  memberPartNames: readonly string[];
}

export interface FloatingIslandPartMembership {
  islandId: string;
  rootPartName: string;
  role: FloatingIslandPartRole;
}

export interface FloatingIslandClassification {
  islands: readonly FloatingIsland[];
  membershipByPartName: ReadonlyMap<string, FloatingIslandPartMembership>;
}

/**
 * Classify free-body islands from the already-normalized URDF joint graph.
 *
 * Every floating edge starts an island at its child. Traversal follows fixed and
 * constrained edges, but stops at another floating edge so nested free bodies
 * receive their own island. Unknown links are ignored defensively. In a malformed
 * graph where a part is reachable from multiple islands, the first floating joint
 * in authoring order owns the part lookup entry.
 */
export function classifyFloatingIslands(spec: UrdfSpec): FloatingIslandClassification {
  const linkNames = new Set(spec.links.map((link) => link.name));
  const outgoingJoints = new Map<string, UrdfJoint[]>();

  for (const joint of spec.joints) {
    if (!linkNames.has(joint.parent) || !linkNames.has(joint.child)) continue;
    const outgoing = outgoingJoints.get(joint.parent);
    if (outgoing) {
      outgoing.push(joint);
    } else {
      outgoingJoints.set(joint.parent, [joint]);
    }
  }

  const islands: FloatingIsland[] = [];
  const membershipByPartName = new Map<string, FloatingIslandPartMembership>();

  for (const rootJoint of spec.joints) {
    if (
      rootJoint.type !== 'floating'
      || !linkNames.has(rootJoint.parent)
      || !linkNames.has(rootJoint.child)
    ) {
      continue;
    }

    const memberPartNames: string[] = [];
    const visited = new Set<string>();
    const pending = [rootJoint.child];

    while (pending.length > 0) {
      const partName = pending.shift();
      if (partName === undefined || visited.has(partName) || !linkNames.has(partName)) continue;

      visited.add(partName);
      memberPartNames.push(partName);

      for (const joint of outgoingJoints.get(partName) ?? []) {
        if (joint.type !== 'floating') pending.push(joint.child);
      }
    }

    const island: FloatingIsland = {
      id: rootJoint.name,
      joint: rootJoint,
      parentPartName: rootJoint.parent,
      rootPartName: rootJoint.child,
      memberPartNames,
    };
    islands.push(island);

    for (const partName of memberPartNames) {
      if (membershipByPartName.has(partName)) continue;
      membershipByPartName.set(partName, {
        islandId: island.id,
        rootPartName: island.rootPartName,
        role: partName === island.rootPartName ? 'root' : 'member',
      });
    }
  }

  return { islands, membershipByPartName };
}
