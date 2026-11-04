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

import type { JSX } from "react";
import { RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FreeBodyCard, JointSlider } from "@/components/inspector/JointSlider";
import { classifyFloatingIslands } from "@/components/viewer3d/floating-islands";
import type { UrdfJoint, UrdfSpec } from "@/components/viewer3d/urdf-parser";

type JointPanelProps = {
  urdfSpec: UrdfSpec | null;
  jointValues: Map<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetAll: () => void;
};

function isConstrainedJoint(joint: UrdfJoint): boolean {
  return joint.type !== "fixed" && joint.type !== "floating" && !joint.mimic;
}

export function JointPanel({ urdfSpec, jointValues, onJointChange, onResetAll }: JointPanelProps): JSX.Element {
  const constrainedJoints = (urdfSpec?.joints ?? []).filter(isConstrainedJoint);
  const scalarJointCount = constrainedJoints.filter(
    (joint) => joint.type === "revolute" || joint.type === "continuous" || joint.type === "prismatic",
  ).length;
  const floatingIslands = urdfSpec ? classifyFloatingIslands(urdfSpec).islands : [];

  if (!urdfSpec || (constrainedJoints.length === 0 && floatingIslands.length === 0)) {
    return (
      <div className="flex h-32 items-center justify-center">
        <p className="text-[11px] text-[#bbb]">No motion relationships</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-2 pb-1">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] tabular-nums text-[#999]">
            {constrainedJoints.length} constrained · {floatingIslands.length}{" "}
            {floatingIslands.length === 1 ? "free body" : "free bodies"} · {scalarJointCount} control
            {scalarJointCount !== 1 ? "s" : ""}
          </span>
          {scalarJointCount > 0 ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={onResetAll}
              className="h-6 gap-1 px-2 text-[11px] text-[#999] hover:text-[#666]"
            >
              <RotateCcw className="size-3" />
              Reset
            </Button>
          ) : null}
        </div>

        {constrainedJoints.length > 0 ? (
          <section aria-labelledby="constrained-joints-heading" className="space-y-2">
            <div className="flex items-center gap-2 pt-1">
              <h3
                id="constrained-joints-heading"
                className="text-[9px] font-medium uppercase tracking-[0.05em] text-[#999]"
              >
                Constrained joints
              </h3>
              <span className="font-mono text-[9px] tabular-nums text-[#bbb]">
                {constrainedJoints.length}
              </span>
              <div className="h-px flex-1 bg-[#ececec]" />
            </div>
            {constrainedJoints.map((joint) => (
              <JointSlider
                key={joint.name}
                joint={joint}
                value={jointValues.get(joint.name) ?? 0}
                onChange={onJointChange}
              />
            ))}
          </section>
        ) : null}

        {floatingIslands.length > 0 ? (
          <section aria-labelledby="free-bodies-heading" className="space-y-2 pt-1">
            <div className="flex items-center gap-2">
              <h3
                id="free-bodies-heading"
                className="text-[9px] font-medium uppercase tracking-[0.05em] text-[#9a4a18]"
              >
                Free bodies
              </h3>
              <span className="font-mono text-[9px] tabular-nums text-[#b47a57]">
                {floatingIslands.length}
              </span>
              <div className="h-px flex-1 bg-[#c06020]/20" />
            </div>
            {floatingIslands.map((island) => (
              <FreeBodyCard
                key={island.id}
                joint={island.joint}
                memberCount={island.memberPartNames.length}
              />
            ))}
          </section>
        ) : null}
      </div>
    </ScrollArea>
  );
}
