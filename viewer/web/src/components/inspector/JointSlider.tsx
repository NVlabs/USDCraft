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

import { Slider } from "@/components/ui/slider";
import type { UrdfJoint } from "@/components/viewer3d/urdf-parser";

// Keep the existing type export available to callers while using the parser's canonical contract.
export type { UrdfJoint } from "@/components/viewer3d/urdf-parser";

type JointSliderProps = {
  joint: UrdfJoint;
  value: number;
  onChange: (name: string, value: number) => void;
};

type FreeBodyCardProps = {
  joint: UrdfJoint;
  memberCount?: number;
};

function radToDeg(rad: number): string {
  return (rad * (180 / Math.PI)).toFixed(1);
}

function jointRange(joint: UrdfJoint): [number, number] {
  if (joint.type === "continuous") {
    return [-Math.PI, Math.PI];
  }

  const lower = joint.limit?.lower ?? 0;
  const upper = joint.limit?.upper ?? 0;
  return [lower, upper];
}

function formatValue(joint: UrdfJoint, value: number): string {
  if (joint.type === "revolute" || joint.type === "continuous") {
    return `${radToDeg(value)}°`;
  }
  return `${value.toFixed(3)}m`;
}

export function FreeBodyCard({ joint, memberCount = 1 }: FreeBodyCardProps): JSX.Element {
  return (
    <div className="space-y-2.5 rounded-sm border border-[#c06020]/25 bg-[#c06020]/[0.04] px-2.5 py-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate font-mono text-[11px] text-[#1e1e1e]" title={joint.child}>
            {joint.child}
          </p>
          <p className="mt-0.5 truncate font-mono text-[9px] text-[#aaa]" title={joint.name}>
            {joint.name}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-[#c06020]/10 px-1.5 py-0.5 font-mono text-[9px] font-medium text-[#9a4a18]">
          FREE · 6DOF
        </span>
      </div>

      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-1 font-mono text-[9px]">
        <dt className="text-[#aaa]">child</dt>
        <dd className="truncate text-[#666]" title={joint.child}>{joint.child}</dd>
        <dt className="text-[#aaa]">parent</dt>
        <dd className="truncate text-[#666]" title={joint.parent}>{joint.parent}</dd>
      </dl>

      <p className="text-[10px] leading-4 text-[#8a664e]">
        Unconstrained placement · no physical joint
        {memberCount > 1 ? ` · ${memberCount} parts in island` : ""}
      </p>
    </div>
  );
}

export function JointSlider({ joint, value, onChange }: JointSliderProps): JSX.Element {
  if (joint.type === "floating") {
    return <FreeBodyCard joint={joint} />;
  }

  const [min, max] = jointRange(joint);
  const scalarControllable = (
    joint.type === "revolute" || joint.type === "continuous" || joint.type === "prismatic"
  );

  return (
    <div className="space-y-2 rounded-sm border border-[#e8e8e8] bg-white px-2.5 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[11px] text-[#1e1e1e]" title={joint.name}>
          {joint.name}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-[#aaa]">
          {joint.type}
        </span>
      </div>

      {scalarControllable ? (
        <>
          <Slider
            min={min}
            max={max}
            step={0.01}
            value={[value]}
            onValueChange={([v]) => onChange(joint.name, v)}
          />

          <div className="flex justify-between font-mono text-[10px] text-[#bbb]">
            <span>{formatValue(joint, min)}</span>
            <span className="text-[#666]">{formatValue(joint, value)}</span>
            <span>{formatValue(joint, max)}</span>
          </div>
        </>
      ) : joint.type === "spherical" ? (
        <p className="text-[10px] text-[#999]">
          3-axis cone{joint.coneDeg != null ? ` · ${joint.coneDeg.toFixed(1)}°` : ""}
        </p>
      ) : null}
    </div>
  );
}
