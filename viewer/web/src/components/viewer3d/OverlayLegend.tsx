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

import { X } from "lucide-react";
import type { JSX } from "react";

type LegendItem = {
  code: string;
  color: string;
  label: string;
};

const KINEMATIC_ITEMS: LegendItem[] = [
  { code: "REV", color: "#5b9cf5", label: "Revolute · bounded rotation" },
  { code: "PRI", color: "#f0a030", label: "Prismatic · linear travel" },
  { code: "CONT", color: "#3ec9a8", label: "Continuous · unbounded rotation" },
  { code: "SPH", color: "#eab308", label: "Spherical · 3-axis cone rotation" },
];

const FREE_BODY_ITEMS: LegendItem[] = [
  { code: "FREE", color: "#c06020", label: "Unconstrained 6DOF · no physical joint" },
];

function LegendSection({ title, items }: { title: string; items: LegendItem[] }): JSX.Element {
  return (
    <div>
      <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-white/45">
        {title}
      </p>
      <div className="space-y-1">
        {items.map((item) => (
          <div key={item.code} className="flex items-center gap-2">
            <span className="inline-flex h-5 min-w-11 items-center gap-1.5 rounded-full bg-black/35 px-2 font-mono text-[9px] font-semibold text-white/90">
              <span className="size-1.5 rounded-full" style={{ backgroundColor: item.color }} />
              {item.code}
            </span>
            <span className="text-[9.5px] leading-tight text-white/60">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type OverlayLegendProps = {
  onClose: () => void;
  showJointOverlay?: boolean;
  showFloatingBodies?: boolean;
};

export function OverlayLegend({
  onClose,
  showJointOverlay = true,
  showFloatingBodies = false,
}: OverlayLegendProps): JSX.Element {
  return (
    <aside className="absolute left-3 top-3 z-[5] w-[252px] rounded-xl border border-white/10 bg-[#080c14]/85 p-3 shadow-xl backdrop-blur-md">
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <div>
          <p className="text-[10px] font-semibold text-white/90">Overlay legend</p>
          {showJointOverlay ? (
            <p className="mt-0.5 text-[9px] text-white/40">SPH is visualized but not auto-animated</p>
          ) : null}
          {showFloatingBodies ? (
            <>
              <p className="mt-0.5 text-[9px] text-white/40">
                Select a FREE part + Preview motion for one-shot lift / wiggle / return
              </p>
              <p className="mt-0.5 text-[9px] text-white/40">
                Kinematic review gesture · not dynamics
              </p>
            </>
          ) : null}
        </div>
        <button
          type="button"
          aria-label="Close overlay legend"
          onClick={onClose}
          className="flex size-6 items-center justify-center rounded-md text-white/45 transition hover:bg-white/10 hover:text-white/80"
        >
          <X className="size-3.5" />
        </button>
      </div>
      <div className="space-y-3">
        {showJointOverlay ? <LegendSection title="Constrained joints" items={KINEMATIC_ITEMS} /> : null}
        {showFloatingBodies ? <LegendSection title="Free bodies" items={FREE_BODY_ITEMS} /> : null}
      </div>
    </aside>
  );
}
