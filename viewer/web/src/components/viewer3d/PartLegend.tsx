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

import { ChevronRight } from "lucide-react";
import { type JSX } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

import type { FloatingIslandPartRole } from "./floating-islands";
import type { UrdfBodyModel, UrdfBodyRealization } from "./urdf-parser";

export type PartLegendSelection = {
  partName: string;
  subpartKey: string | null;
};

export type PartLegendSubpartItem = {
  key: string;
  name: string;
  color: string;
};

export type PartLegendItem = {
  name: string;
  color: string;
  subparts: PartLegendSubpartItem[];
  bodyModel: UrdfBodyModel;
  bodyRealization: UrdfBodyRealization;
  floatingMembership: {
    role: FloatingIslandPartRole;
    rootPartName: string;
  } | null;
};

export type BodyModelBadge = {
  label: string;
  title: string;
  showStaticRest: boolean;
};

export type FloatingBodyBadge = {
  label: string;
  title: string;
};

export function shouldIncludePartInLegend(
  visualCount: number,
  floatingMembership: PartLegendItem['floatingMembership'],
): boolean {
  return visualCount > 0 || floatingMembership != null;
}

export function describeFloatingMembership(
  role: FloatingIslandPartRole,
  rootPartName: string,
): FloatingBodyBadge {
  return role === "root"
    ? {
        label: "FREE",
        title: "Unconstrained 6DOF free-body island root",
      }
    : {
        label: "free member",
        title: `Member of unconstrained island rooted at ${rootPartName}`,
      };
}

export function describeBodyModel(
  bodyModel: UrdfBodyModel,
  bodyRealization: UrdfBodyRealization,
): BodyModelBadge {
  const labels: Record<UrdfBodyModel, string> = {
    rigid: "rigid",
    flexible_1d: "flex 1D",
    surface_deformable: "surface",
    volume_deformable: "volume",
  };

  return {
    label: labels[bodyModel],
    title: bodyModel,
    showStaticRest: bodyRealization === "static_rest_pose",
  };
}

type PartLegendProps = {
  items: PartLegendItem[];
  selection: PartLegendSelection | null;
  onSelectPart: (selection: PartLegendSelection | null) => void;
};

export function PartLegend({
  items,
  selection,
  onSelectPart,
}: PartLegendProps): JSX.Element | null {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="pointer-events-auto absolute right-3 top-3 z-20 w-[min(14rem,calc(100%-1.5rem))] overflow-hidden rounded-[18px] bg-[rgba(245,245,245,0.94)] p-1.5 shadow-[0_10px_28px_rgba(17,17,17,0.06)] backdrop-blur-xl">
      <ScrollArea className="max-h-[min(46vh,15rem)]">
        <div className="space-y-0.5">
          {items.map((item) => {
            const isSelectedPart = selection?.partName === item.name;
            const isExpanded = isSelectedPart && item.subparts.length > 0;
            const bodyModelBadge = describeBodyModel(item.bodyModel, item.bodyRealization);
            const floatingBadge = item.floatingMembership
              ? describeFloatingMembership(
                  item.floatingMembership.role,
                  item.floatingMembership.rootPartName,
                )
              : null;

            return (
              <div key={item.name} className="space-y-0.5">
                <button
                  type="button"
                  aria-pressed={isSelectedPart && selection?.subpartKey == null}
                  onClick={() => onSelectPart(
                    isSelectedPart && selection?.subpartKey == null
                      ? null
                      : { partName: item.name, subpartKey: null },
                  )}
                  className={cn(
                    "group flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors duration-150",
                    isSelectedPart
                      ? "bg-white/[0.96] shadow-[inset_0_0_0_1px_rgba(12,140,233,0.14)]"
                      : "bg-transparent hover:bg-white/[0.7]",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span
                    className="min-w-0 flex-1 truncate font-mono text-[10px] text-[var(--text-primary)]"
                    title={item.name}
                  >
                    {item.name}
                  </span>
                  <span
                    className="shrink-0 rounded-full bg-[var(--surface-2)] px-1.5 py-0.5 font-mono text-[8px] text-[var(--text-tertiary)]"
                    title={bodyModelBadge.title}
                  >
                    {bodyModelBadge.label}
                  </span>
                  {bodyModelBadge.showStaticRest ? (
                    <span
                      className="shrink-0 rounded-full bg-amber-100 px-1.5 py-0.5 font-mono text-[8px] text-amber-800"
                      title="Static rest-pose visualization; deformation is not simulated"
                    >
                      static rest
                    </span>
                  ) : null}
                  {floatingBadge ? (
                    <span
                      className="shrink-0 rounded-full bg-orange-100 px-1.5 py-0.5 font-mono text-[8px] font-semibold text-orange-800"
                      title={floatingBadge.title}
                    >
                      {floatingBadge.label}
                    </span>
                  ) : null}
                  {item.subparts.length > 0 ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--surface-2)] px-1.5 py-0.5 font-mono text-[9px] text-[var(--text-tertiary)]">
                      <ChevronRight
                        aria-hidden="true"
                        className={cn(
                          "h-2.5 w-2.5 transition-transform duration-150",
                          isExpanded ? "rotate-90" : "rotate-0",
                        )}
                      />
                      {item.subparts.length}
                    </span>
                  ) : null}
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-1 w-1 shrink-0 rounded-full transition-opacity duration-150",
                      isSelectedPart
                        ? "bg-[var(--accent)] opacity-100"
                        : "bg-[var(--border-strong)] opacity-0 group-hover:opacity-80",
                    )}
                  />
                </button>

                {isExpanded ? (
                  <div className="ml-3 space-y-0.5 border-l border-[var(--border-subtle)] pl-2">
                    {item.subparts.map((subpart) => {
                      const isSelectedSubpart =
                        selection?.partName === item.name && selection.subpartKey === subpart.key;

                      return (
                        <button
                          key={subpart.key}
                          type="button"
                          aria-pressed={isSelectedSubpart}
                          onClick={() => onSelectPart(
                            isSelectedSubpart
                              ? { partName: item.name, subpartKey: null }
                              : { partName: item.name, subpartKey: subpart.key },
                          )}
                          className={cn(
                            "group flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors duration-150",
                            isSelectedSubpart
                              ? "bg-white/[0.94] shadow-[inset_0_0_0_1px_rgba(12,140,233,0.12)]"
                              : "bg-transparent hover:bg-white/[0.65]",
                          )}
                        >
                          <span
                            aria-hidden="true"
                            className="h-1.5 w-1.5 shrink-0 rounded-full opacity-80"
                            style={{ backgroundColor: subpart.color }}
                          />
                          <span
                            className="min-w-0 flex-1 truncate font-mono text-[9.5px] text-[var(--text-secondary)]"
                            title={subpart.name}
                          >
                            {subpart.name}
                          </span>
                          <span
                            aria-hidden="true"
                            className={cn(
                              "h-1 w-1 shrink-0 rounded-full transition-opacity duration-150",
                              isSelectedSubpart
                                ? "bg-[var(--accent)] opacity-100"
                                : "bg-[var(--border-strong)] opacity-0 group-hover:opacity-75",
                            )}
                          />
                        </button>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
