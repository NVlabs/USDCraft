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

import { Crosshair, Layers3 } from "lucide-react";
import type { JSX } from "react";

import type { ReconstructionOverlaySettings } from "@/components/reconstruction/ReconstructionRecordOverlayCanvas";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

function LayerControl({
  label,
  color,
  visible,
  opacity,
  available,
  onVisible,
  onOpacity,
}: {
  label: string;
  color: string;
  visible: boolean;
  opacity: number;
  available: boolean;
  onVisible: (value: boolean) => void;
  onOpacity: (value: number) => void;
}): JSX.Element {
  return (
    <div className="rounded-lg border border-[var(--border-default)] bg-[var(--surface-1)] p-2.5">
      <div className="flex items-center gap-2">
        <span className="size-2.5 rounded-full" style={{ backgroundColor: color }} />
        <span className="min-w-0 flex-1 text-[10.5px] font-medium text-[var(--text-primary)]">
          {label}
        </span>
        <Switch
          aria-label={`Show ${label}`}
          checked={visible && available}
          disabled={!available}
          onCheckedChange={onVisible}
        />
      </div>
      <div className="mt-2.5 flex items-center gap-2">
        <Slider
          aria-label={`${label} opacity`}
          min={0}
          max={100}
          step={1}
          value={[Math.round(opacity * 100)]}
          disabled={!available || !visible}
          onValueChange={([value]) => onOpacity(value / 100)}
          className="flex-1"
        />
        <span className="w-8 text-right font-mono text-[9px] text-[var(--text-tertiary)]">
          {Math.round(opacity * 100)}%
        </span>
      </div>
    </div>
  );
}

export function ReconstructionOverlayControls({
  enabled,
  settings,
  sourceAvailable,
  candidateAvailable,
  diagnosticsAvailable = false,
  onEnabledChange,
  onSettingsChange,
}: {
  enabled: boolean;
  settings: ReconstructionOverlaySettings;
  sourceAvailable: boolean;
  candidateAvailable: boolean;
  /** The record carries check_axial readings (Provider runs only). */
  diagnosticsAvailable?: boolean;
  onEnabledChange: (enabled: boolean) => void;
  onSettingsChange: (settings: ReconstructionOverlaySettings) => void;
}): JSX.Element {
  const available = sourceAvailable && candidateAvailable;

  return (
    <div className="space-y-3">
      <div className={!available ? "opacity-60" : undefined}>
        <div className="flex items-center justify-between gap-3 py-1.5">
          <div className="min-w-0">
            <Label htmlFor="reconstruction-source-overlay" className="text-[12px] text-[var(--text-primary)]">
              Source overlay
            </Label>
            <p className="text-[10px] leading-[1.3] text-[var(--text-tertiary)]">
              Compare the normalized input source with the q0 candidate
            </p>
          </div>
          <Switch
            id="reconstruction-source-overlay"
            checked={enabled && available}
            disabled={!available}
            onCheckedChange={onEnabledChange}
          />
        </div>
        {!available ? (
          <p className="mt-1 rounded-lg bg-amber-500/[0.08] px-3 py-2 text-[10px] leading-4 text-amber-800/75">
            Overlay needs both a displayable candidate and its normalized source carrier. The compiled candidate viewer remains available when possible.
          </p>
        ) : null}
      </div>
      {enabled && available ? (
        <div className="space-y-2 rounded-lg border border-[var(--border-default)] bg-[var(--surface-0)] p-2.5">
          <div className="flex items-start gap-2 text-[10px] leading-4 text-[var(--text-tertiary)]">
            <Layers3 className="mt-0.5 size-3 shrink-0" />
            <p>
              Comparison is shown only on this Render tab. Turn Source overlay off, or leave Render, to use the standard display, motion and snapshot controls.
            </p>
          </div>
          <LayerControl
            label="Normalized source"
            color="#00a6ed"
            visible={settings.sourceVisible}
            opacity={settings.sourceOpacity}
            available={sourceAvailable}
            onVisible={(sourceVisible) => onSettingsChange({ ...settings, sourceVisible })}
            onOpacity={(sourceOpacity) => onSettingsChange({ ...settings, sourceOpacity })}
          />
          <LayerControl
            label="Reconstructed candidate"
            color="#d946ef"
            visible={settings.candidateVisible}
            opacity={settings.candidateOpacity}
            available={candidateAvailable}
            onVisible={(candidateVisible) => onSettingsChange({ ...settings, candidateVisible })}
            onOpacity={(candidateOpacity) => onSettingsChange({ ...settings, candidateOpacity })}
          />
          <div
            className={
              "rounded-lg border border-[var(--border-default)] bg-[var(--surface-1)] p-2.5" +
              (diagnosticsAvailable ? "" : " opacity-60")
            }
          >
            <div className="flex items-center gap-2">
              <Crosshair className="size-3 shrink-0 text-[var(--text-tertiary)]" />
              <div className="min-w-0 flex-1">
                <Label
                  htmlFor="reconstruction-overlay-diagnostics"
                  className="text-[10.5px] font-medium text-[var(--text-primary)]"
                >
                  check_axial readings
                </Label>
                <p className="text-[9.5px] leading-[1.3] text-[var(--text-tertiary)]">
                  {diagnosticsAvailable
                    ? "Exterior-excess slices as amber boxes"
                    : "Only Provider records with a check_axial call carry readings"}
                </p>
              </div>
              <Switch
                id="reconstruction-overlay-diagnostics"
                aria-label="Show check_axial readings"
                checked={settings.diagnosticsVisible && diagnosticsAvailable}
                disabled={!diagnosticsAvailable}
                onCheckedChange={(diagnosticsVisible) =>
                  onSettingsChange({ ...settings, diagnosticsVisible })
                }
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
