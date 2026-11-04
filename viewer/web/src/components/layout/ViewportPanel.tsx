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

import { FolderOpen, Ruler } from "lucide-react";
import { useEffect, useRef, useState, type ChangeEvent, type JSX, type ReactNode } from "react";
import type * as THREE from "three";

import { SceneCanvas, type SnapshotExporter } from "@/components/viewer3d/SceneCanvas";
import type { RenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";
import { cn } from "@/lib/utils";

interface ViewportPanelProps {
  baseFileUrl: string | null;
  assetRevisionKey: string | null;
  selectionKey: string | null;
  jointPoseSignal: Map<string, number>;
  renderOptions: RenderOptions;
  onUrdfSpecChange: (spec: UrdfSpec | null, jointNodes: Map<string, THREE.Object3D> | null) => void;
  onInvalidateReady?: (invalidate: (() => void) | null) => void;
  onSnapshotReady?: (snapshot: SnapshotExporter | null) => void;
  onLoadStateChange?: (state: {
    loading: boolean;
    error: string | null;
    missingArtifacts: boolean;
    collisionGeometryState: {
      declaredCount: number;
      rigidRuntimeCount: number;
      deformableRestShapeCount: number;
      missingDeformableRestShapePartCount: number;
      missingDeformableRestShapePartNames: string[];
      primitiveCount: number;
      meshCount: number;
      loadedMeshCount: number;
      failedMeshCount: number;
      loading: boolean;
    };
  }) => void;
  overlayNotice?: ReactNode;
  disabledOverlay?: ReactNode;
  allowLocalUsdFile?: boolean;
}

/** Width × depth × height of the loaded model, in metres (scene Y is up). */
export function formatModelDimensions(size: { x: number; y: number; z: number }): string {
  const metres = (value: number): string =>
    value >= 1 ? `${value.toFixed(2)} m` : `${Math.round(value * 1000)} mm`;
  return `${metres(size.x)} × ${metres(size.z)} × ${metres(size.y)} (w × d × h)`;
}

export function ViewportPanel({
  baseFileUrl,
  assetRevisionKey,
  selectionKey,
  jointPoseSignal,
  renderOptions,
  onUrdfSpecChange,
  onInvalidateReady,
  onSnapshotReady,
  onLoadStateChange,
  overlayNotice,
  disabledOverlay,
  allowLocalUsdFile = true,
}: ViewportPanelProps): JSX.Element {
  const [localUsdFile, setLocalUsdFile] = useState<File | null>(null);
  const [modelSize, setModelSize] = useState<{ x: number; y: number; z: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    setLocalUsdFile(null);
  }, [selectionKey]);

  const handleLocalUsdFile = (event: ChangeEvent<HTMLInputElement>): void => {
    setLocalUsdFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };

  // Map from useRenderOptions format to RenderOptionsPanel format for SceneCanvas
  const sceneRenderOptions = {
    fancyGraphics: renderOptions.fancyGraphics,
    reviewProfile: renderOptions.reviewProfile,
    showEdges: renderOptions.showEdges,
    showGrid: renderOptions.showGrid,
    showCollisions: renderOptions.showCollisions,
    showSegmentColors: renderOptions.showSegmentColors,
    doubleSided: renderOptions.doubleSided,
    autoAnimate: renderOptions.autoAnimate,
    showJointOverlay: renderOptions.showJointOverlay,
    showFloatingBodies: renderOptions.showFloatingBodies,
  };

  return (
    <section className="stage-grid relative h-full w-full min-w-0 bg-[var(--surface-2)]">
      <div className="absolute inset-0 overflow-hidden">
        <div
          className={cn(
            "h-full w-full transition duration-200",
            disabledOverlay ? "pointer-events-none select-none blur-[8px] saturate-50 opacity-35" : "",
          )}
        >
          <SceneCanvas
            baseFileUrl={baseFileUrl}
            assetRevisionKey={assetRevisionKey}
            selectionKey={selectionKey}
            jointPoseSignal={jointPoseSignal}
            renderOptions={sceneRenderOptions}
            onUrdfSpecChange={onUrdfSpecChange}
            onInvalidateReady={onInvalidateReady}
            onSnapshotReady={onSnapshotReady}
            onBoundsChange={setModelSize}
            onLoadStateChange={onLoadStateChange}
            localUsdFile={localUsdFile}
          />
        </div>
      </div>
      {modelSize && !disabledOverlay ? (
        <div
          className="pointer-events-none absolute bottom-3 left-3 z-[6] flex items-center gap-1.5 rounded-md border border-[var(--border-default)] bg-[var(--surface-glass)] px-2.5 py-1.5 font-mono text-[9px] text-[var(--text-secondary)] shadow-[0_6px_20px_rgba(0,0,0,0.18)] backdrop-blur-md"
          aria-label="Model dimensions"
          data-model-dimensions
        >
          <Ruler className="size-3 text-[var(--accent)]" />
          {formatModelDimensions(modelSize)}
        </div>
      ) : null}
      {allowLocalUsdFile ? <div className="absolute right-3 top-3 z-[6] flex items-center gap-2">
        {localUsdFile ? (
          <span className="max-w-48 truncate rounded bg-black/55 px-2 py-1 font-mono text-[9px] text-white/80">
            {localUsdFile.name}
          </span>
        ) : null}
        <input
          ref={fileInputRef}
          type="file"
          accept=".usd,.usda,.usdc,.usdz"
          className="hidden"
          onChange={handleLocalUsdFile}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--surface-glass)] px-2.5 py-1.5 text-[9.5px] font-medium text-[var(--text-secondary)] shadow-[0_6px_20px_rgba(0,0,0,0.18)] backdrop-blur-md transition hover:border-[var(--accent)] hover:text-[var(--text-primary)]"
        >
          <FolderOpen className="size-3 text-[var(--accent)]" />
          Open local USD
        </button>
      </div> : null}
      {overlayNotice ? <div className="absolute inset-x-0 bottom-0 z-[5] p-3">{overlayNotice}</div> : null}
      {disabledOverlay ? <div className="absolute inset-0 z-10">{disabledOverlay}</div> : null}
    </section>
  );
}
