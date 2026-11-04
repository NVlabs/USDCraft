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

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, type JSX } from "react";
import * as THREE from "three";

import { applyOverlayAppearance } from "@/components/reconstruction/overlay-appearance";
import { computeFit, updateCameraClipping } from "@/components/viewer3d/camera-utils";
import { positionGroundHelpers } from "@/components/viewer3d/lighting";
import { useThreeScene } from "@/components/viewer3d/useThreeScene";
import { useUrdfLoader } from "@/components/viewer3d/useUrdfLoader";
import { excessBoxes } from "@/lib/axial-geometry";
import type { AxialGeometry, ReconstructionRecordDetail } from "@/lib/types";
import { reconstructionAxialGeometryQueryOptions } from "@/lib/viewer-queries";

export type ReconstructionOverlaySettings = {
  sourceVisible: boolean;
  sourceOpacity: number;
  candidateVisible: boolean;
  candidateOpacity: number;
  /** Draw the last check_axial readings (exterior-excess slices) in the shared frame. */
  diagnosticsVisible: boolean;
};

const Q0_JOINTS = new Map<string, number>();

export const DIAGNOSTIC_COLORS = {
  excess: "#f59e0b",
} as const;

/**
 * Build the diagnostic geometry group in recon_world (Z up). The caller places it with the same
 * transform the URDF loader applies to the candidate, so it lands in the shared q0 frame.
 */
export function buildDiagnosticGroup(geometry: AxialGeometry): THREE.Group {
  const group = new THREE.Group();
  group.name = "axial-diagnostics";
  const excessMaterial = new THREE.LineBasicMaterial({
    color: DIAGNOSTIC_COLORS.excess,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  for (const box of excessBoxes(geometry)) {
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(...box.size));
    const lines = new THREE.LineSegments(edges, excessMaterial);
    lines.position.set(...box.center);
    lines.renderOrder = 30;
    lines.name = box.label;
    group.add(lines);
  }
  return group;
}

function disposeGroup(group: THREE.Object3D): void {
  const materials = new Set<THREE.Material>();
  group.traverse((object) => {
    const geometry = (object as Partial<THREE.Mesh>).geometry;
    if (geometry) geometry.dispose();
    const material = (object as Partial<THREE.Mesh>).material;
    if (Array.isArray(material)) material.forEach((item) => materials.add(item));
    else if (material) materials.add(material);
  });
  materials.forEach((material) => material.dispose());
}

function useTintedLayer(
  group: THREE.Object3D | null,
  options: { color: string; opacity: number; visible: boolean; renderOrder: number },
  invalidate: () => void,
): void {
  useEffect(() => {
    if (!group) return;
    const previousVisibility = group.visible;
    group.visible = options.visible;
    const restore = applyOverlayAppearance(group, options);
    invalidate();
    return () => {
      restore();
      group.visible = previousVisibility;
      invalidate();
    };
  }, [group, invalidate, options]);
}

export function ReconstructionRecordOverlayCanvas({
  detail,
  settings,
}: {
  detail: ReconstructionRecordDetail;
  settings: ReconstructionOverlaySettings;
}): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fittedKeyRef = useRef<string | null>(null);
  const { scene, camera, controls, gridGroup, axisGroup, invalidate, sceneReady } =
    useThreeScene(containerRef, { fancyGraphics: true, maxPixelRatio: 1.5 });
  const source = useUrdfLoader(
    detail.source.available ? detail.source.base_file_url : null,
    detail.source.revision_key,
    Q0_JOINTS,
    false,
    scene,
    null,
    null,
    null,
    null,
    invalidate,
  );
  const candidate = useUrdfLoader(
    detail.candidate.available ? detail.candidate.base_file_url : null,
    detail.candidate.revision_key,
    Q0_JOINTS,
    false,
    scene,
    null,
    null,
    null,
    null,
    invalidate,
  );

  useTintedLayer(
    source.robotGroup,
    {
      color: "#00a6ed",
      opacity: settings.sourceOpacity,
      visible: settings.sourceVisible,
      renderOrder: 10,
    },
    invalidate,
  );
  useTintedLayer(
    candidate.robotGroup,
    {
      color: "#d946ef",
      opacity: settings.candidateOpacity,
      visible: settings.candidateVisible,
      renderOrder: 20,
    },
    invalidate,
  );

  const diagnosticsWanted =
    settings.diagnosticsVisible && detail.summary.axial_diagnostic !== null;
  const geometryQuery = useQuery({
    ...reconstructionAxialGeometryQueryOptions(detail.summary.record_id),
    enabled: diagnosticsWanted,
  });
  const geometry = diagnosticsWanted ? (geometryQuery.data ?? null) : null;
  const diagnosticCounts = useMemo(
    () =>
      geometry
        ? { excess: geometry.excess_slices.length }
        : null,
    [geometry],
  );
  useEffect(() => {
    if (!scene || !geometry || !candidate.robotGroup) return;
    const group = buildDiagnosticGroup(geometry);
    group.rotation.copy(candidate.robotGroup.rotation);
    group.position.copy(candidate.robotGroup.position);
    group.scale.copy(candidate.robotGroup.scale);
    scene.add(group);
    invalidate();
    return () => {
      scene.remove(group);
      disposeGroup(group);
      invalidate();
    };
  }, [candidate.robotGroup, geometry, invalidate, scene]);

  const sourceSettled =
    !detail.source.available || Boolean(source.loadedAssetKey || source.error);
  const candidateSettled =
    !detail.candidate.available || Boolean(candidate.loadedAssetKey || candidate.error);
  useEffect(() => {
    if (
      !sceneReady ||
      !camera ||
      !controls ||
      !gridGroup ||
      !axisGroup ||
      !sourceSettled ||
      !candidateSettled
    ) {
      return;
    }
    const targets = [source.robotGroup, candidate.robotGroup].filter(
      (value): value is THREE.Group => value !== null,
    );
    if (targets.length === 0) return;
    const key = targets.map((target) => target.uuid).join(":");
    if (fittedKeyRef.current === key) return;
    const fit = computeFit(camera, targets);
    camera.position.copy(fit.position);
    camera.near = fit.near;
    camera.far = fit.far;
    camera.updateProjectionMatrix();
    controls.target.copy(fit.target);
    controls.update();
    positionGroundHelpers(gridGroup, axisGroup, targets);
    updateCameraClipping(camera, [...targets, gridGroup, axisGroup]);
    fittedKeyRef.current = key;
    invalidate();
  }, [
    axisGroup,
    camera,
    candidate.robotGroup,
    candidateSettled,
    controls,
    gridGroup,
    invalidate,
    sceneReady,
    source.robotGroup,
    sourceSettled,
  ]);

  return (
    <section className="relative h-full min-h-0 bg-[var(--surface-2)]">
      <div
        ref={containerRef}
        className="h-full w-full"
        role="img"
        aria-label="Input source mesh and reconstructed candidate overlay"
      />
      <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-3 py-2 text-white/90 backdrop-blur">
        <p className="text-[10px] font-medium">
          Post-isolation normalized source / candidate · q0 overlay
        </p>
        <div className="mt-1.5 flex gap-3 text-[9px] text-white/75">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-[#00a6ed]" />
            Normalized source {source.loading ? "loading" : source.error ? "error" : "ready"}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-[#d946ef]" />
            Candidate{" "}
            {!detail.candidate.available
              ? detail.summary.materialization_status
              : candidate.loading
                ? "loading"
                : candidate.error
                  ? "error"
                  : "ready"}
          </span>
          {settings.diagnosticsVisible ? (
            <span className="flex items-center gap-1.5" data-overlay-diagnostics>
              <span className="size-2 rounded-full" style={{ backgroundColor: DIAGNOSTIC_COLORS.excess }} />
              check_axial{" "}
              {detail.summary.axial_diagnostic === null
                ? "no readings"
                : geometryQuery.isPending
                  ? "loading"
                  : geometryQuery.error
                    ? "error"
                    : diagnosticCounts
                      ? `${diagnosticCounts.excess} excess slices`
                      : "ready"}
            </span>
          ) : null}
        </div>
      </div>
      {source.error || candidate.error ? (
        <div className="pointer-events-none absolute bottom-3 left-3 max-w-lg rounded-md border border-red-400/25 bg-black/65 px-3 py-2 font-mono text-[9px] leading-4 text-red-200 backdrop-blur">
          {source.error ? `Source: ${source.error}` : null}
          {source.error && candidate.error ? <br /> : null}
          {candidate.error ? `Candidate: ${candidate.error}` : null}
        </div>
      ) : null}
    </section>
  );
}
