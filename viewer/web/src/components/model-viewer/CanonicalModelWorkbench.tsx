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

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type JSX,
  type ReactNode,
} from "react";
import { AlertTriangle, ChevronLeft } from "lucide-react";
import type { PanelImperativeHandle, PanelSize } from "react-resizable-panels";
import * as THREE from "three";

import { updateUrlSearchParams } from "@/lib/url";
import { resolveMaterializationPresentation } from "@/lib/materialization-presentation";
import { Inspector } from "@/components/layout/Inspector";
import { Sidebar } from "@/components/layout/Sidebar";
import { ViewportPanel } from "@/components/layout/ViewportPanel";
import { MissingArtifactsOverlay } from "@/components/layout/MissingArtifactsOverlay";
import { ArtifactContractWarning } from "@/components/layout/ArtifactContractWarning";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import {
  useRenderOptions,
  type RenderOptions as InspectorRenderOptions,
} from "@/components/viewer3d/useRenderOptions";
import { useJointController } from "@/components/viewer3d/useJointController";
import type { SnapshotExporter } from "@/components/viewer3d/SceneCanvas";
import type { UrdfJoint, UrdfSpec } from "@/components/viewer3d/urdf-parser";
import type { ListMaterializationStatus, MaterializationIssue } from "@/lib/types";

const JOINT_POSE_QUERY_PARAM = "pose";
const INSPECTOR_QUERY_PARAM = "inspector";
const JOINT_POSE_PRECISION = 1e5;
const PREVIEW_UI_SYNC_MS = 100;
const PREVIEW_MIN_CYCLE_SECONDS = 2.6;
const PREVIEW_LINEAR_SPEED_MPS = 0.08;
const PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 4;
const PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND = Math.PI / 5;
const PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS = 0.24;
const PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS = (Math.PI * 2) / 3;

type JointPoseSnapshot = {
  recordId: string;
  values: Record<string, number>;
};

export type CanonicalModelCollisionSupport = {
  available: boolean;
  summary: string;
  detail: string;
  rigidRuntimeCount: number;
  deformableRestShapeCount: number;
  missingDeformableRestShapePartCount: number;
  missingDeformableRestShapePartNames: string[];
};

export type CanonicalMaterializationSummary = {
  // Either product may pass its "not_validated" list placeholder while the full canonical
  // check has not yet run for this record; opening a record's detail view always resolves a
  // real status before this component sees it, so this placeholder should not reach here.
  materialization_status: ListMaterializationStatus;
  materialization_displayable: boolean;
  materialization_issues: MaterializationIssue[];
  has_compile_report: boolean;
};

export type CanonicalModelInspectorContext = {
  urdfSpec: UrdfSpec | null;
  jointValues: Map<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetAll: () => void;
  renderOptions: InspectorRenderOptions;
  onRenderOptionChange: <K extends keyof InspectorRenderOptions>(
    key: K,
    value: InspectorRenderOptions[K],
  ) => void;
  onSnapshot: SnapshotExporter | null;
  collisionSupport: CanonicalModelCollisionSupport | null;
};

export type CanonicalModelWorkbenchProps = {
  recordId: string | null;
  materializationSummary: CanonicalMaterializationSummary | null;
  candidateBaseFileUrl: string | null;
  assetRevisionKey: string | null;
  selectionKey: string | null;
  pending: boolean;
  sidebar: ReactNode;
  renderInspector: (context: CanonicalModelInspectorContext) => ReactNode;
  viewportOverride?: ReactNode;
  allowLocalUsdFile?: boolean;
  productLabel?: string;
};

const EMPTY_COLLISION_GEOMETRY_STATE = {
  declaredCount: 0,
  rigidRuntimeCount: 0,
  deformableRestShapeCount: 0,
  missingDeformableRestShapePartCount: 0,
  missingDeformableRestShapePartNames: [] as string[],
  primitiveCount: 0,
  meshCount: 0,
  loadedMeshCount: 0,
  failedMeshCount: 0,
  loading: false,
} as const;

type PreviewJointMotion = {
  joint: UrdfJoint;
  cycleSeconds: number;
  phaseOffset: number;
};

function readInspectorCollapsedFromUrl(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    return new URLSearchParams(window.location.search).get(INSPECTOR_QUERY_PARAM) === "closed";
  } catch {
    return false;
  }
}

function syncInspectorCollapsedToUrl(collapsed: boolean): void {
  updateUrlSearchParams((params) => {
    if (collapsed) {
      params.set(INSPECTOR_QUERY_PARAM, "closed");
    } else {
      params.delete(INSPECTOR_QUERY_PARAM);
    }
  });
}

function roundJointValue(value: number): number {
  return Math.round(value * JOINT_POSE_PRECISION) / JOINT_POSE_PRECISION;
}

function readJointPoseFromUrl(
  recordId: string | null,
  validJointNames: Set<string>,
): Map<string, number> {
  if (typeof window === "undefined" || !recordId) {
    return new Map();
  }

  try {
    const raw = new URLSearchParams(window.location.search).get(JOINT_POSE_QUERY_PARAM);
    if (!raw) {
      return new Map();
    }

    const parsed = JSON.parse(raw) as Partial<JointPoseSnapshot>;
    if (parsed.recordId !== recordId || !parsed.values || typeof parsed.values !== "object") {
      return new Map();
    }

    const nextValues = new Map<string, number>();
    for (const [jointName, value] of Object.entries(parsed.values)) {
      if (!validJointNames.has(jointName) || typeof value !== "number" || !Number.isFinite(value)) {
        continue;
      }
      if (Math.abs(value) <= 1 / JOINT_POSE_PRECISION) {
        continue;
      }
      nextValues.set(jointName, roundJointValue(value));
    }

    return nextValues;
  } catch {
    return new Map();
  }
}

function syncJointPoseToUrl(recordId: string | null, jointValues: Map<string, number>): void {
  updateUrlSearchParams((params) => {
    if (!recordId) {
      params.delete(JOINT_POSE_QUERY_PARAM);
      return;
    }

    const values = Object.fromEntries(
      Array.from(jointValues.entries())
        .map(([jointName, value]) => [jointName, roundJointValue(value)] as const)
        .filter(([, value]) => Number.isFinite(value) && Math.abs(value) > 1 / JOINT_POSE_PRECISION),
    );

    if (Object.keys(values).length === 0) {
      params.delete(JOINT_POSE_QUERY_PARAM);
      return;
    }

    params.set(
      JOINT_POSE_QUERY_PARAM,
      JSON.stringify({
        recordId,
        values,
      } satisfies JointPoseSnapshot),
    );
  });
}

function isPreviewJoint(joint: UrdfJoint): boolean {
  const mimic = (joint as UrdfJoint & { mimic?: unknown }).mimic;
  return (
    !mimic
    && (joint.type === "revolute" || joint.type === "continuous" || joint.type === "prismatic")
  );
}

function buildRootLinks(joints: UrdfJoint[]): string[] {
  const parents = new Set<string>();
  const children = new Set<string>();

  for (const joint of joints) {
    parents.add(joint.parent);
    children.add(joint.child);
  }

  const roots = Array.from(parents).filter((name) => !children.has(name));
  return roots.length > 0 ? roots : Array.from(new Set(joints.map((joint) => joint.parent)));
}

function buildPreviewJointSequence(urdfSpec: UrdfSpec): UrdfJoint[] {
  const jointsByParent = new Map<string, UrdfJoint[]>();

  for (const joint of urdfSpec.joints) {
    const current = jointsByParent.get(joint.parent) ?? [];
    current.push(joint);
    jointsByParent.set(joint.parent, current);
  }

  const orderedJoints: UrdfJoint[] = [];
  const seenLinks = new Set<string>();

  const visitLink = (linkName: string): void => {
    if (seenLinks.has(linkName)) {
      return;
    }
    seenLinks.add(linkName);

    for (const joint of jointsByParent.get(linkName) ?? []) {
      if (isPreviewJoint(joint)) {
        orderedJoints.push(joint);
      }
      visitLink(joint.child);
    }
  };

  for (const rootLink of buildRootLinks(urdfSpec.joints)) {
    visitLink(rootLink);
  }

  const seenJointNames = new Set(orderedJoints.map((joint) => joint.name));
  for (const joint of urdfSpec.joints) {
    if (isPreviewJoint(joint) && !seenJointNames.has(joint.name)) {
      orderedJoints.push(joint);
    }
  }

  return orderedJoints;
}

function previewJointTravelSpan(joint: UrdfJoint): number {
  if (joint.type === "continuous") {
    return Math.PI * 2;
  }

  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number"
    && Number.isFinite(lower)
    && typeof upper === "number"
    && Number.isFinite(upper)
    && upper > lower;

  if (hasRange) {
    return upper - lower;
  }

  if (joint.type === "prismatic") {
    return PREVIEW_FALLBACK_PRISMATIC_TRAVEL_METERS;
  }

  return PREVIEW_FALLBACK_REVOLUTE_TRAVEL_RADIANS;
}

function previewJointCycleSeconds(joint: UrdfJoint): number {
  const span = previewJointTravelSpan(joint);

  if (joint.type === "continuous") {
    return span / PREVIEW_CONTINUOUS_ANGULAR_SPEED_RAD_PER_SECOND;
  }

  const speed =
    joint.type === "prismatic" ? PREVIEW_LINEAR_SPEED_MPS : PREVIEW_ANGULAR_SPEED_RAD_PER_SECOND;
  return Math.max(PREVIEW_MIN_CYCLE_SECONDS, (span * 2) / speed);
}

function previewJointPhaseOffset(jointName: string, index: number): number {
  let hash = 0;
  for (const character of jointName) {
    hash = (hash * 33 + character.charCodeAt(0)) % 4096;
  }

  return THREE.MathUtils.euclideanModulo((hash / 4096) + (index * 0.61803398875), 1);
}

function buildPreviewJointMotions(urdfSpec: UrdfSpec): PreviewJointMotion[] {
  return buildPreviewJointSequence(urdfSpec).map((joint, index) => ({
    joint,
    cycleSeconds: previewJointCycleSeconds(joint),
    phaseOffset: previewJointPhaseOffset(joint.name, index),
  }));
}

function previewJointValue(joint: UrdfJoint, phase: number): number {
  if (joint.type === "continuous") {
    return THREE.MathUtils.euclideanModulo((phase * Math.PI * 2) + Math.PI, Math.PI * 2) - Math.PI;
  }

  const lower = joint.limit?.lower;
  const upper = joint.limit?.upper;
  const hasRange =
    typeof lower === "number"
    && Number.isFinite(lower)
    && typeof upper === "number"
    && Number.isFinite(upper)
    && upper > lower;

  if (hasRange) {
    const normalized = 0.5 - (0.5 * Math.cos(phase * Math.PI * 2));
    return THREE.MathUtils.lerp(lower, upper, normalized);
  }

  const wave = Math.sin(phase * Math.PI * 2);

  if (joint.type === "prismatic") {
    return wave * 0.12;
  }

  return wave * (Math.PI / 3);
}

export function CanonicalModelWorkbench({
  recordId,
  materializationSummary,
  candidateBaseFileUrl,
  assetRevisionKey,
  selectionKey,
  pending,
  sidebar,
  renderInspector,
  viewportOverride,
  allowLocalUsdFile = true,
  productLabel,
}: CanonicalModelWorkbenchProps): JSX.Element {
  const { options: renderOptions, setOption: setRenderOption } = useRenderOptions();

  const [urdfSpec, setUrdfSpec] = useState<UrdfSpec | null>(null);
  const [jointNodes, setJointNodes] = useState<Map<string, THREE.Object3D> | null>(null);
  const [previewJointValues, setPreviewJointValues] = useState<Map<string, number>>(new Map());
  const [inspectorCollapsed, setInspectorCollapsed] = useState(readInspectorCollapsedFromUrl);
  const [snapshotExporter, setSnapshotExporter] = useState<SnapshotExporter | null>(null);
  const [modelLoadState, setModelLoadState] = useState<{
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
  }>({
    loading: false,
    error: null,
    missingArtifacts: false,
    collisionGeometryState: EMPTY_COLLISION_GEOMETRY_STATE,
  });
  const inspectorPanelRef = useRef<PanelImperativeHandle | null>(null);
  const initialInspectorCollapsedRef = useRef(inspectorCollapsed);
  const jointValuesRef = useRef<Map<string, number>>(new Map());
  const previewJointValuesRef = useRef<Map<string, number>>(new Map());
  const previewUiLastSyncRef = useRef(0);
  const viewportInvalidateRef = useRef<(() => void) | null>(null);

  const { jointValues, setJointValue, applyJointValues, resetAll } = useJointController(
    jointNodes,
    urdfSpec,
  );

  const handleUrdfSpecChange = useCallback(
    (spec: UrdfSpec | null, nodes: Map<string, THREE.Object3D> | null) => {
      setUrdfSpec(spec);
      setJointNodes(nodes);
    },
    [],
  );

  const handleRenderOptionChange = useCallback(
    <K extends keyof InspectorRenderOptions>(key: K, value: InspectorRenderOptions[K]) => {
      setRenderOption(key, value);
    },
    [setRenderOption],
  );

  const handleInspectorResize = useCallback((size: PanelSize) => {
    setInspectorCollapsed(size.inPixels <= 24);
  }, []);

  const handleInspectorExpand = useCallback(() => {
    setInspectorCollapsed(false);
    inspectorPanelRef.current?.expand();
  }, []);

  const handleViewportInvalidateReady = useCallback((invalidate: (() => void) | null) => {
    viewportInvalidateRef.current = invalidate;
  }, []);

  const handleSnapshotReady = useCallback((snapshot: SnapshotExporter | null) => {
    setSnapshotExporter(() => snapshot);
  }, []);

  useEffect(() => {
    jointValuesRef.current = jointValues;
  }, [jointValues]);

  useEffect(() => {
    if (!initialInspectorCollapsedRef.current) {
      return;
    }

    inspectorPanelRef.current?.collapse();
  }, []);

  useEffect(() => {
    syncInspectorCollapsedToUrl(inspectorCollapsed);
  }, [inspectorCollapsed]);

  useEffect(() => {
    if (!recordId || !urdfSpec) {
      return;
    }

    const validJointNames = new Set(urdfSpec.joints.map((joint) => joint.name));
    const nextPose = readJointPoseFromUrl(recordId, validJointNames);
    applyJointValues(nextPose, { commit: true });
  }, [applyJointValues, recordId, urdfSpec]);

  useEffect(() => {
    syncJointPoseToUrl(recordId, jointValues);
  }, [jointValues, recordId]);

  useEffect(() => {
    if (!urdfSpec || !renderOptions.autoAnimate) {
      previewJointValuesRef.current = new Map();
      previewUiLastSyncRef.current = 0;
      setPreviewJointValues(new Map());
      applyJointValues(jointValuesRef.current);
      return;
    }

    const previewMotions = buildPreviewJointMotions(urdfSpec);
    if (previewMotions.length === 0) {
      setPreviewJointValues(new Map());
      return;
    }

    let frameId = 0;

    const tick = (now: number) => {
      const elapsedSeconds = now / 1000;
      const nextValues = new Map<string, number>();

      for (const motion of previewMotions) {
        const jointPhase = THREE.MathUtils.euclideanModulo(
          (elapsedSeconds / motion.cycleSeconds) + motion.phaseOffset,
          1,
        );
        nextValues.set(motion.joint.name, previewJointValue(motion.joint, jointPhase));
      }

      previewJointValuesRef.current = nextValues;
      if (now - previewUiLastSyncRef.current >= PREVIEW_UI_SYNC_MS) {
        previewUiLastSyncRef.current = now;
        startTransition(() => {
          setPreviewJointValues(new Map(nextValues));
        });
      }
      applyJointValues(nextValues);
      viewportInvalidateRef.current?.();
      frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frameId);
      previewJointValuesRef.current = new Map();
      previewUiLastSyncRef.current = 0;
      setPreviewJointValues(new Map());
      applyJointValues(jointValuesRef.current);
      viewportInvalidateRef.current?.();
    };
  }, [applyJointValues, renderOptions.autoAnimate, urdfSpec]);

  const inspectorRenderOptions = {
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

  const materializationPresentation = useMemo(
    () => materializationSummary
      ? resolveMaterializationPresentation(materializationSummary)
      : null,
    [materializationSummary],
  );
  const materializationBlocksViewer = Boolean(
    materializationPresentation && !materializationPresentation.displayable,
  );
  const missingArtifactsState = useMemo(() => {
    if (!materializationSummary || !materializationPresentation) {
      return null;
    }
    const declaredBlockingStatus = materializationPresentation.blockingStatus;
    if (!declaredBlockingStatus && !modelLoadState.missingArtifacts) {
      return null;
    }

    const detail = !declaredBlockingStatus && modelLoadState.error
      ? `Viewer request failed: ${modelLoadState.error}`
      : null;

    return {
      status: declaredBlockingStatus ?? "missing",
      issues: declaredBlockingStatus ? materializationSummary.materialization_issues : [],
      hasCompileReport: materializationSummary.has_compile_report,
      detail,
    };
  }, [materializationPresentation, materializationSummary, modelLoadState.error, modelLoadState.missingArtifacts]);

  const displayedJointValues = renderOptions.autoAnimate ? previewJointValues : jointValues;
  const collisionSupport = useMemo<CanonicalModelCollisionSupport | null>(() => {
    if (
      !recordId
      || !urdfSpec
      || modelLoadState.loading
      || modelLoadState.error
      || pending
    ) {
      return null;
    }

    const {
      declaredCount,
      rigidRuntimeCount,
      deformableRestShapeCount,
      missingDeformableRestShapePartCount,
      missingDeformableRestShapePartNames,
      primitiveCount,
      meshCount,
      loadedMeshCount,
      failedMeshCount,
      loading: collisionLoading,
    } = modelLoadState.collisionGeometryState;
    const availableCollisionCount = primitiveCount + loadedMeshCount;
    const countBreakdown = [
      rigidRuntimeCount > 0 ? `${rigidRuntimeCount} rigid runtime` : null,
      deformableRestShapeCount > 0
        ? `${deformableRestShapeCount} deformable rest-shape`
        : null,
      missingDeformableRestShapePartCount > 0
        ? `${missingDeformableRestShapePartCount} deformable ${
          missingDeformableRestShapePartCount === 1 ? "part" : "parts"
        } missing rest-shape collision`
        : null,
    ].filter((entry): entry is string => entry !== null).join(" · ");
    const declaredSummary = `${declaredCount} collision ${declaredCount === 1 ? "shape" : "shapes"}${
      countBreakdown ? ` · ${countBreakdown}` : ""
    }`;
    const semanticDetails = [
      missingDeformableRestShapePartCount > 0
        ? `Rest-shape collision geometry is missing for: ${missingDeformableRestShapePartNames.join(", ")}.`
        : null,
      deformableRestShapeCount > 0
        ? "Deformable collision geometry is shown in its rest configuration; this viewer does not simulate live deformation or contact response."
        : null,
    ].filter((entry): entry is string => entry !== null);
    const detailWithSemantics = (base: string): string => [base, ...semanticDetails].join(" ");
    const supportCounts = {
      rigidRuntimeCount,
      deformableRestShapeCount,
      missingDeformableRestShapePartCount,
      missingDeformableRestShapePartNames,
    };

    if (declaredCount === 0) {
      return {
        available: false,
        summary: "No collision geometry declared",
        detail: detailWithSemantics("This asset currently provides visual geometry only."),
        ...supportCounts,
      };
    }

    if (collisionLoading && availableCollisionCount === 0) {
      return {
        available: true,
        summary: `${declaredSummary} · loading meshes…`,
        detail: detailWithSemantics("The viewer is verifying collision meshes for this asset."),
        ...supportCounts,
      };
    }

    if (availableCollisionCount > 0) {
      const hasPartialFailures = failedMeshCount > 0;
      return {
        available: true,
        summary: declaredSummary,
        detail: hasPartialFailures
          ? detailWithSemantics("Some collision meshes are unavailable, but partial collision geometry is loaded.")
          : detailWithSemantics("Collision geometry is available for inspection."),
        ...supportCounts,
      };
    }

    if (meshCount > 0 && failedMeshCount === 0) {
      return {
        available: true,
        summary: declaredSummary,
        detail: detailWithSemantics("Collision meshes will load when collision rendering is enabled."),
        ...supportCounts,
      };
    }

    return {
      available: false,
      summary: failedMeshCount > 0 ? "Collision meshes failed to load" : "No collision geometry declared",
      detail: failedMeshCount > 0
        ? detailWithSemantics("The URDF references collision meshes, but the viewer could not load them from disk.")
        : detailWithSemantics("This asset currently provides visual geometry only."),
      ...supportCounts,
    };
  }, [modelLoadState.collisionGeometryState, modelLoadState.error, modelLoadState.loading, pending, recordId, urdfSpec]);

  useEffect(() => {
    if (!collisionSupport || collisionSupport.available || !renderOptions.showCollisions) {
      return;
    }
    setRenderOption("showCollisions", false);
  }, [collisionSupport, renderOptions.showCollisions, setRenderOption]);

  const collisionCoverageIncomplete = Boolean(
    collisionSupport && collisionSupport.missingDeformableRestShapePartCount > 0,
  );
  const collisionNoticeIsPartial = Boolean(
    collisionSupport?.available && collisionCoverageIncomplete,
  );
  const collisionNotice = collisionSupport
    && (!collisionSupport.available || collisionCoverageIncomplete)
    && !missingArtifactsState ? (
    <div className="group relative inline-flex cursor-default">
      <div
        className="inline-flex items-center gap-1.5 rounded-md bg-amber-500/[0.08] px-2.5 py-1 text-[11px] font-medium text-amber-600 shadow-sm backdrop-blur-sm"
        aria-label={`${collisionNoticeIsPartial ? "Partial collision coverage" : "Missing collision geometry"}. See Render Tab on the right.`}
      >
        <AlertTriangle className="size-3 shrink-0 opacity-70" />
        <span>{collisionNoticeIsPartial ? "Partial Collision Coverage" : "Missing Collision Geometry"}</span>
      </div>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 min-w-max -translate-x-1/2 -translate-y-0.5 rounded-md bg-amber-500/[0.08] px-2.5 py-1.5 text-[11px] font-medium text-amber-600 opacity-0 shadow-sm backdrop-blur-sm transition duration-150 ease-out group-hover:translate-y-0 group-hover:opacity-100">
        See Render Tab on the right
      </span>
    </div>
  ) : null;

  const artifactContractWarning = materializationSummary
    && materializationPresentation?.showContractWarning
    ? (
      <ArtifactContractWarning issues={materializationSummary.materialization_issues} />
    )
    : null;
  const viewportNotice = artifactContractWarning || collisionNotice ? (
    <div className="flex flex-wrap items-end gap-2">
      {artifactContractWarning}
      {collisionNotice}
    </div>
  ) : null;

  useEffect(() => {
    setModelLoadState({
      loading: false,
      error: null,
      missingArtifacts: false,
      collisionGeometryState: EMPTY_COLLISION_GEOMETRY_STATE,
    });
  }, [selectionKey]);

  const inspectorContext: CanonicalModelInspectorContext = {
    urdfSpec,
    jointValues: displayedJointValues,
    onJointChange: setJointValue,
    onResetAll: resetAll,
    renderOptions: inspectorRenderOptions,
    onRenderOptionChange: handleRenderOptionChange,
    onSnapshot: snapshotExporter,
    collisionSupport,
  };
  const hasViewportOverride = viewportOverride !== undefined;
  const missingArtifactsProductLabel =
    productLabel === "reconstruction" ? "Reconstruction" : "Generation";

  return (
    <div
      className="relative min-h-0 flex-1"
      data-model-viewer-product={productLabel || undefined}
    >
      <ResizablePanelGroup orientation="horizontal" className="h-full">
        <ResizablePanel defaultSize="22%" minSize="18%" maxSize="32%" className="min-w-0">
          <Sidebar>{sidebar}</Sidebar>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize="50%" className="min-w-0">
          {hasViewportOverride ? viewportOverride : (
            <ViewportPanel
              baseFileUrl={pending || materializationBlocksViewer ? null : candidateBaseFileUrl}
              assetRevisionKey={assetRevisionKey}
              selectionKey={selectionKey}
              jointPoseSignal={displayedJointValues}
              renderOptions={renderOptions}
              onUrdfSpecChange={handleUrdfSpecChange}
              onInvalidateReady={handleViewportInvalidateReady}
              onSnapshotReady={handleSnapshotReady}
              onLoadStateChange={setModelLoadState}
              overlayNotice={viewportNotice}
              allowLocalUsdFile={allowLocalUsdFile}
              disabledOverlay={
                missingArtifactsState ? (
                  <MissingArtifactsOverlay
                    status={missingArtifactsState.status}
                    issues={missingArtifactsState.issues}
                    hasCompileReport={missingArtifactsState.hasCompileReport}
                    detail={missingArtifactsState.detail}
                    productLabel={missingArtifactsProductLabel}
                  />
                ) : undefined
              }
            />
          )}
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel
          defaultSize="28%"
          minSize="20%"
          maxSize="40%"
          className="min-w-0"
          collapsible
          panelRef={inspectorPanelRef}
          onResize={handleInspectorResize}
        >
          <Inspector
            disabledOverlay={
              missingArtifactsState ? (
                <MissingArtifactsOverlay
                  status={missingArtifactsState.status}
                  issues={missingArtifactsState.issues}
                  hasCompileReport={missingArtifactsState.hasCompileReport}
                  detail={missingArtifactsState.detail}
                  compact
                  productLabel={missingArtifactsProductLabel}
                />
              ) : undefined
            }
          >
            {renderInspector(inspectorContext)}
          </Inspector>
        </ResizablePanel>
      </ResizablePanelGroup>
      {inspectorCollapsed ? (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={handleInspectorExpand}
          aria-label="Expand inspector panel"
          className="absolute right-3 top-1/2 z-20 h-8 w-8 -translate-y-1/2 rounded-full bg-[var(--surface-0)] p-0 text-[var(--text-tertiary)] shadow-[0_2px_8px_rgba(0,0,0,0.08)] hover:text-[var(--text-primary)]"
        >
          <ChevronLeft className="size-4" />
        </Button>
      ) : null}
    </div>
  );
}
