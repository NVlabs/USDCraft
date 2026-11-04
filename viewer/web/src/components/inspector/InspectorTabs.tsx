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

import { Boxes } from "lucide-react";
import { Suspense, useState, type JSX, type ReactNode } from "react";

import { lazyWithChunkReload } from "@/lib/lazy";
import type { InspectorTab } from "@/lib/types";
import { useViewer, useViewerDispatch } from "@/lib/viewer-context";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { InspectPanel } from "@/components/inspector/InspectPanel";
import { RenderOptionsPanel } from "@/components/inspector/RenderOptionsPanel";
import {
  inspectorTabs,
  type InspectorPanelTab,
  type InspectorRecordAdapter,
  type RenderOptionsExtraSection,
} from "@/components/inspector/record-adapter";
import { defaultRenderOptions, type RenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { SnapshotExporter } from "@/components/viewer3d/SceneCanvas";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";

const CodePanel = lazyWithChunkReload(() =>
  import("@/components/inspector/CodePanel").then((module) => ({
    default: module.CodePanel,
  })),
);

const MetadataPanel = lazyWithChunkReload(() =>
  import("@/components/inspector/MetadataPanel").then((module) => ({
    default: module.MetadataPanel,
  })),
);

export type InspectorTabsProps = {
  urdfSpec?: UrdfSpec | null;
  jointValues?: Map<string, number>;
  onJointChange?: (name: string, value: number) => void;
  onResetAll?: () => void;
  renderOptions?: RenderOptions;
  onRenderOptionChange?: <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => void;
  onSnapshot?: SnapshotExporter | null;
  activeTab?: InspectorPanelTab;
  onActiveTabChange?: (tab: InspectorPanelTab) => void;
  inputsContent?: ReactNode;
  diagnosticsContent?: ReactNode;
  recordAdapter?: InspectorRecordAdapter;
  renderExtraSections?: readonly RenderOptionsExtraSection[];
  collisionSupport?: {
    available: boolean;
    summary: string;
    detail: string;
    rigidRuntimeCount: number;
    deformableRestShapeCount: number;
    missingDeformableRestShapePartCount: number;
    missingDeformableRestShapePartNames: string[];
  } | null;
};

function InspectorTabFallback(): JSX.Element {
  return (
    <div className="flex h-32 items-center justify-center">
      <p className="text-[10px] text-[var(--text-quaternary)]">Loading...</p>
    </div>
  );
}

export function InspectorTabs({
  urdfSpec = null,
  jointValues = new Map(),
  onJointChange = () => {},
  onResetAll = () => {},
  renderOptions,
  onRenderOptionChange,
  onSnapshot = null,
  activeTab,
  onActiveTabChange,
  inputsContent,
  diagnosticsContent,
  recordAdapter,
  renderExtraSections = [],
  collisionSupport = null,
}: InspectorTabsProps): JSX.Element {
  const {
    selectedInspectorTab: generationInspectorTab,
    selectedRecordId: generationRecordId,
  } = useViewer();
  const dispatch = useViewerDispatch();

  const [localOptions, setLocalOptions] = useState<RenderOptions>(defaultRenderOptions);
  const [localPanelTab, setLocalPanelTab] = useState<InspectorPanelTab | null>(null);
  const activeOptions = renderOptions ?? localOptions;
  const handleOptionChange = onRenderOptionChange ?? (<K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => {
    setLocalOptions((prev) => ({ ...prev, [key]: value }));
  });

  const tabLabels: Record<InspectorPanelTab, string> = {
    inspect: "Model",
    render: "View",
    code: "Source",
    metadata: "Runtime",
    inputs: "Evidence",
    diagnostics: "Diagnostics",
  };
  const hasInputs = inputsContent !== undefined;
  const hasDiagnostics = diagnosticsContent !== undefined;
  const tabs = inspectorTabs(hasInputs, hasDiagnostics);
  const selectedInspectorTab = activeTab ?? localPanelTab ?? generationInspectorTab;
  const selectedRecordId = recordAdapter?.recordId ?? generationRecordId;

  const handleTabChange = (value: string): void => {
    const nextTab = value as InspectorPanelTab;
    if (onActiveTabChange) {
      onActiveTabChange(nextTab);
      return;
    }
    if (activeTab !== undefined) return;
    if (nextTab === "inputs" || nextTab === "diagnostics") {
      setLocalPanelTab(nextTab);
      return;
    }
    setLocalPanelTab(null);
    dispatch({ type: "SET_INSPECTOR_TAB", payload: nextTab as InspectorTab });
  };

  return (
    <Tabs
      value={selectedInspectorTab}
      onValueChange={handleTabChange}
      className="relative flex h-full flex-col gap-0"
    >
      {/* Tab bar */}
      <div className="custom-scrollbar flex min-h-11 items-center overflow-x-auto border-b border-[var(--border-default)] bg-[linear-gradient(180deg,var(--surface-0),var(--surface-1))] px-1 2xl:px-2">
        <TabsList className="flex h-auto shrink-0 gap-0 rounded-none border-none bg-transparent p-0">
          {tabs.map((tab) => (
            <TabsTrigger
              key={tab}
              value={tab}
              className="relative rounded-none px-1.5 py-3 text-[10px] font-medium text-[var(--text-tertiary)] transition-colors duration-150 hover:text-[var(--text-secondary)] data-[state=active]:bg-transparent data-[state=active]:text-[var(--accent-strong)] data-[state=active]:shadow-none data-[state=active]:after:absolute data-[state=active]:after:bottom-0 data-[state=active]:after:left-1.5 data-[state=active]:after:right-1.5 data-[state=active]:after:h-0.5 data-[state=active]:after:rounded-full data-[state=active]:after:bg-[var(--accent)] 2xl:px-2.5 2xl:text-[10.5px] 2xl:data-[state=active]:after:left-2.5 2xl:data-[state=active]:after:right-2.5"
            >{tabLabels[tab]}</TabsTrigger>
          ))}
        </TabsList>
      </div>

      <TabsContent value="inspect" className="min-h-0 flex-1 overflow-hidden px-3 pb-3 pt-3">
        <InspectPanel
          urdfSpec={urdfSpec}
          jointValues={jointValues}
          onJointChange={onJointChange}
          onResetAll={onResetAll}
          recordAdapter={recordAdapter}
        />
      </TabsContent>

      <TabsContent value="render" className="min-h-0 flex-1 overflow-hidden px-3 pb-3 pt-3">
        <RenderOptionsPanel
          options={activeOptions}
          onOptionChange={handleOptionChange}
          onSnapshot={onSnapshot}
          collisionSupport={collisionSupport}
          extraSections={renderExtraSections}
        />
      </TabsContent>

      <TabsContent value="code" className="min-h-0 flex-1 overflow-hidden px-3 pb-3 pt-3">
        {selectedInspectorTab === "code" ? (
          <Suspense fallback={<InspectorTabFallback />}>
            <CodePanel recordAdapter={recordAdapter} />
          </Suspense>
        ) : null}
      </TabsContent>

      <TabsContent value="metadata" className="min-h-0 flex-1 overflow-hidden px-3 pb-3 pt-3">
        {selectedInspectorTab === "metadata" ? (
          <Suspense fallback={<InspectorTabFallback />}>
            <MetadataPanel recordAdapter={recordAdapter} />
          </Suspense>
        ) : null}
      </TabsContent>

      {hasInputs ? (
        <TabsContent value="inputs" className="min-h-0 flex-1 overflow-hidden">
          {inputsContent}
        </TabsContent>
      ) : null}

      {hasDiagnostics ? (
        <TabsContent value="diagnostics" className="min-h-0 flex-1 overflow-hidden">
          {diagnosticsContent}
        </TabsContent>
      ) : null}

      {!selectedRecordId && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--surface-0)]/95 backdrop-blur-sm">
          <div className="max-w-56 px-5 text-center">
            <span className="mx-auto mb-3 grid size-9 place-items-center rounded-xl border border-[var(--border-default)] bg-[var(--surface-1)] text-[var(--accent)]">
              <Boxes className="size-4" />
            </span>
            <p className="text-[11px] font-medium text-[var(--text-secondary)]">Choose an asset</p>
            <p className="mt-1 text-[9.5px] leading-4 text-[var(--text-quaternary)]">
              Model structure, source, and runtime details will appear here.
            </p>
          </div>
        </div>
      )}
    </Tabs>
  );
}
