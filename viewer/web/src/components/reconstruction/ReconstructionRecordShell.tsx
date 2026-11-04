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
import { useEffect, useMemo, useState, type JSX, type ReactNode } from "react";

import { InspectorTabs } from "@/components/inspector/InspectorTabs";
import {
  buildInspectorMetadataPresentation,
  type InspectorPanelTab,
  type InspectorRecordAdapter,
} from "@/components/inspector/record-adapter";
import {
  CanonicalModelWorkbench,
  type CanonicalModelInspectorContext,
} from "@/components/model-viewer/CanonicalModelWorkbench";
import { ReconstructionInputsPanel } from "@/components/reconstruction/ReconstructionInputsPanel";
import { ReconstructionCompareView } from "@/components/reconstruction/ReconstructionCompareView";
import { ReconstructionDiagnosticsPanel } from "@/components/reconstruction/ReconstructionDiagnosticsPanel";
import { ReconstructionOverlayControls } from "@/components/reconstruction/ReconstructionOverlayControls";
import { ReconstructionRecordBrowser } from "@/components/reconstruction/ReconstructionRecordBrowser";
import {
  ReconstructionRecordOverlayCanvas,
  type ReconstructionOverlaySettings,
} from "@/components/reconstruction/ReconstructionRecordOverlayCanvas";
import { fetchReconstructionRecordTextFile } from "@/lib/api";
import { reconstructionRunSettings } from "@/lib/run-settings";
import {
  readReconstructionRecord,
  syncReconstructionRecordToUrl,
} from "@/lib/reconstruction-record-view";
import {
  DEFAULT_RECONSTRUCTION_INSPECTOR_TAB,
  resolveReconstructionViewport,
} from "@/lib/reconstruction-viewport";
import type { ReconstructionRecordDetail, ReconstructionRecordSummary } from "@/lib/types";
import {
  reconstructionRecordQueryOptions,
  reconstructionRecordsQueryOptions,
} from "@/lib/viewer-queries";

const DEFAULT_OVERLAY: ReconstructionOverlaySettings = {
  sourceVisible: true,
  sourceOpacity: 0.42,
  candidateVisible: true,
  candidateOpacity: 0.72,
  diagnosticsVisible: false,
};

function Message({ children }: { children: string }): JSX.Element {
  return (
    <div className="flex h-full items-center justify-center px-6 text-center">
      <p className="max-w-md text-[11px] leading-5 text-[var(--text-quaternary)]">{children}</p>
    </div>
  );
}

function buildRecordAdapter(detail: ReconstructionRecordDetail): InspectorRecordAdapter {
  return {
    recordId: detail.summary.record_id,
    summary: detail.summary,
    requestText: detail.request_text,
    textLoader: fetchReconstructionRecordTextFile,
    ...(detail.candidate.base_file_url
      ? {
          materializedFileUrl: (_recordId: string, filePath: string) =>
            `${detail.candidate.base_file_url}/${filePath}`,
        }
      : {}),
    codeFilePaths: {
      "model.py": "model.py",
      "model.urdf": "model.urdf",
    },
    metadataPresentation: buildInspectorMetadataPresentation(detail.summary, {
      runSettings: reconstructionRunSettings(detail),
      artifacts: [
        {
          filePath: "provenance.json",
          label: "provenance.json",
          available: detail.summary.has_provenance,
        },
        {
          filePath: "compile_report.json",
          label: "compile_report.json",
          available: detail.summary.has_compile_report,
        },
      ],
    }),
  };
}

function ReconstructionInspector({
  context,
  detail,
  records,
  compareId,
  activeTab,
  overlayEnabled,
  overlaySettings,
  onActiveTabChange,
  onOverlayEnabledChange,
  onOverlaySettingsChange,
  onCompareChange,
}: {
  context: CanonicalModelInspectorContext;
  detail: ReconstructionRecordDetail;
  records: readonly ReconstructionRecordSummary[];
  compareId: string | null;
  activeTab: InspectorPanelTab;
  overlayEnabled: boolean;
  overlaySettings: ReconstructionOverlaySettings;
  onActiveTabChange: (tab: InspectorPanelTab) => void;
  onOverlayEnabledChange: (enabled: boolean) => void;
  onOverlaySettingsChange: (settings: ReconstructionOverlaySettings) => void;
  onCompareChange: (recordId: string | null) => void;
}): JSX.Element {
  const recordAdapter = useMemo(() => buildRecordAdapter(detail), [detail]);
  const overlayControls = (
    <ReconstructionOverlayControls
      enabled={overlayEnabled}
      settings={overlaySettings}
      sourceAvailable={detail.source.available}
      candidateAvailable={detail.candidate.available}
      diagnosticsAvailable={detail.summary.axial_diagnostic !== null}
      onEnabledChange={onOverlayEnabledChange}
      onSettingsChange={onOverlaySettingsChange}
    />
  );

  return (
    <InspectorTabs
      urdfSpec={context.urdfSpec}
      jointValues={context.jointValues}
      onJointChange={context.onJointChange}
      onResetAll={context.onResetAll}
      renderOptions={context.renderOptions}
      onRenderOptionChange={context.onRenderOptionChange}
      onSnapshot={context.onSnapshot}
      collisionSupport={context.collisionSupport}
      activeTab={activeTab}
      onActiveTabChange={onActiveTabChange}
      recordAdapter={recordAdapter}
      inputsContent={<ReconstructionInputsPanel inputs={detail.inputs} />}
      diagnosticsContent={
        <ReconstructionDiagnosticsPanel
          detail={detail}
          records={records}
          compareId={compareId}
          onCompareChange={onCompareChange}
        />
      }
      renderExtraSections={[
        {
          key: "reconstruction-overlay",
          label: "Comparison",
          content: overlayControls,
        },
      ]}
    />
  );
}

export default function ReconstructionRecordShell(): JSX.Element {
  const recordsQuery = useQuery(reconstructionRecordsQueryOptions());
  const records = useMemo(() => recordsQuery.data ?? [], [recordsQuery.data]);
  const [requestedId, setRequestedId] = useState<string | null>(readReconstructionRecord);
  const [activeTab, setActiveTab] = useState<InspectorPanelTab>(
    DEFAULT_RECONSTRUCTION_INSPECTOR_TAB,
  );
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const [overlaySettings, setOverlaySettings] =
    useState<ReconstructionOverlaySettings>(DEFAULT_OVERLAY);
  const [requestedCompareId, setRequestedCompareId] = useState<string | null>(null);
  const recordId = useMemo(() => {
    if (requestedId && records.some((record) => record.record_id === requestedId)) {
      return requestedId;
    }
    return records[0]?.record_id ?? null;
  }, [records, requestedId]);
  const selectedSummary = useMemo(
    () => records.find((record) => record.record_id === recordId) ?? null,
    [recordId, records],
  );
  const detailQuery = useQuery({
    ...reconstructionRecordQueryOptions(recordId ?? "missing"),
    enabled: Boolean(recordId),
  });
  const detail = detailQuery.data ?? null;
  const compareId =
    requestedCompareId &&
    requestedCompareId !== recordId &&
    records.some((record) => record.record_id === requestedCompareId)
      ? requestedCompareId
      : null;

  useEffect(() => {
    if (!recordId || requestedId === recordId) return;
    setRequestedId(recordId);
    syncReconstructionRecordToUrl(recordId);
  }, [recordId, requestedId]);

  useEffect(() => {
    if (detail?.source.available && detail.candidate.available) return;
    setOverlayEnabled(false);
  }, [detail]);

  const selectRecord = (nextId: string): void => {
    setRequestedId(nextId);
    setOverlayEnabled(false);
    setOverlaySettings(DEFAULT_OVERLAY);
    setRequestedCompareId(null);
    syncReconstructionRecordToUrl(nextId);
  };
  const recordsError = recordsQuery.error instanceof Error ? recordsQuery.error.message : null;
  const detailError = detailQuery.error instanceof Error ? detailQuery.error.message : null;
  const viewportMode = detail
    ? resolveReconstructionViewport({
        activeTab,
        overlayEnabled,
        sourceAvailable: detail.source.available,
        candidateAvailable: detail.candidate.available,
      })
    : "candidate";

  let viewportOverride: ReactNode | undefined;
  if (detailQuery.isPending && recordId) {
    viewportOverride = <Message>Loading Reconstruction record…</Message>;
  } else if (detailError) {
    viewportOverride = <Message>{detailError}</Message>;
  } else if (!recordId) {
    viewportOverride = <Message>Select a canonical Reconstruction record.</Message>;
  } else if (detail && viewportMode === "overlay") {
    viewportOverride = (
      <ReconstructionRecordOverlayCanvas detail={detail} settings={overlaySettings} />
    );
  } else if (detail && compareId) {
    viewportOverride = (
      <ReconstructionCompareView
        left={detail}
        rightRecordId={compareId}
        onClose={() => setRequestedCompareId(null)}
      />
    );
  }

  const renderInspector = (context: CanonicalModelInspectorContext): ReactNode => {
    if (!detail) {
      return <Message>{detailError ?? "Select a canonical Reconstruction record."}</Message>;
    }
    return (
      <ReconstructionInspector
        context={context}
        detail={detail}
        records={records}
        compareId={compareId}
        activeTab={activeTab}
        overlayEnabled={overlayEnabled}
        overlaySettings={overlaySettings}
        onActiveTabChange={setActiveTab}
        onOverlayEnabledChange={setOverlayEnabled}
        onOverlaySettingsChange={setOverlaySettings}
        onCompareChange={setRequestedCompareId}
      />
    );
  };

  return (
    <CanonicalModelWorkbench
      recordId={recordId}
      materializationSummary={detail?.summary ?? selectedSummary}
      candidateBaseFileUrl={detail?.candidate.base_file_url ?? null}
      assetRevisionKey={detail?.candidate.revision_key ?? null}
      selectionKey={recordId}
      pending={Boolean(recordId && !detail)}
      sidebar={(
        <ReconstructionRecordBrowser
          records={records}
          selectedRecordId={recordId}
          loading={recordsQuery.isPending}
          error={recordsError}
          onSelect={selectRecord}
        />
      )}
      renderInspector={renderInspector}
      viewportOverride={viewportOverride}
      productLabel="reconstruction"
    />
  );
}
