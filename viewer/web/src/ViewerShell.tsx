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

import type { JSX } from "react";

import { useViewer } from "@/lib/viewer-context";
import { RecordBrowser } from "@/components/browser/RecordBrowser";
import { GenerationInputsPanel } from "@/components/inspector/GenerationInputsPanel";
import { InspectorTabs } from "@/components/inspector/InspectorTabs";
import {
  CanonicalModelWorkbench,
  type CanonicalModelInspectorContext,
} from "@/components/model-viewer/CanonicalModelWorkbench";

function renderGenerationInspector(
  context: CanonicalModelInspectorContext,
  recordId: string | null,
): JSX.Element {
  return (
    <InspectorTabs
      inputsContent={<GenerationInputsPanel recordId={recordId} />}
      urdfSpec={context.urdfSpec}
      jointValues={context.jointValues}
      onJointChange={context.onJointChange}
      onResetAll={context.onResetAll}
      renderOptions={context.renderOptions}
      onRenderOptionChange={context.onRenderOptionChange}
      onSnapshot={context.onSnapshot}
      collisionSupport={context.collisionSupport}
    />
  );
}

export default function ViewerShell(): JSX.Element {
  const { selectedRecordId, selectedRecordSummary } = useViewer();
  const candidateBaseFileUrl = selectedRecordId
    ? `/api/records/${encodeURIComponent(selectedRecordId)}/files`
    : null;

  return (
    <CanonicalModelWorkbench
      recordId={selectedRecordId}
      materializationSummary={selectedRecordSummary}
      candidateBaseFileUrl={candidateBaseFileUrl}
      assetRevisionKey={selectedRecordSummary?.viewer_asset_updated_at ?? null}
      selectionKey={selectedRecordId}
      pending={Boolean(selectedRecordId && !selectedRecordSummary)}
      sidebar={<RecordBrowser />}
      renderInspector={(context) => renderGenerationInspector(context, selectedRecordId)}
      productLabel="generation"
    />
  );
}
