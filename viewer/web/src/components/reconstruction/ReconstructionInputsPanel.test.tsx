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

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ReconstructionInputsPanel } from "@/components/reconstruction/ReconstructionInputsPanel";
import type { ReconstructionRecordEvidence } from "@/lib/types";

function evidence(
  overrides: Partial<ReconstructionRecordEvidence>,
): ReconstructionRecordEvidence {
  return {
    role: "semantic_subject_evidence",
    name: "semantic_image",
    label: "Semantic image",
    relative_path: "inputs/semantic.png",
    file_url: "/api/reconstruction-records/rec_reconstruction_001/files/inputs/semantic.png",
    media_kind: "image",
    byte_count: 2048,
    sha256: null,
    metadata: {
      key: "semantic_image",
      author_input: true,
      authority: "declared_semantic_evidence",
    },
    ...overrides,
  };
}

describe("ReconstructionInputsPanel", () => {
  it("shows only the three ordered primary input groups", () => {
    const html = renderToStaticMarkup(
      <ReconstructionInputsPanel
        inputs={[
          evidence({
            role: "host_preprocessing_audit",
            name: "observation_audit_0008",
            label: "Observation Audit 0008",
            relative_path:
              "inputs/preprocessing/audit/observation/source/preview/orientation_preview.png",
            metadata: {
              key: "observation_audit_0008",
              author_input: false,
              authority: "host_preprocessing_audit",
            },
          }),
          evidence({
            role: "metric_surface_evidence",
            name: "axial",
            label: "AXIAL",
            relative_path: "inputs/axial.txt",
            file_url: "/api/reconstruction-records/rec_reconstruction_001/files/inputs/axial.txt",
            media_kind: "text",
            byte_count: 8192,
            metadata: {
              key: "axial",
              author_input: true,
              authority: "surface_only_metric_evidence",
            },
          }),
          evidence({
            role: "user_supplied_source_surface_entry",
            name: "original_source",
            label: "Source mesh",
            relative_path: "inputs/source/original/input_mesh.glb",
            file_url:
              "/api/reconstruction-records/rec_reconstruction_001/files/inputs/source/original/input_mesh.glb",
            media_kind: "mesh",
            metadata: {
              key: "original_source",
              author_input: false,
              authority: "surface_evidence_provenance_only",
              source_resolved_path: "/private/source/input_mesh.glb",
            },
          }),
          evidence({
            role: "subject_isolation_audit",
            name: "subject_isolation_audit_0012",
            label: "Subject Isolation Audit 0012",
            relative_path:
              "inputs/preprocessing/audit/subject_isolation/evidence/component_id/azimuth_135.png",
            metadata: {
              key: "subject_isolation_audit_0012",
              author_input: false,
              authority: "host_preprocessing_audit",
            },
          }),
          evidence({}),
          evidence({
            role: "user_supplied_source_surface_entry",
            name: "original_source_duplicate",
            label: "Duplicate source mesh",
            relative_path: "inputs/source/original/duplicate.glb",
            media_kind: "mesh",
            metadata: {
              key: "original_source",
              author_input: false,
              authority: "surface_evidence_provenance_only",
            },
          }),
        ]}
      />,
    );

    const sourceIndex = html.indexOf("Source Mesh");
    const semanticIndex = html.indexOf("Semantic Input Image");
    const axialIndex = html.indexOf('data-input-group="axial"');

    expect(sourceIndex).toBeGreaterThan(-1);
    expect(semanticIndex).toBeGreaterThan(sourceIndex);
    expect(axialIndex).toBeGreaterThan(semanticIndex);
    expect(html.match(/data-input-group=/g)).toHaveLength(3);
    expect(html).toContain("Source mesh");
    expect(html).toContain("Semantic image");
    expect(html).toContain('src="/api/reconstruction-records/rec_reconstruction_001/files/inputs/semantic.png"');
    expect(html).toContain("Author input");
    expect(html).toContain("AXIAL");
    expect(html).toContain("Host input");
    expect(html).toContain("surface_only_metric_evidence");
    expect(html).toContain("8.0 KiB");
    expect(html).not.toContain("Observation Audit 0008");
    expect(html).not.toContain("orientation_preview.png");
    expect(html).not.toContain("Subject Isolation Audit 0012");
    expect(html).not.toContain("component_id/azimuth_135.png");
    expect(html).not.toContain("Duplicate source mesh");
    expect(html).not.toContain("source_resolved_path");
    expect(html).toContain("audit artifacts remain in the record provenance");
  });

  it("keeps the three primary groups visible when a record is incomplete", () => {
    const html = renderToStaticMarkup(<ReconstructionInputsPanel inputs={[]} />);

    expect(html.match(/data-input-group=/g)).toHaveLength(3);
    expect(html.match(/Not available in this record\./g)).toHaveLength(3);
  });
});
