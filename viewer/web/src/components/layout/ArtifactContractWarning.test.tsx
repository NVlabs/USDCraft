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

import { ArtifactContractWarning } from "./ArtifactContractWarning";

describe("ArtifactContractWarning", () => {
  it("uses a click-expandable status warning and lists structured issues", () => {
    const html = renderToStaticMarkup(
      <ArtifactContractWarning
        issues={[
          {
            path: "compile_report.json",
            message: "checks_run is missing: generation_root_floating",
          },
          {
            path: "isaac/model.usdc",
            message: "artifact hash does not match",
          },
        ]}
      />,
    );

    expect(html).toContain("<details");
    expect(html).toContain("<summary");
    expect(html).toContain("Artifact contract is invalid");
    expect(html).toContain("A displayable model is available");
    expect(html).toContain("compile_report.json");
    expect(html).toContain("generation_root_floating");
    expect(html).toContain("isaac/model.usdc");
  });

  it("keeps the warning useful when the API reports no structured details", () => {
    const html = renderToStaticMarkup(<ArtifactContractWarning issues={[]} />);

    expect(html).toContain("Artifact contract is invalid");
    expect(html).toContain("No structured contract details were reported");
  });
});
