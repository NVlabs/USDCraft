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

import { MissingArtifactsOverlay } from "./MissingArtifactsOverlay";

describe("MissingArtifactsOverlay", () => {
  it("shows a failed compilation as a blocking error", () => {
    const html = renderToStaticMarkup(
      <MissingArtifactsOverlay
        status="compile_failed"
        issues={[
          {
            path: "compile_report.json",
            message: "model.py exited with status 1",
          },
        ]}
        hasCompileReport
      />,
    );

    expect(html).toContain("Compilation Failed");
    expect(html).toContain("Artifact compilation failed");
    expect(html).toContain("Generation compile report records a failed artifact build");
    expect(html).toContain("model.py exited with status 1");
    expect(html).toContain("compile_report.json remains available");
    expect(html).not.toContain("Artifact Contract Invalid");
  });

  it("keeps the no-materialization state distinct", () => {
    const html = renderToStaticMarkup(
      <MissingArtifactsOverlay
        status="missing"
        issues={[{ path: "compile_report.json", message: "missing JSON object" }]}
        hasCompileReport={false}
      />,
    );

    expect(html).toContain("Materialization Required");
    expect(html).toContain("Viewer model files are missing");
    expect(html).toContain("saved model.urdf entry");
    expect(html).not.toContain("Compilation Failed");
  });
});
