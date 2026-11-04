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

import { describe, expect, it } from "vitest";
import { reconstructionDuration } from "./reconstruction-duration";
import type { ReconstructionRecordSummary } from "./types";

const record = (duration_seconds?: number) => ({ duration_seconds }) as ReconstructionRecordSummary;

describe("record duration badge", () => {
  it("shows the approximate run time", () => {
    expect(reconstructionDuration(record(599.8))?.text).toBe("≈ 10m 00s");
  });
  it("omits missing timing but preserves a recorded zero", () => {
    expect(reconstructionDuration(record())).toBeNull();
    expect(reconstructionDuration(record(NaN))).toBeNull();
    expect(reconstructionDuration(record(0))?.text).toBe("≈ 0m 00s");
  });
});
