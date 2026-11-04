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

import { formatModelDimensions } from "./ViewportPanel";

describe("model dimensions HUD", () => {
  it("prints width, depth and height in metres, millimetres under one metre", () => {
    expect(formatModelDimensions({ x: 1.938, y: 1.103, z: 2.133 })).toBe(
      "1.94 m × 2.13 m × 1.10 m (w × d × h)",
    );
    expect(formatModelDimensions({ x: 0.345, y: 0.202, z: 0.298 })).toBe(
      "345 mm × 298 mm × 202 mm (w × d × h)",
    );
  });
});
