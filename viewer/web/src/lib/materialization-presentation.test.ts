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

import { resolveMaterializationPresentation } from "./materialization-presentation";

describe("resolveMaterializationPresentation", () => {
  it("keeps contract-invalid artifacts displayable while requesting a warning", () => {
    expect(resolveMaterializationPresentation({
      materialization_status: "contract_invalid",
      materialization_displayable: true,
    })).toEqual({
      displayable: true,
      blockingStatus: null,
      showContractWarning: true,
    });
  });

  it("blocks records whose model files are missing", () => {
    expect(resolveMaterializationPresentation({
      materialization_status: "missing",
      materialization_displayable: false,
    })).toEqual({
      displayable: false,
      blockingStatus: "missing",
      showContractWarning: false,
    });
  });

  it("blocks failed compilations with their distinct status", () => {
    expect(resolveMaterializationPresentation({
      materialization_status: "compile_failed",
      materialization_displayable: false,
    })).toEqual({
      displayable: false,
      blockingStatus: "compile_failed",
      showContractWarning: false,
    });
  });

  it("does not layer a warning over a contradictory non-displayable contract state", () => {
    expect(resolveMaterializationPresentation({
      materialization_status: "contract_invalid",
      materialization_displayable: false,
    })).toEqual({
      displayable: false,
      blockingStatus: "missing",
      showContractWarning: false,
    });
  });
});
