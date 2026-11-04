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

import type { ListMaterializationStatus, MaterializationStatus } from "@/lib/types";

export type BlockingMaterializationStatus = Extract<
  MaterializationStatus,
  "missing" | "compile_failed"
>;

export type MaterializationPresentation = {
  displayable: boolean;
  blockingStatus: BlockingMaterializationStatus | null;
  showContractWarning: boolean;
};

export function resolveMaterializationPresentation(record: {
  materialization_status: ListMaterializationStatus;
  materialization_displayable: boolean;
}): MaterializationPresentation {
  const displayable = record.materialization_displayable;

  return {
    displayable,
    blockingStatus: displayable
      ? null
      : record.materialization_status === "compile_failed"
        ? "compile_failed"
        : "missing",
    showContractWarning: displayable && record.materialization_status === "contract_invalid",
  };
}
