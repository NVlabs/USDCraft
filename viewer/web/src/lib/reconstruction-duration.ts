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

import type { ReconstructionRecordSummary } from "@/lib/types";

export function reconstructionDuration(record: ReconstructionRecordSummary): { text: string; title: string } | null {
  const value = record.duration_seconds;
  if (value == null || !Number.isFinite(value) || value < 0) return null;
  const seconds = Math.round(value);
  const minutes = Math.floor(seconds / 60);
  const time = `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
  return {
    text: `≈ ${time}`,
    title: "Run time from first stage to record creation; excludes queue and final persistence tail",
  };
}
