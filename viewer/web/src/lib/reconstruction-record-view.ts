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

import { updateUrlSearchParams } from "@/lib/url";

const RECORD_ID_RE = /^rec_[A-Za-z0-9_-]+$/;

export function reconstructionRecordFromSearch(search: string): string | null {
  const value = new URLSearchParams(search).get("recon-record");
  return value && RECORD_ID_RE.test(value) ? value : null;
}

export function readReconstructionRecord(): string | null {
  return typeof window === "undefined"
    ? null
    : reconstructionRecordFromSearch(window.location.search);
}

export function syncReconstructionRecordToUrl(recordId: string): void {
  updateUrlSearchParams((params) => {
    params.set("recon-record", recordId);
  });
}
