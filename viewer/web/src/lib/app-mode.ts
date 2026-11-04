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

import type { AppMode } from "@/lib/types";
import { updateUrlSearchParams } from "@/lib/url";

const RECONSTRUCTION_MODE = "reconstruction";
// Mode-specific URL state that must not leak into the other product.
const RECONSTRUCTION_PARAMS = ["recon-record"] as const;
function applyMode(params: URLSearchParams, mode: AppMode): void {
  if (mode === "generation") {
    params.delete("mode");
    for (const key of RECONSTRUCTION_PARAMS) params.delete(key);
  } else {
    params.set("mode", RECONSTRUCTION_MODE);
  }
}

export function appModeFromSearch(search: string): AppMode {
  const mode = new URLSearchParams(search).get("mode");
  return mode === RECONSTRUCTION_MODE ? "reconstruction" : "generation";
}

export function searchWithAppMode(search: string, mode: AppMode): string {
  const params = new URLSearchParams(search);
  applyMode(params, mode);
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

export function readAppMode(): AppMode {
  return typeof window === "undefined" ? "generation" : appModeFromSearch(window.location.search);
}

export function syncAppModeToUrl(mode: AppMode): void {
  updateUrlSearchParams((params) => applyMode(params, mode));
}
