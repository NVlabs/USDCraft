// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
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

import { lazy, type ComponentType, type LazyExoticComponent } from "react";

const CHUNK_RELOAD_KEY_PREFIX = "usdcraft:chunk-reload";

function isChunkLoadError(error: unknown): error is Error {
  if (!(error instanceof Error)) {
    return false;
  }

  const message = error.message;
  return (
    message.includes("Failed to fetch dynamically imported module")
    || message.includes("error loading dynamically imported module")
    || message.includes("Importing a module script failed")
    || message.includes("ChunkLoadError")
  );
}

function reloadKeyForError(error: Error): string {
  const failedUrl = error.message.match(/https?:\/\/\S+/)?.[0] ?? error.message;
  return `${CHUNK_RELOAD_KEY_PREFIX}:${failedUrl}`;
}

// React.lazy's public type uses `ComponentType<any>` for generic component props.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function lazyWithChunkReload<T extends ComponentType<any>>(
  loader: () => Promise<{ default: T }>,
): LazyExoticComponent<T> {
  return lazy(async () => {
    try {
      return await loader();
    } catch (error) {
      if (isChunkLoadError(error) && typeof window !== "undefined") {
        const storageKey = reloadKeyForError(error);
        if (!window.sessionStorage.getItem(storageKey)) {
          window.sessionStorage.setItem(storageKey, String(Date.now()));
          window.location.reload();
          return new Promise<{ default: T }>(() => {});
        }
      }

      throw error;
    }
  });
}
