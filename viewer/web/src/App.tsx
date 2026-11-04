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

import { Suspense, useCallback, useState, type JSX } from "react";

import { readAppMode, syncAppModeToUrl } from "@/lib/app-mode";
import { lazyWithChunkReload } from "@/lib/lazy";
import type { AppMode } from "@/lib/types";
import { ViewerProvider } from "@/lib/viewer-context";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppHeader } from "@/components/layout/AppHeader";

const ViewerShell = lazyWithChunkReload(() => import("@/ViewerShell"));
const ReconstructionRecordShell = lazyWithChunkReload(
  () => import("@/components/reconstruction/ReconstructionRecordShell"),
);

function AppLoadingFallback(): JSX.Element {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center">
      <p className="text-[12px] text-[var(--text-quaternary)]">Loading...</p>
    </div>
  );
}

export default function App(): JSX.Element {
  const [appMode, setAppMode] = useState<AppMode>(readAppMode);
  const handleAppModeChange = useCallback((mode: AppMode) => {
    setAppMode(mode);
    syncAppModeToUrl(mode);
  }, []);

  return (
    <TooltipProvider>
      <ViewerProvider appMode={appMode}>
        <div
          className="flex h-screen flex-col bg-[var(--surface-2)]"
          data-app-mode={appMode}
        >
          <AppHeader appMode={appMode} onAppModeChange={handleAppModeChange} />
          <Suspense fallback={<AppLoadingFallback />}>
            {appMode === "reconstruction" ? <ReconstructionRecordShell /> : <ViewerShell />}
          </Suspense>
        </div>
      </ViewerProvider>
    </TooltipProvider>
  );
}
