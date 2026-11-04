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

import { type JSX } from "react";
import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { Box, RefreshCw, ScanLine } from "lucide-react";

import { viewerQueryKeys } from "@/lib/viewer-queries";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { AppMode } from "@/lib/types";

const PRODUCTS: ReadonlyArray<{
  mode: AppMode;
  label: string;
  icon: typeof Box;
}> = [
  { mode: "generation", label: "Generation", icon: Box },
  {
    mode: "reconstruction",
    label: "Reconstruction",
    icon: ScanLine,
  },
];

function UsdCraftMark(): JSX.Element {
  return (
    <img
      src="/usdcraft-mark.png"
      alt=""
      className="size-9 shrink-0 object-contain drop-shadow-[0_4px_10px_rgba(70,112,0,0.22)]"
      aria-hidden="true"
    />
  );
}

type AppHeaderProps = {
  appMode: AppMode;
  onAppModeChange: (mode: AppMode) => void;
};

export function AppHeader({ appMode, onAppModeChange }: AppHeaderProps): JSX.Element {
  const queryClient = useQueryClient();
  const activeFetchCount = useIsFetching({ queryKey: viewerQueryKeys.root() });

  return (
    <header className="flex h-[60px] shrink-0 items-center gap-5 border-b border-[var(--border-default)] bg-[var(--surface-0)] px-4 shadow-[0_1px_0_rgba(0,0,0,0.12)]">
      <div className="flex shrink-0 items-center gap-2.5">
        <UsdCraftMark />
        <span className="leading-none">
          <span className="wordmark block text-[15px] text-[var(--text-primary)]">USDCraft</span>
          <span className="wordmark-kicker mt-1 block text-[var(--text-quaternary)]">
            USD asset workbench
          </span>
        </span>
      </div>

      <span className="hidden h-6 w-px bg-[var(--border-default)] sm:block" aria-hidden="true" />

      <nav
        className="flex shrink-0 items-center gap-1 rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-2)] p-1"
        aria-label="Product"
      >
        {PRODUCTS.map((item) => {
          const active = item.mode === appMode;
          const ProductIcon = item.icon;
          return (
            <button
              key={item.mode}
              type="button"
              aria-pressed={active}
              onClick={() => onAppModeChange(item.mode)}
              className={`relative flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] transition-all ${
                active
                  ? "bg-[var(--accent-soft)] font-medium text-[var(--accent-strong)] shadow-[inset_0_0_0_1px_var(--accent-soft)]"
                  : "text-[var(--text-tertiary)] hover:bg-[var(--surface-1)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <ProductIcon className="size-3" strokeWidth={1.8} />
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="min-w-0 flex-1" />

      <div className="flex items-center gap-1">
        <span className="mr-2 hidden items-center gap-1.5 font-mono text-[8.5px] uppercase tracking-[0.08em] text-[var(--text-quaternary)] xl:flex">
          <span className="size-1.5 rounded-full bg-[var(--success)] shadow-[0_0_6px_var(--success)]" />
          Local viewer
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                void queryClient.invalidateQueries({ queryKey: viewerQueryKeys.root() })
              }
              disabled={activeFetchCount > 0}
              aria-label="Refresh records"
              className="h-7 w-7 rounded-md p-0 text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
            >
              <RefreshCw className={`size-3.5 ${activeFetchCount > 0 ? "animate-spin" : ""}`} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Refresh records</TooltipContent>
        </Tooltip>
      </div>
    </header>
  );
}
