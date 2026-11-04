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

import { type JSX, type ReactNode } from "react";

import { useViewer } from "@/lib/viewer-context";
import { cn } from "@/lib/utils";

interface InspectorProps {
  children: ReactNode;
  disabledOverlay?: ReactNode;
}

export function Inspector({ children, disabledOverlay }: InspectorProps): JSX.Element {
  const { inspectorOpen } = useViewer();

  if (!inspectorOpen) {
    return <aside className="h-full min-w-0 bg-[var(--surface-0)]" />;
  }

  return (
    <aside className="relative flex h-full min-w-0 flex-col overflow-hidden bg-[var(--surface-0)] shadow-[-4px_0_24px_rgba(0,0,0,0.08)]">
      <div
        className={cn(
          "flex h-full min-w-0 flex-col overflow-hidden transition duration-200",
          disabledOverlay ? "pointer-events-none select-none blur-[7px] saturate-50 opacity-35" : "",
        )}
      >
        {children}
      </div>
      {disabledOverlay ? <div className="absolute inset-0 z-10">{disabledOverlay}</div> : null}
    </aside>
  );
}
