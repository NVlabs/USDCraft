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

import { Search, X } from "lucide-react";
import { type JSX } from "react";

import { useViewer, useViewerDispatch } from "@/lib/viewer-context";
import { Input } from "@/components/ui/input";

export function RecordSearch(): JSX.Element {
  const { searchQuery } = useViewer();
  const dispatch = useViewerDispatch();

  return (
    <div className="relative">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3 -translate-y-1/2 text-[var(--text-quaternary)]" />
      <Input
        value={searchQuery}
        onChange={(event) => dispatch({ type: "SET_SEARCH", payload: event.target.value })}
        placeholder="Search records…"
        className="h-8 rounded-md border-[var(--border-default)] bg-[var(--surface-1)] pl-7 pr-8 text-[10px] shadow-[inset_0_1px_0_rgba(0,0,0,0.08)]"
      />
      {searchQuery ? (
        <button
          type="button"
          onClick={() => dispatch({ type: "SET_SEARCH", payload: "" })}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-[var(--text-quaternary)] hover:text-[var(--text-secondary)]"
          aria-label="Clear search"
        >
          <X className="size-3" />
        </button>
      ) : null}
    </div>
  );
}
