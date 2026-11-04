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

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type Dispatch,
  type JSX,
  type ReactNode,
} from "react";
import { useQuery } from "@tanstack/react-query";

import { readAppMode } from "@/lib/app-mode";
import { bootstrapQueryOptions, recordSummaryQueryOptions } from "@/lib/viewer-queries";
import { reconcileWorkbenchSelection } from "@/lib/workbench-filters";
import type {
  AppMode,
  InputKindFilter,
  InspectorTab,
  RatingFilter,
  RecordSummary,
  SourceFilter,
  ViewerAction,
  ViewerState,
} from "@/lib/types";

const ViewerContext = createContext<ViewerState | null>(null);
const ViewerDispatchContext = createContext<Dispatch<ViewerAction> | null>(null);
const INSPECTOR_TABS = new Set<InspectorTab>(["inspect", "render", "code", "metadata"]);

function readInitialUrlState(): Pick<
  ViewerState,
  | "selectedRecordId"
  | "selectedInspectorTab"
  | "searchQuery"
  | "sourceFilter"
  | "modelFilter"
  | "inputKindFilter"
  | "ratingFilter"
> {
  if (typeof window === "undefined") {
    return {
      selectedRecordId: null,
      selectedInspectorTab: "inspect",
      searchQuery: "",
      sourceFilter: null,
      modelFilter: null,
      inputKindFilter: null,
      ratingFilter: null,
    };
  }
  const params = new URLSearchParams(window.location.search);
  const tab = params.get("tab") as InspectorTab | null;
  const inputValue = params.get("input");
  const inputKindFilter: InputKindFilter =
    inputValue === "text" || inputValue === "text_image" ? inputValue : null;
  const ratingValue = params.get("rating");
  const numericRating = ratingValue == null ? Number.NaN : Number(ratingValue);
  const ratingFilter: RatingFilter =
    ratingValue === "unrated"
      ? "unrated"
      : Number.isInteger(numericRating) && numericRating >= 1 && numericRating <= 5
        ? (numericRating as 1 | 2 | 3 | 4 | 5)
        : null;
  const sourceValue = params.get("source");
  const sourceFilter: SourceFilter =
    sourceValue === "codex" || sourceValue === "provider" ? sourceValue : null;
  return {
    selectedRecordId: params.get("record")?.trim() || null,
    selectedInspectorTab: tab && INSPECTOR_TABS.has(tab) ? tab : "inspect",
    searchQuery: params.get("q") ?? "",
    sourceFilter,
    modelFilter: params.get("model")?.trim() || null,
    inputKindFilter,
    ratingFilter,
  };
}

const initialUrlState = readInitialUrlState();
const initialState: ViewerState = {
  bootstrap: null,
  recordCache: {},
  selectedRecordId: initialUrlState.selectedRecordId,
  selectedRecordSummary: null,
  selectedInspectorTab: initialUrlState.selectedInspectorTab,
  inspectorOpen: true,
  loading: true,
  error: null,
  searchQuery: initialUrlState.searchQuery,
  sourceFilter: initialUrlState.sourceFilter,
  modelFilter: initialUrlState.modelFilter,
  inputKindFilter: initialUrlState.inputKindFilter,
  ratingFilter: initialUrlState.ratingFilter,
  workbenchNotice: null,
};

function indexRecords(records: Array<RecordSummary | null>): Record<string, RecordSummary> {
  const indexed: Record<string, RecordSummary> = {};
  for (const record of records) {
    if (record) indexed[record.record_id] = record;
  }
  return indexed;
}

function withUpdatedBootstrapRecords(
  state: ViewerState,
  records: Record<string, RecordSummary>,
): ViewerState["bootstrap"] {
  if (!state.bootstrap) return null;
  return {
    ...state.bootstrap,
    workbench_entries: state.bootstrap.workbench_entries.map((entry) => ({
      ...entry,
      record: records[entry.record_id] ?? entry.record,
    })),
  };
}

export function viewerReducer(state: ViewerState, action: ViewerAction): ViewerState {
  switch (action.type) {
    case "SET_BOOTSTRAP": {
      const incoming = indexRecords(action.payload.workbench_entries.map((entry) => entry.record));
      const recordCache = { ...state.recordCache, ...incoming };
      return {
        ...state,
        bootstrap: action.payload,
        recordCache,
        selectedRecordSummary: state.selectedRecordId
          ? recordCache[state.selectedRecordId] ?? state.selectedRecordSummary
          : null,
        loading: false,
        error: null,
      };
    }
    case "UPSERT_RECORDS": {
      const recordCache = { ...state.recordCache, ...indexRecords(action.payload) };
      return {
        ...state,
        bootstrap: withUpdatedBootstrapRecords(state, recordCache),
        recordCache,
        selectedRecordSummary: state.selectedRecordId
          ? recordCache[state.selectedRecordId] ?? null
          : null,
      };
    }
    case "SET_SELECTED_RECORD_SUMMARY": {
      const record = action.payload;
      if (!record) return { ...state, selectedRecordSummary: null };
      if (record.record_id !== state.selectedRecordId) return state;
      const recordCache = { ...state.recordCache, [record.record_id]: record };
      return {
        ...state,
        bootstrap: withUpdatedBootstrapRecords(state, recordCache),
        recordCache,
        selectedRecordSummary: record,
      };
    }
    case "SELECT_RECORD":
      return {
        ...state,
        selectedRecordId: action.payload,
        selectedRecordSummary: action.payload ? state.recordCache[action.payload] ?? null : null,
      };
    case "SET_INSPECTOR_TAB":
      return { ...state, selectedInspectorTab: action.payload };
    case "TOGGLE_INSPECTOR":
      return { ...state, inspectorOpen: !state.inspectorOpen };
    case "SET_LOADING":
      return { ...state, loading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload, loading: false };
    case "SET_SEARCH":
      return { ...state, searchQuery: action.payload };
    case "SET_SOURCE_FILTER":
      return { ...state, sourceFilter: action.payload };
    case "SET_MODEL_FILTER":
      return { ...state, modelFilter: action.payload };
    case "SET_INPUT_KIND_FILTER":
      return { ...state, inputKindFilter: action.payload };
    case "SET_RATING_FILTER":
      return { ...state, ratingFilter: action.payload };
    case "SET_WORKBENCH_NOTICE":
      return { ...state, workbenchNotice: action.payload };
    case "DELETE_RECORD_LOCAL": {
      const { recordId, nextRecordId } = action.payload;
      const remainingEntries =
        state.bootstrap?.workbench_entries.filter((entry) => entry.record_id !== recordId) ?? [];
      const recordCache = { ...state.recordCache };
      delete recordCache[recordId];
      const requestedNext = remainingEntries.find(
        (entry) => entry.record_id === nextRecordId && !entry.archived && entry.record,
      );
      const selectedRecordId =
        state.selectedRecordId === recordId
          ? requestedNext?.record_id ?? null
          : state.selectedRecordId;
      return {
        ...state,
        bootstrap: state.bootstrap
          ? { ...state.bootstrap, workbench_entries: remainingEntries }
          : null,
        recordCache,
        selectedRecordId,
        selectedRecordSummary: selectedRecordId ? recordCache[selectedRecordId] ?? null : null,
      };
    }
  }
}

function syncUrl(
  selectedRecordId: string | null,
  selectedInspectorTab: InspectorTab,
  searchQuery: string,
  sourceFilter: SourceFilter,
  modelFilter: string | null,
  inputKindFilter: InputKindFilter,
  ratingFilter: RatingFilter,
): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (selectedRecordId) url.searchParams.set("record", selectedRecordId);
  else url.searchParams.delete("record");
  if (selectedInspectorTab === "inspect") url.searchParams.delete("tab");
  else url.searchParams.set("tab", selectedInspectorTab);
  if (searchQuery) url.searchParams.set("q", searchQuery);
  else url.searchParams.delete("q");
  if (sourceFilter) url.searchParams.set("source", sourceFilter);
  else url.searchParams.delete("source");
  if (modelFilter) url.searchParams.set("model", modelFilter);
  else url.searchParams.delete("model");
  if (inputKindFilter) url.searchParams.set("input", inputKindFilter);
  else url.searchParams.delete("input");
  if (ratingFilter != null) url.searchParams.set("rating", String(ratingFilter));
  else url.searchParams.delete("rating");
  window.history.replaceState(null, "", url);
}

/**
 * The selected record needs its full summary when it is not cached yet, or when the cached list row
 * is a lazy `not_validated` placeholder: only the full check lets the inspector offer downloads.
 */
export function needsSelectedRecordSummary(
  state: Pick<ViewerState, "selectedRecordId" | "recordCache">,
): boolean {
  if (!state.selectedRecordId) return false;
  const cached = state.recordCache[state.selectedRecordId];
  return !cached || cached.materialization_status === "not_validated";
}

export function ViewerProvider({
  children,
  appMode = readAppMode(),
}: {
  children: ReactNode;
  appMode?: AppMode;
}): JSX.Element {
  const [state, dispatch] = useReducer(viewerReducer, initialState);
  // The Generation bootstrap (`/api/bootstrap`) scans every canonical Generation record; the
  // Reconstruction Viewer never renders it, so fetching it in Recon mode only adds concurrent
  // disk load to the unrelated Reconstruction record list fetch.
  const bootstrapQuery = useQuery({ ...bootstrapQueryOptions(), enabled: appMode !== "reconstruction" });
  const selectedRecordQuery = useQuery({
    ...recordSummaryQueryOptions(state.selectedRecordId ?? ""),
    enabled: needsSelectedRecordSummary(state),
  });
  const reconciledRecordId = useMemo(() => {
    if (!state.bootstrap) return state.selectedRecordId;
    return reconcileWorkbenchSelection(
      state.bootstrap.workbench_entries,
      {
        searchQuery: state.searchQuery,
        sourceFilter: state.sourceFilter,
        modelFilter: state.modelFilter,
        inputKindFilter: state.inputKindFilter,
        ratingFilter: state.ratingFilter,
      },
      state.selectedRecordId,
    );
  }, [
    state.bootstrap,
    state.inputKindFilter,
    state.modelFilter,
    state.ratingFilter,
    state.searchQuery,
    state.selectedRecordId,
    state.sourceFilter,
  ]);

  useEffect(() => {
    if (bootstrapQuery.data) dispatch({ type: "SET_BOOTSTRAP", payload: bootstrapQuery.data });
  }, [bootstrapQuery.data]);

  useEffect(() => {
    if (bootstrapQuery.error) {
      dispatch({
        type: "SET_ERROR",
        payload: bootstrapQuery.error instanceof Error ? bootstrapQuery.error.message : "Failed to load viewer",
      });
    }
  }, [bootstrapQuery.error]);

  useEffect(() => {
    if (selectedRecordQuery.data) {
      dispatch({ type: "SET_SELECTED_RECORD_SUMMARY", payload: selectedRecordQuery.data });
    }
  }, [selectedRecordQuery.data]);

  useEffect(() => {
    if (!state.bootstrap || state.selectedRecordId === reconciledRecordId) return;
    dispatch({ type: "SELECT_RECORD", payload: reconciledRecordId });
  }, [reconciledRecordId, state.bootstrap, state.selectedRecordId]);

  useEffect(() => {
    syncUrl(
      state.selectedRecordId,
      state.selectedInspectorTab,
      state.searchQuery,
      state.sourceFilter,
      state.modelFilter,
      state.inputKindFilter,
      state.ratingFilter,
    );
  }, [
    state.inputKindFilter,
    state.modelFilter,
    state.ratingFilter,
    state.searchQuery,
    state.selectedInspectorTab,
    state.selectedRecordId,
    state.sourceFilter,
  ]);

  const value = useMemo(() => state, [state]);
  return (
    <ViewerContext.Provider value={value}>
      <ViewerDispatchContext.Provider value={dispatch}>
        {children}
      </ViewerDispatchContext.Provider>
    </ViewerContext.Provider>
  );
}

export function useViewer(): ViewerState {
  const context = useContext(ViewerContext);
  if (!context) throw new Error("useViewer must be used within ViewerProvider");
  return context;
}

export function useViewerDispatch(): Dispatch<ViewerAction> {
  const context = useContext(ViewerDispatchContext);
  if (!context) throw new Error("useViewerDispatch must be used within ViewerProvider");
  return context;
}
