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

import { useQueryClient } from "@tanstack/react-query";
import { Image, Loader2, Star, Trash2, Type, X } from "lucide-react";
import { useEffect, useId, useRef, useState, type JSX } from "react";

import { deleteRecord, saveRecordRating } from "@/lib/api";
import { useViewerReadOnly } from "@/lib/use-viewer-read-only";
import type { RecordSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useViewerDispatch } from "@/lib/viewer-context";
import { viewerQueryKeys } from "@/lib/viewer-queries";

type RecordListItemProps = {
  recordId: string;
  record: RecordSummary | null;
  isSelected: boolean;
  nextRecordIdOnDelete: string | null;
  onSelect: (recordId: string) => void;
};

const STAR_VALUES = [1, 2, 3, 4, 5] as const;
type RowMutation = "rating" | "delete";

export function recordListPrimaryText(record: RecordSummary | null, recordId: string): string {
  return record?.title?.trim() || record?.prompt_preview?.trim() || recordId;
}

export function recordListMetadata(record: RecordSummary | null): { model: string | null } {
  return { model: record?.model_id?.trim() || null };
}

export function recordRatingLabel(rating: number | null | undefined): string {
  return rating == null ? "Unrated" : `${rating} / 5`;
}

export function RecordListItem({
  recordId,
  record,
  isSelected,
  nextRecordIdOnDelete,
  onSelect,
}: RecordListItemProps): JSX.Element {
  const dispatch = useViewerDispatch();
  const queryClient = useQueryClient();
  const readOnly = useViewerReadOnly();
  const [mutationPending, setMutationPending] = useState<RowMutation | null>(null);
  const [ratingError, setRatingError] = useState<string | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const mutationRef = useRef<RowMutation | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteCancelRef = useRef<HTMLButtonElement | null>(null);
  const deleteDialogId = useId();
  const deleteTitleId = `${deleteDialogId}-title`;
  const deleteDescriptionId = `${deleteDialogId}-description`;
  const metadata = recordListMetadata(record);
  const InputIcon = record?.input_kind === "text_image" ? Image : Type;
  const ratingPending = mutationPending === "rating";
  const deletePending = mutationPending === "delete";

  useEffect(() => {
    if (deleteConfirmOpen) deleteCancelRef.current?.focus();
  }, [deleteConfirmOpen]);

  const closeDeleteConfirm = (): void => {
    if (deletePending) return;
    setDeleteConfirmOpen(false);
    window.requestAnimationFrame(() => deleteTriggerRef.current?.focus());
  };

  const handleRating = async (rating: (typeof STAR_VALUES)[number]): Promise<void> => {
    if (!record || deleteConfirmOpen || mutationRef.current) return;
    const nextRating = record.rating === rating ? null : rating;
    mutationRef.current = "rating";
    setMutationPending("rating");
    setRatingError(null);
    try {
      const updated = await saveRecordRating(recordId, nextRating);
      dispatch({ type: "UPSERT_RECORDS", payload: [updated] });
      queryClient.setQueryData(viewerQueryKeys.recordSummary(recordId), updated);
      void queryClient.invalidateQueries({ queryKey: viewerQueryKeys.bootstrap() });
    } catch (error) {
      setRatingError(error instanceof Error ? error.message : "Failed to save rating");
    } finally {
      if (mutationRef.current === "rating") {
        mutationRef.current = null;
        setMutationPending(null);
      }
    }
  };

  const handleDelete = async (): Promise<void> => {
    if (mutationRef.current) return;
    mutationRef.current = "delete";
    setMutationPending("delete");
    setDeleteError(null);
    try {
      const result = await deleteRecord(recordId);
      dispatch({
        type: "DELETE_RECORD_LOCAL",
        payload: { recordId, nextRecordId: nextRecordIdOnDelete },
      });
      if (result.cleanup_pending) {
        dispatch({
          type: "SET_WORKBENCH_NOTICE",
          payload: `Record ${recordId} was removed from Workbench, but final cleanup of its staged files is still pending. Some files may remain on disk until cleanup succeeds.`,
        });
      }
      queryClient.removeQueries({ queryKey: viewerQueryKeys.recordSummary(recordId) });
      void queryClient.invalidateQueries({ queryKey: viewerQueryKeys.bootstrap() });
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : "Failed to delete record");
      mutationRef.current = null;
      setMutationPending(null);
    }
  };

  return (
    <div
      className={cn(
        "group border-b border-l-2 border-b-[var(--border-subtle)] px-3 py-3 transition-colors",
        isSelected
          ? "border-l-[var(--accent)] bg-[var(--accent-soft)]"
          : "border-l-transparent hover:bg-[var(--surface-1)]",
      )}
      data-record-list-item={recordId}
    >
      <div className="flex items-start gap-1">
        <button
          type="button"
          onClick={() => onSelect(recordId)}
          className="flex min-w-0 flex-1 items-start gap-2 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]"
          aria-current={isSelected ? "true" : undefined}
        >
          <InputIcon
            className={cn(
              "mt-0.5 size-3 shrink-0 text-[var(--text-quaternary)]",
              isSelected && "text-[var(--accent)]",
            )}
          />
          <div className="min-w-0 flex-1">
            <p className="line-clamp-2 text-[11.5px] font-medium leading-[1.5] tracking-[-0.01em] text-[var(--text-primary)]">
              {recordListPrimaryText(record, recordId)}
            </p>
            {metadata.model ? (
              <p className="mt-1.5 truncate text-[9.5px] leading-[1.5] text-[var(--text-tertiary)]">
                {metadata.model}
              </p>
            ) : null}
          </div>
        </button>

        {readOnly ? null : (
        <button
          ref={deleteTriggerRef}
          type="button"
          onClick={() => {
            if (mutationRef.current) return;
            setDeleteError(null);
            setDeleteConfirmOpen(true);
          }}
          disabled={mutationPending !== null}
          className={cn(
            "flex size-6 shrink-0 items-center justify-center rounded text-[var(--text-quaternary)] outline-none transition-colors hover:bg-[rgba(209,52,21,0.08)] hover:text-[var(--destructive)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)]",
            isSelected && "text-[var(--text-tertiary)]",
          )}
          aria-label={`Delete record ${recordId}`}
          aria-expanded={deleteConfirmOpen}
          aria-controls={deleteConfirmOpen ? deleteDialogId : undefined}
          title="Delete record"
        >
          <Trash2 className="size-3" />
        </button>
        )}
      </div>

      {record && !deleteConfirmOpen ? (
        <div className="mt-2 ml-5 flex min-w-0 items-center justify-between gap-2">
          <span className="font-mono text-[8px] uppercase tracking-[0.06em] text-[var(--text-quaternary)]">
            {readOnly ? "Rating" : "Your rating"}
          </span>
          <div className="flex items-center" aria-label={`Rating: ${recordRatingLabel(record.rating)}`}>
            {STAR_VALUES.map((starValue) => {
              const active = record.rating != null && starValue <= record.rating;
              const clearsRating = record.rating === starValue;
              return (
                <button
                  key={starValue}
                  type="button"
                  disabled={mutationPending !== null || readOnly}
                  onClick={() => void handleRating(starValue)}
                  className={cn(
                    "rounded-sm p-0.5 text-[var(--text-quaternary)] outline-none transition-colors hover:text-[var(--accent-strong)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)] disabled:opacity-60",
                    readOnly ? "disabled:cursor-default" : "disabled:cursor-wait",
                  )}
                  title={readOnly ? "Read-only viewer" : undefined}
                  aria-label={clearsRating ? "Clear rating" : `Rate ${starValue} of 5`}
                  aria-pressed={record.rating === starValue}
                >
                  <Star
                    className={cn(
                      "size-3",
                      active && "fill-[var(--accent)] text-[var(--accent)]",
                    )}
                  />
                </button>
              );
            })}
          </div>
          {ratingPending ? (
            <span
              className="text-[9px] text-[var(--text-quaternary)]"
              role="status"
              aria-live="polite"
            >
              Saving…
            </span>
          ) : null}
        </div>
      ) : null}

      {ratingError ? (
        <p
          className="mt-1 ml-5 text-[9.5px] leading-4 text-[var(--destructive)]"
          role="alert"
        >
          {ratingError}
        </p>
      ) : null}

      {deleteConfirmOpen ? (
        <div
          id={deleteDialogId}
          className="mt-2 rounded-md border border-[rgba(209,52,21,0.24)] bg-[rgba(209,52,21,0.05)] p-2"
          role="alertdialog"
          aria-labelledby={deleteTitleId}
          aria-describedby={deleteDescriptionId}
          aria-busy={deletePending}
          onKeyDown={(event) => {
            if (event.key === "Escape" && !deletePending) {
              event.preventDefault();
              closeDeleteConfirm();
            }
          }}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <p
                id={deleteTitleId}
                className="text-[10.5px] font-medium text-[var(--destructive)]"
              >
                Permanently delete this record?
              </p>
              <p className="mt-1 break-all font-mono text-[9px] leading-3.5 text-[var(--text-tertiary)]">
                {recordId}
              </p>
              <p
                id={deleteDescriptionId}
                className="mt-1 text-[9.5px] leading-4 text-[var(--text-quaternary)]"
              >
                The record and its materialized files will be removed.
              </p>
            </div>
            <button
              type="button"
              onClick={closeDeleteConfirm}
              disabled={deletePending}
              className="rounded p-0.5 text-[var(--text-quaternary)] hover:text-[var(--text-secondary)] disabled:opacity-50"
              aria-label="Cancel record deletion"
            >
              <X className="size-3" />
            </button>
          </div>
          {deleteError ? (
            <p className="mt-1 text-[9.5px] leading-4 text-[var(--destructive)]" role="alert">
              {deleteError}
            </p>
          ) : null}
          <div className="mt-2 flex justify-end gap-1.5">
            <button
              ref={deleteCancelRef}
              type="button"
              onClick={closeDeleteConfirm}
              disabled={deletePending}
              className="h-6 rounded border border-[var(--border-subtle)] px-2 text-[9.5px] text-[var(--text-secondary)] hover:bg-[var(--surface-2)] disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void handleDelete()}
              disabled={deletePending}
              className="flex h-6 items-center gap-1 rounded bg-[var(--destructive)] px-2 text-[9.5px] font-medium text-white hover:opacity-90 disabled:cursor-wait disabled:opacity-60"
            >
              {deletePending ? <Loader2 className="size-3 animate-spin" /> : <Trash2 className="size-3" />}
              <span role="status" aria-live="polite">
                {deletePending ? "Deleting…" : "Delete record"}
              </span>
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
