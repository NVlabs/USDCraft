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

/**
 * Batch grouping for the record lists. Records that share a `batch_id` (one CSV run of
 * `generation-batch` / `reconstruction-batch`) fold into one collapsible group anchored where the
 * batch's first record appears in the current sort order; everything else stays a flat row.
 */

export type BatchGroup<T> = { kind: "group"; batchId: string; records: T[] };
export type BatchListItem<T> = { kind: "record"; record: T } | BatchGroup<T>;

/** Batches smaller than this stay flat: a group of one is just noise. */
export const BATCH_GROUP_MIN_SIZE = 2;

export function groupByBatch<T>(
  records: readonly T[],
  batchIdOf: (record: T) => string | null | undefined,
  minSize: number = BATCH_GROUP_MIN_SIZE,
): BatchListItem<T>[] {
  const groups = new Map<string, BatchGroup<T>>();
  const items: BatchListItem<T>[] = [];
  for (const record of records) {
    const batchId = batchIdOf(record);
    if (!batchId) {
      items.push({ kind: "record", record });
      continue;
    }
    const existing = groups.get(batchId);
    if (existing) {
      existing.records.push(record);
      continue;
    }
    const group: BatchGroup<T> = { kind: "group", batchId, records: [record] };
    groups.set(batchId, group);
    items.push(group);
  }
  return items.flatMap((item) =>
    item.kind === "group" && item.records.length < minSize
      ? item.records.map((record) => ({ kind: "record" as const, record }))
      : [item],
  );
}

/** Flatten grouped items back into the visible record order (collapsed groups contribute none). */
export function visibleRecords<T>(
  items: readonly BatchListItem<T>[],
  isExpanded: (group: BatchGroup<T>) => boolean,
): T[] {
  return items.flatMap((item) =>
    item.kind === "record" ? [item.record] : isExpanded(item) ? item.records : [],
  );
}

const BATCH_ID_STAMP = /^(.*)_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/;

export type BatchLabel = { name: string; startedAt: Date | null };

/** `round2_20260905T140000Z` → name `round2`, started 2026-09-05 14:00:00 UTC. */
export function parseBatchId(batchId: string): BatchLabel {
  const match = BATCH_ID_STAMP.exec(batchId);
  if (!match) return { name: batchId, startedAt: null };
  const [, name, year, month, day, hour, minute, second] = match;
  const startedAt = new Date(
    Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
    ),
  );
  return { name, startedAt: Number.isNaN(startedAt.getTime()) ? null : startedAt };
}

const STARTED_FORMATTER = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export function formatBatchStart(label: BatchLabel): string | null {
  return label.startedAt ? STARTED_FORMATTER.format(label.startedAt) : null;
}

export type BatchStats = {
  count: number;
  rated: number;
  meanRating: number | null;
  meanRecall: number | null;
};

export function batchStats<T>(
  records: readonly T[],
  ratingOf: (record: T) => number | null | undefined,
  recallOf: (record: T) => number | null | undefined = () => null,
): BatchStats {
  const ratings = records.map(ratingOf).filter((value): value is number => value != null);
  const recalls = records.map(recallOf).filter((value): value is number => value != null);
  const mean = (values: number[]): number | null =>
    values.length > 0 ? values.reduce((total, value) => total + value, 0) / values.length : null;
  return {
    count: records.length,
    rated: ratings.length,
    meanRating: mean(ratings),
    meanRecall: mean(recalls),
  };
}

/** One-line group summary: `12 records · recall avg 0.81 · 3 rated (avg 4.0)`. */
export function describeBatchStats(stats: BatchStats): string {
  return [
    `${stats.count} record${stats.count === 1 ? "" : "s"}`,
    stats.meanRecall != null ? `recall avg ${stats.meanRecall.toFixed(2)}` : null,
    stats.rated > 0
      ? `${stats.rated} rated${stats.meanRating != null ? ` (avg ${stats.meanRating.toFixed(1)})` : ""}`
      : "unrated",
  ]
    .filter((item): item is string => Boolean(item))
    .join(" · ");
}
