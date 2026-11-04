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

import { describe, expect, it } from "vitest";

import {
  batchStats,
  describeBatchStats,
  groupByBatch,
  parseBatchId,
  visibleRecords,
} from "./batch-groups";

type Row = { id: string; batch: string | null; rating: number | null; recall: number | null };

const rows: Row[] = [
  { id: "a", batch: null, rating: 4, recall: null },
  { id: "b", batch: "round2_20260905T140000Z", rating: null, recall: 0.9 },
  { id: "c", batch: "solo_20260905T150000Z", rating: 2, recall: 0.5 },
  { id: "d", batch: "round2_20260905T140000Z", rating: 5, recall: 0.7 },
  { id: "e", batch: null, rating: null, recall: null },
];

describe("batch grouping", () => {
  it("folds records sharing a batch id at the first occurrence and keeps singletons flat", () => {
    const items = groupByBatch(rows, (row) => row.batch);
    expect(items.map((item) => (item.kind === "group" ? `G:${item.batchId}` : item.record.id))).toEqual([
      "a",
      "G:round2_20260905T140000Z",
      "c",
      "e",
    ]);
    const group = items[1];
    expect(group.kind === "group" && group.records.map((row) => row.id)).toEqual(["b", "d"]);
  });

  it("flattens collapsed groups out of the visible order", () => {
    const items = groupByBatch(rows, (row) => row.batch);
    expect(visibleRecords(items, () => false).map((row) => row.id)).toEqual(["a", "c", "e"]);
    expect(visibleRecords(items, () => true).map((row) => row.id)).toEqual(["a", "b", "d", "c", "e"]);
  });

  it("parses the csv-stem plus UTC stamp batch id", () => {
    const label = parseBatchId("reconstruction_batch_round2_20260905T140000Z");
    expect(label.name).toBe("reconstruction_batch_round2");
    expect(label.startedAt?.toISOString()).toBe("2026-09-05T14:00:00.000Z");
    expect(parseBatchId("custom-id")).toEqual({ name: "custom-id", startedAt: null });
  });

  it("summarizes a group with counts, mean recall and ratings", () => {
    const group = rows.filter((row) => row.batch === "round2_20260905T140000Z");
    const stats = batchStats(group, (row) => row.rating, (row) => row.recall);
    expect(stats).toEqual({ count: 2, rated: 1, meanRating: 5, meanRecall: 0.8 });
    expect(describeBatchStats(stats)).toBe("2 records · recall avg 0.80 · 1 rated (avg 5.0)");
    expect(describeBatchStats(batchStats([rows[4]], (row) => row.rating))).toBe("1 record · unrated");
  });
});
