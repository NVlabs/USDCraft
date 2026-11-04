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

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  deleteRecord,
  fetchReconstructionRecordTextFile,
  saveRecordRating,
} from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("record curation API", () => {
  it("sends an exact record-id confirmation in the delete body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: "deleted",
        record_id: "rec_one",
        materialization_deleted: true,
        cleanup_pending: false,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await deleteRecord("rec_one");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/records/rec_one",
      expect.objectContaining({
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_record_id: "rec_one" }),
      }),
    );
  });

  it("can clear a rating with null", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ record_id: "rec_one", rating: null }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await saveRecordRating("rec_one", null);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/records/rec_one/rating",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ rating: null }),
      }),
    );
  });

  it("uses an object-shaped API detail message instead of stringifying the object", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            detail: {
              message: "Delete the descendant records first",
              descendant_record_ids: ["rec_child"],
            },
          },
          409,
        ),
      ),
    );

    await expect(deleteRecord("rec_parent")).rejects.toThrow(
      "Delete the descendant records first",
    );
  });
});

describe("Reconstruction record text API", () => {
  it("keeps Code and Metadata reads on the Reconstruction endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        record_id: "rec_reconstruction_001",
        file_path: "model.py",
        content: "object_model = build()\n",
        truncated: false,
        byte_count: 23,
        preview_byte_limit: 131072,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchReconstructionRecordTextFile(
      "rec_reconstruction_001",
      "model.py",
      { previewBytes: 131072 },
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/reconstruction-records/rec_reconstruction_001/text/model.py?preview_bytes=131072",
      { signal: undefined },
    );
    expect(fetchMock.mock.calls[0]?.[0]).not.toContain("/api/records/");
  });
});
