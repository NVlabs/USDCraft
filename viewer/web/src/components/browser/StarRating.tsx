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

import { Star } from "lucide-react";
import type { JSX } from "react";

import { cn } from "@/lib/utils";

export const STAR_VALUES = [1, 2, 3, 4, 5] as const;
export type StarValue = (typeof STAR_VALUES)[number];

export function ratingLabel(rating: number | null | undefined): string {
  return rating == null ? "Unrated" : `${rating} / 5`;
}

/** Five-star rating control shared by the Generation and Reconstruction record lists. */
export function StarRating({
  rating,
  disabled = false,
  pending = false,
  onRate,
}: {
  rating: number | null;
  disabled?: boolean;
  pending?: boolean;
  onRate: (rating: StarValue) => void;
}): JSX.Element {
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <div className="flex items-center" aria-label={`Rating: ${ratingLabel(rating)}`}>
        {STAR_VALUES.map((starValue) => {
          const active = rating != null && starValue <= rating;
          const clearsRating = rating === starValue;
          return (
            <button
              key={starValue}
              type="button"
              disabled={disabled}
              onClick={() => onRate(starValue)}
              className="rounded-sm p-0.5 text-[var(--text-quaternary)] outline-none transition-colors hover:text-[var(--accent-strong)] focus-visible:ring-2 focus-visible:ring-[var(--accent-soft)] disabled:cursor-wait disabled:opacity-60"
              aria-label={clearsRating ? "Clear rating" : `Rate ${starValue} of 5`}
              aria-pressed={rating === starValue}
            >
              <Star
                className={cn(
                  "size-3.5",
                  active && "fill-[var(--accent)] text-[var(--accent)]",
                )}
              />
            </button>
          );
        })}
      </div>
      <span className="text-[9.5px] text-[var(--text-quaternary)]" role="status" aria-live="polite">
        {pending ? "Saving…" : ratingLabel(rating)}
      </span>
    </div>
  );
}
