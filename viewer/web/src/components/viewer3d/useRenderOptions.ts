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

import { useState, useCallback, useEffect, useRef } from 'react';

import { updateUrlSearchParams } from '@/lib/url';

export interface RenderOptions {
  fancyGraphics: boolean;
  // Authored-PBR profile: suppresses the studio-shading material override so
  // polished metal and glass keep their authored parameters. Transient toggle.
  reviewProfile: boolean;
  showEdges: boolean;
  showGrid: boolean;
  showCollisions: boolean;
  showSegmentColors: boolean;
  doubleSided: boolean;
  autoAnimate: boolean;
  showJointOverlay: boolean;
  showFloatingBodies: boolean;
}

const STORAGE_KEY = 'usdcraft-render-options';
const STORAGE_SCHEMA_VERSION = 1;
const RENDER_QUERY_PARAM = 'render';
// Boolean options bit-packed into the `?render=` query param. Order is stable;
// new boolean flags are appended and an unused slot stays `null` (written as
// '0', ignored when read) so every share link keeps its bit positions.
type BooleanRenderOptionKey = Exclude<
  keyof RenderOptions,
  | 'reviewProfile'
>;
const RENDER_OPTION_KEYS: Array<BooleanRenderOptionKey | null> = [
  'showEdges',
  'showGrid',
  'showCollisions',
  'showSegmentColors',
  'doubleSided',
  'autoAnimate',
  'showJointOverlay',
  null, // unused slot
  'fancyGraphics',
  'showFloatingBodies',
];

// The shortest accepted render query has seven options. Because options are
// append-only, a shorter query decodes as a prefix with defaults for the rest.
const MIN_SUPPORTED_RENDER_QUERY_LENGTH = 7;

export const defaultRenderOptions: RenderOptions = {
  fancyGraphics: false,
  reviewProfile: false,
  showEdges: true,
  showGrid: true,
  showCollisions: false,
  showSegmentColors: false,
  doubleSided: true,
  autoAnimate: false,
  showJointOverlay: false,
  showFloatingBodies: false,
};

const DEFAULT_RENDER_QUERY_VALUE = serializeOptions(defaultRenderOptions);

export interface RenderOptionsState {
  options: RenderOptions;
  setOption: <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => void;
  resetOptions: () => void;
}

/**
 * Read persisted render options from localStorage, falling back to defaults.
 */
function loadOptions(): RenderOptions {
  let options = { ...defaultRenderOptions };

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<RenderOptions> & { schemaVersion?: number };
      if (parsed.schemaVersion === STORAGE_SCHEMA_VERSION) {
        options = { ...defaultRenderOptions, ...parsed };
      } else {
        options = {
          ...defaultRenderOptions,
          ...parsed,
          showSegmentColors: false,
          doubleSided: true,
          fancyGraphics: false,
        };
      }
    }
  } catch {
    // Ignore corrupt or inaccessible storage.
  }

  if (typeof window === 'undefined') {
    return options;
  }

  try {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = parseOptions(params.get(RENDER_QUERY_PARAM));
    if (fromUrl) {
      return fromUrl;
    }
  } catch {
    // Ignore malformed URLs.
  }

  return options;
}

function serializeOptions(options: RenderOptions): string {
  return RENDER_OPTION_KEYS.map((key) => (key && options[key] ? '1' : '0')).join('');
}

function parseOptions(value: string | null): RenderOptions | null {
  if (!value) {
    return null;
  }

  const nextOptions = { ...defaultRenderOptions };
  const keys =
    value.length >= MIN_SUPPORTED_RENDER_QUERY_LENGTH
    && value.length <= RENDER_OPTION_KEYS.length
      ? RENDER_OPTION_KEYS.slice(0, value.length)
      : null;
  if (!keys) {
    return null;
  }

  for (const [index, key] of keys.entries()) {
    const bit = value[index];
    if (bit !== '0' && bit !== '1') {
      return null;
    }
    if (key) {
      nextOptions[key] = bit === '1';
    }
  }

  return nextOptions;
}

/**
 * Hook for managing render option toggles.
 *
 * State is persisted to localStorage under the key 'usdcraft-render-options'.
 */
export function useRenderOptions(): RenderOptionsState {
  const [options, setOptions] = useState<RenderOptions>(loadOptions);
  const isInitialMount = useRef(true);

  // Persist to localStorage whenever options change (skip the initial mount).
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        ...options,
        schemaVersion: STORAGE_SCHEMA_VERSION,
      }));
    } catch {
      // Storage full or unavailable -- silently ignore.
    }
  }, [options]);

  useEffect(() => {
    const serialized = serializeOptions(options);
    updateUrlSearchParams((params) => {
      if (serialized === DEFAULT_RENDER_QUERY_VALUE) {
        params.delete(RENDER_QUERY_PARAM);
      } else {
        params.set(RENDER_QUERY_PARAM, serialized);
      }
    });
  }, [options]);

  const setOption = useCallback(
    <K extends keyof RenderOptions>(key: K, value: RenderOptions[K]) => {
      setOptions((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const resetOptions = useCallback(() => {
    setOptions({ ...defaultRenderOptions });
  }, []);

  return { options, setOption, resetOptions };
}
