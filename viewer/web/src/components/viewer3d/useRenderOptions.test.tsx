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

import { renderToStaticMarkup } from 'react-dom/server';
import type { JSX } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useRenderOptions } from './useRenderOptions';

function RenderOptionsProbe(): JSX.Element {
  const { options } = useRenderOptions();
  return (
    <span>
      {String(options.showJointOverlay)}:{String(options.fancyGraphics)}:
      {String(options.showFloatingBodies)}
    </span>
  );
}

function renderOptionsFromSearch(search: string): string {
  vi.stubGlobal('localStorage', {
    getItem: () => null,
    setItem: () => undefined,
  });
  vi.stubGlobal('window', { location: { search } });
  return renderToStaticMarkup(<RenderOptionsProbe />);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useRenderOptions query compatibility', () => {
  it.each([
    ['?render=0000001', 'true:false:false'],
    ['?render=00000011', 'true:false:false'],
    ['?render=000000111', 'true:true:false'],
  ])('defaults trailing fields missing from short query %s', (search, expected) => {
    expect(renderOptionsFromSearch(search)).toContain(expected);
  });

  it('reads floating bodies from the new trailing bit', () => {
    expect(renderOptionsFromSearch('?render=0000001111')).toContain('true:true:true');
  });
});
