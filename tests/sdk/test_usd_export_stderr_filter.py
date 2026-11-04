# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re

import pytest

from sdk._core._usd_export import _USDZ_PACKAGING_NOISE, _filtered_native_stderr


def test_usdz_packaging_noise_pattern_matches_the_openusd_relocation_warning() -> None:
    line = (
        "Warning: in _EnqueueDependency at line 108 of .../assetLocalization.cpp -- Failed to "
        "resolve reference @0/decal_f087814853c14212.png@ with computed asset path "
        "@0/decal_f087814853c14212.png@ found in layer @/x/build/isaac/model.usdc@."
    )
    assert _USDZ_PACKAGING_NOISE.search(line)
    assert not _USDZ_PACKAGING_NOISE.search("Warning: something else about textures/decal.png")


def test_filtered_native_stderr_drops_only_matching_native_lines(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with _filtered_native_stderr(re.compile(r"NOISE")):
        os.write(2, b"NOISE line one\nkeep this line\nNOISE line two\n")
    captured = capfd.readouterr()
    assert captured.err == "keep this line\n"


def test_filtered_native_stderr_restores_stderr_after_an_exception(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError):
        with _filtered_native_stderr(re.compile(r"NOISE")):
            os.write(2, b"NOISE\nsurvivor\n")
            raise RuntimeError("boom")
    os.write(2, b"after\n")
    assert capfd.readouterr().err == "survivor\nafter\n"
