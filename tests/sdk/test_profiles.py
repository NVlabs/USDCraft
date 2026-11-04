# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

from pathlib import Path

from sdk._profiles import get_sdk_profile

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sdk_docs_profile_includes_base_and_cadquery_topics() -> None:
    base = get_sdk_profile("sdk")

    base_docs = {path.as_posix() for path in base.docs_full}

    assert "sdk/_docs/common/40_assets.md" in base_docs
    assert "sdk/_docs/base/41_panels_and_grilles.md" in base_docs
    assert "sdk/_docs/base/44_knobs_and_controls.md" in base_docs
    assert "sdk/_docs/base/48_wheels_and_tires.md" in base_docs
    assert "sdk/_docs/base/46_section_lofts.md" in base_docs
    assert "sdk/_docs/base/55_common_assemblies.md" in base_docs
    assert "sdk/_docs/cadquery/35_cadquery.md" in base_docs
    assert "sdk/_docs/cadquery/39d_cadquery_gears.md" in base_docs
    assert "sdk/_docs/base/47_shell_partition.md" not in base_docs
    assert "sdk/_docs/cadquery/39a_cadquery_examples.md" not in base_docs


def test_sdk_docs_profile_exactly_matches_repository_documents() -> None:
    profile = get_sdk_profile("sdk")
    actual = {
        path.relative_to(_REPO_ROOT) for path in (_REPO_ROOT / "sdk" / "_docs").glob("*/*.md")
    }

    assert len(profile.docs_full) == len(set(profile.docs_full))
    assert set(profile.docs_full) == actual

    for path in profile.docs_full:
        assert not path.is_absolute()
        assert ".." not in path.parts
        resolved = (_REPO_ROOT / path).resolve()
        assert resolved.is_relative_to(_REPO_ROOT)
        assert resolved.is_file()
