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

from collections.abc import Iterator

import pytest

from sdk._core import assets


@pytest.fixture(autouse=True)
def isolated_sdk_asset_session() -> Iterator[None]:
    """Start every test without an ambient SDK asset session.

    The SDK lazily creates a process-wide ephemeral asset session the first time model code
    needs one (for example when a test executes a documentation snippet). Without this reset,
    that session leaks into later tests and changes where their relative mesh paths resolve,
    so a test's result would depend on which tests ran before it.
    """

    token = assets._CURRENT_ASSET_SESSION.set(None)
    try:
        yield
    finally:
        assets._CURRENT_ASSET_SESSION.reset(token)
