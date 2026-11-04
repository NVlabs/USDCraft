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

from contextlib import suppress

from .common import (
    AllowedOverlap,
    DisconnectedGeometryClassification,
    TestFailure,
    TestReport,
)
from .context import TestContext

__all__ = [
    "AllowedOverlap",
    "DisconnectedGeometryClassification",
    "TestContext",
    "TestFailure",
    "TestReport",
]

_COMPAT_MODULE = "sdk._core.testing"
for _name in __all__:
    _obj = globals().get(_name)
    if getattr(_obj, "__module__", "").startswith("sdk._core._testing"):
        with suppress(Exception):
            _obj.__module__ = _COMPAT_MODULE

del _COMPAT_MODULE, _name, _obj
