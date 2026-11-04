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

import importlib
from typing import Any

_REPO_SYNC_COMMAND = "uv sync --group dev"


def _install_command_hint() -> str:
    return f"Run `{_REPO_SYNC_COMMAND}` from the repository root, then retry."


def _missing_dependency_message(*, feature: str, dependency: str) -> str:
    return (
        f"{feature} requires the `{dependency}` package, but it is not installed in the current "
        f"environment. {_install_command_hint()}"
    )


def require_cadquery(*, feature: str) -> Any:
    try:
        return importlib.import_module("cadquery")
    except Exception as exc:
        raise RuntimeError(
            _missing_dependency_message(feature=feature, dependency="cadquery")
        ) from exc
