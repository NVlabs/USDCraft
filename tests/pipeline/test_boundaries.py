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

"""Keep shared execution independent of either product implementation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("directory", "forbidden"),
    [
        ("pipeline", ("usdcraft.generation", "usdcraft.reconstruction")),
        ("render", ("usdcraft.generation", "usdcraft.reconstruction")),
        ("generation", ("usdcraft.reconstruction",)),
        ("reconstruction", ("usdcraft.generation",)),
    ],
)
def test_dependency_direction(directory: str, forbidden: tuple[str, ...]) -> None:
    violations = []
    for path in (ROOT / "usdcraft" / directory).rglob("*.py"):
        package = ".".join(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = importlib.util.resolve_name("." * node.level + name, package)
                names = [name, *(name + "." + alias.name for alias in node.names)]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)
