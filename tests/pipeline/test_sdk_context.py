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

"""The compact entry reference is usable without extra prerequisite document reads."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from sdk._profiles import get_sdk_profile
from usdcraft.pipeline.provider import ProviderToolResult
from usdcraft.pipeline.sdk_context import PRELOADED_SDK_DOC_PATHS, render_sdk_document_inventory
from usdcraft.pipeline.tools import ProviderToolExecutor

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_paths_work_for_provider_and_shell(tmp_path: Path) -> None:
    inventory = render_sdk_document_inventory(ROOT, mark_preloaded=True)
    paths = re.findall(r"^- `([^`]+)`", inventory, re.MULTILINE)
    assert paths == [p.as_posix() for p in get_sdk_profile("sdk").docs_full]
    assert inventory.count(str(ROOT)) == 1
    marked = [line.split("`")[1] for line in inventory.splitlines() if "(preloaded above)" in line]
    assert marked == list(PRELOADED_SDK_DOC_PATHS)
    workspace = SimpleNamespace(
        repo_root=ROOT,
        root=tmp_path,
        model_path=tmp_path / "model.py",
        build_dir=tmp_path / "build",
    )
    executor = ProviderToolExecutor(
        workspace,
        finish_tool_name="finish_generation",
        check_runner=lambda _: ProviderToolResult.ok("unused"),
        compile_runner=lambda *args, **kwargs: ProviderToolResult.ok("unused"),
    )
    for path in paths:
        reply = executor.execute("read_file", {"path": path, "limit": 1}).payload
        assert reply["ok"], reply
        # Shell Authors join the declared root and relative path even when cwd is the workspace.
        assert reply["output"]["content"] == "L1: " + (ROOT / path).read_text().splitlines()[0]
