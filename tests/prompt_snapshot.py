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

"""Golden snapshots of rendered Author tasks.

A task text is compared with a checked-in snapshot after normalizing the run-specific parts
(repository root, workspace paths, run ids) and dropping the two sections that are derived from
other files and tested on their own (the preloaded SDK documents and the SDK document inventory).
Set ``USDCRAFT_UPDATE_GOLDENS=1`` to rewrite the snapshots after a deliberate task change; the
diff of the golden file is then the review artifact.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "prompts"
_UPDATE = os.environ.get("USDCRAFT_UPDATE_GOLDENS", "").strip() not in {"", "0", "false"}


def normalize_task(text: str, *, paths: dict[str, str]) -> str:
    """Replace run-specific paths with placeholders and drop file-derived sections."""

    for value, placeholder in sorted(paths.items(), key=lambda item: -len(item[0])):
        if value:
            text = text.replace(value, placeholder)
    # PRELOADED SDK REFERENCES ... up to the next section header the arms use after it.
    text = re.sub(
        r"PRELOADED SDK REFERENCES\n.*?(?=\n(?:SDK REFERENCE DISCOVERY|PUBLIC SDK REFERENCES)\n)",
        "PRELOADED SDK REFERENCES\n<preloaded documents: tested in test_prompt / test_authoring_policy>\n",
        text,
        flags=re.DOTALL,
    )
    # The SDK document inventory (Generation marks it; Reconstruction lists it after its header).
    text = re.sub(
        r"<!-- sdk-doc-inventory:start -->\n.*?<!-- sdk-doc-inventory:end -->",
        "<!-- sdk-doc-inventory: tested in test_prompt -->",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(The following is the complete canonical SDK documentation inventory for this run:\n)"
        r"(?:- `[^\n]*`\n)+",
        r"\1<sdk document inventory: tested in test_profiles>\n",
        text,
    )
    return text


def assert_matches_golden(name: str, text: str, *, paths: dict[str, str]) -> None:
    """Compare ``text`` with ``tests/golden/prompts/<name>.md`` (or rewrite it on request)."""

    normalized = normalize_task(text, paths=paths)
    golden = GOLDEN_DIR / f"{name}.md"
    if _UPDATE:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(normalized, encoding="utf-8")
        return
    assert golden.is_file(), (
        f"missing golden snapshot {golden}; run with USDCRAFT_UPDATE_GOLDENS=1 to create it"
    )
    expected = golden.read_text(encoding="utf-8")
    if normalized != expected:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                normalized.splitlines(),
                fromfile=str(golden.relative_to(GOLDEN_DIR.parents[1])),
                tofile="rendered",
                lineterm="",
                n=2,
            )
        )
        raise AssertionError(
            f"task text differs from {golden.name}; review the diff and, if the change is "
            f"intended, rerun with USDCRAFT_UPDATE_GOLDENS=1:\n{diff}"
        )


__all__ = ["GOLDEN_DIR", "assert_matches_golden", "normalize_task"]
