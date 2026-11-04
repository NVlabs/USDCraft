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

"""Pin complete current task content and preserve public tool contracts on both arms."""

import hashlib
import json
import os
from pathlib import Path

from tests.generation.test_prompt import CHECK, COMPILE, INVENTORY, _image, _repo, _workspace
from tests.reconstruction.test_shared_harness import render_case
from usdcraft.generation.contract import GenerationRequest
from usdcraft.generation.prompt import build_generation_prompt
from usdcraft.generation.provider_tools import build_generation_tool_schemas
from usdcraft.reconstruction.provider_tools import build_reconstruction_tool_schemas


def test_initial_context_matches_golden(tmp_path: Path) -> None:
    result = {}

    def save(key, text, paths):
        for p, repl in sorted(paths.items(), key=lambda x: -len(x[0])):
            text = text.replace(p, repl)
        result[key] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "bytes": len(text.encode()),
            "text": text,
        }

    root = tmp_path
    for arm in ["provider_tools", "shell"]:
        for i, image_observation in enumerate((False, True)):
            p = root / f"{arm}-{i}"
            p.mkdir()
            prompt, paths, preflight, ws = render_case(p, image_observation, arm)
            save(f"recon-{arm}-{i}", prompt, paths)
            result[f"tools-recon-{i}"] = build_reconstruction_tool_schemas()
        for image in [False, True]:
            p = root / f"gen-{arm}-{image}"
            p.mkdir()
            repo = _repo(p)
            req = GenerationRequest(
                prompt="a cabinet with three drawers",
                repo_root=repo,
                image_path=_image(p) if image else None,
            )
            ws = _workspace(p, req, "fixed")
            prompt = build_generation_prompt(
                req,
                ws,
                check_command="check_model" if arm == "provider_tools" else CHECK,
                compile_command="compile_model" if arm == "provider_tools" else COMPILE,
                reference_access=arm,
                image_observation_text=INVENTORY if image else None,
            )
            save(
                f"gen-{arm}-{image}",
                prompt,
                {str(ws.root): "<workspace>", str(repo.resolve()): "<repo>"},
            )
    result["tools-gen"] = build_generation_tool_schemas()

    golden = Path(__file__).parents[1] / "golden/prompts/pipeline_context_baseline.json"
    expected = json.loads(golden.read_text())
    actual = {
        k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
        for k, v in result.items()
    }
    if os.environ.get("USDCRAFT_UPDATE_GOLDENS") == "1":
        # A text update never silently accepts a change to public tool schemas.
        assert {k: v for k, v in actual.items() if k.startswith("tools-")} == {
            k: v for k, v in expected.items() if k.startswith("tools-")
        }
        golden.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    else:
        assert actual == expected
