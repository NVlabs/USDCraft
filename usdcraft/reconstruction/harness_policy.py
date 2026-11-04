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

"""One model-independent Reconstruction harness with explicit capability switches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The Provider arm's default Author turn budget.
DEFAULT_MAX_TURNS = 80


@dataclass(frozen=True, slots=True)
class ReconstructionHarnessPolicy:
    image_observation: bool = True
    # The Provider Author turn budget. None on the Codex arm, which has no turn limit; the recorded
    # harness then carries no ``max_turns`` at all.
    max_turns: int | None = DEFAULT_MAX_TURNS

    def to_dict(self) -> dict:
        payload: dict[str, object] = {"image_observation": self.image_observation}
        if self.max_turns is not None:
            payload["max_turns"] = self.max_turns
        return payload


def resolve_harness_policy(
    *,
    image_observation: bool = True,
    max_turns: int | None = None,
    turn_budget: bool = True,
) -> ReconstructionHarnessPolicy:
    """Validate the capability switches; ``turn_budget=False`` resolves a Codex-arm policy."""

    if not isinstance(image_observation, bool):
        raise ValueError("image_observation must be a boolean")
    if not turn_budget:
        if max_turns is not None:
            raise ValueError("max_turns is a Provider-only option; the Codex arm has no turn limit")
        return ReconstructionHarnessPolicy(image_observation, None)
    turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS
    if isinstance(turns, bool) or not isinstance(turns, int) or not 1 <= turns <= 200:
        raise ValueError("max_turns must be an integer between 1 and 200")
    return ReconstructionHarnessPolicy(image_observation, turns)


def policy_for_workspace(root: Path) -> ReconstructionHarnessPolicy:
    path = root / "harness_policy.json"
    if not path.is_file():
        return ReconstructionHarnessPolicy()
    recorded = json.loads(path.read_text())
    # A Codex-arm workspace records no turn budget.
    return resolve_harness_policy(**recorded, turn_budget="max_turns" in recorded)
