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

"""Repository documentation routing, without pinning editorial wording."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_root_agent_entrypoint_routes_instead_of_copying_component_specs() -> None:
    agents_path = REPOSITORY_ROOT / "AGENTS.md"
    agents = agents_path.read_text(encoding="utf-8")
    # The root README is the user quickstart; the evidence contract is routed from the agent
    # entrypoint's knowledge-routing table, not from the user-facing page.

    assert agents_path.stat().st_size < 16 * 1024
    assert "usdcraft/reconstruction/README.md" in agents
    assert "usdcraft/reconstruction/EVIDENCE.md" in agents
    assert "usdcraft/generation/README.md" in agents
    assert "docs/architecture.md" in agents
    assert "storage/README.md" in agents
    assert "viewer/README.md" in agents
    assert "suspected_source_reconstruction_artifact" not in agents
    assert "MICRO_SLICE_BOUNDS_Z192" not in agents
    assert "PhysxJointAxisAPI" not in agents
