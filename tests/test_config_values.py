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

import os

import pytest

from usdcraft.config import load_repo_env
from usdcraft.values import normalize_reasoning_level

_ENV_KEYS = (
    "USDCRAFT_COMPILE_TIMEOUT_SECONDS",
    "USDCRAFT_CODEX_BIN",
    "NVIDIA_API_KEY",
    "NVIDIA_MODEL",
    "USDCRAFT_RECONSTRUCTION_WORKSPACE",
    "USDCRAFT_GENERATION_WORKSPACE",
    "USDCRAFT_REPO_ROOT",
    "USDCRAFT_VIEWER_READ_ONLY",
    "USDCRAFT_AUTHOR_WORKSPACE",
    "USDCRAFT_MP_START_METHOD",
    "USDCRAFT_PARAMETRIC_MESH_CACHE",
    "OBJECT_GEOMETRY_OVERLAP_TOL",
    "URDF_GEOMETRY_OVERLAP_TOL",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        # Make pytest track cleanup even when the key was initially absent; load_repo_env writes
        # directly to os.environ after this setup.
        monkeypatch.setenv(key, "__pytest_absent_sentinel__")
        monkeypatch.delenv(key, raising=False)


def test_repo_env_loads_whitelisted_settings_without_provider_secrets(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "USDCRAFT_COMPILE_TIMEOUT_SECONDS=42\n"
        "NVIDIA_API_KEY=secret-that-must-not-enter-the-process\n"
        "NVIDIA_MODEL=some/model\n"
        "USDCRAFT_CODEX_BIN=${NVIDIA_API_KEY}\n",
        encoding="utf-8",
    )
    _clear_env(monkeypatch)

    load_repo_env(tmp_path)

    assert os.environ["USDCRAFT_COMPILE_TIMEOUT_SECONDS"] == "42"
    # No dotenv interpolation: a whitelisted value cannot expand a secret from the same file.
    assert os.environ["USDCRAFT_CODEX_BIN"] == "${NVIDIA_API_KEY}"
    assert "NVIDIA_API_KEY" not in os.environ
    assert "NVIDIA_MODEL" not in os.environ


def test_repo_env_ignores_keys_outside_the_whitelist(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "USDCRAFT_RECONSTRUCTION_WORKSPACE=/tmp/hijack\n"
        "USDCRAFT_GENERATION_WORKSPACE=/tmp/hijack\n"
        "USDCRAFT_REPO_ROOT=/tmp/hijack\n"
        "USDCRAFT_VIEWER_READ_ONLY=0\n"
        "USDCRAFT_AUTHOR_WORKSPACE=/tmp/hijack\n"
        "USDCRAFT_MP_START_METHOD=fork\n"
        "USDCRAFT_PARAMETRIC_MESH_CACHE=0\n"
        "OBJECT_GEOMETRY_OVERLAP_TOL=0.01\n"
        "URDF_GEOMETRY_OVERLAP_TOL=0.01\n",
        encoding="utf-8",
    )
    _clear_env(monkeypatch)

    load_repo_env(tmp_path)

    for key in _ENV_KEYS:
        assert key not in os.environ, key


def test_repo_env_never_overrides_existing_public_setting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "USDCRAFT_COMPILE_TIMEOUT_SECONDS=42\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS", "84")

    load_repo_env(tmp_path)

    assert os.environ["USDCRAFT_COMPILE_TIMEOUT_SECONDS"] == "84"


def test_codex_requests_have_no_run_clock(tmp_path) -> None:
    """The Codex arm has no time limit: its request carries no timeout at all."""

    from usdcraft.generation.contract import GenerationRequest

    request = GenerationRequest(prompt="lamp", repo_root=tmp_path)
    assert not hasattr(request, "timeout_seconds")
    assert "timeout_seconds" not in request.to_dict()["codex"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_reasoning_normalization_never_invents_a_default(value: object) -> None:
    with pytest.raises(ValueError, match="string|empty"):
        normalize_reasoning_level(value)  # type: ignore[arg-type]


def test_reasoning_normalization_keeps_alias_and_future_values() -> None:
    assert normalize_reasoning_level(" med ") == "medium"
    assert normalize_reasoning_level(" Future-Tier-Max ") == "Future-Tier-Max"
