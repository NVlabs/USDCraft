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
from pathlib import Path

from dotenv import dotenv_values

# The only repository `.env` names copied into the process environment. `.env` also holds the
# NVIDIA_* and OPENAI_* Provider settings, which the Provider client reads directly and which are
# never injected. Internal control variables (workspace roots, repository root, Viewer mode) are set by
# the program for its own subprocesses and are never read from `.env`.
REPO_ENV_INJECTED_KEYS = frozenset({"USDCRAFT_COMPILE_TIMEOUT_SECONDS", "USDCRAFT_CODEX_BIN"})


def load_repo_env(repo_root: Path) -> None:
    """Load the whitelisted non-secret USDCraft settings into the process environment.

    Provider credentials live in the same repository-local ``.env`` file, but Codex
    and compile subprocesses must not inherit them. Provider code reads its own
    explicitly named settings without mutating ``os.environ``. A variable already set in the
    process environment wins over ``.env``.
    """

    env_path = Path(repo_root).resolve() / ".env"
    if env_path.is_file():
        # Do not let a public-looking variable interpolate a secret value from the same file.
        for key, value in dotenv_values(env_path, interpolate=False).items():
            if value is not None and key in REPO_ENV_INJECTED_KEYS and key not in os.environ:
                os.environ[key] = value
