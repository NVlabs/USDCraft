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

import multiprocessing as mp
from typing import Final

# Compile workers always start with ``spawn`` (available on every CPython platform): heavy compile
# state stays in short-lived workers, and the credential-holding Provider process is never forked.
MP_START_METHOD: Final = "spawn"


def get_mp_context() -> mp.context.SpawnContext:
    return mp.get_context(MP_START_METHOD)
