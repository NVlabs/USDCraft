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

import hashlib
from pathlib import Path

from usdcraft.pipeline.hashing import sha256_bytes, sha256_file, sha256_text


def test_shared_sha256_helpers_match_standard_library(tmp_path: Path) -> None:
    payload = (b"usdcraft\x00hashing\xff" * (64 * 1024)) + b"tail"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_bytes(payload) == expected
    assert sha256_file(path) == expected

    text = "USDCraft generation \N{CRAB}"
    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
