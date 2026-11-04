# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

from pathlib import Path

TEXT_MEDIA_TYPES = {
    ".urdf": "application/xml",
    ".xml": "application/xml",
    ".py": "text/x-python",
    ".txt": "text/plain",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

MEDIA_TYPE_MAP = {
    **TEXT_MEDIA_TYPES,
    **IMAGE_MEDIA_TYPES,
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".stl": "model/stl",
    ".obj": "model/obj",
    ".usd": "model/vnd.usd",
    ".usda": "model/vnd.usd",
    ".usdc": "model/vnd.usd",
    ".usdz": "model/vnd.usdz+zip",
}

TRACE_MEDIA_TYPE_MAP = {
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain",
    ".json": "application/json",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".zst": "application/zstd",
}


def file_media_type(target: Path) -> str:
    return MEDIA_TYPE_MAP.get(target.suffix.lower(), "application/octet-stream")


def trace_media_type(target: Path) -> str:
    return TRACE_MEDIA_TYPE_MAP.get(target.suffix.lower(), file_media_type(target))


def file_cache_control(*, immutable: bool) -> str:
    return "public, max-age=31536000, immutable" if immutable else "no-store"


def read_text_file_payload(
    target: Path, *, preview_bytes: int, full: bool
) -> tuple[str, bool, int]:
    byte_count = target.stat().st_size
    truncated = False
    if full or byte_count <= preview_bytes:
        raw = target.read_bytes()
    else:
        with target.open("rb") as handle:
            raw = handle.read(preview_bytes + 1)
        truncated = byte_count > preview_bytes
        raw = raw[:preview_bytes]
    return raw.decode("utf-8", errors="replace"), truncated, byte_count
