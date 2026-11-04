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

"""Content binding for advisory measurements; missing/stale readings never gate promotion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def usd_fingerprint(path: Path | None) -> str | None:
    """Hash authored layer content and asset bytes, independent of crate serialization."""
    if path is None or not path.is_file():
        return None
    from pxr import Sdf, UsdUtils

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(str(path.resolve()))
        )
        if unresolved or not layers:
            return None
        layer_paths = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
        files = layer_paths | {Path(asset).resolve() for asset in assets}

        entries = sorted((os.path.relpath(f, path.resolve().parent), f) for f in files)
        digest = hashlib.sha256()
        for name, file in entries:
            if file in layer_paths:
                # Reload from disk without sharing a cached layer from a previous build.
                layer = Sdf.Layer.OpenAsAnonymous(str(file))
                if layer is None:
                    return None
                content = layer.ExportToString().encode()
            else:
                content = file.read_bytes()
            digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
        return digest.hexdigest()
    except (OSError, RuntimeError):
        return None


def diagnostic_binding(source: Path | None, candidate: Path) -> dict[str, str | None]:
    return {
        "source_sha256": usd_fingerprint(source),
        "candidate_sha256": usd_fingerprint(candidate),
        "pose": "authored_q0",
        "hash_scope": "usd_authored_dependency_closure",
    }
