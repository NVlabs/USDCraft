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

"""Same-frame source-overlay carrier for Reconstruction.

``prepare_source_overlay`` exports the exact post-isolation normalized scene used by AXIAL as
a URDF plus GLB carrier and a small manifest. The Viewer overlays it on the candidate, and the
Author workspace stages the same carrier for on-demand source renders; this module starts no
renderer and produces no images. The carrier writer comes from ``orientation.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from usdcraft.pipeline.hashing import sha256_file as _sha256
from usdcraft.reconstruction.orientation import _normalized_identity, _write_carrier

if TYPE_CHECKING:
    from usdcraft.reconstruction.source import NormalizedSourceArtifacts, NormalizedSourceScene


@dataclass(frozen=True, slots=True)
class SourceOverlayArtifacts:
    root: Path
    carrier_urdf: Path
    carrier_glb: Path
    manifest: Path


def prepare_source_overlay(
    scene: NormalizedSourceScene, normalized_artifacts: NormalizedSourceArtifacts, output_dir: Path
) -> SourceOverlayArtifacts:
    """Export same-frame source geometry for Viewer and later Author-selected renders.

    This does not initialize a renderer or generate image evidence.
    """
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = _normalized_identity(normalized_artifacts)
    urdf, glb, stats = _write_carrier(scene, root / "carrier")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "reconstruction_source_overlay",
                "source_identity": identity,
                "target_frame": {"meters_per_unit": 1.0, "up_axis": "+Z", "front_axis": "-Y"},
                "carrier": {
                    "urdf": "carrier/model.urdf",
                    "urdf_sha256": _sha256(urdf),
                    "glb": "carrier/assets/source.glb",
                    "glb_sha256": _sha256(glb),
                    **stats,
                },
                "images": [],
                "render_policy": "author_requested",
            },
            indent=2,
        )
        + "\n"
    )
    return SourceOverlayArtifacts(root, urdf, glb, manifest)


__all__ = ["SourceOverlayArtifacts", "prepare_source_overlay"]
