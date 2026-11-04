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

"""Asset Structure file names and small schema helpers shared by the USD exporter modules."""

from __future__ import annotations

from pxr import Usd

from .errors import ValidationError
from .types import Articulation, ArticulationType

_PHYSICS_LAYER_NAME = "physics.usdc"
_PHYSX_LAYER_NAME = "physx.usda"
_NEWTON_MJC_LAYER_NAME = "mujoco.usda"
_ASSET_INTERFACE_NAME = "asset.usda"
_PHYSX_ENTRY_NAME = "model.usdc"
_NEWTON_ENTRY_NAME = "model_newton.usdc"
_MJC_PARAMETER_COUNT = 10
_MJC_MIN_VALUE = 1e-15


def _apply_api_schema_token(prim: Usd.Prim, schema_name: str) -> None:
    """Apply an API token even when its provider plugin is unavailable.

    The standalone OpenUSD runtime used by the Host does not ship NVIDIA's
    PhysxSchema Python module. ``Usd.Prim.AddAppliedSchema`` authors the token
    through USD's list editor, preserving any explicit, prepend, append,
    delete, and reorder operations already present on ``apiSchemas``.
    """

    if not prim.AddAppliedSchema(schema_name):
        raise ValidationError(
            f"Failed to apply API schema token {schema_name!r} to {prim.GetPath()}"
        )


def _joint_type(joint: Articulation) -> ArticulationType:
    return (
        joint.articulation_type
        if isinstance(joint.articulation_type, ArticulationType)
        else ArticulationType(str(joint.articulation_type))
    )
