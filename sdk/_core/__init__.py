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

"""Shared internal SDK implementation."""

# ruff: noqa: F401
from .articulated_object import ArticulatedObject
from .assets import AssetContext
from .errors import SDKError, ValidationError
from .geometry_scale import scale_geometry_to_size
from .placement import (
    SurfaceFrame,
    SurfaceWrapMapping,
    align_centers,
    part_local_aabb,
    place_on_face,
    place_on_face_uv,
    place_on_surface,
    proud_for_flush_mount,
    surface_frame,
    wrap_mesh_onto_surface,
    wrap_profile_onto_surface,
)
from .segmented_flexible_member import (
    PlanarSegmentJoint,
    SegmentedFlexibleMember,
    SegmentedFlexibleMemberSpec,
    SpatialSegmentJoint,
    add_segmented_flexible_member,
)
from .testing import TestContext, TestFailure, TestReport
from .types import (
    Articulation,
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    CollisionApproximation,
    CollisionProperties,
    Cylinder,
    Inertia,
    Inertial,
    JointDrive,
    Material,
    Mesh,
    Mimic,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    Part,
    PassiveJointSpring,
    PhysicsMaterial,
    PhysxJointFriction,
    PhysxJointFrictionCoefficient,
    PhysxJointResistance,
    RigidBodyProperties,
    Sphere,
    TextureInput,
    Visual,
    estimate_mass_from_density,
)
