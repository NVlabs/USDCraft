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

"""Authoring policy shared by every USDCraft Author task.

The physical-authoring rules below — the flexible-member rule, explicit mass and materials, contact
presets, joint limits and resistance, gravity hold, drives, the radians contract and the backend
capability facts — are the same for a generated asset and a reconstructed one: they describe how a simulator-ready asset is
authored with the public SDK, not what the asset is or which evidence decides its shape. Both
products render this one text into their Author tasks (the Generation DELIVERABLE list and the
Reconstruction SIMULATOR-READY MECHANICAL AUTHORING section) so the guidance cannot drift between
them. Product semantics stay in each product's own ``prompt.py``.
"""

from __future__ import annotations

_FLEXIBLE_MEMBER_BULLETS = """- In __PRODUCT__, each hose/rope/cable/cord/wire/chain/strap/belt/flexible suspension is one
  BodyModel.FLEXIBLE_1D Part: smooth static q=0 rest shape, FIXED to its appropriate parent.
  Rigid fittings/nozzles/plugs/handles stay separate RIGID Parts with their real relations.
  Keep flexible visuals collidable for deformable_contact inspection; use allow_overlap for
  intended contact, not collidable=False.
- Distributed flexible dynamics are unsupported in __PRODUCT__: neither
  add_segmented_flexible_member nor an equivalent hand-built rigid-segment joint chain is valid.
  Curvature/drape/aiming/repositioning use the static member. Block only for explicit must-have
  distributed bending/payout/serial load transfer/flexible contact dynamics it cannot represent.
  Separate rigid fittings/carriers may have meaningful low-order DOFs, without segmentation.
  Choose RIGID/SURFACE_DEFORMABLE/VOLUME_DEFORMABLE roles deliberately."""

_MECHANICAL_AUTHORING_BULLETS = """- Every rigid Part, including root/moving members, needs explicit positive-mass Inertial and
  RigidBodyProperties contact binding, separate from visual PBR. Choose contact by external role:
  static/dynamic pairs metal .25/.18, painted .40/.30, plastic .50/.35, wood .55/.40, rubber .90/.70,
  guide .15/.10; static >= dynamic >= 0, usually restitution 0-.1.
- Derive mass visibly from modeled dimensions and plausible density using
  estimate_mass_from_density(Box/Cylinder/Sphere, density, fill_fraction=...). Justify hollow/thin
  construction by fill fraction or area*thickness*density; sum 1-2 major compound components.
  Mass literals need a supplied-specification source comment. Exclude people/contents/cargo unless
  requested. A representative primitive Inertial.from_geometry suffices.
  Runtime COM/inertia come from final colliders; preserve plausible geometry and mass instead of
  changing COM, inertia or collision offsets to silence diagnostics.
- For scalar REVOLUTE/CONTINUOUS/PRISMATIC joints, explicitly choose positive effort/velocity,
  damping and resistance. Bounded limits are finite and contain q=0. Select PhysxJointFriction
  for absolute breakaway/drag (0 <= dynamic <= static), or PhysxJointFrictionCoefficient only for
  intentionally load-proportional bearing/shaft/slider resistance; coefficient 0 declares a
  frictionless joint. Angular effort is N*m, viscosity N*m*s/rad; linear effort is N, viscosity
  N*s/m. Newton maps absolute static effort to mjc:frictionloss and adds its viscosity to native
  damping, losing separate static/dynamic thresholds. MotionProperties.friction is the Newton
  fallback without absolute resistance and the URDF friction value. Choose it deliberately;
  backend friction models are not physically equivalent.
- Mark every scalar joint meta={"gravity_hold": True} if it must stay where released, or False
  for deliberate gravity-driven/free motion or modeled contact support. Zero-axis-load joints may
  remain True. Across the full limits, compute signed gravity effort from all mechanically
  connected downstream bodies, other joints at q=0, including joint-origin rotation and only the
  modeled default payload. Exclude independent FLOATING branches. Angular load is
  sum(axis_world dot ((COM_world - joint_world) cross (mass * gravity))); linear load is
  sum(axis_world dot (mass * gravity)). Use coarse primitive-volume-weighted collider centroids,
  not object/AABB height as a lever arm. Vertical hinges and horizontal slides have zero axis load.
- For hold joints sample max_q |gravity(q) + passive_spring(q) + force_mode_drive(q)|.
  Spring effort is stiffness*(reference-q); clamp drive effort by its effective max force.
  PhysxJointFriction.stay_put(max_passive_effort) adds its margins to that maximum residual.
  A variable payload needs a stated/tested design load; otherwise size the empty asset. Add a
  ctx.check proving static resistance covers the residual, then run __CHECK__.
  For positive residual held by dry friction, also set
  NewtonJointSolverTuning(solimp_friction=(0.999, 0.9999, 0.001, 0.5, 2.0)); d0 >= .999 is
  required by __PRODUCT__ pose-hold policy, and dwidth affects damping. Newton's default soft
  impedance can creep even with adequate frictionloss. This tuning sets no friction magnitude;
  stay_put supplies no tuning or spring. Zero residual and gravity_hold=False do not require it.
  Coverage/impedance diagnostics are authoring checks: explicit undercoverage fails repair, missing
  legacy markers warn. They add no semantic Host gate and prove no runtime stability.
- Choose plausible resistance: free cabinet hinge 1-3 N at handle radius, small knob .005-.08 N*m,
  reduced-order drawer 3-10 N with dynamic/static ratio .6-.9, free wheel/shaft load coefficient
  .01-.05. Model a real counterbalance with PassiveJointSpring plus residual friction, never a
  spring added solely to suppress a diagnostic. Spring stiffness/reference use N*m/rad and radians
  (linear N/m and metres); it is mutually exclusive with JointDrive. Static friction is not a
  detent: clicks require discrete targets, angle-dependent effort or contact geometry. Avoid
  double-counting guide friction when real rail contact already supplies that resistance.
- Passive damping belongs in MotionProperties.damping. With an active JointDrive, set that field
  to 0.0 and use JointDrive.damping alone. Omit target drives for passive hand-operated mechanisms;
  use them only for actual actuation/counterbalance/return. max_force=None uses MotionLimits.effort;
  explicit max_force is finite, nonnegative and <= effort. Author public SDK fields; compiled USD
  and customData are not places to patch essential physics.
- Screen extreme inertia disparity within each connected native articulation component using
  check_generation_physx_armature_stability. Its limited q=0 proxy uses source inertia about the
  scalar axis through link COM, or masses for slides; it excludes parallel-axis, subtree, contact
  and runtime collider effects. For flagged ratios >100, total physx_armature is
  max(0, J_max/100-J_child) kg*m^2 or max(0, m_max/100-m_child) kg. This empirical __PRODUCT__
  screen has limited Isaac Sim 6 evidence; 100:1 is untested, not an NVIDIA threshold or Host gate.
  Leave ordinary joints at zero. Comment numerical compensation vs real rotor inertia J*G^2;
  preserve plausible mass/inertia/COM. The SDK field selects the correct export API, without
  angular conversion. It is PhysX-only: no inferred Newton value. Validate runtime stability.
- Keep angular limits/targets in radians, velocity in rad/s and gains per radian; linear values
  stay SI. Exporters handle USD degree conversion. For matching static active-controller structure
  prefer force mode, omit target_velocity and use position PD or zero-stiffness positive damping.
  Newton has no verified static equivalent for target velocity, acceleration mode, pure-P or
  velocity limits: acceleration becomes direct force without static gain/bias, velocity limits
  are metadata only. Passive damping/springs use native Newton properties, not active actuators.
  __PRODUCT__ does not run an Isaac Sim import/runtime test; static exports prove no motion result.
- SPHERICAL is native USDC, has no Workbench controls and only semantic URDF fallback;
  no scalar limits/damping/friction/mimic/drive. D6 is unsupported; spherical/floating is no D6
  substitute. Full types/backends: sdk/_docs/common/20_core_types.md; diagnostics: 80_testing.md
  in that directory."""


def render_flexible_member_bullets(*, product: str) -> str:
    """The shared flexible-member rule for both products.

    Every hose, rope, cable, cord, wire, chain, strap or belt is one static-rest
    ``BodyModel.FLEXIBLE_1D`` Part; distributed flexible dynamics are an SDK/export limitation, so
    no Author segments a member into a rigid-body joint chain.
    """

    name = str(product).strip()
    if not name:
        raise ValueError("Authoring policy product name must not be empty.")
    return _FLEXIBLE_MEMBER_BULLETS.replace("__PRODUCT__", name)


def render_mechanical_authoring_bullets(
    *, product: str, check_command: str = "usdcraft check"
) -> str:
    """The shared physical-authoring bullets.

    ``product`` is named where a policy is attributed; ``check_command`` is the arm's authoring
    check (the shell command or the Provider ``check_model`` tool) named in the gravity-hold rule.
    """

    name = str(product).strip()
    if not name:
        raise ValueError("Authoring policy product name must not be empty.")
    check = str(check_command).strip()
    if not check:
        raise ValueError("Authoring policy check command must not be empty.")
    return _MECHANICAL_AUTHORING_BULLETS.replace("__PRODUCT__", name).replace("__CHECK__", check)


def render_mechanical_authoring_block(
    *, product: str, intro: str | None = None, check_command: str = "usdcraft check"
) -> str:
    """The bullets under their own section header, with an optional product-owned lead sentence."""

    lead = f"{str(intro).strip()}\n" if intro and str(intro).strip() else ""
    flexible = render_flexible_member_bullets(product=product)
    bullets = render_mechanical_authoring_bullets(product=product, check_command=check_command)
    return f"""SIMULATOR-READY MECHANICAL AUTHORING
{lead}{flexible}
{bullets}
"""


__all__ = [
    "render_flexible_member_bullets",
    "render_mechanical_authoring_block",
    "render_mechanical_authoring_bullets",
]
