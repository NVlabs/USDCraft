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

"""High-frequency compound geometry families for articulated-object authoring."""

from __future__ import annotations

from math import atan2, hypot, pi
from typing import Iterable, List, Sequence, cast

from .common import Vec3, sample_catmull_rom_spline_3d
from .primitives import (
    BoxGeometry,
    CylinderGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
    TorusGeometry,
)
from .sweeps import WirePolylineGeometry, beam_between


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _strip_solid(profiles: list[list[tuple[float, float, float]]]) -> MeshGeometry:
    """Connect equal-sized cross sections without assuming an XY profile plane."""
    geometry = MeshGeometry()
    ring_size = len(profiles[0])
    for profile in profiles:
        if len(profile) != ring_size:
            raise ValueError("strip profiles must have equal vertex counts")
        for vertex in profile:
            geometry.add_vertex(*vertex)
    for ring in range(len(profiles) - 1):
        start = ring * ring_size
        next_start = start + ring_size
        for index in range(ring_size):
            following = (index + 1) % ring_size
            a, b = start + index, start + following
            c, d = next_start + following, next_start + index
            geometry.add_face(a, b, c)
            geometry.add_face(a, c, d)
    first = 0
    last = (len(profiles) - 1) * ring_size
    for index in range(1, ring_size - 1):
        geometry.add_face(first, first + index + 1, first + index)
        geometry.add_face(last, last + index, last + index + 1)
    return geometry


class FlexibleHoseGeometry(MeshGeometry):
    """Smooth static hose routed through semantic control points.

    This models the observed/reference shape of a flexible hose. Physics-driven
    deformation is intentionally outside the URDF geometry contract.
    """

    def __init__(
        self,
        control_points: Iterable[Sequence[float]],
        *,
        radius: float,
        radial_segments: int = 14,
        samples_per_segment: int = 8,
        cap_ends: bool = True,
    ) -> None:
        super().__init__()
        points = cast(
            List[Vec3], [tuple(float(value) for value in point) for point in control_points]
        )
        if len(points) < 2 or any(len(point) != 3 for point in points):
            raise ValueError("control_points must contain at least two XYZ points")
        radius = _positive(radius, name="radius")
        if len(points) >= 3:
            points = sample_catmull_rom_spline_3d(
                points, samples_per_segment=max(2, int(samples_per_segment))
            )
        self.merge(
            WirePolylineGeometry(
                points,
                radius=radius,
                radial_segments=max(6, int(radial_segments)),
                cap_ends=cap_ends,
                corner_mode="miter",
            )
        )


class CasterGeometry(MeshGeometry):
    """Compact swivel-caster visual with stem, fork, axle, and rolling wheel."""

    def __init__(
        self,
        *,
        wheel_radius: float,
        wheel_width: float,
        fork_thickness: float | None = None,
        stem_height: float | None = None,
        clearance: float | None = None,
        radial_segments: int = 28,
    ) -> None:
        super().__init__()
        radius = _positive(wheel_radius, name="wheel_radius")
        width = _positive(wheel_width, name="wheel_width")
        fork = _positive(fork_thickness or radius * 0.16, name="fork_thickness")
        stem = _positive(stem_height or radius * 0.75, name="stem_height")
        gap = _positive(clearance or radius * 0.12, name="clearance")
        wheel = CylinderGeometry(radius, width, radial_segments=max(12, int(radial_segments)))
        wheel.rotate_y(pi / 2.0).translate(0.0, 0.0, radius)
        self.merge(wheel)
        side_x = width / 2.0 + gap + fork / 2.0
        fork_height = radius * 1.35
        for x in (-side_x, side_x):
            self.merge(
                BoxGeometry((fork, radius * 0.35, fork_height)).translate(x, 0.0, radius * 1.35)
            )
        bridge_width = 2.0 * side_x + fork
        bridge_z = radius * 2.0 + gap
        self.merge(BoxGeometry((bridge_width, radius * 0.55, fork)).translate(0.0, 0.0, bridge_z))
        self.merge(
            CylinderGeometry(fork * 0.65, stem, radial_segments=20).translate(
                0.0, 0.0, bridge_z + stem / 2.0
            )
        )
        axle_length = bridge_width
        self.merge(
            CylinderGeometry(fork * 0.42, axle_length, radial_segments=16)
            .rotate_y(pi / 2.0)
            .translate(0.0, 0.0, radius)
        )


class CasterForkGeometry(MeshGeometry):
    """Offset swivel yoke authored separately from its wheel.

    The wheel center is ``(0, 0, wheel_radius)`` and its axle runs along X.
    ``trail_offset`` moves the swivel stem along +Y, producing the recognizable
    swept caster silhouette.  Keep the wheel in a separate visual or articulated
    part when tire and metal need different materials.
    """

    def __init__(
        self,
        *,
        wheel_radius: float,
        wheel_width: float,
        fork_thickness: float | None = None,
        stem_height: float | None = None,
        clearance: float | None = None,
        trail_offset: float | None = None,
        radial_segments: int = 20,
    ) -> None:
        super().__init__()
        radius = _positive(wheel_radius, name="wheel_radius")
        width = _positive(wheel_width, name="wheel_width")
        fork = _positive(fork_thickness or radius * 0.16, name="fork_thickness")
        stem = _positive(stem_height or radius * 0.72, name="stem_height")
        gap = _positive(clearance or radius * 0.12, name="clearance")
        trail = float(trail_offset if trail_offset is not None else radius * 0.42)
        if trail < 0.0:
            raise ValueError("trail_offset must be non-negative")

        side_x = width / 2.0 + gap + fork / 2.0
        bridge_z = radius * 2.05 + gap
        for x in (-side_x, side_x):
            self.merge(
                beam_between(
                    (x, 0.0, radius),
                    (x, trail, bridge_z),
                    width=fork,
                    height=fork * 0.72,
                    up_hint=(1.0, 0.0, 0.0),
                )
            )
        self.merge(
            beam_between(
                (-side_x, trail, bridge_z),
                (side_x, trail, bridge_z),
                width=fork,
                height=fork * 0.82,
            )
        )
        race_radius = max(fork * 1.35, width * 0.22)
        self.merge(
            CylinderGeometry(race_radius, fork * 0.72, radial_segments=radial_segments).translate(
                0.0, trail, bridge_z + fork * 0.36
            )
        )
        self.merge(
            CylinderGeometry(fork * 0.62, stem, radial_segments=radial_segments).translate(
                0.0, trail, bridge_z + fork * 0.72 + stem / 2.0
            )
        )
        self.merge(
            CylinderGeometry(fork * 0.43, 2.0 * side_x + fork, radial_segments=radial_segments)
            .rotate_y(pi / 2.0)
            .translate(0.0, 0.0, radius)
        )


class MoldedSeatGeometry(MeshGeometry):
    """Continuous shallow seat pan and reclined back with rounded side rims.

    Local X is seat width, +Y points toward the back, and the nominal seat
    surface is around Z=0.  This family covers cart child seats, high-chair
    inserts, stadium shells, and compact molded chair buckets without reducing
    them to two intersecting slabs.
    """

    def __init__(
        self,
        *,
        width: float,
        depth: float,
        back_height: float,
        thickness: float,
        seat_sag: float = 0.0,
        back_recline: float = 0.0,
        side_rim_radius: float | None = None,
        sections: int = 10,
    ) -> None:
        super().__init__()
        width = _positive(width, name="width")
        depth = _positive(depth, name="depth")
        back_height = _positive(back_height, name="back_height")
        thickness = _positive(thickness, name="thickness")
        sag = float(seat_sag)
        recline = float(back_recline)
        if sag < 0.0 or sag >= depth * 0.45:
            raise ValueError("seat_sag must be non-negative and smaller than 45% of depth")
        if abs(recline) >= back_height:
            raise ValueError("abs(back_recline) must be smaller than back_height")
        count = max(5, int(sections))
        rim = _positive(side_rim_radius or thickness * 0.72, name="side_rim_radius")

        seat_profiles: list[list[tuple[float, float, float]]] = []
        left_rim: list[tuple[float, float, float]] = []
        right_rim: list[tuple[float, float, float]] = []
        for index in range(count):
            t = index / (count - 1)
            y = -depth / 2.0 + depth * t
            z = -4.0 * sag * t * (1.0 - t)
            half_width = width * (0.46 + 0.04 * t)
            seat_profiles.append(
                [
                    (-half_width, y, z - thickness / 2.0),
                    (half_width, y, z - thickness / 2.0),
                    (half_width, y, z + thickness / 2.0),
                    (-half_width, y, z + thickness / 2.0),
                ]
            )
            left_rim.append((-half_width, y, z + rim * 0.35))
            right_rim.append((half_width, y, z + rim * 0.35))
        self.merge(_strip_solid(seat_profiles))

        back_profiles: list[list[tuple[float, float, float]]] = []
        for index in range(count):
            t = index / (count - 1)
            z = back_height * t
            center_y = depth / 2.0 + recline * t
            half_width = width * (0.50 - 0.055 * t)
            back_profiles.append(
                [
                    (-half_width, center_y - thickness / 2.0, z),
                    (half_width, center_y - thickness / 2.0, z),
                    (half_width, center_y + thickness / 2.0, z),
                    (-half_width, center_y + thickness / 2.0, z),
                ]
            )
        self.merge(_strip_solid(back_profiles))

        top_half_width = width * (0.50 - 0.055)
        left_rim.extend(
            [
                (-width / 2.0, depth / 2.0, 0.0),
                (-top_half_width, depth / 2.0 + recline, back_height),
            ]
        )
        right_rim.extend(
            [
                (width / 2.0, depth / 2.0, 0.0),
                (top_half_width, depth / 2.0 + recline, back_height),
            ]
        )
        for path in (left_rim, right_rim):
            self.merge(
                WirePolylineGeometry(
                    path,
                    radius=rim,
                    radial_segments=12,
                    cap_ends=True,
                    corner_mode="fillet",
                )
            )


class PostBeamBracketGeometry(MeshGeometry):
    """Open saddle bracket for a beam/post or beam/rafter connection.

    The supported member runs along local X.  Two side plates sit on +/-Y,
    joined by a lower saddle, with transverse bolt shanks along Y.  The helper
    is suitable for playgrounds, pergolas, furniture frames, racks, and other
    exposed structural joinery.
    """

    def __init__(
        self,
        *,
        member_width: float,
        member_height: float,
        plate_length: float,
        plate_thickness: float,
        bolt_radius: float,
        bolt_count: int = 2,
    ) -> None:
        super().__init__()
        member_width = _positive(member_width, name="member_width")
        member_height = _positive(member_height, name="member_height")
        plate_length = _positive(plate_length, name="plate_length")
        plate = _positive(plate_thickness, name="plate_thickness")
        bolt_radius = _positive(bolt_radius, name="bolt_radius")
        count = int(bolt_count)
        if count < 1 or count > 6:
            raise ValueError("bolt_count must be between 1 and 6")
        if plate * 2.0 >= member_width:
            raise ValueError("plate_thickness is too large for member_width")

        side_y = member_width / 2.0 + plate / 2.0
        for y in (-side_y, side_y):
            self.merge(
                RoundedBoxGeometry(
                    (plate_length, plate, member_height),
                    min(plate * 0.45, member_height * 0.08),
                    round_axis="y",
                ).translate(0.0, y, 0.0)
            )
        self.merge(
            BoxGeometry((plate_length, member_width + 2.0 * plate, plate)).translate(
                0.0, 0.0, -member_height / 2.0 + plate / 2.0
            )
        )
        total_depth = member_width + 2.0 * plate
        for index in range(count):
            x = plate_length * ((index + 1) / (count + 1) - 0.5)
            self.merge(
                CylinderGeometry(bolt_radius, total_depth + plate * 0.3, radial_segments=16)
                .rotate_x(pi / 2.0)
                .translate(x, 0.0, member_height * 0.12)
            )


class GableRoofGeometry(MeshGeometry):
    """Two pitched roof panels with ridge cap and eave fascia.

    Local Z=0 is the eave plane, the ridge is centered at X=0, and roof depth
    runs along Y.  This is a compact family for playhouses, sheds, kiosks, pet
    houses, and small architectural props.
    """

    def __init__(
        self,
        *,
        width: float,
        depth: float,
        rise: float,
        panel_thickness: float,
        overhang: float = 0.0,
        fascia_height: float | None = None,
        edge_radius: float | None = None,
    ) -> None:
        super().__init__()
        width = _positive(width, name="width")
        depth = _positive(depth, name="depth")
        rise = _positive(rise, name="rise")
        panel = _positive(panel_thickness, name="panel_thickness")
        overhang = float(overhang)
        if overhang < 0.0:
            raise ValueError("overhang must be non-negative")
        fascia = _positive(fascia_height or panel * 2.2, name="fascia_height")
        radius = _positive(edge_radius or panel * 0.35, name="edge_radius")

        run = width / 2.0 + overhang
        slope = hypot(run, rise)
        angle = atan2(rise, run)
        panel_depth = depth + 2.0 * overhang
        left = RoundedBoxGeometry((slope, panel_depth, panel), radius, round_axis="y").rotate_y(
            -angle
        )
        left.translate(-run / 2.0, 0.0, rise / 2.0)
        right = RoundedBoxGeometry((slope, panel_depth, panel), radius, round_axis="y").rotate_y(
            angle
        )
        right.translate(run / 2.0, 0.0, rise / 2.0)
        self.merge(left)
        self.merge(right)
        self.merge(
            CylinderGeometry(panel * 0.72, panel_depth, radial_segments=18)
            .rotate_x(pi / 2.0)
            .translate(0.0, 0.0, rise + panel * 0.35)
        )
        for x in (-run, run):
            self.merge(BoxGeometry((panel, panel_depth, fascia)).translate(x, 0.0, 0.0))


class BeltSeatGeometry(MeshGeometry):
    """Symmetric flexible swing seat with a shallow longitudinal sag."""

    def __init__(
        self,
        *,
        width: float,
        depth: float,
        thickness: float,
        sag: float,
        sections: int = 9,
    ) -> None:
        super().__init__()
        width = _positive(width, name="width")
        depth = _positive(depth, name="depth")
        thickness = _positive(thickness, name="thickness")
        sag = float(sag)
        if sag < 0.0 or sag >= depth:
            raise ValueError("sag must be non-negative and smaller than depth")
        count = max(3, int(sections))
        profiles: list[list[tuple[float, float, float]]] = []
        for index in range(count):
            t = index / (count - 1)
            y = -depth / 2.0 + depth * t
            z = -4.0 * sag * t * (1.0 - t)
            profiles.append(
                [
                    (-width / 2.0, y, z - thickness / 2.0),
                    (width / 2.0, y, z - thickness / 2.0),
                    (width / 2.0, y, z + thickness / 2.0),
                    (-width / 2.0, y, z + thickness / 2.0),
                ]
            )
        self.merge(_strip_solid(profiles))


class SlideChuteGeometry(MeshGeometry):
    """Curved descending slide bed with raised continuous side rails."""

    def __init__(
        self,
        *,
        width: float,
        length: float,
        drop: float,
        thickness: float,
        rail_radius: float,
        sections: int = 16,
    ) -> None:
        super().__init__()
        width = _positive(width, name="width")
        length = _positive(length, name="length")
        drop = _positive(drop, name="drop")
        thickness = _positive(thickness, name="thickness")
        rail_radius = _positive(rail_radius, name="rail_radius")
        count = max(5, int(sections))
        profiles: list[list[tuple[float, float, float]]] = []
        left: list[tuple[float, float, float]] = []
        right: list[tuple[float, float, float]] = []
        for index in range(count):
            t = index / (count - 1)
            y = length * t
            z = -drop * (3.0 * t * t - 2.0 * t * t * t)
            profiles.append(
                [
                    (-width / 2.0, y, z - thickness),
                    (width / 2.0, y, z - thickness),
                    (width / 2.0, y, z),
                    (-width / 2.0, y, z),
                ]
            )
            left.append((-width / 2.0, y, z + rail_radius))
            right.append((width / 2.0, y, z + rail_radius))
        self.merge(_strip_solid(profiles))
        for path in (left, right):
            self.merge(
                WirePolylineGeometry(
                    path,
                    radius=rail_radius,
                    radial_segments=12,
                    cap_ends=True,
                    corner_mode="miter",
                )
            )


class HangingChainGeometry(MeshGeometry):
    """Presentation-only oval-link mesh centered around the Z axis.

    The links are merged into one static visual mesh; this helper does not author
    rigid bodies, colliders, or joints. USDCraft Authors (Generation and
    Reconstruction) use this kind of static rest shape on one
    ``BodyModel.FLEXIBLE_1D`` part and do not discretize a chain into rigid
    segments and joints. Direct SDK and downstream callers may use
    ``add_segmented_flexible_member`` explicitly; that helper is outside the
    USDCraft authoring contract.
    """

    def __init__(
        self,
        *,
        length: float,
        link_count: int,
        link_radius: float,
        wire_radius: float,
    ) -> None:
        super().__init__()
        length = _positive(length, name="length")
        link_radius = _positive(link_radius, name="link_radius")
        wire_radius = _positive(wire_radius, name="wire_radius")
        count = int(link_count)
        if count < 2:
            raise ValueError("link_count must be at least 2")
        if wire_radius >= link_radius:
            raise ValueError("wire_radius must be smaller than link_radius")
        for index in range(count):
            z = -length * index / (count - 1)
            link = TorusGeometry(
                link_radius,
                wire_radius,
                radial_segments=12,
                tubular_segments=24,
            )
            link.rotate_x(pi / 2.0 if index % 2 == 0 else 0.0).translate(0.0, 0.0, z)
            self.merge(link)


class LadderGeometry(MeshGeometry):
    """Two rails and repeated rungs fitted from base/top anchor dimensions.

    Local X spans the ladder width. The base is centered at ``(0, 0, 0)`` and
    the top is centered at ``(0, run, rise)``.
    """

    def __init__(
        self,
        *,
        width: float,
        rise: float,
        run: float,
        rail_section: float,
        rung_count: int,
        rung_section: float | None = None,
    ) -> None:
        super().__init__()
        width = _positive(width, name="width")
        rise = _positive(rise, name="rise")
        run = float(run)
        if run < 0.0:
            raise ValueError("run must be non-negative")
        rail = _positive(rail_section, name="rail_section")
        rung = _positive(rung_section or rail * 0.78, name="rung_section")
        count = int(rung_count)
        if count < 2:
            raise ValueError("rung_count must be at least 2")
        for x in (-width / 2.0, width / 2.0):
            self.merge(
                beam_between(
                    (x, 0.0, 0.0),
                    (x, run, rise),
                    width=rail,
                )
            )
        for index in range(count):
            t = (index + 1) / (count + 1)
            y = run * t
            z = rise * t
            self.merge(
                beam_between(
                    (-width / 2.0, y, z),
                    (width / 2.0, y, z),
                    width=rung,
                )
            )


class AFrameGeometry(MeshGeometry):
    """Two splayed legs plus a spreader, parameterized by structural anchors.

    The feet lie on Z=0 at ``Y=±footprint_width/2`` and the apex is at
    ``(0, 0, height)``. Add the supported top beam as a separate visual.
    """

    def __init__(
        self,
        *,
        height: float,
        footprint_width: float,
        leg_section: float,
        spreader_height: float,
        diagonal_brace: bool = False,
    ) -> None:
        super().__init__()
        height = _positive(height, name="height")
        footprint = _positive(footprint_width, name="footprint_width")
        section = _positive(leg_section, name="leg_section")
        spreader_z = float(spreader_height)
        if not 0.0 < spreader_z < height:
            raise ValueError("spreader_height must lie between zero and height")
        left_foot = (0.0, -footprint / 2.0, 0.0)
        right_foot = (0.0, footprint / 2.0, 0.0)
        apex = (0.0, 0.0, height)
        self.merge(beam_between(left_foot, apex, width=section))
        self.merge(beam_between(right_foot, apex, width=section))
        t = spreader_z / height
        half_span = footprint * (1.0 - t) / 2.0
        self.merge(
            beam_between(
                (0.0, -half_span, spreader_z),
                (0.0, half_span, spreader_z),
                width=section * 0.72,
            )
        )
        if diagonal_brace:
            low_t, high_t = 0.18, 0.68
            self.merge(
                beam_between(
                    (0.0, -footprint * (1.0 - low_t) / 2.0, height * low_t),
                    (0.0, footprint * (1.0 - high_t) / 2.0, height * high_t),
                    width=section * 0.58,
                )
            )
