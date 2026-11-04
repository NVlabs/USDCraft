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

"""INTERIOR_ISLANDS: sampled-surface islands standing free inside a slab's body outline.

``MICRO_SLICE_BOUNDS_Z192`` lists every sampled-surface island of every stack slab as an anonymous
box, and the brief reads the largest island of a slab as the body and the rest as parts standing
beside it. A dishwasher rack, an oven shelf or a coffee-machine water tank is neither: its islands
lie strictly inside the body's outline and touch no wall, and a bare island count would not tell
the Author that the shell holds a fitting, at what height or how large. This section groups those
nested islands across slabs and reports each group's height band, footprint, centre and character
(a grid of thin members, a plate, a hollow shell, a block) in metres.

The physical contract is untouched: the islands are sampled surfaces, the interior beyond them stays
UNKNOWN, and a group is the Host's deterministic grouping HYPOTHESIS — not a Part, a solid or a
motion claim. A fitting that touches the walls (a shelf on side rails) or rests on the floor of its
own slab (a turntable) merges with the body island and is not listed; absence proves nothing.

Typical groups are dishwasher rack levels, oven shelf levels as free bars, a hollow water-tank
shell, refrigerator drawers as shells and raised turntables as plates; wall-hung shelves and a
turntable lying in its floor slab stay silent. The section never changes a source's stack, micro
selection or contours.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

import numpy as np

# An island counts as interior when its box sits at least this many micro cells inside the body box
# on every side, so a wall-attached fitting never qualifies.
INTERIOR_MARGIN_CELLS = 1
# An island inset from the body box by no more than this on every side is the inner face of the
# shell's own wall (a thick tub, a basin bowl), not a fitting.
WALL_INSET_MAX_CELLS = 4
WALL_INSET_MAX_SHARE = 0.05
# A slab whose largest island covers less of the footprint than this is legs or a stem, not a shell
# with an inside.
BODY_FOOTPRINT_MIN_SHARE = 0.25
# Same-size islands in one slab form a grid (rack wires, shelf bars, upright tines seen end-on)
# from this many members; a grid reports up to this many member size classes, largest count first.
# A sparse grid of few specks (three scan dots across an empty cavity) is noise: a grid stands only
# with many members or a footprint it fills to at least a tenth.
GRID_MIN_MEMBERS = 3
GRID_DENSE_MIN_MEMBERS = 6
GRID_MIN_FILL = 0.10
GRID_MAX_CLASSES = 3
SIZE_MATCH_MIN_CELLS = 2
SIZE_MATCH_FRACTION = 0.20
# A lone island must be this large to be reported (hinge pins and scan speckle are not fittings),
# unless it is a bar spanning this share of the body's side (one free wire of a rack).
SINGLE_MIN_BODY_SHARE = 0.04
SINGLE_MIN_SIDE_M = 0.025
BAR_MIN_SPAN_SHARE = 0.40
BAR_MAX_SHORT_SIDE_M = 0.015
BAR_MIN_ASPECT = 8.0
SHELL_MIN_SIDE_M = 0.040
PLATE_FILL_MIN = 0.60
SHELL_FILL_MAX = 0.25
# Entities merge (within a slab and across adjacent slabs) when their footprints, grown by the
# tolerance, overlap by this share of the smaller one: a rack's base plate, its wire grid and the
# frame bars hugging the grid's edge are one fitting.
OVERLAP_MIN_SHARE = 0.80
MERGE_TOLERANCE_CELLS = 3
REPEAT_SIZE_TOLERANCE = 0.15
INTERIOR_MAX_GROUPS = 8
# Governs the group lines only (the header, semantics and coords lines are fixed); the smallest
# groups are dropped first and the header records detected versus shown.
INTERIOR_ISLANDS_BYTE_BUDGET = 1024


class _IslandLike(Protocol):
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    cell_count: int

    @property
    def bbox_cells(self) -> int: ...


class _RunLike(Protocol):
    @property
    def stack_z_start(self) -> int: ...

    @property
    def stack_z_end(self) -> int: ...

    @property
    def micro_z_start(self) -> int: ...

    @property
    def micro_z_end(self) -> int: ...

    @property
    def islands(self) -> tuple[Any, ...]: ...


class _FrameLike(Protocol):
    @property
    def pitch(self) -> float: ...

    @property
    def dimensions(self) -> np.ndarray: ...

    @property
    def bounds_min(self) -> np.ndarray: ...


@dataclass(frozen=True)
class InteriorIslandGroup:
    index: int
    z_m: tuple[float, float]
    x_m: tuple[float, float]
    y_m: tuple[float, float]
    kind: str  # grid | plate | shell | block | bar | frame
    kinds: tuple[str, ...]
    # Grid member classes as (count, width_m, depth_m), largest count first; empty for a lone island.
    member_classes: tuple[tuple[int, float, float], ...]
    fill: float
    same_as: int | None = None
    dz_m: float | None = None

    @property
    def members(self) -> int:
        return sum(count for count, _, _ in self.member_classes) if self.member_classes else 1

    @property
    def centre_m(self) -> tuple[float, float]:
        return ((self.x_m[0] + self.x_m[1]) / 2.0, (self.y_m[0] + self.y_m[1]) / 2.0)

    @property
    def size_m(self) -> tuple[float, float]:
        return (self.x_m[1] - self.x_m[0], self.y_m[1] - self.y_m[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"g{self.index}",
            "z_m": [round(self.z_m[0], 4), round(self.z_m[1], 4)],
            "x_m": [round(self.x_m[0], 4), round(self.x_m[1], 4)],
            "y_m": [round(self.y_m[0], 4), round(self.y_m[1], 4)],
            "centre_m": [round(self.centre_m[0], 4), round(self.centre_m[1], 4)],
            "size_mm": [round(self.size_m[0] * 1000), round(self.size_m[1] * 1000)],
            "kind": self.kind,
            "kinds": list(self.kinds),
            "members": self.members,
            "member_classes": [
                {"count": count, "size_mm": [round(w * 1000), round(d * 1000)]}
                for count, w, d in self.member_classes
            ],
            "fill": round(self.fill, 2),
            "same_as": None if self.same_as is None else f"g{self.same_as}",
            "dz_m": None if self.dz_m is None else round(self.dz_m, 4),
        }


@dataclass(frozen=True)
class InteriorIslandSummary:
    groups: tuple[InteriorIslandGroup, ...]
    pitch_m: float
    detected: int = 0
    section_bytes: int = 0


@dataclass
class _Entity:
    x0: int
    x1: int
    y0: int
    y1: int
    cells: int
    kind: str
    classes: list[tuple[int, int, int]]  # (count, width_cells, depth_cells) per grid size class

    @property
    def bbox_cells(self) -> int:
        return max((self.x1 - self.x0) * (self.y1 - self.y0), 1)

    @property
    def members(self) -> int:
        return sum(count for count, _, _ in self.classes) if self.classes else 1


@dataclass
class _Group:
    entities: list[tuple[int, int, int, int, _Entity]]  # stack_z0, stack_z1, micro_z0, micro_z1

    @property
    def last_stack_z_end(self) -> int:
        return self.entities[-1][1]

    @property
    def last(self) -> _Entity:
        return self.entities[-1][4]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            min(e.x0 for *_, e in self.entities),
            max(e.x1 for *_, e in self.entities),
            min(e.y0 for *_, e in self.entities),
            max(e.y1 for *_, e in self.entities),
        )

    @property
    def score(self) -> int:
        x0, x1, y0, y1 = self.bbox
        return (x1 - x0) * (y1 - y0) * len(self.entities)


def _overlap_share(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Share of the smaller box covered by the other, both grown by the merge tolerance."""

    t = MERGE_TOLERANCE_CELLS
    a = (a[0] - t, a[1] + t, a[2] - t, a[3] + t)
    b = (b[0] - t, b[1] + t, b[2] - t, b[3] + t)
    inter_x = min(a[1], b[1]) - max(a[0], b[0])
    inter_y = min(a[3], b[3]) - max(a[2], b[2])
    if inter_x <= 0 or inter_y <= 0:
        return 0.0
    smaller = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
    return inter_x * inter_y / max(smaller, 1)


def _size_matches(w: int, h: int, ref_w: int, ref_h: int) -> bool:
    return abs(w - ref_w) <= max(SIZE_MATCH_MIN_CELLS, SIZE_MATCH_FRACTION * max(w, ref_w)) and abs(
        h - ref_h
    ) <= max(SIZE_MATCH_MIN_CELLS, SIZE_MATCH_FRACTION * max(h, ref_h))


def _is_bar(island: _IslandLike, pitch: float) -> bool:
    width = (island.x_end - island.x_start) * pitch
    depth = (island.y_end - island.y_start) * pitch
    short, long = min(width, depth), max(width, depth)
    return short <= BAR_MAX_SHORT_SIDE_M and long >= BAR_MIN_ASPECT * max(short, pitch)


def _single_kind(island: _IslandLike, pitch: float) -> str:
    if _is_bar(island, pitch):
        return "bar"
    fill = island.cell_count / max(island.bbox_cells, 1)
    if fill >= PLATE_FILL_MIN:
        return "plate"
    width = (island.x_end - island.x_start) * pitch
    depth = (island.y_end - island.y_start) * pitch
    if fill <= SHELL_FILL_MAX and min(width, depth) >= SHELL_MIN_SIDE_M:
        return "shell"
    return "block"


def _merge_entities(entities: list[_Entity]) -> list[_Entity]:
    """Merge entities whose footprints overlap (largest first) so one fitting is one entity.

    Two or more free bars in one slab are a frame (a rack's rails) whose rectangle then gathers the
    tines or plate it bounds.
    """

    bars = [e for e in entities if e.kind == "bar"]
    if len(bars) >= 2:
        frame = _Entity(
            x0=min(e.x0 for e in bars),
            x1=max(e.x1 for e in bars),
            y0=min(e.y0 for e in bars),
            y1=max(e.y1 for e in bars),
            cells=sum(e.cells for e in bars),
            kind="frame",
            classes=[],
        )
        entities = [e for e in entities if e.kind != "bar"] + [frame]
    merged: list[_Entity] = []
    for entity in sorted(entities, key=lambda e: (-e.bbox_cells, e.x0, e.y0)):
        box = (entity.x0, entity.x1, entity.y0, entity.y1)
        for target in merged:
            if (
                _overlap_share((target.x0, target.x1, target.y0, target.y1), box)
                >= OVERLAP_MIN_SHARE
            ):
                target.x0, target.x1 = min(target.x0, entity.x0), max(target.x1, entity.x1)
                target.y0, target.y1 = min(target.y0, entity.y0), max(target.y1, entity.y1)
                target.cells += entity.cells
                target.classes = sorted(target.classes + entity.classes, key=lambda c: -c[0])
                if entity.kind == "grid":
                    target.kind = "grid"
                break
        else:
            merged.append(entity)
    return merged


def _slab_entities(run: _RunLike, frame: _FrameLike) -> list[_Entity]:
    islands = tuple(run.islands)
    if len(islands) < 2:
        return []
    body = max(islands, key=lambda i: i.bbox_cells)
    footprint_cells = int(frame.dimensions[0]) * int(frame.dimensions[1])
    if body.bbox_cells < BODY_FOOTPRINT_MIN_SHARE * footprint_cells:
        return []
    margin = INTERIOR_MARGIN_CELLS
    wall_x = max(WALL_INSET_MAX_CELLS, WALL_INSET_MAX_SHARE * (body.x_end - body.x_start))
    wall_y = max(WALL_INSET_MAX_CELLS, WALL_INSET_MAX_SHARE * (body.y_end - body.y_start))
    nested = [
        island
        for island in islands
        if island is not body
        and island.x_start >= body.x_start + margin
        and island.x_end <= body.x_end - margin
        and island.y_start >= body.y_start + margin
        and island.y_end <= body.y_end - margin
        and not (
            island.x_start - body.x_start <= wall_x
            and body.x_end - island.x_end <= wall_x
            and island.y_start - body.y_start <= wall_y
            and body.y_end - island.y_end <= wall_y
        )
    ]
    if not nested:
        return []
    pitch = float(frame.pitch)
    clusters: list[list[_IslandLike]] = []
    for island in sorted(nested, key=lambda i: (-i.bbox_cells, i.x_start, i.y_start)):
        w, h = island.x_end - island.x_start, island.y_end - island.y_start
        for cluster in clusters:
            ref = cluster[0]
            if _size_matches(w, h, ref.x_end - ref.x_start, ref.y_end - ref.y_start):
                cluster.append(island)
                break
        else:
            clusters.append([island])
    entities: list[_Entity] = []
    min_single_cells = max(
        SINGLE_MIN_BODY_SHARE * body.bbox_cells, (SINGLE_MIN_SIDE_M / pitch) ** 2
    )
    for cluster in clusters:
        if len(cluster) >= GRID_MIN_MEMBERS:
            widths = sorted(i.x_end - i.x_start for i in cluster)
            heights = sorted(i.y_end - i.y_start for i in cluster)
            grid = _Entity(
                x0=min(i.x_start for i in cluster),
                x1=max(i.x_end for i in cluster),
                y0=min(i.y_start for i in cluster),
                y1=max(i.y_end for i in cluster),
                cells=sum(i.cell_count for i in cluster),
                kind="grid",
                classes=[(len(cluster), widths[len(widths) // 2], heights[len(heights) // 2])],
            )
            if (
                len(cluster) >= GRID_DENSE_MIN_MEMBERS
                or grid.cells / grid.bbox_cells >= GRID_MIN_FILL
            ):
                entities.append(grid)
            continue
        for island in cluster:
            spans_body = _is_bar(island, pitch) and (
                island.x_end - island.x_start >= BAR_MIN_SPAN_SHARE * (body.x_end - body.x_start)
                or island.y_end - island.y_start >= BAR_MIN_SPAN_SHARE * (body.y_end - body.y_start)
            )
            if island.bbox_cells < min_single_cells and not spans_body:
                continue
            entities.append(
                _Entity(
                    x0=island.x_start,
                    x1=island.x_end,
                    y0=island.y_start,
                    y1=island.y_end,
                    cells=island.cell_count,
                    kind=_single_kind(island, pitch),
                    classes=[],
                )
            )
    return _merge_entities(entities)


def _link_groups(runs: Sequence[_RunLike], frame: _FrameLike) -> list[_Group]:
    groups: list[_Group] = []
    open_groups: list[_Group] = []
    for run in sorted(runs, key=lambda r: r.stack_z_start):
        entities = _slab_entities(run, frame)
        still_open: list[_Group] = []
        taken: set[int] = set()
        for group in open_groups:
            if group.last_stack_z_end != run.stack_z_start:
                continue
            last = group.last
            best: tuple[float, int] | None = None
            for index, entity in enumerate(entities):
                if index in taken:
                    continue
                share = _overlap_share(
                    (last.x0, last.x1, last.y0, last.y1),
                    (entity.x0, entity.x1, entity.y0, entity.y1),
                )
                if share >= OVERLAP_MIN_SHARE and (best is None or share > best[0]):
                    best = (share, index)
            if best is not None:
                taken.add(best[1])
                group.entities.append(
                    (
                        run.stack_z_start,
                        run.stack_z_end,
                        run.micro_z_start,
                        run.micro_z_end,
                        entities[best[1]],
                    )
                )
                still_open.append(group)
        for index, entity in enumerate(entities):
            if index in taken:
                continue
            group = _Group(
                [(run.stack_z_start, run.stack_z_end, run.micro_z_start, run.micro_z_end, entity)]
            )
            groups.append(group)
            still_open.append(group)
        open_groups = still_open
    return groups


def _group_record(index: int, group: _Group, frame: _FrameLike) -> InteriorIslandGroup:
    origin = np.asarray(frame.bounds_min, dtype=float)
    pitch = float(frame.pitch)
    x0, x1, y0, y1 = group.bbox
    per_slab = [e.kind for *_, e in group.entities]
    kind = Counter(per_slab).most_common(1)[0][0]
    kinds = tuple(dict.fromkeys(per_slab))
    grid_entities = [e for *_, e in group.entities if e.kind == "grid"]
    classes: tuple[tuple[int, float, float], ...] = ()
    if grid_entities:
        # The slab with the most members describes the grid; a rack's frame bars and its upright
        # tines seen end-on both appear as their own size class.
        best = max(grid_entities, key=lambda e: (e.members, e.cells))
        classes = tuple(
            (count, w * pitch, d * pitch) for count, w, d in best.classes[:GRID_MAX_CLASSES]
        )
    cells = sum(e.cells for *_, e in group.entities)
    boxes = sum(e.bbox_cells for *_, e in group.entities)
    return InteriorIslandGroup(
        index=index,
        z_m=(
            float(origin[2] + group.entities[0][2] * pitch),
            float(origin[2] + group.entities[-1][3] * pitch),
        ),
        x_m=(float(origin[0] + x0 * pitch), float(origin[0] + x1 * pitch)),
        y_m=(float(origin[1] + y0 * pitch), float(origin[1] + y1 * pitch)),
        kind=kind,
        kinds=kinds,
        member_classes=classes,
        fill=min(1.0, cells / max(boxes, 1)),
    )


def _mark_repeats(groups: list[InteriorIslandGroup]) -> list[InteriorIslandGroup]:
    out: list[InteriorIslandGroup] = []
    for group in groups:
        same_as = None
        dz = None
        for earlier in out:
            # A repeated level stands above the earlier one; two fittings side by side in the same
            # slabs are neighbours, not repeats.
            if earlier.kind != group.kind or group.z_m[0] < earlier.z_m[1]:
                continue
            size_a, size_b = earlier.size_m, group.size_m
            if all(
                abs(a - b) <= REPEAT_SIZE_TOLERANCE * max(a, b, 1e-9)
                for a, b in zip(size_a, size_b, strict=True)
            ):
                same_as = earlier.same_as if earlier.same_as is not None else earlier.index
                dz = group.z_m[0] - next(g for g in out if g.index == same_as).z_m[0]
                break
        out.append(replace(group, same_as=same_as, dz_m=dz))
    return out


def detect_interior_islands(runs: Sequence[_RunLike], frame: _FrameLike) -> InteriorIslandSummary:
    """Group the islands standing free inside each slab's body outline across the selected stack."""

    pitch = float(frame.pitch)
    linked = _link_groups(runs, frame)
    detected = len(linked)
    if not linked:
        return InteriorIslandSummary((), pitch)
    ranked = sorted(linked, key=lambda g: (-g.score, g.entities[0][0], g.bbox))

    def _records(kept: Sequence[_Group]) -> tuple[InteriorIslandGroup, ...]:
        ordered = sorted(kept, key=lambda g: (g.entities[0][2], g.bbox))
        return tuple(_mark_repeats([_group_record(i + 1, g, frame) for i, g in enumerate(ordered)]))

    kept = ranked[:INTERIOR_MAX_GROUPS]
    while True:
        groups = _records(kept)
        lines = format_interior_islands(InteriorIslandSummary(groups, pitch, detected))
        section_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines)
        group_bytes = sum(len(line.encode("utf-8")) + 1 for line in lines[FIXED_LINES:])
        if group_bytes <= INTERIOR_ISLANDS_BYTE_BUDGET or len(kept) <= 1:
            return InteriorIslandSummary(groups, pitch, detected, section_bytes)
        kept = kept[:-1]


FIXED_LINES = 3  # header, semantics, coords


def _fill_text(value: float) -> str:
    text = f"{value:.2f}"
    return text[1:] if text.startswith("0") else text


def format_interior_islands(summary: InteriorIslandSummary) -> tuple[str, ...]:
    """Serialize the section; empty when no nested island group exists (absence proves nothing)."""

    if not summary.groups:
        return ()
    lines = [
        (
            f"[INTERIOR_ISLANDS parent=MICRO_SLICE_BOUNDS_Z192 shown={len(summary.groups)} "
            f"detected={summary.detected} margin={INTERIOR_MARGIN_CELLS}cell]"
        ),
        (
            "semantics: sampled-surface islands lying strictly inside the slab body's outline at each "
            "height, grouped across slabs; Host HYPOTHESIS for fittings standing free of the walls "
            "inside a shell (racks, shelves, trays, tanks, baskets); anonymous, nonsemantic, NOT "
            "Parts; interiors beyond sampled surfaces stay UNKNOWN; a fitting touching the walls or "
            "resting on the floor of its slab merges with the body and is NOT listed — absence "
            "proves nothing"
        ),
        (
            "coords: world metres q=0; z[]=slabs the group spans (extent is a lower bound at slab "
            "pitch); x[] y[]=union footprint; kind=grid(n members, members=count@size per size "
            "class)|plate|shell(hollow outline)|block|bar(one free wire)|frame(rails bounding the "
            "footprint); f=sampled/bbox cells; "
            "same_as=a group with the same footprint at another height (a repeated level), dz=level "
            "spacing"
        ),
    ]
    for group in summary.groups:
        centre = group.centre_m
        size = group.size_m
        text = (
            f"g{group.index} z[{group.z_m[0]:+.3f},{group.z_m[1]:+.3f}] "
            f"x[{group.x_m[0]:+.3f},{group.x_m[1]:+.3f}] y[{group.y_m[0]:+.3f},{group.y_m[1]:+.3f}] "
            f"centre({centre[0]:+.3f},{centre[1]:+.3f}) size={size[0] * 1000:.0f}x{size[1] * 1000:.0f}mm "
            f"kind={group.kind}"
        )
        if group.member_classes:
            text += (
                f" n={group.members} members="
                + "+".join(
                    f"{count}@{w * 1000:.0f}x{d * 1000:.0f}" for count, w, d in group.member_classes
                )
                + "mm"
            )
        text += f" f={_fill_text(group.fill)}"
        if len(group.kinds) > 1:
            text += " kinds=" + "+".join(group.kinds)
        if group.same_as is not None and group.dz_m is not None:
            text += f" same_as=g{group.same_as} dz={group.dz_m:+.3f}"
        lines.append(text)
    return tuple(lines)


_GROUP_LINE = re.compile(
    r"^g(?P<index>\d+) z\[(?P<z0>[+-][0-9.]+),(?P<z1>[+-][0-9.]+)\] "
    r"x\[(?P<x0>[+-][0-9.]+),(?P<x1>[+-][0-9.]+)\] y\[(?P<y0>[+-][0-9.]+),(?P<y1>[+-][0-9.]+)\] "
    r"centre\([+-][0-9.]+,[+-][0-9.]+\) size=\d+x\d+mm "
    r"kind=(?P<kind>grid|plate|shell|block|bar|frame)"
    r"(?: n=\d+ members=(?P<members>[0-9@x+]+)mm)? f=(?P<fill>[0-9.]+)"
    r"(?: kinds=(?P<kinds>[a-z+]+))?(?: same_as=g(?P<same>\d+) dz=(?P<dz>[+-][0-9.]+))?$"
)


def parse_interior_islands(text: str) -> tuple[InteriorIslandGroup, ...]:
    """Read serialized INTERIOR_ISLANDS group lines back into groups."""

    out: list[InteriorIslandGroup] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("[INTERIOR_ISLANDS"):
            inside = True
            continue
        if inside and line.startswith("["):
            break
        if not inside:
            continue
        match = _GROUP_LINE.match(line.strip())
        if match is None:
            continue
        kind = match["kind"]
        kinds = tuple(match["kinds"].split("+")) if match["kinds"] else (kind,)
        classes: list[tuple[int, float, float]] = []
        for token in (match["members"] or "").split("+"):
            if "@" not in token:
                continue
            count, _, size = token.partition("@")
            w, _, d = size.partition("x")
            classes.append((int(count), int(w) / 1000.0, int(d) / 1000.0))
        out.append(
            InteriorIslandGroup(
                index=int(match["index"]),
                z_m=(float(match["z0"]), float(match["z1"])),
                x_m=(float(match["x0"]), float(match["x1"])),
                y_m=(float(match["y0"]), float(match["y1"])),
                kind=kind,
                kinds=kinds,
                member_classes=tuple(classes),
                fill=float(match["fill"]),
                same_as=int(match["same"]) if match["same"] else None,
                dz_m=float(match["dz"]) if match["dz"] else None,
            )
        )
    return tuple(out)


__all__ = [
    "INTERIOR_ISLANDS_BYTE_BUDGET",
    "INTERIOR_MAX_GROUPS",
    "InteriorIslandGroup",
    "InteriorIslandSummary",
    "detect_interior_islands",
    "format_interior_islands",
    "parse_interior_islands",
]
