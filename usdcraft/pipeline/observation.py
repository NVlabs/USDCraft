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

"""Shared image-observation engine: one clean-context semantic-inventory Agent per run.

Both products run the same observation stage: one fresh-context vision call reads the run's
declared primary image (plus any declared auxiliary views) and returns one schema-validated
semantic inventory of the primary subject. The inventory is an auditable external first reading
for the later Author, never authority: it adds no visibility, proves no absence, and the primary
image wins every conflict. A failed or blocked observation degrades the run to the previous
Author-only reading with a warning; it never blocks a run and never enters an approval
fingerprint.

This module owns everything product-independent: the result schema, deterministic payload
validation, prompt rendering, atomic artifact publishing, and the Codex transport. Product
adapters (`usdcraft/reconstruction/observation.py`, `usdcraft/generation/observation.py`) own the
task wording, the image set, the scope strings, and any non-Codex transport.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence, TypedDict, cast

from usdcraft.pipeline.codex import CLEAN_CONTEXT_DISABLED_CODEX_FEATURES
from usdcraft.values import normalize_reasoning_level

if TYPE_CHECKING:
    from collections.abc import Sized

    from usdcraft.pipeline.provider.client import NvidiaProviderConfig

# Runtime/transport dependencies are loaded only when an observation execution needs them.

OBSERVATION_MAX_ATTEMPTS = 2
OBSERVATION_DISABLED_CODEX_FEATURES: tuple[str, ...] = CLEAN_CONTEXT_DISABLED_CODEX_FEATURES

_PART_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PART_KEYS = {
    "name",
    "visible_count",
    "mobility",
    "level",
    "visibility",
    "geometry",
    "arrangement",
    "affordances",
    "attaches_to",
    "duplicate_of",
    "mirror_of",
    "carried_as",
    "notes",
}
_MOBILITY_VALUES = {"fixed", "movable", "uncertain"}
# Carried contents (goods on shelves, items in racks, food in a compartment) are subject Parts the
# inventory records by kind; `carried_as` says where they sit so the Author can decide what to
# build: `displayed` on open shelving/racks/hooks/top surfaces (part of the outward form),
# `enclosed` inside a compartment behind a door, lid, glass or panel. Every other entry — the
# subject's own body, panels, controls, shelves and hardware — is `none`.
_CARRIED_AS_VALUES = {"none", "displayed", "enclosed"}
_LEVEL_VALUES = {"primary_body", "major_part", "sub_feature", "micro_feature"}
# "inferred" marks a structurally required entry no image shows (a review-added connecting
# member); it is a hypothesis for the Author, never a visibility claim.
_VISIBILITY_VALUES = {"visible", "partially_occluded", "inferred"}

OBSERVATION_AGENT_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["complete", "blocked"]},
        "parts": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # The snake_case pattern is enforced at generation time by the transport
                    # schema so a name-format slip never costs a full vision retry; the
                    # deterministic validator below re-checks it as defense in depth.
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "pattern": _PART_NAME_PATTERN.pattern,
                    },
                    "visible_count": {"type": "integer", "minimum": 1, "maximum": 999},
                    "mobility": {"type": "string", "enum": sorted(_MOBILITY_VALUES)},
                    "level": {"type": "string", "enum": sorted(_LEVEL_VALUES)},
                    "visibility": {"type": "string", "enum": sorted(_VISIBILITY_VALUES)},
                    "geometry": {"type": "string", "minLength": 1, "maxLength": 400},
                    "arrangement": {"type": "string", "minLength": 1, "maxLength": 400},
                    "affordances": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                    # Categorical attachment: the EARLIER entry this part physically meets.
                    # Null only for the primary/backbone entry and genuinely free-standing
                    # ground-supported members. Never coordinates, offsets, or numbers.
                    "attaches_to": {
                        "type": ["string", "null"],
                        "maxLength": 64,
                        "pattern": _PART_NAME_PATTERN.pattern,
                    },
                    "duplicate_of": {
                        "type": ["string", "null"],
                        "maxLength": 64,
                        "pattern": _PART_NAME_PATTERN.pattern,
                    },
                    "mirror_of": {
                        "type": ["string", "null"],
                        "maxLength": 64,
                        "pattern": _PART_NAME_PATTERN.pattern,
                    },
                    "carried_as": {"type": "string", "enum": sorted(_CARRIED_AS_VALUES)},
                    "notes": {"type": ["string", "null"], "maxLength": 400},
                },
                "required": sorted(_PART_KEYS),
            },
        },
        # Explicit named removals with reasons: the review call's only removal channel. The
        # first observation call leaves this empty; omitting an entry from `parts` is never a
        # removal.
        "removals": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "pattern": _PART_NAME_PATTERN.pattern,
                    },
                    "reason": {"type": "string", "minLength": 1, "maxLength": 400},
                },
                "required": ["name", "reason"],
            },
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 1600},
    },
    "required": ["status", "parts", "removals", "summary"],
}


class ImageObservationError(RuntimeError):
    """The observation Agent failed; the caller degrades to the Author-only reading."""


@dataclass(slots=True, frozen=True)
class ObservationRuntimeSelection:
    model_id: str
    reasoning_effort: str
    runtime_source: str
    selection_source: str


@dataclass(slots=True, frozen=True)
class ObservationProcessResult:
    returncode: int
    stdout: str
    stderr: str


class ObservationProcessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        stdin_text: str,
        cwd: Path,
        timeout_seconds: float | None,
        env: Mapping[str, str],
    ) -> ObservationProcessResult: ...


@dataclass(slots=True, frozen=True)
class ImageObservationArtifacts:
    root: Path
    inventory_path: Path
    manifest_path: Path
    inventory: dict[str, object]
    attempts: int


def _normalized_nonempty(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def resolve_observation_runtime(
    *,
    model: str | None,
    reasoning_effort: str | None,
    product: str,
    environ: Mapping[str, str] | None = None,
) -> ObservationRuntimeSelection:
    """Resolve the one observation subagent runtime without adopting the Author's semantics."""

    if (model is None) != (reasoning_effort is None):
        raise ValueError("--model and --reasoning must be provided together")
    if model is not None and reasoning_effort is not None:
        return ObservationRuntimeSelection(
            model_id=_normalized_nonempty(model, label="Codex model"),
            reasoning_effort=normalize_reasoning_level(
                _normalized_nonempty(reasoning_effort, label="Codex reasoning effort")
            ),
            runtime_source=f"explicit_{product}_observation_request",
            selection_source=f"explicit_{product}_observation_request",
        )
    from usdcraft.pipeline.runtime import current_main_codex_runtime

    inherited = current_main_codex_runtime(environ=environ)
    return ObservationRuntimeSelection(
        model_id=inherited.model_id,
        reasoning_effort=inherited.reasoning_effort,
        runtime_source=inherited.runtime_source,
        selection_source=f"inherited_root_for_{product}_observation",
    )


def validate_observation_payload(payload: object) -> dict[str, object]:
    """Deterministically validate one observation Agent response payload."""

    if not isinstance(payload, dict) or set(payload) != {"status", "parts", "removals", "summary"}:
        raise ImageObservationError("observation Agent response has unexpected fields")
    if payload["status"] not in {"complete", "blocked"}:
        raise ImageObservationError("observation Agent status is invalid")
    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 1600:
        raise ImageObservationError("observation Agent summary is missing or oversized")
    removals = payload["removals"]
    if not isinstance(removals, list) or len(removals) > 8:
        raise ImageObservationError("removals must be an array of at most 8 entries")
    for index, removal in enumerate(removals):
        if not isinstance(removal, dict) or set(removal) != {"name", "reason"}:
            raise ImageObservationError(f"removals[{index}] has unexpected fields")
        if not isinstance(removal["name"], str) or not _PART_NAME_PATTERN.fullmatch(
            removal["name"]
        ):
            raise ImageObservationError(f"removals[{index}] name must be stable snake_case")
        reason = removal["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 400:
            raise ImageObservationError(f"removals[{index}] reason must be a short string")
    parts = payload["parts"]
    if not isinstance(parts, list) or len(parts) > 64:
        raise ImageObservationError("parts must be an array of at most 64 entries")
    if payload["status"] == "blocked":
        if parts:
            raise ImageObservationError("a blocked observation must list no parts")
        raise ImageObservationError(f"observation Agent is blocked: {summary.strip()}")
    if not parts:
        raise ImageObservationError("a complete observation must list at least one Part")
    names: set[str] = set()
    for index, part in enumerate(parts):
        label = f"parts[{index}]"
        if not isinstance(part, dict) or set(part) != _PART_KEYS:
            raise ImageObservationError(f"{label} has unexpected fields")
        name = part["name"]
        if not isinstance(name, str) or not _PART_NAME_PATTERN.fullmatch(name):
            raise ImageObservationError(f"{label} name must be stable snake_case")
        if name in names:
            raise ImageObservationError(f"{label} duplicates part name {name!r}")
        names.add(name)
        count = part["visible_count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 999:
            raise ImageObservationError(f"{label} visible_count must be an integer from 1 to 999")
        if part["mobility"] not in _MOBILITY_VALUES:
            raise ImageObservationError(f"{label} mobility is invalid")
        if part["level"] not in _LEVEL_VALUES:
            raise ImageObservationError(f"{label} level is invalid")
        if part["visibility"] not in _VISIBILITY_VALUES:
            raise ImageObservationError(f"{label} visibility is invalid")
        if part["carried_as"] not in _CARRIED_AS_VALUES:
            raise ImageObservationError(f"{label} carried_as is invalid")
        geometry = part["geometry"]
        if not isinstance(geometry, str) or not geometry.strip() or len(geometry) > 400:
            raise ImageObservationError(f"{label} geometry must be a short non-empty phrase")
        arrangement = part["arrangement"]
        if not isinstance(arrangement, str) or not arrangement.strip() or len(arrangement) > 400:
            raise ImageObservationError(f"{label} arrangement must be a short non-empty phrase")
        affordances = part["affordances"]
        if not isinstance(affordances, list) or len(affordances) > 8:
            raise ImageObservationError(f"{label} affordances must be an array of at most 8 items")
        for item in affordances:
            if not isinstance(item, str) or not item.strip() or len(item) > 200:
                raise ImageObservationError(f"{label} affordances must be short strings")
        for reference_key in ("attaches_to", "duplicate_of", "mirror_of"):
            reference = part[reference_key]
            if reference is None:
                continue
            if not isinstance(reference, str) or reference == name:
                raise ImageObservationError(f"{label} {reference_key} is invalid")
        attaches_to = part["attaches_to"]
        if attaches_to is not None and attaches_to not in names - {name}:
            # `names` currently holds this entry and every earlier one, so this enforces the
            # build-order contract deterministically: attachments point strictly backwards.
            raise ImageObservationError(
                f"{label} attaches_to must name an earlier entry (order is the build sequence)"
            )
        note = part["notes"]
        if note is not None and (not isinstance(note, str) or len(note) > 400):
            raise ImageObservationError(f"{label} notes must be null or a short string")
    for index, part in enumerate(parts):
        for reference_key in ("duplicate_of", "mirror_of"):
            reference = part[reference_key]
            if reference is not None and reference not in names:
                raise ImageObservationError(
                    f"parts[{index}] {reference_key} references unknown part {reference!r}"
                )
    return payload


def parse_observation_result(path: Path) -> dict[str, object]:
    """Read and deterministically validate one observation Agent final-response file."""

    if not path.is_file():
        raise ImageObservationError("observation Agent did not write its final response")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageObservationError("observation Agent response is invalid JSON") from exc
    return validate_observation_payload(payload)


_SHARPEN_PART_KEYS = ("geometry", "arrangement", "affordances", "notes", "level", "carried_as")
_FROZEN_PART_KEYS = ("mobility", "visibility", "attaches_to", "duplicate_of", "mirror_of")


def merge_reviewed_inventory(
    base: Mapping[str, object],
    reviewed: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Deterministically merge one review response into the base inventory.

    Additions and descriptive sharpening are accepted, and counts may only grow (finding more of
    a visible repeat is an addition); identity-bearing fields stay frozen. Removal has exactly one
    channel: an explicit `removals` entry naming the part with a concrete reason. It is honored
    only when the reviewer did not simultaneously keep the entry in `parts` and no surviving entry
    references the removed name. Dropping an entry by omission is rejected (the original is
    restored) and a round that attempts any omission-drop also loses its additions, because those
    are usually merged replacements that would double-count structures. When the merged result
    fails deterministic validation the base inventory is returned unchanged.
    """

    base_parts = list(cast("list[dict[str, Any]]", base["parts"]))
    reviewed_parts = list(cast("list[dict[str, Any]]", reviewed["parts"]))
    base_names = {part["name"] for part in base_parts}
    review_by_name = {part["name"]: part for part in reviewed_parts}

    requested_removals = {
        entry["name"]: entry["reason"]
        for entry in cast("list[dict[str, Any]]", reviewed.get("removals", []))
    }
    rejected_removals: list[str] = []
    # A removal contradicted by the reviewer's own kept entry, or naming an unknown part, is
    # rejected outright; reference safety is checked after the surviving set is known.
    candidate_removals = set()
    for name in requested_removals:
        if name in review_by_name or name not in base_names:
            rejected_removals.append(name)
        else:
            candidate_removals.add(name)
    surviving_names = {
        part["name"] for part in base_parts if part["name"] not in candidate_removals
    }
    surviving_names.update(
        part["name"] for part in reviewed_parts if part["name"] not in base_names
    )
    removed: list[str] = []
    for name in sorted(candidate_removals):
        referenced = any(
            part.get(key) == name
            for part in [*base_parts, *reviewed_parts]
            if part["name"] in surviving_names
            for key in ("attaches_to", "duplicate_of", "mirror_of")
        )
        if referenced:
            rejected_removals.append(name)
            candidate_removals.discard(name)
        else:
            removed.append(name)

    sharpened: list[str] = []
    rejected_changes: list[str] = []
    merged_parts: list[dict[str, object]] = []
    for part in base_parts:
        if part["name"] in candidate_removals:
            continue
        review = review_by_name.get(part["name"])
        if review is None:
            merged_parts.append(dict(part))
            continue
        merged = dict(part)
        changed = False
        for key in _SHARPEN_PART_KEYS:
            if review[key] != part[key]:
                merged[key] = review[key]
                changed = True
        review_count = review["visible_count"]
        if isinstance(review_count, int) and review_count > part["visible_count"]:
            merged["visible_count"] = review_count
            changed = True
        elif review_count != part["visible_count"]:
            rejected_changes.append(f"{part['name']}.visible_count")
        for key in _FROZEN_PART_KEYS:
            if review[key] != part[key]:
                rejected_changes.append(f"{part['name']}.{key}")
        if changed:
            sharpened.append(str(part["name"]))
        merged_parts.append(merged)

    rejected_drops = [
        str(part["name"])
        for part in base_parts
        if part["name"] not in review_by_name and part["name"] not in removed
    ]
    added: list[str] = []
    if not rejected_drops:
        for part in reviewed_parts:
            name = part["name"]
            if name in base_names or any(item["name"] == name for item in merged_parts):
                continue
            merged_parts.append(dict(part))
            added.append(str(name))

    merged_payload: dict[str, object] = {
        "status": "complete",
        "parts": merged_parts,
        "removals": [],
        "summary": base["summary"],
    }
    report: dict[str, object] = {
        "status": "applied" if added or sharpened or removed else "unchanged",
        "added": added,
        "sharpened": sharpened,
        "removed": removed,
        "removal_reasons": {name: requested_removals[name] for name in removed},
        "rejected_removals": sorted(rejected_removals),
        "rejected_drops": rejected_drops,
        "rejected_changes": rejected_changes,
    }
    try:
        validate_observation_payload(json.loads(json.dumps(merged_payload)))
    except ImageObservationError as exc:
        return dict(base), {
            **report,
            "status": "merge_invalid_base_kept",
            "merge_error": str(exc),
        }
    return merged_payload, report


@dataclass(slots=True, frozen=True)
class ObservationTaskProfile:
    """Product-owned semantics for the shared observation task text.

    The engine owns the inventory-contract wording shared by every observation reading; a product
    supplies only what genuinely differs: who the agent is, what evidence arrives with what
    authority, which downstream stage owns dimension numbers, and how auxiliary renders are
    handled. Sharing the contract body does not merge product semantics — request authority,
    evidence authority, and provenance stay entirely inside the product-supplied fields.
    """

    task_preamble: str
    review_preamble: str
    evidence_bullets: tuple[str, ...]
    review_evidence_bullets: tuple[str, ...]
    prohibitions_paragraph: str
    views: str
    views_show: str
    views_plainly_show: str
    dimension_authority_clause: str
    completion_trailing_sentence: str
    review_category_knowledge_bullet: str
    blocked_subject: str


def build_observation_task(profile: ObservationTaskProfile) -> str:
    """Render the complete first-reading task from the shared contract body and one profile."""

    # Concise fields keep latency down while preserving uncertainty and visible controls; the
    # task sets no fixed part cap because complex subjects need more entries.
    evidence = "\n".join(profile.evidence_bullets)
    return f"""{profile.task_preamble}

ALLOWED EVIDENCE
{evidence}
- This task and the response schema are the only other inputs.

{profile.prohibitions_paragraph}

READING BUDGET
Inspect the attached evidence in one pass and return the final structured inventory. The Host
handles result files; this image-only task needs no shell, filesystem or network calls.
Keep each geometry, arrangement and notes field to a short phrase carrying distinct information;
omit redundant affordances. Preserve small controls, openings, pose cues and explicit uncertainty.
Use `uncertain` when the image does not establish mobility, and retain the reason for each
`inferred` completion. Group repeated details with `visible_count`. Once the inventory is
consistent, submit it without iterating on prose or starting another review.

INVENTORY CONTRACT
- List as `visible` or `partially_occluded` only Parts of the primary subject that are positively
  visible in attached image 1. Crop, occlusion, backside invisibility, ambiguity or
  non-observation never establishes absence, and an unseen part enters the inventory only through
  the bounded completion rules below — never as a `visible` entry. Incidental scene dressing,
  backgrounds and props are not subject Parts.
- Contents the subject visibly carries are subject Parts, not scene dressing: goods stocked on
  its shelves, items held in its racks, trays or hooks, food in its compartments. Inventory them
  by visually distinct kind — one entry per kind with `visible_count` for the repeats, not one
  entry per instance — and classify each with `carried_as`: `displayed` when it sits on open
  shelving, a rack, a hook, a rail or a top surface and so forms part of the subject's outward
  appearance; `enclosed` when it sits inside a compartment, cavity or drawer behind a door, lid,
  glass or panel and is visible only because that closure stands open or is transparent. Every
  other entry — the subject's own body, panels, doors, shelves, controls and hardware — is
  `carried_as: none`. Only freestanding surroundings (floor, walls, nearby furniture, backdrops)
  are incidental scene dressing. Whether carried contents get modeled is a later decision made
  with the user's text and this classification; your job is to record them.
- Printed labels, stickers, logos, dial faces, painted stripes and other surface graphics are the
  appearance of the Part they sit on, never Parts of their own: describe them in that Part's
  `notes` (what they show, their colours, where on the surface they sit) and do not list them as
  entries. Only a physically separate applied object — a hang tag, a clipped card, a strapped
  band — is a Part.
- Every entry needs a stable `snake_case` name, the visible count, fixed/movable/uncertain
  mobility, a granularity level (`primary_body` for the main body, `major_part`, `sub_feature`,
  `micro_feature`), `visible` or `partially_occluded` visibility, a `geometry` phrase, a coarse
  arrangement phrase, an `attaches_to` declaration, and the separately usable or graspable
  affordances. Mark ambiguity and occlusion explicitly in `notes` instead of guessing.
- `attaches_to` names the earlier entry this part physically meets — the part it is mounted on,
  rests on, hinges on, or is sleeved into; state the interface kind (rests on, bolted to, hinged
  on, slides in) and where on that part the attachment sits, as a categorical position (top-rear
  corner, high and outboard, centre of the front edge), in `arrangement` or `notes`. The
  declaration is categorical: never coordinates, offsets, or numbers of any kind. Only the
  `primary_body` entry, and a genuinely free-standing member supported by the ground rather than
  by the subject, uses null.
- Order the inventory with the `primary_body`/backbone first, then outward, so every entry
  references an earlier entry. This order keeps references unambiguous; the later Author owns
  construction order and may revise the proposed grouping and attachments.
- `geometry` records what {profile.views_show} about the entry's form, in categorical terms only:
  its concrete shape character (boxy, cylindrical, domed, bell-shaped, tapered, flared, and where
  the silhouette changes), its coarse aspect (long-and-thin, squat, flat, broad, ...), and its
  pose or orientation, always with the lean direction stated relative to the subject's own front
  and back (upright, tilted rearward away from the head, splayed outboard, axle across the
  subject). Read these cues from
  {profile.views} and be specific; a generic phrase helps nobody. Never state a size, proportion,
  ratio, or any other number: {profile.dimension_authority_clause} For the `primary_body` entry,
  describe the overall silhouette in the same categorical terms (wider than tall, low and squat,
  upright and slender) and, when the subject is articulated or multi-stage, the overall assembly
  silhouette its pose forms (a compressed Z, a forward cantilever, an arch). A
  `duplicate_of`/`mirror_of` entry describes only its own pose and
  placement in `geometry`; the shape lives on the referenced entry.
- Visible mechanism evidence is part of what {profile.views_show}: a hinge line and which side it
  sits on, slide rails, a gap that reveals the opening direction, a visible pivot or latch.
  Record it in `geometry`, `arrangement`, or `notes` as the visible observation it is, without
  concluding a joint type or motion range.
- Keep visibly distinct constructions distinct: a separately graspable or operable affordance and
  every visibly distinct physical construction or support piece is its own entry even when rigidly
  fixed. Appearance-only texture boundaries, painted markings and decorative seams are not Parts.
- When one entry is a same-shape repeat or a mirrored twin of another, set `duplicate_of` or
  `mirror_of` to that entry's exact name instead of describing the shape twice.
- Name each part with its category's standard term (carcass, plinth, turntable, crown, apron);
  use positional prefixes only to distinguish repeats (upper_left_...). Standard names keep the
  inventory stable across readings and let downstream stages refer to the same part.
- Use only the distinct entries the evidence supports; there is no target part count.
  Keep micro-detail inside its parent entry's `geometry`/`notes` instead of new entries
  unless it is separately graspable or operable.
- The `removals` field belongs to the later review pass: leave it empty.

BOUNDED COMPLETION
After the visible inventory, complete the parts the subject needs but {profile.views} cannot
show. Each completion entry sets `visibility` to `inferred` and is exactly one of these two kinds:
- the unseen counterpart of a visible part: set `mirror_of` or `duplicate_of` to the visible
  entry's exact name, and describe only its own pose and placement in `geometry` — the shape
  lives on the visible entry;
- a support or closure structure the subject cannot stand, hold together, or close without: a
  rear panel, a bottom board, the occluded leg of a set — with the structural reason in `notes`.
Every completion entry declares `attaches_to` like any other entry and obeys the build-sequence
ordering.
The inventory covers only surfaces that are visible or become visible in normal use (an opened
drawer's box, the inside of a door that opens). Never list internal machinery, wiring, or
components enclosed behind panels that normal use does not open. Complete only what the subject
genuinely needs, never at `micro_feature` granularity, and never invent decorative or merely
plausible detail.{profile.completion_trailing_sentence}

OUTPUT
Return only the schema-conforming JSON response. Use `status: complete` with the inventory and a
short summary. Use `status: blocked` only when {profile.blocked_subject} itself cannot be read or
shows no identifiable primary subject; a blocked result must list no parts.
"""


def build_observation_review_task(
    profile: ObservationTaskProfile,
    *,
    inventory_text: str,
) -> str:
    """Render the add-and-sharpen review task from the shared contract body and one profile."""

    review_evidence = "\n".join(profile.review_evidence_bullets)
    return f"""{profile.review_preamble} A deterministic merge enforces your allowed changes:
renames are rejected, dropping an entry by omission is rejected with the original restored — and
a response that drops any entry by omission also loses its additions. Removing an entry has
exactly one channel, the explicit `removals` field.

Review the attached evidence once, preserve precise fields verbatim, and keep additions or
corrections concise. Preserve small controls, pose cues and uncertainty; there is no target
part count. Submit the full result when consistent, without another review or prose polishing.

ALLOWED EVIDENCE
{review_evidence}
- This task, the inventory below, and the response schema are the only other inputs. Do not
  inspect any other filesystem path, delegate, or start another agent.

YOU MAY
- ADD an entry for a positively visible primary-subject Part that no existing entry covers.
  Positive visibility in attached image 1 is required; category expectation, occlusion or
  non-observation never justifies an addition. This is your main job: find what is MISSING. Every
  added entry declares `attaches_to` naming an entry that precedes it in the returned list
  (categorical, never numbers).
- ADD entries for visibly carried or stocked contents of the subject (goods on shelves, items in
  racks, food in a compartment) that no entry covers, one entry per visually distinct kind with
  the visible count and its `carried_as` classification (`displayed` on open shelving, racks,
  hooks or top surfaces; `enclosed` behind a door, lid, glass or panel).
- ADD a missing bounded-completion entry with `visibility: inferred`, of exactly these kinds:
  the unseen counterpart of a visible part (`mirror_of`/`duplicate_of` set to the visible entry,
  `geometry` describing only its own pose and placement); a support or closure structure the
  subject cannot stand, hold together, or close without (rear panel, bottom board, occluded leg);
  or a connecting member the listed parts cannot operate without (an axle between a wheel and its
  body, a bracket, a hinge pin), with the structural reason in `notes`. The inventory covers only
  surfaces that are visible or become visible in normal use: never add internal machinery,
  wiring, or components enclosed behind panels that normal use does not open, never add at
  `micro_feature` granularity, and never add decorative or merely plausible detail.
- REMOVE an entry only by naming it in `removals` with a concrete reason, and only when
  {profile.views_plainly_show} the subject has no such part — be very sure. Hallucinated,
  double-counted, scene-dressing, or surface-graphic entries (a printed label or logo listed as a
  Part) are the target. Keeping the same name in `parts`
  cancels its removal, and omitting an entry from `parts` is never a removal.
- RAISE a `visible_count` that undercounts clearly distinguishable visible repeats.
- SHARPEN `geometry`, `arrangement`, `affordances`, `notes`, or correct the granularity `level`
  or the `carried_as` classification of an existing entry. A sharpened `geometry` adds the specific shape-character, aspect, or pose
  cue {profile.views_show}, in categorical terms only — never a size, proportion, ratio, or any
  other number; the later Author owns every dimension reading. Keep already-precise fields
  verbatim.

YOU MAY NOT
- merge or rename entries; existing `name` values are frozen.
- drop an entry by omitting it from `parts`; omission is restored and forfeits your additions.
- lower any `visible_count`, or change `mobility`, `visibility`, `attaches_to`,
  `duplicate_of` or `mirror_of`.
{profile.review_category_knowledge_bullet}

OUTPUT
Return only the schema-conforming JSON response containing the FULL inventory: every original
entry with its exact name, unchanged fields byte-for-byte, plus your allowed changes. When the
inventory is already complete, return it unchanged. Never use `status: blocked`.

=== FIRST INVENTORY (JSON) ===
{inventory_text}
"""


def render_inventory_for_prompt(payload: Mapping[str, object]) -> str:
    """Serialize one validated inventory as the compact text block an Author task embeds."""

    lines: list[str] = []
    for part in cast("list[dict[str, Any]]", payload["parts"]):
        header = (
            f"- {part['name']}: count={part['visible_count']}, {part['mobility']},"
            f" {part['level']}, {part['visibility']}"
        )
        if part["attaches_to"]:
            header += f", attaches to {part['attaches_to']}"
        if part["duplicate_of"]:
            header += f", duplicate of {part['duplicate_of']}"
        if part["mirror_of"]:
            header += f", mirror of {part['mirror_of']}"
        if part["carried_as"] != "none":
            header += f", {part['carried_as']} content"
        lines.append(header)
        lines.append(f"  geometry: {part['geometry']}")
        lines.append(f"  arrangement: {part['arrangement']}")
        affordances = part["affordances"]
        if affordances:
            lines.append("  affordances: " + "; ".join(affordances))
        if part["notes"]:
            lines.append(f"  notes: {part['notes']}")
    return "\n".join(lines)


def publish_observation_artifacts(
    *,
    staging: Path,
    output: Path,
    payload: dict[str, object],
    manifest_schema: str,
    scope: str,
    agent_context: str,
    primary_image_path: Path,
    runtime: ObservationRuntimeSelection,
    attempts: int,
    started_monotonic: float,
    extra_manifest: Mapping[str, object] | None = None,
    review_report: Mapping[str, object] | None = None,
) -> ImageObservationArtifacts:
    """Write the validated inventory plus manifest into staging and atomically publish it."""

    from usdcraft.pipeline.hashing import sha256_file as _sha256

    inventory_path = staging / "observation_inventory.json"
    inventory_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": manifest_schema,
        "status": "ready",
        "scope": scope,
        "canonical_record": False,
        "agent_run": True,
        "agent_context": agent_context,
        "authority": "external_first_reading_not_authority",
        "primary_image_sha256": _sha256(primary_image_path),
        "inventory_sha256": _sha256(inventory_path),
        "attempts": attempts,
        "runtime": {
            "model": runtime.model_id,
            "reasoning_effort": runtime.reasoning_effort,
            "runtime_source": runtime.runtime_source,
            "selection_source": runtime.selection_source,
        },
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "part_count": len(cast("Sized", payload["parts"])),
        **({"review": dict(review_report)} if review_report is not None else {}),
        **dict(extra_manifest or {}),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    return ImageObservationArtifacts(
        root=output,
        inventory_path=output / "observation_inventory.json",
        manifest_path=output / "manifest.json",
        inventory=payload,
        attempts=attempts,
    )


def build_observation_codex_command(
    runtime: ObservationRuntimeSelection,
    workspace: Path,
    *,
    executable: str = "codex",
) -> tuple[str, ...]:
    schema = workspace / ".codex-run" / "final.schema.json"
    final = workspace / ".codex-run" / "final.json"
    attached = sorted(workspace.glob("input_*.png")) + sorted(workspace.glob("input_*.jpg"))
    primary = [path for path in attached if path.name.startswith("input_1_")]
    if not schema.is_file() or not primary:
        raise ImageObservationError("observation Agent workspace is missing its staged inputs")
    from usdcraft.pipeline.codex import CODEX_PROJECT_DOC_CONFIG

    command = [
        executable,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--json",
    ]
    for feature in OBSERVATION_DISABLED_CODEX_FEATURES:
        command.extend(("--disable", feature))
    command.extend(
        (
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(final),
            "-C",
            str(workspace),
            "--model",
            runtime.model_id,
            "-c",
            f"model_reasoning_effort={json.dumps(runtime.reasoning_effort, ensure_ascii=False)}",
            "-c",
            CODEX_PROJECT_DOC_CONFIG,
        )
    )
    for path in attached:
        command.extend(("--image", str(path)))
    command.append("-")
    return tuple(command)


def _default_process_runner(
    command: Sequence[str],
    *,
    stdin_text: str,
    cwd: Path,
    timeout_seconds: float | None,
    env: Mapping[str, str],
) -> ObservationProcessResult:
    """Run one Codex child; image observation passes ``timeout_seconds=None`` (no limit)."""

    codex_dir = cwd / ".codex-run"
    codex_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = codex_dir / "stdout.jsonl"
    stderr_path = codex_dir / "stderr.log"
    # Child writes go directly to disk, including partial lines and output before a timeout.
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=cwd,
            env=dict(env),
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            process.communicate(stdin_text, timeout=timeout_seconds)
        except BaseException as exc:
            # The child runs in its own session, so an interrupt never reaches it: stop the group.
            with suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(OSError):
                process.kill()
            process.communicate()
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            raise subprocess.TimeoutExpired(
                cmd=exc.cmd,
                timeout=cast(float, timeout_seconds),
                output=stdout_path.read_text(encoding="utf-8", errors="replace"),
                stderr=stderr_path.read_text(encoding="utf-8", errors="replace"),
            ) from exc
    return ObservationProcessResult(
        returncode=int(process.returncode),
        stdout=stdout_path.read_text(encoding="utf-8", errors="replace"),
        stderr=stderr_path.read_text(encoding="utf-8", errors="replace"),
    )


def _write_agent_workspace(
    workspace: Path,
    *,
    image_paths: Sequence[Path],
    build_task: Callable[[Sequence[str]], str],
    agent_context_schema: str,
    agent_context: str,
    extra_context: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    if not workspace.is_dir() or any(workspace.iterdir()):
        raise ImageObservationError("observation Agent workspace must be a new empty directory")
    codex_dir = workspace / ".codex-run"
    codex_dir.mkdir()
    staged: list[Path] = []
    for index, source in enumerate(image_paths, start=1):
        destination = workspace / f"input_{index}_{source.name}"
        shutil.copy2(source, destination)
        staged.append(destination)
    task = build_task([path.name for path in staged])
    (workspace / "TASK.md").write_text(task.rstrip() + "\n", encoding="utf-8")
    (codex_dir / "final.schema.json").write_text(
        json.dumps(OBSERVATION_AGENT_RESULT_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (workspace / "context_manifest.json").write_text(
        json.dumps(
            {
                "schema": agent_context_schema,
                "context": agent_context,
                "allowed_evidence": [
                    "TASK.md",
                    *[path.name for path in staged],
                    ".codex-run/final.schema.json",
                ],
                "attached_images": [path.name for path in staged],
                "primary_image": staged[0].name,
                "prior_attempts": False,
                "candidate_or_evaluator_feedback": False,
                "user_config_loaded": False,
                "project_documents_loaded": False,
                "disabled_codex_features": list(OBSERVATION_DISABLED_CODEX_FEATURES),
                "authority": "external_first_reading_not_authority",
                "filesystem_isolation": "evidence_protocol_not_os_security_boundary",
                **dict(extra_context),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace / "TASK.md", codex_dir / "final.json", codex_dir


class _ObservationCallKwargs(TypedDict):
    """Keyword arguments shared by every ``_invoke_codex_observation`` call of one run."""

    image_paths: Sequence[Path]
    runtime: ObservationRuntimeSelection
    executable: str
    environ: Mapping[str, str]
    process_runner: ObservationProcessRunner
    agent_context_schema: str
    agent_context: str
    extra_context: Mapping[str, object]


def _invoke_codex_observation(
    *,
    image_paths: Sequence[Path],
    build_task: Callable[[Sequence[str]], str],
    runtime: ObservationRuntimeSelection,
    executable: str,
    environ: Mapping[str, str],
    process_runner: ObservationProcessRunner,
    agent_context_schema: str,
    agent_context: str,
    extra_context: Mapping[str, object],
    audit_dir: Path,
) -> dict[str, object]:
    """Run one Codex observation call; the complete agent workspace lands in ``audit_dir``."""

    agent_workspace = audit_dir
    agent_workspace.mkdir(parents=True, exist_ok=False)
    task_path, final_path, codex_dir = _write_agent_workspace(
        agent_workspace,
        image_paths=list(image_paths),
        build_task=build_task,
        agent_context_schema=agent_context_schema,
        agent_context=agent_context,
        extra_context=dict(extra_context),
    )
    command = build_observation_codex_command(runtime, agent_workspace, executable=executable)
    process_env = dict(environ)
    process_env["PYTHONUNBUFFERED"] = "1"
    try:
        process = process_runner(
            command,
            stdin_text=task_path.read_text(encoding="utf-8"),
            cwd=agent_workspace,
            timeout_seconds=None,
            env=process_env,
        )
    except FileNotFoundError as exc:
        raise ImageObservationError(
            "image observation requires the `codex` executable; install/login to "
            "Codex CLI or set USDCRAFT_CODEX_BIN"
        ) from exc
    (codex_dir / "stdout.jsonl").write_text(process.stdout, encoding="utf-8")
    (codex_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        details = (process.stderr or process.stdout or "(no output)").strip()
        raise ImageObservationError(
            f"image-observation Agent exited with status {process.returncode}: {details[-4000:]}"
        )
    return parse_observation_result(final_path)


OBSERVATION_TOOL_NAME = "submit_image_observation"


def observation_tool_schema() -> dict[str, object]:
    """The one strict function tool a Provider observation call may invoke."""

    schema = json.loads(json.dumps(OBSERVATION_AGENT_RESULT_SCHEMA))
    return {
        "type": "function",
        "name": OBSERVATION_TOOL_NAME,
        "description": (
            "Submit the complete semantic inventory of the depicted primary subject exactly once."
        ),
        "parameters": schema,
        "strict": True,
    }


def extract_observation_arguments(output: Sequence[Mapping[str, object]]) -> str:
    calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == OBSERVATION_TOOL_NAME
    ]
    if len(calls) != 1:
        raise ImageObservationError(
            f"observation Agent must call {OBSERVATION_TOOL_NAME} exactly once "
            f"(got {len(calls)} calls)"
        )
    arguments = calls[0].get("arguments")
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(arguments, str) or not arguments.strip():
        raise ImageObservationError("observation tool call arguments must be JSON text")
    return arguments


def _provider_user_message(text: str, image_paths: Sequence[Path]) -> dict[str, object]:
    from usdcraft.pipeline.provider.harness import image_path_to_data_url

    content: list[dict[str, object]] = [{"type": "input_text", "text": text}]
    for path in image_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": image_path_to_data_url(Path(path)),
                "detail": "high",
            }
        )
    return {"role": "user", "content": content}


def run_image_observation_via_provider(
    *,
    image_paths: Sequence[Path],
    output_dir: Path,
    build_task: Callable[[], str],
    build_review_task: Callable[[str], str],
    manifest_schema: str,
    scope: str,
    agent_context: str,
    config: object,
    selection_source: str,
    extra_manifest: Mapping[str, object] | None = None,
    cancellation_event: Event | None = None,
    client: object | None = None,
) -> ImageObservationArtifacts:
    """Run one clean-context observation call through the NVIDIA Responses client.

    One user message (task text plus every image, primary first) with a single strict function
    tool whose parameters are the shared observation schema. Retries the complete call once on an
    invalid response, mirroring the Codex engine; one best-effort add-and-sharpen review call
    follows and is merged deterministically. Raises ``ImageObservationError`` on failure so the
    caller degrades to the Author-only reading; a set cancellation event always re-raises.
    """

    from usdcraft.pipeline.provider import NvidiaProviderError, NvidiaResponsesClient

    if not image_paths:
        raise ValueError("image observation requires at least the primary image")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ImageObservationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_client = cast(
        "NvidiaResponsesClient",
        client
        or NvidiaResponsesClient(
            cast("NvidiaProviderConfig", config), cancellation_event=cancellation_event
        ),
    )
    runtime = ObservationRuntimeSelection(
        model_id=str(getattr(config, "model")),
        reasoning_effort=str(getattr(config, "reasoning_effort")),
        runtime_source="provider_repository_config",
        selection_source=selection_source,
    )
    images = [Path(path) for path in image_paths]
    task = build_task()
    input_items = [_provider_user_message(task, images)]
    tools = [observation_tool_schema()]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    started = time.monotonic()
    last_error: ImageObservationError | None = None

    def _call(items: list[dict[str, object]], *, label: str) -> object:
        try:
            return resolved_client.create_response(input_items=items, tools=tools)
        except NvidiaProviderError as exc:
            if cancellation_event is not None and cancellation_event.is_set():
                raise
            raise ImageObservationError(
                f"provider image-observation {label} failed: {exc}"
            ) from exc

    def _dump(name: str, response: object) -> None:
        (staging / "agent" / name).write_text(
            json.dumps(
                {
                    "status": getattr(response, "status"),
                    "output": list(getattr(response, "output")),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        (staging / "agent").mkdir()
        (staging / "agent" / "TASK.md").write_text(task.rstrip() + "\n", encoding="utf-8")
        for attempt in range(1, OBSERVATION_MAX_ATTEMPTS + 1):
            try:
                response = _call(input_items, label="call")
                _dump(f"response_attempt_{attempt}.json", response)
                arguments = extract_observation_arguments(getattr(response, "output"))
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ImageObservationError(
                        "observation tool call arguments are invalid JSON"
                    ) from exc
                payload = validate_observation_payload(decoded)
            except ImageObservationError as exc:
                last_error = exc
                if attempt == OBSERVATION_MAX_ATTEMPTS:
                    raise
                continue

            review_report: dict[str, object]
            inventory_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            review_task = build_review_task(inventory_text)
            (staging / "agent" / "review_task.md").write_text(
                review_task.rstrip() + "\n", encoding="utf-8"
            )
            try:
                review_response = _call(
                    [_provider_user_message(review_task, images)], label="review call"
                )
                _dump("review_response.json", review_response)
                review_arguments = extract_observation_arguments(getattr(review_response, "output"))
                try:
                    review_decoded = json.loads(review_arguments)
                except json.JSONDecodeError as exc:
                    raise ImageObservationError(
                        "observation review tool call arguments are invalid JSON"
                    ) from exc
                reviewed = validate_observation_payload(review_decoded)
            except ImageObservationError as exc:
                review_report = {"status": "review_failed", "error": str(exc)[:800]}
            else:
                payload, review_report = merge_reviewed_inventory(payload, reviewed)
            (staging / "review_merge.json").write_text(
                json.dumps(review_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return publish_observation_artifacts(
                staging=staging,
                output=output,
                payload=payload,
                manifest_schema=manifest_schema,
                scope=scope,
                agent_context=agent_context,
                primary_image_path=images[0],
                runtime=runtime,
                attempts=attempt,
                started_monotonic=started,
                extra_manifest={"arm": "provider", **dict(extra_manifest or {})},
                review_report=review_report,
            )
        raise last_error or ImageObservationError("provider image-observation call failed")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_image_observation_via_codex(
    *,
    image_paths: Sequence[Path],
    output_dir: Path,
    build_task: Callable[[Sequence[str]], str],
    product: str,
    manifest_schema: str,
    scope: str,
    agent_context_schema: str,
    agent_context: str,
    build_review_task: Callable[[Sequence[str], str], str] | None = None,
    extra_manifest: Mapping[str, object] | None = None,
    extra_context: Mapping[str, object] | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
    process_runner: ObservationProcessRunner = _default_process_runner,
) -> ImageObservationArtifacts:
    """Run one clean-context Codex observation Agent and atomically publish its inventory.

    Retries the complete call once on an invalid response. Raises ``ImageObservationError`` when
    the Agent fails, blocks, or returns an invalid inventory twice; callers degrade to the
    Author-only reading and never fail the run for it. Like the rest of the Codex arm, the call
    and its review have no time limit.

    With ``build_review_task`` (staged image names, exact inventory JSON text → review task), one
    add-and-sharpen review call follows the first valid inventory and is merged deterministically
    by :func:`merge_reviewed_inventory`. The review is best-effort: a failed, blocked, or invalid
    review keeps the first inventory and never fails the observation.
    """

    if not image_paths:
        raise ValueError("image observation requires at least the primary image")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ImageObservationError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = resolve_observation_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        product=product,
        environ=environ,
    )
    selected_environ = os.environ if environ is None else environ
    resolved_executable = (
        executable or selected_environ.get("USDCRAFT_CODEX_BIN", "").strip() or "codex"
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    started = time.monotonic()
    call_kwargs: _ObservationCallKwargs = {
        "image_paths": list(image_paths),
        "runtime": runtime,
        "executable": resolved_executable,
        "environ": selected_environ,
        "process_runner": process_runner,
        "agent_context_schema": agent_context_schema,
        "agent_context": agent_context,
        "extra_context": dict(extra_context or {}),
    }
    last_error: ImageObservationError | None = None
    try:
        payload: dict[str, object] | None = None
        attempts = 0
        for attempt in range(1, OBSERVATION_MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                payload = _invoke_codex_observation(
                    build_task=build_task,
                    audit_dir=staging / f"agent_attempt_{attempt}",
                    **call_kwargs,
                )
            except ImageObservationError as exc:
                last_error = exc
                if attempt == OBSERVATION_MAX_ATTEMPTS:
                    raise
                continue
            break
        if payload is None:
            raise last_error or ImageObservationError("image-observation Agent failed")
        os.replace(staging / f"agent_attempt_{attempts}", staging / "agent")

        review_report: dict[str, object] | None = None
        if build_review_task is not None:
            inventory_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            try:
                reviewed = _invoke_codex_observation(
                    build_task=lambda names: build_review_task(names, inventory_text),
                    audit_dir=staging / "review_agent",
                    **call_kwargs,
                )
            except ImageObservationError as exc:
                # The review is best-effort: a failed, blocked, or invalid review keeps the first
                # inventory and never fails the run.
                review_report = {"status": "review_failed", "error": str(exc)[:800]}
            else:
                payload, review_report = merge_reviewed_inventory(payload, reviewed)
            (staging / "review_merge.json").write_text(
                json.dumps(review_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return publish_observation_artifacts(
            staging=staging,
            output=output,
            payload=payload,
            manifest_schema=manifest_schema,
            scope=scope,
            agent_context=agent_context,
            primary_image_path=Path(image_paths[0]),
            runtime=runtime,
            attempts=attempts,
            started_monotonic=started,
            extra_manifest=extra_manifest,
            review_report=review_report,
        )
    except BaseException:
        os.replace(staging, output)
        raise


__all__ = [
    "OBSERVATION_AGENT_RESULT_SCHEMA",
    "OBSERVATION_DISABLED_CODEX_FEATURES",
    "OBSERVATION_MAX_ATTEMPTS",
    "ImageObservationArtifacts",
    "ImageObservationError",
    "ObservationProcessResult",
    "ObservationProcessRunner",
    "ObservationRuntimeSelection",
    "ObservationTaskProfile",
    "build_observation_codex_command",
    "build_observation_review_task",
    "build_observation_task",
    "merge_reviewed_inventory",
    "parse_observation_result",
    "publish_observation_artifacts",
    "render_inventory_for_prompt",
    "resolve_observation_runtime",
    "validate_observation_payload",
]
