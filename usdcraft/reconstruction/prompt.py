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

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from usdcraft.pipeline.policy import render_mechanical_authoring_block
from usdcraft.pipeline.sdk_context import render_preloaded_sdk_docs
from usdcraft.pipeline.sdk_context import (
    render_sdk_document_inventory as _render_sdk_document_inventory,
)
from usdcraft.reconstruction.workspace import ReconstructionAuthorWorkspace

RUN_INPUTS_HEADER = "\nRUN INPUTS\n"


def reconstruction_cache_prefix(prompt: str) -> str:
    """Public task text before instance-specific inputs, for a separate cache breakpoint."""
    prefix, separator, _ = prompt.partition(RUN_INPUTS_HEADER)
    if not separator:
        raise ValueError("Reconstruction task has no run-input boundary")
    return prefix


def _command(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        result = value.strip()
    else:
        result = shlex.join(str(item) for item in value)
    if not result:
        raise ValueError("Reconstruction author command must not be empty.")
    return result


@dataclass(slots=True, frozen=True)
class ArmWording:
    """The words that differ between the two execution arms of one shared task.

    The doctrine, evidence, design order, observation readings and repair loop are identical for
    the Codex Author (shell commands, schema-conforming final response) and the NVIDIA Provider
    Author (bounded function tools, ``finish_reconstruction``); only how the per-face measurement
    and tools are reached is named differently.
    """

    provider_tools: bool
    # Display form of the per-face measurement, with backticks: "`measure_face`" or the command.
    measure_text: str
    # Raw shell command of the per-face measurement (Codex arm only; the Author appends the face).
    measure_command: str


# Reconstruction's own lead into the shared physical-authoring policy: physics values are Author
# hypotheses and never move the geometry the evidence decides.
MECHANICAL_AUTHORING_INTRO = (
    "Mass, inertia, contact material, friction, damping, resistance and gravity behavior are "
    "explicit Author hypotheses unless user-supplied guidance directly specifies them: keep each "
    "consequential derivation visible in `model.py` through named variables or a simple "
    "expression, never present these hypotheses as source facts, and never use them to alter "
    "evidence-bound Part identity or count, q=0 geometry and bounds, openings, clearances or "
    "articulation."
)

PROVIDER_ARM = ArmWording(
    provider_tools=True,
    measure_text="`measure_face`",
    measure_command="measure_face",
)


def _build_free_author_prompt(
    *,
    arm: ArmWording,
    workspace: ReconstructionAuthorWorkspace,
    user_guidance: str | None,
    check_text: str,
    compile_text: str,
    axial_text: str,
    image_observation_text: str | None,
    request_text: str | None,
    scale_note: str | None,
) -> str:
    """One Author owns evidence interpretation, geometry and visual repair in both arms."""
    access_lines = []
    if arm.provider_tools:
        access_lines.append(
            "Read exact SDK references with read_file, search_docs and find_examples. "
            "Use write_model(content) for first construction or complete replacement of model.py: "
            "send the complete source without Markdown fences or read_file line labels. Keep one "
            "build_object_model definition, one object_model binding and your run_tests. "
            "Use apply_patch for local edits with exact current context or removed lines in each "
            "hunk; read the relevant region when its current text is unknown."
        )
        access_lines.append(
            "measure_mesh queries frozen source vertices: bounds=[xmin,ymin,zmin,xmax,ymax,zmax] in metres, or null for all. max_points bounds stored samples; replies preview 16 XYZ points and provide detail_path, quantiles, plane fit and orthographic images. Check projections and residuals before treating a fitted normal as a panel orientation."
        )
        access_lines.append(
            "render_source shows the frozen source mesh at azimuth/elevation in degrees, in the normalized metre/Z-up frame. Request a useful view when shape or occlusion is unclear; its image is attached to your next turn. It needs no candidate or compile. Source renders support qualitative shape reading; use measurements for dimensions."
        )
        access_lines.append(
            "render_candidate requires a successful compile for the current model revision and returns its image: azimuth/elevation in degrees, joints=[{name,position}] in radians/metres or [] for q=0."
        )
        access_lines.append(
            "compare_part requires a successful compile for the current model revision. It compares part=<candidate link>, element=<visual name or null>, face=<+/-X/Y/Z>, bounds=<six source metres>. Inspect its local overlay and residuals; correspondence is your hypothesis."
        )
        access_lines.append(
            "measure_face queries automatic AXIAL detections on a face, optionally within z_band. It is not an independent remeasurement."
        )
        validation = f"Use validate_model for a complete candidate checkpoint: it runs {check_text}, {compile_text}, {axial_text} and candidate rendering in one call. Use individual tools for a specific failure, local comparison or additional view instead of repeating the whole checkpoint. A source-changing write_model or apply_patch invalidates the current compiled revision; compile_model or validate_model must succeed before check_axial, compare_part, render_candidate. ok reports check/compile success; diagnostics_complete reports diagnostic execution, not shape quality. validation_state identifies the revision covered by each operation. Inspect the returned results and images before choosing a repair."
        finish = "Call finish_reconstruction with complete or blocked and your summary. State which final-revision inspections you performed and any missing or stale diagnostics shown by validation_state."
    else:
        tokens = shlex.split(arm.measure_command)
        prefix = (
            tokens[: tokens.index("recon-measure-face")]
            if "recon-measure-face" in tokens
            else ["usdcraft"]
        )

        def command(name: str) -> str:
            return shlex.join([*prefix, name])

        access_lines.append(
            "Use shell tools to read exact SDK references and edit model.py. Work only with this run's staged evidence; prior cases, Viewer state and hidden evaluators are excluded to keep the run independent."
        )
        access_lines.append(
            f"Inspect inputs/source_mesh.usdc with Python/OpenUSD or `{command('recon-measure-mesh')} inputs/source_mesh.usdc --output author_feedback/mesh_001`. Add --bounds xmin ymin zmin xmax ymax zmax and --max-points N (up to 4096) for a region. Replies contain quantiles, bounded XYZ samples, projections and plane-fit residuals. Preserve metres/Z-up and inspect returned projections."
        )
        access_lines.append(
            "CLI bounds take six space-separated numbers, for example `--bounds -0.10 -0.20 0.00 0.10 0.20 0.30`; choose values for your source region. Do not use commas or attach the first value with '=': each coordinate is a separate argument."
        )
        access_lines.append(
            f"`{command('recon-render-source')} inputs/source_render --output author_feedback/source_001 --azimuth 45 --elevation 30` shows the frozen source mesh in normalized metre/Z-up coordinates. Choose a useful camera when shape or occlusion is unclear and open the returned image. Use a new output directory per query. It needs no candidate or compile; use measurements for dimensions."
        )
        access_lines.append(
            f"`{command('recon-render')} build --output author_feedback/view_001 --azimuth 45 --elevation 30` renders your candidate. Add --joint NAME=VALUE for an articulated pose in radians/metres. View the returned image. Use a new output directory per query."
        )
        access_lines.append(
            f"`{command('recon-compare-part')} inputs/source_mesh.usdc build --part NAME --face=-Y --bounds xmin ymin zmin xmax ymax zmax --output author_feedback/part_001` compares local shape; optionally add --element NAME. Inspect the overlay; correspondence is your hypothesis."
        )
        access_lines.append(
            f"`{arm.measure_command} --face=<face>` queries automatic AXIAL feature detections."
        )
        validation = f"Run `{check_text} && {compile_text} && {axial_text}` after substantive changes, batching the ordered validation steps in one shell call. Repair code/export failures and repeat. AXIAL findings are advisory."
        finish = "Return the schema-conforming complete or blocked response and your summary."
    image_delivery = (
        "Inspect the images attached to the next turn after tool results; Provider delivers them automatically."
        if arm.provider_tools
        else "Open returned image files with your image-viewing tool; a PNG path is not an inspection."
    )
    inspection = f"Inspect images of the final compiled geometry for silhouette, openings, thin surfaces, materials and missing parts. Resolve visible overlap/flicker and unintended gaps. At useful joint poses, verify that the functional assembly moves with its control and retains required clearance/support. {image_delivery} After consequential edits, compile and reinspect the affected views. A render failure is a tool limitation to report, not a reason to discard a valid model. Candidate images are your output, never new target evidence; posing does not test dynamics."
    access = "\n".join(access_lines)
    interior_inspection = """SOURCE INTERIOR INSPECTION
For compartments or occluded functional assemblies, inspect source component bounds and localized
interior slices with the mesh queries before deciding what to reconstruct. Mesh-query
replies preview component bounds; read their detail_path for the stored list and narrow the query
when truncated. Follow sampled internal surfaces and repeated levels, not just the exterior envelope.
A closed door or an incomplete image inventory is not evidence of an empty interior.
Use focused regions to distinguish fittings
from the enclosing shell. Component boundaries are geometry clues, not semantic parts or joints;
combine them with the image and normal use, and mark unresolved identities or motion as hypotheses.
"""
    design = "FREE DESIGN ORDER\nRead the semantic image for object identity, shape character, visible parts, pose and normal function. Choose your construction and diagnostic order around that interpretation, obtaining reliable source measurements for its dimensions. Build a coarse executable candidate early, then use focused evidence and SDK queries to resolve consequential uncertainty."
    source_declaration = "- inputs/source_mesh.usdc: frozen normalized source, metres, Z-up, q=0."
    observation = (
        """INDEPENDENT IMAGE OBSERVATION
The following image-only first reading is advisory, not a settled part list. Verify uncertain
entries against the image and source geometry. You may rename, merge, split, add or discard
entries while following the design-order guidance. An observed item is not necessarily a rigid
body. Missing observations do not prove absence; inferred entries are hypotheses. You own the
final interpretation and need not repeat every observation or reproduce this inventory as tests.
"""
        + image_observation_text
        + "\n\n"
        if image_observation_text
        else ""
    )
    return (
        f"""You are the one clean-context Reconstruction Author for this USDCraft run.

OUTCOME
Reconstruct the declared object as an editable, executable SDK model in `model.py` with usable
articulated parts and deliberate physics. You own planning, evidence interpretation, geometry,
visual inspection and repair. Work within this session; do not launch other agents, because one
Author must own the final model. The Host compiles, validates mechanically and persists it.

EVIDENCE AND DECISIONS
The request defines the target. Interpret the image and mesh together: the image establishes
visible identity, shape character, required parts, appearance and use; the mesh supplies reliable
observable geometry and metric support. AXIAL is a lossy source digest and diagnostic, not a
construction blueprint or an independent semantic target.
Preserve reliable source dimensions. A measurement of a selected region is not proof that the
region is the entire semantic part. Automatic feature matching, inventory and slab grouping
are fallible interpretations. Remeasure or inspect another view when they conflict with the
object; you decide part identity, count, shape, attachment, pose and the final construction.
Use the image for identity and qualitative shape checks, not numerical measurements from
pixels or perspective ratios. Constrain unresolved metrics with measured hosts and normal use. Distinguish measured values from lower bounds and hypotheses. Missing surfaces
and empty query regions do not prove empty space. Infer hidden walls, cavities, joint limits and
mechanisms needed for normal use; do not present them as observed. Preserve the declared scale
and frame; do not change them merely to reduce a residual. Briefly document consequential
assumptions or rejected measurements in model comments and your final summary.

FROZEN SOURCE FRAME
The staged source metre/Z-up world frame is the common frame for AXIAL, mesh measurements
and source renders, and your compiled candidate. Preserve its origin, scale, handedness and yaw;
do not rotate, mirror, recenter or re-ground the whole reconstruction to match the semantic
image's camera. Local Part and joint frames must compose into this same world frame at q=0.
Image-relative left/right/front are not world-axis labels. When source front is unresolved,
world -Y is a coordinate label, not evidence of the object's semantic front. Before placing
direction-sensitive parts or joint anchors, match an asymmetric visible feature to its signed
source region using available measurements and views. If correspondence remains ambiguous,
state that uncertainty and keep the frozen frame instead of inventing a global reorientation.

{design}

AUTHOR TOOLS
{access}

FUNCTIONAL MOBILITY REVIEW
Review each functional assembly before construction and again after repair: fixed structure,
attachment carried by another moving member, operating joint, detachable part, or unresolved.
Consider setup adjustment and secondary/support motion as well as the main action. A clamp may
hold an adjustable guide still during use; its photographed clamped state or a hidden slot does
not establish a permanent weld. Relate controls, pads, guides and supports to what they operate;
use instance geometry, the image, the request and normal use to choose a minimal mechanism.
Use these families to look for candidates, not as mandatory parts or joint-count templates:
- Storage/appliances: doors, lids, drawers, pull-out racks and access fittings.
- Tools: adjustable guides/stops, clamps and their release/adjustment relations.
- Mobile supports: wheel rolling, caster steering and supported locks where justified.
- Furniture: folding, height, tilt and other supported positioning adjustments.
- Containers: latches, caps, closures and genuinely detachable members.
For consequential choices, briefly state the motion or fixed relation and its evidence in model
comments; identify the carried assembly, axis/travel and use sequence when motion is supported.
If the request explicitly requires rolling/mobile support and the scan omits it, reconstruct a
plausible support mechanism with its count, dimensions and placement marked as hypotheses in the
frozen source frame. A generic category alone does not require wheels or any other unobserved
mechanism. When instance evidence and requested function do not resolve a relation, record it
as unknown rather than adding arbitrary DOFs. A moving appearance or separate mesh component
alone does not prove an independent joint; a required adjustment must not be lost just because
the supplied pose is stationary.

AUTHORING AND REPAIR
Choose a compact construction that captures the object's characteristic shapes and functionality.
Use the SDK's geometry and mesh facilities; avoid one primitive per evidence sample, which
creates complexity without a coherent editable model. `model.py` is the executable source;
edit it rather than derived USD/URDF so the Host rebuild reproduces your changes. Preserve
usable handles, interior clearance, openings and contact surfaces. Group rigidly attached
visual details into appropriate bodies; only required independent motion needs articulation.
Assign carried functional members to the moving assembly, not just the external control.
An empty usable compartment excludes loose payload unless requested; it retains supported
structural and functional fittings such as shelves, guides, trays and racks. Preserve these even
when occluded in the semantic image; unobserved space remains unknown, not proof of absence.
Displayed loose objects may be separate floating bodies. Surface graphics belong on their host surfaces.
Use `BaseMode.FLOATING` for the standalone root; fixation and the simulation scene belong to the
consumer. Author enough focused `run_tests()` checks for decisive motion/clearance properties;
self-written checks do not establish visual correctness or independently validate your assumptions.

{interior_inspection}
Use `inputs/axial_brief.txt` when you need its frame, envelope or a relevant feature anchor;
use mesh queries for direct measurements of disputed regions. Map each measurement to
the visually identified part before choosing dimensions. Preserve reliable source dimensions;
revise uncertain detector-to-part assignments instead of reshaping the object to match a digest.
Keep unknown or hidden construction explicit as a hypothesis.
Use the preloaded SDK foundations and the specific constructors you need, then build a coarse executable
candidate early. Expand documentation searches only for a concrete API question or failure.
Before polishing materials or hidden mechanics, compare the main silhouette and visible controls.
An envelope match does not establish the body's profile: query a few relevant height/face regions
for taper, shoulder curvature and control projection depth. A graspable control still needs
mesh-supported depth where available; its visible function does not justify extra protrusion.

{validation}
{inspection}
PRESERVE FUNCTION THROUGH REPAIR
Keep evidence-supported sliding and rotating assemblies operable while repairing contact,
support or collision defects. Correct their geometry, joint frames, clearances and support;
do not replace required PRISMATIC/REVOLUTE motion with FIXED or FLOATING, delete the member or
collapse its useful travel just to pass checks. Such changes alter the mechanism; revise it
only when evidence supports a different physical relation, and explain that evidence.
Check a normal-use sequence and its combined poses, such as opening a door before extending a
guided rack. Supply all interacting joint positions together in one ctx.pose mapping or render
request; nested ctx.pose calls do not accumulate their outer joint settings. For a reported
collision, inspect the actual pose and named geometry: independent joint sampling can combine
positions outside the intended operating sequence. Repair genuine interference on the use path,
and document any required sequencing and specifically justified contact allowances. Do not hide
reachable collisions with broad exemptions or claim that a posed check proves dynamic safety.
Treat diagnostic findings as questions to investigate, not automatic edit instructions: repair
supported defects, remeasure uncertain regions, and explain important intentional differences.
For an underside residual, verify whether the selected surface is the physical underside,
rounded shell or incomplete scan before raising the body or adding feet. Sparse ground coverage
or a bottom-plane median alone does not establish foot height. Preserve the grounded frame.
For consequential visible parts (including small controls and thin flaps), connect the image
identity to a source face/region and your candidate part. Check local profile as well as size and
placement; when correspondence is uncertain, inspect a focused view and query that region.
A detector's recess footprint may be only the deepest subset of a larger tray; verify the rim
before resizing. Preserve independently observed repeated-part angles rather than imposing symmetry.
Keep model comments focused on important retained discrepancies and unsupported dimensions. AXIAL recall is not a target or completion threshold. A source artifact must not be copied just
to improve agreement. Rerun relevant checks on the final revision. Identify any checks or views that still describe an earlier revision in the closing summary.
Keep queries and repair bounded by the run budget and focus on consequential errors.
Zero numbered discrepancies is not a shape-quality verdict. Resolve consequential local residuals
before finishing: repair them, or explain a checked correspondence/occlusion limitation. Do not claim Isaac Sim runtime success without an actual runtime test.
{finish} Summarize what you built, checks and visual inspections actually performed, important
assumptions and remaining limitations. Block only for a concrete inability to produce a valid
editable asset, not for an unresolved semantic disagreement or AXIAL residual.

{render_mechanical_authoring_block(product="Reconstruction", intro=MECHANICAL_AUTHORING_INTRO, check_command=check_text)}

{render_preloaded_sdk_docs(workspace.repo_root, task_noun="Reconstruction Author", reread_clause="reopen a preloaded document only for an exact detail")}
PUBLIC SDK REFERENCES
The SDK quickstart is included above. Read additional exact references for the
geometry families and APIs your model needs; the remaining inventory is a routing map.
{_render_sdk_document_inventory(workspace.repo_root, mark_preloaded=True)}
{RUN_INPUTS_HEADER}
REQUEST
{request_text or "Reconstruct the primary subject shown in the semantic image and source mesh."}

USER-SUPPLIED GUIDANCE
{user_guidance or "No supplemental user guidance."}

SCALE
{scale_note or "Use the staged source metre scale; record uncertainty when scale is estimated."}

DECLARED EVIDENCE AND IMAGE ORDER
- Attached image 1: `inputs/{workspace.semantic_image_path.name}`, the target semantic image.

{source_declaration}
- `evidence_manifest.json`: input hashes and frame binding.
- `inputs/axial_brief.txt`: compact automatic geometry digest; `inputs/axial.txt` feeds
  the AXIAL diagnostic. Query only what helps; reading or transcribing the whole text is not required.


{observation}
""".rstrip()
        + "\n"
    )


def build_reconstruction_author_prompt(
    *,
    workspace: ReconstructionAuthorWorkspace,
    user_guidance: str | None,
    check_command: str | Sequence[str],
    compile_command: str | Sequence[str],
    axial_reencode_command: str | Sequence[str],
    measure_command: str | Sequence[str] | None = None,
    image_observation_text: str | None = None,
    reference_access: Literal["shell", "provider_tools"] = "shell",
    request_text: str | None = None,
    scale_note: str | None = None,
) -> str:
    """Build the complete, case-neutral Reconstruction Author task.

    ``reference_access`` selects the arm: ``shell`` is the Codex Author, which runs the check,
    compile, AXIAL diagnostic and per-face measurement as shell commands (the four command
    arguments) and finishes with the schema-conforming final response; ``provider_tools`` is the
    NVIDIA Provider Author with bounded function tools (the command arguments are then the tool
    names) and ``finish_reconstruction``. Evidence interpretation, optional observation advice and the in-session repair loop
    share one policy; direct measurement and rendering share one implementation.
    """

    if reference_access not in {"shell", "provider_tools"}:
        raise ValueError(f"Unsupported Reconstruction reference access mode: {reference_access!r}.")
    check_text = _command(check_command)
    compile_text = _command(compile_command)
    axial_text = _command(axial_reencode_command)
    if reference_access == "provider_tools":
        arm = PROVIDER_ARM
    else:
        if measure_command is None:
            raise ValueError(
                "The Codex Reconstruction Author needs the per-face measurement command."
            )
        measure_text = _command(measure_command)
        arm = ArmWording(
            provider_tools=False,
            measure_text=f"`{measure_text} --face=<face>`",
            measure_command=measure_text,
        )
    return _build_free_author_prompt(
        arm=arm,
        workspace=workspace,
        user_guidance=user_guidance,
        check_text=check_text,
        compile_text=compile_text,
        axial_text=axial_text,
        image_observation_text=image_observation_text,
        request_text=request_text,
        scale_note=scale_note,
    )


__all__ = ["ArmWording", "PROVIDER_ARM", "build_reconstruction_author_prompt"]
