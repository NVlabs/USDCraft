# Reconstruction evidence contract

This document describes the evidence available to the Reconstruction preprocessing and Author
stages and what each channel may and may not establish. It is not a generic mesh truth model. The
executable sources are `source.py`, `isolation_agent.py`, `provider_isolation.py`, `isolation.py`,
`mesh_preprocess.py`,
`axial.py`, `face_features.py`, `face_openings.py` (shared face grid in `_face_grid.py`),
`interior_islands.py`, `axial_brief.py`,
`provider_axial_audit.py`, `local_surface_audit.py`, `author_tools.py`, `diagnostics.py` and the
case-neutral task in `prompt.py`. Commands and the pipeline are described in [README.md](README.md).

## Authority matrix

| Channel | Positive authority | Does not establish |
| --- | --- | --- |
| Semantic image | Primary-subject category, positively visible fixed/movable Part identity and count, appearance, affordance, normal function | Absence in cropped, occluded, ambiguous, backside or unseen regions; metric geometry |
| Normalized mesh and `measure_mesh` | Selected source vertices in the metre/Z-up q=0 frame; direct bounds, quantiles, plane fits and projections | Semantic segmentation, solid occupancy, hidden structure or object-scale calibration |
| `AXIAL` and its brief | Derived bounds, positions and serialized sampled source surfaces; automatic detections are revisable interpretations | Solid fill, inside/outside, thickness, cavity contents, free space, material, Part semantics |
| Author-requested source renders | Qualitative visible surface form, topology, openings, thin members, separation and localized malformed-surface cues | Part name/function/joint/count, independent dimensions or independent semantic corroboration |
| Image-observation inventory | An advisory external first reading of the semantic image | Evidence authority, a binding part list, metric values |
| Isolation decision | Which complete source-surface components are retained | Part identity, geometry repair, semantics for the Author, or proof that excluded space is empty |
| User guidance | Supplemental instructions bound into the request | Undeclared evidence or permission to override observed evidence |
| Candidate model, renders and collision shapes | The Author's explicit reconstruction hypothesis and its rendered appearance | Retroactive authority over source interiors or missing source evidence |

Incidental scene dressing is not a primary-subject Part. Evidence channels remain separate: a
source render cannot become a second semantic image, and the semantic image does not supply
calibrated metric support. Image proportions are a consistency check or an explicitly uncertain
estimate. The Author interprets conflicts using the image and mesh-derived evidence, preserving
reliable measurements while revising automatic region-to-part matches. Harness switches control
channel availability, not evidence authority: disabling source views or observation does not turn
image pixels into metric evidence, and the semantic image and AXIAL always remain.

## Surface-only source model

Every accepted source mesh or point set is evidence only where its surface is sampled. Closed,
watertight, manifold or positive-volume topology does not establish solid material; ray parity and
surface integrals are topology diagnostics, not occupancy. Every nonsurface location is `UNKNOWN`
(neither occupied nor clear), including the interior of a closed mesh. The source does not prove
inside versus outside, fill or wall thickness, cavity contents, depth or rear closure, collision
occupancy or free space, or continuity between separated samples, islands, components or slices.

Missing, torn, fused or stretched surfaces remain unknown gaps. An image- or category-supported
continuation across a gap is a documented Author hypothesis; candidate solids and collisions confer
no authority back onto the source. A source artifact is not copied merely to improve agreement;
consequential intentional differences are explained in model comments and the closing summary.

For compartments and occluded assemblies, the Author inspects component bounds and interior slices
with `measure_mesh`, including surfaces hidden from the semantic camera; image nonvisibility and an
incomplete inventory do not erase them. An empty compartment excludes loose payload, not supported
fittings such as guides, shelves, trays and racks. Component boundaries do not assign identity or
motion.

## Semantic image

Every request declares exactly one semantic image with the role `reconstruction_target`: the
semantic record of the subject being reconstructed. Positive visible evidence establishes subject
requirements; metric attributes come from reliable source measurements or explicitly constrained
hypotheses, not image pixels or ratios. Positive visibility is bounded: a visible item establishes
its existence and visible count, but an unseen region cannot establish absence or count zero. The
Author records ambiguity rather than filling it with negative claims.

Reconstruction targets the declared subject instance rather than using the image as inspiration.
Every clear positive semantic fact is an Author requirement; category knowledge may complete
unknown regions but may not replace visible identity, count, arrangement, affordance, appearance or
function with a merely category-plausible alternative.

## Subject isolation

Isolation is optional mesh-artifact cleanup (`--mesh-cleanup` or a true `mesh_cleanup` CSV cell,
available on both arms, default off); otherwise the Host freezes a deterministic retain-all
declaration, which never proves the scene artifact-free. Both arms run the same Agent task,
evidence, decision schema and validation; only the transport differs (Codex CLI with no time limit,
or one NVIDIA request bounded by the Provider per-request timeout). It runs after only
the declared axis/unit decode and before orientation, recentering, ground snap, autoscale and AXIAL.
The Agent receives exactly a path-free component inventory, one native-appearance contact sheet and
one component-ID-colour sheet (azimuths 45/135/225/315 at elevation 35, plus a top view) — no
semantic image, AXIAL, source file, SDK, Author task, candidate or ground truth. It may exclude only
whole components (or a deterministic axis empty-band group expanding to explicit component IDs) as
clear background, incidental or outlier evidence; it cannot clip, cut, repair, fill, infer solids,
choose Parts or transform the subject. Unresolved and fused components are retained.

The Host hash-binds the decision to the source closure, inventory and frame and applies the same
retained set to preprocessing and the source carrier. Excluded surfaces never prove space empty.
Isolation artifacts are Host audit data withheld from the Author; context isolation is an
auditable protocol, not an OS security boundary.

## Author-requested source views

The Author starts with the semantic image only. Preprocessing exports a geometry-only source carrier
(with available native appearance) and no camera images; input manifests declare
`source_view_policy: author_requested`. The Author requests cameras via Provider `render_source`
or Codex `recon-render-source`; each result records camera, carrier hashes,
backend capabilities and fallback.

Source views are same-source qualitative surface evidence. They may reveal an opening, rear
assembly, thin member or malformed patch the semantic camera cannot see; such a structure remains
semantically unresolved geometry and does not by itself establish another Part, and it is not an
artifact merely because the semantic camera does not show it. Views cannot name Parts, infer
functions or joints, supply independent dimensions or override AXIAL.

## `AXIAL`

`AXIAL` is deterministic, lossy, surface-only geometry evidence stored as `axial.txt`. Its first
line is exactly `AXIAL`; the following header lines declare the normalized q=0 pose, metre-scale
Z-up frame, source bounds, micro-grid frame, source types, negative-space limitations, selected
stack policy and cell legend. Projected solid/clear intervals are not serialized.

The micro grid uses 192 cells along its longest axis. `~` means sampled source-surface support and
`.` means unknown; neither the fine grid nor a coarser stack cell is occupancy. `SLICE_STACK`
encodes fixed horizontal XY planes ordered along Z; repeated patterns do not establish continuous
walls or material between planes.

### Budget and stack selection

- The stack resolution is the finest of 32, 24 and 16 cells along the longest grid axis whose
  **stack-bound core** — header, `MICRO_SLICE_BOUNDS_Z192`, `BODY_WALLS_Z` and the
  `SLICE_STACK`/`PATTERN` raster — fits 32 KiB UTF-8.
- The face-derived sections (`FACE_FEATURES`, `FACE_OPENINGS`, `MEMBER_TRACKS192`,
  `INTERIOR_ISLANDS`) sit above that ceiling under their own caps, so they never change the
  selected stack. The complete text is fused at 56 KiB (core ceiling plus a 24 KiB section
  allowance), which is also the Host gate's maximum for `axial.txt`.
- Presence before budget: island bounds and small-component extents are always serialized
  completely. Budgets trade only detail (contour payload, stack resolution).
- The ceiling is a runaway fuse, not a truncation trigger: if no candidate fits, preprocessing
  fails loudly. A section is serialized completely or marked omitted under its bounded policy.
- The stack axis is always Z and every plane is horizontal XY.
- The manifest records `agent_core_selection_reason` (`finest_within_core_ceiling` or
  `coarser_for_core_ceiling`).

### `MICRO_SLICE_BOUNDS_Z192`

For every selected Z slice, this section lists every 8-connected sampled-surface island in the XY
plane with its XY bounds. An entry may append `(f=…;p=…)`: `p` is a simplified outer boundary on
the micro cell-corner lattice (at most 12 integer vertices), emitted only where it materially
differs from the bbox; `f` is sampled cells divided by bbox cells. The 3 KiB budget governs only
contour payload: when exceeded, contours drop first (largest islands keep theirs longest,
`selection=contours_partial`) down to a `boxes_only` floor. No island is ever dropped, and an
omitted contour is unknown detail, never absence.

Islands are anonymous and nonsemantic. The bbox is the extent authority; a contour is a quantized
simplification that calibrates a smooth profile, never a polyline to loft verbatim. Contour and box
interiors remain unknown and may bridge unsampled cells by about one micro cell. Overlapping boxes
on adjacent slices do not establish identity, connectivity, extrusion or a wall; the only
sanctioned cross-slice linking is `MEMBER_TRACKS192`. A local protrusion extremum must not be
applied to the enclosing body.

### `BODY_WALLS_Z`

For every stack slab and each XY side this section gives the **wall** (median over the slab's
columns of the outermost sampled cell) and the **extreme** (bbox edge), in metres. When an open
door, lid or drawer spreads a slab's column extremes and drags the median outward of a clearly
defined face plane that the slab still partly shows, the wall takes that face plane. The Author
builds the body to the wall values; a wall/extreme gap reads as "something protrudes here".

A slab whose outline follows an elliptical width profile (mean/max column width below 0.90 and a
small deviation from the fitted profile) is serialized with `section=round` and its walls set to
its extremes, which then name the diameter or rounded extent.

Each row carries `cover`, the share of the XY footprint with sampled surface. The `support:` line
gives lowest-slab coverage, `body_full_width_from_z` and a descriptive verdict
(`flat_scanned_bottom`, `contact_patch_or_unscanned_underside` or `partial_bottom`). These do not
distinguish separate feet, a rounded lower shell or an unscanned underside. The grounded source
establishes the contact plane; without identified supports the Author completes the observed lower
profile to the ground with the simplest compatible closure. Low coverage alone does not justify
lifting the shell or inventing tall feet.

### `FACE_FEATURES`

For each of the six axis-aligned faces, this section lists compact protrusions separated from a
locally fitted face base: a first-surface depth map (pitch `max(2 mm, longest extent / 400)`), a
6 cm median-window base and residual blobs of 8–120 mm with median height at least 3 mm or 1.5
pitches, peak at least twice that floor, at most 80 mm high, filled and off the silhouette. A blob
whose peak is between 1.2× and 2× the floor is kept only as a button-like cap: flat-topped, on an
object whose floor is within 6 mm, and with a like-sized sibling within 150 mm on the same face.

Levels: **L1** on the whole face; **L2** with a 3 cm window inside a large, band-shaped or
silhouette-touching L1 region; **L3** keeps cells above half the peak inside a band that still
fuses a control with ribs or a step. **L0** reads the face against a 40 cm base for protrusions the
6 cm window cannot report whole (pulls, dials, door rails): up to 400 mm long with one side at most
250 mm, 8–150 mm high, at most twelve per face, largest first. L1–L3 entries inside an L0 footprint
are its corners or steps, not separate controls; L0 may wrap a knob together with its bezel. Long
bands (steps, tiers, door edges), openings and cliffs over 6 cm are never listed.

Each entry gives `c(u,v)` centre in world metres, `size` in mm, `h` max/median height in mm and its
level. Entries are anonymous and add no authority, but are the direct metric anchor for a knob,
button, handle, lever or foot once the image names it: the image decides that a control exists and
what it is; the entry says where and how large. Scans erode tips, so `h` is a lower bound; a
missing entry never proves a face flat or a control absent. Thresholds are physical, so the source
must be in true scale (see the scale step in the README).

### `FACE_OPENINGS`

The complement of `FACE_FEATURES`: enclosed regions where a face steps in (a slot, basin, well,
open top or open door). The reference plane is the face rim when continuous (`rim`), the
silhouette outline when the face plane is missing (`outline`), otherwise the dominant depth
(`mode`). A `recess` steps in at least 6 mm; an `opening` at least 15 mm or has no surface; a recess
carrying openings is a recessed panel (`in=N`). A region covering most of the face is `open_face`
unless a solid patch of at least a tenth of the face area lies on the reference plane. Entries give
centre and plan size in metres; visible depth is a **lower bound**. Anonymous, nonsemantic, at most
six per face. Known limits: shallow recesses on flat cooktops, grids and wire objects.

### `INTERIOR_ISLANDS`

Derived from the selected stack's micro islands: in each slab whose largest island covers at least
a quarter of the footprint, islands wholly inside that body box (one-cell margin; an island
inset by at most 5% or four cells is the wall's own inner face) are free-standing fittings.
Same-size islands form a `grid` from three members (sparse grids need six members or 10% fill);
two or more free bars form a `frame`; a lone island is a `plate` (fill ≥ 0.6), a `shell` (fill ≤
0.25, at least 40 mm per side), a `bar` (one wire spanning 40% of the body) or a `block`. Specks
below 4% of the body box (or 25 mm) are dropped. Entities overlapping by 80% merge within and
across adjacent slabs; a later group of the same kind and footprint (±15%) above an earlier one is
marked `same_as=gK dz=…`. At most eight groups and 1 KiB of group lines are shown; the header
records `detected` against `shown`.

A group is a deterministic grouping hypothesis over sampled surfaces, not a Part, solid or motion
claim; the image and category say what the fitting is. Fittings touching walls or resting on the
slab floor merge with the body island and are not listed, so absence proves nothing.

### `MEMBER_TRACKS192`

Links per-slab islands across Z into slender-member tracks (nearest-centroid linking gated by
cross-section size and slope, split at sharp bends, the sixteen longest segments). Each track gives
a candidate sweep centerline in world metres — Z span, start/end/via centroids and mean width — for
lamp arms, legs, posts and wires. A track is a deterministic linking hypothesis, not a Part, joint,
solid or motion claim. The Author may adopt it as a centerline where the image and source views
cohere with it; absence of a track never proves a member absent.

### AXIAL brief

At workspace staging the Host derives `inputs/axial_brief.txt`: the header (frame, envelope,
ground, interior/absence rules), one `BODY_Z` line per slab merging its `BODY_WALLS_Z` row with its
body island's centre, size and dominant yaw, the `support:` line, an `others:` line naming up to
four further islands of at least a tenth of the body footprint, then `FACE_FEATURES`,
`FACE_OPENINGS`, `INTERIOR_ISLANDS` and `MEMBER_TRACKS192` verbatim. The raster stack stays
Host-side for `check_axial`. The brief adds nothing the full text lacks; the full text remains the
evidence of record. `measure_face` answers per-face questions from the same evidence.

## Candidate AXIAL diagnostics

After a successful compile the Author runs Provider `check_axial` (also inside `validate_model`) or
Codex `recon-check-axial … --source … --payload-dir …`. Numbered discrepancies are detector
findings, not mandatory edits. Successful execution exits zero even when residuals remain; invalid
input or execution failure exits nonzero. No residual by itself makes the Author `blocked` or
blocks Host promotion. None of these diagnostics read the semantic image, isolation data, ground
truth or evaluator output, and none is a Host gate or simulator-runtime claim. Every call records
its identity and source/candidate USD closure hashes (payloads under `author_feedback/axial_checks`);
the last measurement is marked `current`, `stale` or `unverified` against the final authored USD and
dependency assets. These labels describe binding only, never quality.

### Stack re-encode and recall

The candidate's boundary surfaces are re-encoded in the source AXIAL grid. Missing source-positive
cells are objective lossy-evidence mismatch; candidate-only surfaces in `UNKNOWN` cells remain soft.
Recall closes the report as a reference line; it is not a target or completion threshold. The
Author should not trace cells, add duplicate shells or fill functional openings to maximize it.

### Face depth, envelope and discrepancies

The report compares the normalized source and compiled candidate directly: both are sampled (600k
points each, fixed seeds) into per-face first-surface depth maps on one grid over the union of both
bounds (about eight samples per column, pitch at least 3 mm and at most half a stack pitch). Per
column the signed difference (candidate minus source, positive outward) is on-plane (≤ 5 mm),
shifted plane (≤ 30 mm) or structural; columns inside a source `FACE_OPENINGS` outline count as
openings; candidate-only columns are excess and source-only columns are missing coverage.

Numbered discrepancies are limited to:

- a face whose source is a plane (at least half its columns on one depth) with the candidate's
  median offset over those plane columns at least 8 mm;
- excess above 5% of the candidate's columns, counting only columns more than two columns from any
  source column (one consolidated defect naming the largest face); and
- a matched candidate opening off its source opening (source `open_face` never counts).

Envelope bounds are a reading only: an image-posed part may extend beyond the source bounds.
Controls raise no discrepancies.

### Readings (never discrepancies)

- **Local first-surface regions** (`local_surface_audit.py`): up to 32 source feature/opening
  regions with a one-cell margin, at most 64 cells per side and at least 1 mm pitch. Each reports
  coverage, candidate-only/missing columns, median signed offset, absolute p90 and centred profile
  p90. Fewer than eight common columns or under 25% source coverage is unassessed; open-face entries
  are excluded. This is not Part segmentation or an occupancy metric.
- **Opening placement**: candidate openings/recesses matched to source `FACE_OPENINGS` by face and
  centre within 6 cm, with offset and size ratios, uncovered source openings and candidate-only
  openings. Depth is not compared.
- **Interior fittings**: per source `INTERIOR_ISLANDS` group, candidate surface samples inside its
  footprint (grown by half a stack pitch) and a height window grown by three stack pitches; at
  least 25 samples reads `modelled`, with the median candidate height's offset from the group band.
- **Protrusion placement**: candidate `FACE_FEATURES` entries (at most 120 mm) matched one-to-one
  to the source's per face within 60 mm when sizes are within 0.5–2× of each other, with offsets
  and size ratios; up to six unbuilt source protrusions and three candidate-only protrusions are
  listed. A source protrusion the candidate lacks may be fused, eroded or texture-only; the image
  decides which protrusions are controls.

## Part correspondence

`compare_part` binds an Author-selected source box and face to an explicitly named candidate link
or visual at q=0 and reports first-surface coverage, depth offset, centred profile residual and a
projection overlay. It does not infer correspondence; occlusion, selection errors, scan noise and
pose can explain residuals, and missing source stays unassessed. `measure_mesh` plane fits are
vertex-weighted estimates with residual/eigenvalue evidence, not automatically accepted normals.
Automatic opening footprints are depth regions, not certified rims; wall estimates can mix body and
articulated surfaces. A recess footprint may be only the deepest subset of a larger tray.

## Verification map

Focused tests in `tests/reconstruction/`: `test_reconstruction_source.py`,
`test_reconstruction_isolation.py`, `test_provider_isolation.py`, `test_axial.py`, `test_face_features.py`, `test_face_openings.py`,
`test_interior_islands.py`, `test_local_surface_audit.py`, `test_reconstruction_provider_audit.py`,
`test_reconstruction_diagnostics.py`, `test_reconstruction_diagnostic_cli.py` and
`test_reconstruction_evidence_policy.py` (evidence authority in the task).
