# Testing

Use `TestContext` to test an authored model's decisive geometric and mechanical relationships.
Return `ctx.report()` from `run_tests()`. Invoke the check tool/command provided by the task to
run authored tests and shared diagnostics, then investigate failures while the model is editable.
A passed self-check establishes only its tested relationships; it does not establish target
identity, visual fidelity, or simulator behavior.

```python
from sdk import TestContext, TestReport


def run_tests() -> TestReport:
    ctx = TestContext(object_model, seed=0)
    # Add exact checks for this model's important clearance, contact or motion requirements.
    return ctx.report()
```

## What to test

Choose a small set of consequential requirements: functional clearance, opening direction,
retained insertion, support, closure or an independently established count. Use meaningful
expected values rather than restating the same constants that built the geometry.
Do not turn each observation item or decorative detail into an assertion.

The check workflow supplies model/root/asset checks, disconnected-group and geometry-island
checks, rest-pose overlap and sampled articulated overlap. Avoid duplicating those broad checks.
Use one or two decisive additional poses for a specific mechanism. Shared sampled overlap uses
up to 128 deterministic poses; bounded scalar joints include rest, lower, upper and midpoint
candidates. Sampling is not continuous swept-volume proof.

A finding can expose a real defect, an intentional interface or a limited approximation.
Repair the defect or document a narrowly scoped physical classification. An allowance neither
adds missing structure nor changes colliders. Preserve named visuals referenced by tests, or
update those references in the same edit.

## Parameters and returned values

- Part arguments accept a registered Part object or its name string. Visual arguments accept a
  named Visual or its name on the referenced Part. Use variables or quoted names consistently.
- `axis` is a positive world axis `"x"`, `"y"` or `"z"`; `axes` accepts a string such as `"xy"`
  or a sequence such as `("x", "y")`.
- Distances and tolerances are metres. `name` overrides the recorded check name.
- Assertion methods record a result and return a boolean. Query methods return values without
  asserting them. Missing/unresolvable query geometry can return `None`; check before indexing.

```text
TestContext(model, seed=0)
ctx.report() -> TestReport
ctx.check(name: str, ok: bool, details: str = "") -> bool
ctx.fail(name: str, details: str) -> bool
ctx.warn(text: str) -> None
```

`check` takes the check name first and condition second. `fail` records a false check.
`warn` adds a warning. Construct reports through `ctx.report()` rather than hand-building counters.
Public report fields include `passed`, `checks_run`, `checks`, `failures`, `warnings`, `allowances`,
`allowed_isolated_parts`, `allowed_overlaps` and `disconnected_geometry_classifications`.
`TestFailure` contains `name` and `details`; an `AllowedOverlap` contains `link_a`, `link_b`,
`reason`, `elem_a` and `elem_b`.

## Exact relationship assertions

```text
ctx.expect_contact(link_a, link_b, *, contact_tol=1e-6,
                   elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_gap(positive_link, negative_link, *, axis, min_gap=None, max_gap=None,
               max_penetration=None, positive_elem=None, negative_elem=None,
               elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_overlap(link_a, link_b, *, axes="xy", min_overlap=0.0,
                   elem_a=None, elem_b=None, name=None) -> bool
ctx.expect_within(inner_link, outer_link, *, axes="xy", margin=0.0,
                  inner_elem=None, outer_elem=None, elem_a=None, elem_b=None,
                  name=None) -> bool
ctx.expect_origin_distance(link_a, link_b, *, axes="xy", min_dist=0.0,
                           max_dist=None, name=None) -> bool
ctx.expect_origin_gap(positive_link, negative_link, *, axis, min_gap=0.0,
                      max_gap=None, name=None) -> bool
```

| Assertion | What it measures |
| --- | --- |
| `expect_contact` | Minimum exact geometry distance <= contact_tol; inspect penetration separately |
| `expect_gap` | Signed axis gap: positive geometry minimum minus negative geometry maximum |
| `expect_overlap` | Projected overlap >= min_overlap on every selected axis; not 3D collision/contact |
| `expect_within` | Selected inner bounds fit inside outer bounds plus allowed slack |
| `expect_origin_distance` | Distance between Part origins on selected axes; not surface distance |
| `expect_origin_gap` | Signed origin-to-origin gap along one positive world axis |

Bounds are inclusive. For `expect_gap`, omitted `min_gap` is derived from `max_penetration`;
`max_penetration=d` allows a lower gap of `-d`. `elem_a`/`elem_b` alias
`positive_elem`/`negative_elem`. For `expect_within`, those aliases refer to inner/outer elements.
**Positive `margin` permits slack outside the outer bounds; it does not require internal clearance.**
Use named cavity-wall geometry and explicit gaps to test a usable empty opening. Whole-part
bounds cannot prove that a nested volume is hollow or unobstructed.

## Poses and spatial queries

```text
ctx.pose(joint_positions=None, **joint_name_positions)
ctx.part_world_position(part) -> tuple[float, float, float] | None
ctx.link_world_position(link) -> tuple[float, float, float] | None
ctx.part_world_aabb(part) -> tuple[Vec3, Vec3] | None
ctx.link_world_aabb(link) -> tuple[Vec3, Vec3] | None
ctx.part_element_world_aabb(part, *, elem) -> tuple[Vec3, Vec3] | None
```

`ctx.pose` temporarily overrides articulation positions and restores them on exit. Its mapping
keys can be joint objects or names; keyword arguments use names. Angular positions are radians,
prismatic positions metres. `link_*` methods alias `part_*`. AABB queries return `(minimum, maximum)`
in the current world pose; origin queries return the Part frame origin.

```python
with ctx.pose({"lid_hinge": 0.5}):
    lid_bounds = ctx.part_world_aabb("lid")
    ctx.check("lid rises", lid_bounds is not None and lid_bounds[1][2] > required_open_height)
```

This fragment assumes the named hinge/lid and a justified `required_open_height` from the actual
mechanism. Verify that positive motion opens or extends in the intended direction, not merely
that the child moved somewhere.

Mimic followers derive from their source; pose the source joint, not the follower.
A spherical pose is a QC-only swing angle within ±radians(cone_deg), on one deterministic tangent
slice. Native spherical motion retains three rotational degrees, which this check does not exhaust.
FLOATING poses use `Origin(xyz=..., rpy=...)` in the placement frame. Default floating samples use
only `Origin()`; optional `joint.meta["qc_samples"]=[Origin(...), ...]` supplies other test poses.

## Scoped classifications and allowances

```text
ctx.allow_overlap(link_a, link_b, *, reason, elem_a=None, elem_b=None) -> None
ctx.allow_isolated_part(part, *, reason) -> None
ctx.allow_coplanar_surfaces(link_a, link_b, *, reason, elem_a=None, elem_b=None) -> None
ctx.classify_disconnected_geometry(part, *, classification, elements, reason) -> None
```

`allow_overlap` describes intentional real interpenetration, such as a captured pin in a simplified
bearing or a small seated trim interface. Containment in a footprint or cavity alone does not need
an allowance. Scope moving-part allowances to exact named elements; a broad pair allowance can
hide unrelated collisions throughout its travel. Allowances apply to both rest and sampled checks.

Use `allow_isolated_part` only for a genuinely independent/contact-supported member or a justified
external-ground arrangement. For an intentional multi-Part isolated assembly, identify each Part.
It does not justify a floating handle, missing mount or prop suspended above a shelf.
`allow_coplanar_surfaces` classifies a specific intended coplanar interface.

`classify_disconnected_geometry` accepts only reported named visuals and a physical reason.
Valid classes are `structural_connected`, `articulated_contact`, `intentional_repetition`,
`contained_prop`, `decorative_offset` and `unexpected_floating`. The first five classify a matching
finding as informational; `unexpected_floating`, unnamed and unlisted islands remain warnings.
The declaration does not physically join disconnected geometry.

```python
ctx.classify_disconnected_geometry(
    "basket", classification="intentional_repetition",
    elements=("vertical_wires", "floor_wires"),
    reason="repeated wire visuals meet the modeled perimeter rails",
)
```

Keep exact contact/support checks where that load path is consequential. Classifications are not
substitutes for geometry or permission to hide every disconnected island.

### Nested sliders

For a hollow sleeve, check transverse fit, free clearance and retained insertion. For an explicitly
simplified solid sleeve proxy, only the corresponding named moving interface may receive a scoped
overlap allowance; that approximation does not establish an open cavity.

The fragment below assumes a Z-axis slide and named visuals in an existing model. It checks
retained projected length at two poses; it does not replace clearance/collision inspection:

```python
for travel, retained in ((0.0, 0.080), (slide_upper, 0.030)):
    with ctx.pose({slide_joint: travel}):
        ctx.expect_within(
            inner_stage, outer_stage, axes="xy", margin=0.0,
            inner_elem="inner_member", outer_elem="outer_sleeve",
            name=f"transverse fit at {travel}",
        )
        ctx.expect_overlap(
            inner_stage, outer_stage, axes="z", min_overlap=retained,
            elem_a="inner_member", elem_b="outer_sleeve",
            name=f"retained insertion at {travel}",
        )
```

Choose travel and retained length from the real mechanism; a bound fitting inside another bound
is not proof of clearance from interior walls. Test named seating surfaces or use focused probes
when whole-Part bounds cannot resolve the relationship.

## Optional structural diagnostics

```text
ctx.fail_if_articulation_origin_far_from_geometry(*, tol=0.015,
                                                reason=None, name=None) -> bool
ctx.warn_if_articulation_origin_far_from_geometry(*, tol=0.015,
                                                reason=None, name=None) -> bool
ctx.warn_if_coplanar_surfaces(*, max_pose_samples=32, plane_tol=0.001,
                            min_overlap=0.02, min_overlap_ratio=0.35,
                            ignore_adjacent=True, ignore_fixed=True, name=None) -> bool
```

Joint-origin distance uses an absolute tolerance; it is not scale-aware. Use it to investigate
an uncertain mount, not to force arbitrary geometry within a universal distance. `reason` can
explain a relaxed tolerance. A FLOATING placement edge has no mechanical mount.

Coplanar diagnostics use plane separation, projected distance and area-ratio thresholds.
Use them for an actual ambiguity such as an unintended duplicate surface, and classify intentional
interfaces locally. Heuristic findings need inspection before changing supported geometry.
