from __future__ import annotations

from sdk import ArticulatedObject, BaseMode, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="draft_model", base_mode=BaseMode.FLOATING)
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    # Use it for prompt-specific exact checks, targeted poses, and explicit
    # allowances such as `ctx.allow_overlap(...)` while you can still edit the model.
    # Optional TestContext diagnostics can inspect mesh readiness, isolated groups,
    # geometry islands, current-pose overlap, and sampled articulated poses.
    # For each normal-use mechanism, encode independently derived positive
    # functional clearance with `expect_gap(...)` at decisive poses and use
    # `expect_contact(...)` only for intentional mounts or physical hard stops.
    # If an overlap diagnostic reports an intersection, classify it first: intentional
    # embeddings or nested fits should get a scoped allowance; unintended
    # collisions should be fixed in geometry, support, mount, or pose.

    return ctx.report()


object_model = build_object_model()
