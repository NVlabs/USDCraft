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

import json
from pathlib import Path

import pytest

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    BodyModel,
    Box,
    Inertia,
    Inertial,
    JointDrive,
    MotionLimits,
    MotionProperties,
    NewtonJointSolverTuning,
    Origin,
    PassiveJointSpring,
    PhysxJointFriction,
    RigidBodyProperties,
)
from sdk import (
    TestContext as SdkTestContext,
)
from storage.artifact_contract import REQUIRED_BACKEND_CHECKS, physics_output_contract
from usdcraft import cli as usdcraft_cli
from usdcraft.compile.command import (
    GENERATION_ROOT_FLOATING_CHECK,
    _copy_urdf_assets,
    _require_asset_structure_validation,
    _require_backend_validation,
    _validate_usd_dependency_closure,
    check_model,
    compile_model,
)
from usdcraft.compile.compiler import (
    _run_compiler_owned_baseline_tests,
    compile_timeout_seconds,
    compile_urdf_report_maybe_timeout,
)
from usdcraft.compile.physics_checks import (
    GENERATION_GRAVITY_HOLD_COVERAGE_CHECK,
    GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK,
    GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK,
    GENERATION_MOVABLE_BODY_PHYSICS_CHECK,
    GENERATION_PHYSX_ARMATURE_STABILITY_CHECK,
    _check_generation_gravity_hold_coverage,
    _check_generation_physx_armature_stability,
    _gravity_hold_estimate,
)

MINIMAL_MODEL = """from __future__ import annotations

from sdk import ArticulatedObject, Box, Origin, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="compile_smoke_box")
    model.meta["mode"] = "static"
    root = model.part("body")
    root.visual(Box((0.20, 0.12, 0.08)), origin=Origin(xyz=(0.0, 0.0, 0.04)), name="body")
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
"""

FIXED_ONLY_MODEL = """from __future__ import annotations

from sdk import ArticulatedObject, ArticulationType, BaseMode, Box, Origin, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="fixed_only_model", base_mode=BaseMode.FIXED)
    model.meta["mode"] = "static"
    base = model.part("base")
    base.visual(Box((0.10, 0.10, 0.10)), origin=Origin(xyz=(0.0, 0.0, 0.05)))
    cap = model.part("cap")
    cap.visual(Box((0.10, 0.10, 0.10)), origin=Origin(xyz=(0.0, 0.0, 0.05)))
    model.articulation(
        "fixed_mount",
        ArticulationType.FIXED,
        parent=base,
        child=cap,
        origin=Origin(xyz=(0.0, 0.0, 0.10)),
    )
    return model


def run_tests() -> TestReport:
    return TestContext(object_model).report()


object_model = build_object_model()
"""

LEVER_COLLISION_MODEL = """from __future__ import annotations

from sdk import (
    ArticulatedObject,
    ArticulationType,
    BaseMode,
    Box,
    Inertial,
    MotionLimits,
    MotionProperties,
    Origin,
    PhysxJointFriction,
    RigidBodyProperties,
    TestContext,
    TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="sampled_pose_lever_collision",
        base_mode=BaseMode.FIXED,
    )
    model.meta["mode"] = "articulated"
    model.physics_material(
        "lever_contact",
        static_friction=0.5,
        dynamic_friction=0.4,
    )

    base = model.part("base")
    base.visual(
        Box((0.20, 0.04, 0.02)),
        origin=Origin(xyz=(0.07, 0.0, 0.0)),
        name="fixed_handle",
    )
    base.visual(
        Box((0.02, 0.04, 0.02)),
        origin=Origin(xyz=(-0.05, 0.0, 0.03)),
        name="pivot_pin",
    )

    lever = model.part(
        "lever",
        inertial=Inertial.from_geometry(Box((0.20, 0.04, 0.02)), mass=0.2),
        rigid_body=RigidBodyProperties(physics_material="lever_contact"),
    )
    lever.visual(Box((0.02, 0.04, 0.02)), name="pivot_barrel")
    lever.visual(
        Box((0.20, 0.04, 0.02)),
        origin=Origin(xyz=(0.12, 0.0, 0.015)),
        name="free_handle",
    )
    model.articulation(
        "squeeze",
        ArticulationType.REVOLUTE,
        parent=base,
        child=lever,
        origin=Origin(xyz=(-0.05, 0.0, 0.03)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(effort=20.0, velocity=1.0, lower=0.0, upper=0.6),
        motion_properties=MotionProperties(damping=0.1, friction=0.05),
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(0.1),
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.allow_overlap(
        "base",
        "lever",
        elem_a="pivot_pin",
        elem_b="pivot_barrel",
        reason="The lever barrel is intentionally captured around its pivot pin.",
    )
    return ctx.report()


object_model = build_object_model()
"""


def _baseline_report_for_model(
    object_model: ArticulatedObject,
    tmp_path: Path,
    *,
    name: str,
    authored_report: object | None = None,
) -> object:
    script_path = tmp_path / name / "model.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    # The baseline only needs this source for structural AST checks and asset-root resolution;
    # the already-constructed object model is supplied directly below.
    script_path.write_text("object_model = build_object_model()\n", encoding="utf-8")
    if authored_report is None:
        authored_report = SdkTestContext(object_model).report()
    return _run_compiler_owned_baseline_tests(
        {"object_model": object_model},
        script_path=script_path,
        authored_report=authored_report,
    )


def _complete_prismatic_model(
    *,
    damping: float | None = 0.1,
    friction: float | None = 0.05,
    physx_resistance: PhysxJointFriction | None = None,
    drive: JointDrive | None = None,
) -> ArticulatedObject:
    model = ArticulatedObject("physics_complete", base_mode=BaseMode.FIXED)
    model.physics_material(
        "slider_contact",
        static_friction=0.5,
        dynamic_friction=0.4,
    )
    base = model.part("base")
    base.visual(Box((0.1, 0.1, 0.1)), origin=Origin(xyz=(0.0, 0.0, 0.05)))
    slider_geometry = Box((0.1, 0.1, 0.1))
    slider = model.part(
        "slider",
        inertial=Inertial.from_geometry(slider_geometry, mass=0.2),
        rigid_body=RigidBodyProperties(physics_material="slider_contact"),
    )
    slider.visual(slider_geometry, origin=Origin(xyz=(0.0, 0.0, 0.05)))
    model.articulation(
        "slide",
        ArticulationType.PRISMATIC,
        parent=base,
        child=slider,
        origin=Origin(xyz=(0.0, 0.0, 0.1)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=10.0, velocity=0.5, lower=0.0, upper=0.08),
        motion_properties=MotionProperties(damping=damping, friction=friction),
        physx_joint_resistance=physx_resistance or PhysxJointFriction.from_breakaway_effort(4.0),
        drive=drive,
    )
    return model


def _gravity_hold_hinge_model(
    *,
    name: str,
    mass: float,
    com: tuple[float, float, float],
    static_effort: float,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    limits: tuple[float, float] = (-0.2, 0.2),
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity_hold: bool | None = None,
    passive_spring: PassiveJointSpring | None = None,
    drive: JointDrive | None = None,
    newton_joint_solver_tuning: NewtonJointSolverTuning | None = None,
) -> tuple[ArticulatedObject, object]:
    model = ArticulatedObject(f"{name}_gravity_model", base_mode=BaseMode.FIXED)
    base = model.part("base")
    base.visual(Box((0.1, 0.1, 0.1)))
    body_geometry = Box((0.02, 0.02, 0.02))
    body = model.part(
        "body",
        inertial=Inertial.from_geometry(body_geometry, mass=mass),
    )
    body.visual(body_geometry, origin=Origin(xyz=com))
    meta = {} if gravity_hold is None else {"gravity_hold": gravity_hold}
    articulation = model.articulation(
        name,
        ArticulationType.REVOLUTE,
        base,
        body,
        origin=Origin(rpy=origin_rpy),
        axis=axis,
        motion_limits=MotionLimits(
            effort=100.0,
            velocity=2.0,
            lower=limits[0],
            upper=limits[1],
        ),
        motion_properties=MotionProperties(damping=0.0, friction=0.1),
        newton_joint_solver_tuning=newton_joint_solver_tuning,
        passive_spring=passive_spring,
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(static_effort),
        drive=drive,
        meta=meta,
    )
    return model, articulation


def _gravity_hold_report(model: ArticulatedObject) -> object:
    ctx = SdkTestContext(model)
    _check_generation_gravity_hold_coverage(ctx, model)
    return ctx.report()


def _newton_pose_hold_tuning(*, d0: float = 0.999) -> NewtonJointSolverTuning:
    return NewtonJointSolverTuning(
        solimp_friction=(d0, 0.9999, 0.001, 0.5, 2.0),
    )


def _source_inertial(
    *,
    mass: float,
    diagonal: tuple[float, float, float],
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Inertial:
    return Inertial(
        mass=mass,
        inertia=Inertia(
            ixx=diagonal[0],
            ixy=0.0,
            ixz=0.0,
            iyy=diagonal[1],
            iyz=0.0,
            izz=diagonal[2],
        ),
        origin=Origin(rpy=origin_rpy),
    )


def _physx_armature_report(model: ArticulatedObject) -> object:
    ctx = SdkTestContext(model)
    _check_generation_physx_armature_stability(ctx, model)
    return ctx.report()


def _contact_supported_shelf(*, gap: float = 0.0) -> ArticulatedObject:
    model = ArticulatedObject("contact_supported_shelf", base_mode=BaseMode.FIXED)
    model.physics_material(
        "rack_contact",
        static_friction=0.7,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    model.physics_material(
        "product_contact",
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
    )
    shelf = model.part("shelf")
    shelf.visual(
        Box((1.0, 0.5, 0.1)),
        origin=Origin(xyz=(0.0, 0.0, 0.5)),
        name="shelf_slab",
    )
    product_geometry = Box((0.1, 0.1, 0.2))
    for index, x in enumerate((-0.2, 0.2), start=1):
        product = model.part(
            f"product_{index}",
            inertial=Inertial.from_geometry(
                product_geometry,
                mass=0.25,
                origin=Origin(xyz=(0.0, 0.0, 0.1)),
            ),
            rigid_body=RigidBodyProperties(physics_material="product_contact"),
        )
        product.visual(
            product_geometry,
            origin=Origin(xyz=(0.0, 0.0, 0.1)),
            name="body",
        )
        model.articulation(
            f"place_product_{index}",
            ArticulationType.FLOATING,
            parent=shelf,
            child=product,
            origin=Origin(xyz=(x, 0.0, 0.55 + gap)),
        )
    return model


def _independent_table_and_chairs() -> ArticulatedObject:
    model = ArticulatedObject("independent_table_and_chairs", base_mode=BaseMode.FLOATING)
    model.physics_material(
        "wood_contact",
        static_friction=0.7,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    table_geometry = Box((1.0, 0.6, 0.75))
    table = model.part(
        "table",
        inertial=Inertial.from_geometry(
            table_geometry,
            mass=18.0,
            origin=Origin(xyz=(0.0, 0.0, 0.375)),
        ),
        rigid_body=RigidBodyProperties(physics_material="wood_contact"),
    )
    table.visual(table_geometry, origin=Origin(xyz=(0.0, 0.0, 0.375)), name="body")

    chair_geometry = Box((0.45, 0.45, 0.9))
    for name, x in (("left_chair", -0.8), ("right_chair", 0.8)):
        chair = model.part(
            name,
            inertial=Inertial.from_geometry(
                chair_geometry,
                mass=5.0,
                origin=Origin(xyz=(0.0, 0.0, 0.45)),
            ),
            rigid_body=RigidBodyProperties(physics_material="wood_contact"),
        )
        chair.visual(
            chair_geometry,
            origin=Origin(xyz=(0.0, 0.0, 0.45)),
            name="body",
        )
        model.articulation(
            f"place_{name}",
            ArticulationType.FLOATING,
            parent=table,
            child=chair,
            origin=Origin(xyz=(x, 0.0, 0.0)),
        )
    return model


@pytest.mark.slow
def test_compile_model_runs_strict_urdf_usd_with_render_diagnostics(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL, encoding="utf-8")

    result = compile_model(model, tmp_path / "build")

    assert result.urdf_path.is_file()
    assert result.usdc_path.is_file()
    assert result.newton_usdc_path.is_file()
    assert result.usdz_path.is_file()
    assert len(result.render_paths) == 7
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["schema_version"] == 2
    assert report["validation_profile"] == "generation"
    # Backend-mapping facts are informational notes, kept out of the warnings list.
    assert isinstance(report["notes"], list)
    assert not any("Isaac Sim 6 MJC" in warning for warning in report["warnings"])
    assert report["asset_structure"] == physics_output_contract()
    assert len(report["artifacts"]["isaac/model.usdc"]) == 64
    assert len(report["artifacts"]["isaac/model_newton.usdc"]) == 64
    for path in (
        "isaac/asset.usda",
        "isaac/physics.usdc",
        "isaac/physx.usda",
        "isaac/mujoco.usda",
    ):
        assert len(report["artifacts"][path]) == 64
        assert path in report["usd_dependency_closure"]
    assert "usd_strict" in report["checks_run"]
    assert "physx_usd_strict" in report["checks_run"]
    assert "newton_usd_strict" in report["checks_run"]
    assert "usd_dependency_closure" in report["checks_run"]
    assert "authored_tests" not in report["checks_run"]
    assert "compiler_baseline_qc" not in report["checks_run"]
    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK not in report["checks_run"]
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in report["checks_run"]
    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK not in report["checks_run"]
    assert GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK not in report["checks_run"]
    assert GENERATION_PHYSX_ARMATURE_STABILITY_CHECK not in report["checks_run"]
    assert "generation_render_diagnostics" in report["checks_run"]


def test_check_model_runs_authored_tests_as_noncanonical_diagnostics(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL, encoding="utf-8")

    result = check_model(model)

    assert result.model_path == model.resolve()
    assert result.signal_bundle.status == "success"
    assert not (tmp_path / "compile_report.json").exists()
    assert not (tmp_path / "build").exists()


def test_check_cli_exposes_stable_authoring_only_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL, encoding="utf-8")

    assert usdcraft_cli.main(["check", str(model)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "status=success",
        "scope=authoring_only",
        f"model={model.resolve()}",
    ]
    assert not (tmp_path / "compile_report.json").exists()


@pytest.mark.slow
def test_check_model_reports_failed_authored_test_without_changing_host_compile(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.py"
    model.write_text(
        MINIMAL_MODEL.replace(
            "    return TestContext(object_model).report()",
            "    ctx = TestContext(object_model)\n"
            '    ctx.check("requested_affordance", False, "handle clearance is missing")\n'
            "    return ctx.report()",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="requested_affordance.*handle clearance is missing"):
        check_model(model)

    artifacts = compile_model(model, tmp_path / "build", render=False)
    assert artifacts.report_path.is_file()


@pytest.mark.slow
def test_generation_host_requires_effective_floating_root(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(FIXED_ONLY_MODEL, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Generation assets must export a floating root"):
        compile_model(
            model,
            tmp_path / "generation-build",
            render=False,
            require_generation_root_floating=True,
        )

    # Direct SDK compilation still supports non-Generation fixed-root assets.
    direct = compile_model(model, tmp_path / "direct-build", render=False)
    direct_report = json.loads(direct.report_path.read_text(encoding="utf-8"))
    assert GENERATION_ROOT_FLOATING_CHECK not in direct_report["checks_run"]


def test_generation_host_records_floating_root_check(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL, encoding="utf-8")

    artifacts = compile_model(
        model,
        tmp_path / "build",
        render=False,
        require_generation_root_floating=True,
    )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert GENERATION_ROOT_FLOATING_CHECK in report["checks_run"]


def test_compile_model_does_not_run_authored_semantic_tests(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(
        MINIMAL_MODEL.replace(
            "return TestContext(object_model).report()",
            'raise RuntimeError("semantic assertion should not block export")',
        ),
        encoding="utf-8",
    )

    result = compile_model(model, tmp_path / "build", render=False)

    assert result.usdc_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert "authored_tests" not in report["checks_run"]


def test_compile_model_does_not_require_generation_physics_policy_fields(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(
        LEVER_COLLISION_MODEL.replace(
            "        motion_properties=MotionProperties(damping=0.1, friction=0.05),\n"
            "        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(0.1),\n",
            "",
        ),
        encoding="utf-8",
    )

    result = compile_model(model, tmp_path / "build", render=False)

    assert result.usdc_path.is_file()
    assert result.newton_usdc_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in report["checks_run"]


def test_compile_model_keeps_render_failure_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL, encoding="utf-8")

    def fail_render(*args: object, **kwargs: object) -> tuple[Path, ...]:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr("usdcraft.compile.command.render_urdf_views", fail_render)

    result = compile_model(model, tmp_path / "build")

    assert result.render_paths == ()
    assert result.usdc_path.is_file()
    assert any("renderer unavailable" in warning for warning in result.warnings)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"


def test_compile_model_still_rejects_invalid_python(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SyntaxError"):
        compile_model(model, tmp_path / "build", render=False)


def test_compile_model_still_rejects_invalid_sdk_usage(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(MINIMAL_MODEL.replace("root.visual(", "root.visaul("), encoding="utf-8")

    with pytest.raises(RuntimeError, match="AttributeError.*visaul"):
        compile_model(model, tmp_path / "build", render=False)


def test_compile_model_still_rejects_unexportable_sdk_topology(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(
        MINIMAL_MODEL.replace(
            "    return model\n",
            '    orphan = model.part("orphan")\n'
            '    orphan.visual(Box((0.05, 0.05, 0.05)), name="orphan")\n'
            "    return model\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly one root part"):
        compile_model(model, tmp_path / "build", render=False)


def test_compile_model_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.py not found"):
        compile_model(tmp_path / "missing.py", tmp_path / "build")


def test_copy_urdf_assets_keeps_only_referenced_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    used_mesh = source / "assets" / "meshes" / "used.obj"
    used_texture = source / "assets" / "textures" / "used.png"
    unused_mesh = source / "assets" / "meshes" / "unused.obj"
    stale_output = output / "assets" / "meshes" / "stale.obj"
    for path, content in (
        (used_mesh, b"mesh"),
        (used_texture, b"texture"),
        (unused_mesh, b"unused"),
        (stale_output, b"stale"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    copied = _copy_urdf_assets(
        """
        <robot name="asset_pruning">
          <link name="body">
            <visual>
              <geometry><mesh filename="assets/meshes/used.obj"/></geometry>
              <material name="paint">
                <texture filename="assets/textures/used.png"/>
              </material>
            </visual>
          </link>
        </robot>
        """,
        source_root=source,
        output_dir=output,
    )

    assert tuple(path.relative_to(output).as_posix() for path in copied) == (
        "assets/meshes/used.obj",
        "assets/textures/used.png",
    )
    assert (output / "assets" / "meshes" / "used.obj").read_bytes() == b"mesh"
    assert (output / "assets" / "textures" / "used.png").read_bytes() == b"texture"
    assert not (output / "assets" / "meshes" / "unused.obj").exists()
    assert not stale_output.exists()


def test_dependency_closure_rejects_authored_absolute_asset_path(tmp_path: Path) -> None:
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"texture")
    stage = tmp_path / "model.usda"
    stage.write_text(
        '#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {\n'
        f"    custom asset test = @{texture.as_posix()}@\n"
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="authors an absolute dependency"):
        _validate_usd_dependency_closure(stage, output_dir=tmp_path)


def test_dependency_closure_reports_relative_in_output_files(tmp_path: Path) -> None:
    texture = tmp_path / "textures" / "texture.png"
    texture.parent.mkdir()
    texture.write_bytes(b"texture")
    stage = tmp_path / "model.usda"
    stage.write_text(
        '#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {\n'
        "    custom asset test = @textures/texture.png@\n"
        "}\n",
        encoding="utf-8",
    )

    assert _validate_usd_dependency_closure(stage, output_dir=tmp_path) == (
        "model.usda",
        "textures/texture.png",
    )


def test_backend_validation_evidence_is_required_for_both_backends() -> None:
    with pytest.raises(RuntimeError, match="did not report"):
        _require_backend_validation({})
    with pytest.raises(RuntimeError, match="newton"):
        _require_backend_validation(
            {
                "backend_validation": {
                    "physx": {"status": "success"},
                    "newton": {"status": "failed"},
                }
            }
        )
    _require_backend_validation(
        {
            "backend_validation": {
                backend: {
                    "status": "success",
                    "checks": sorted(checks),
                    "runtime_status": "not_run",
                }
                for backend, checks in REQUIRED_BACKEND_CHECKS.items()
            }
        }
    )
    incomplete = {
        backend: {
            "status": "success",
            "checks": sorted(checks),
            "runtime_status": "not_run",
        }
        for backend, checks in REQUIRED_BACKEND_CHECKS.items()
    }
    incomplete["physx"]["checks"].remove("physx_schema_consistency")
    with pytest.raises(RuntimeError, match="physx"):
        _require_backend_validation({"backend_validation": incomplete})

    missing_root_binding = {
        backend: {
            "status": "success",
            "checks": sorted(checks),
            "runtime_status": "not_run",
        }
        for backend, checks in REQUIRED_BACKEND_CHECKS.items()
    }
    missing_root_binding["newton"]["checks"].remove("articulation_root_body_binding")
    with pytest.raises(RuntimeError, match="newton"):
        _require_backend_validation({"backend_validation": missing_root_binding})


def test_asset_structure_profile_evidence_is_required() -> None:
    expected = {
        "version": "3.0",
        "profile": "minimal_shared_physics",
        "entrypoints_flattened": True,
        "interface_usd": "isaac/asset.usda",
        "layers": {
            "physics": "isaac/physics.usdc",
            "physx": "isaac/physx.usda",
            "newton_mjc": "isaac/mujoco.usda",
        },
    }
    _require_asset_structure_validation({"asset_structure": expected})
    with pytest.raises(RuntimeError, match="minimal_shared_physics"):
        _require_asset_structure_validation({"asset_structure": {**expected, "profile": "other"}})


def test_generation_physics_gate_requires_movable_children_and_fixed_descendants(
    tmp_path: Path,
) -> None:
    model = ArticulatedObject("moving_subassembly", base_mode=BaseMode.FIXED)
    base = model.part("base")
    base.visual(Box((0.2, 0.2, 0.1)))
    bracket = model.part("fixed_bracket")
    bracket.visual(Box((0.1, 0.1, 0.1)))
    door = model.part("door")
    door.visual(Box((0.2, 0.02, 0.2)))
    handle = model.part("door_handle")
    handle.visual(Box((0.05, 0.02, 0.02)))
    model.articulation("mount", ArticulationType.FIXED, base, bracket)
    model.articulation(
        "hinge",
        ArticulationType.REVOLUTE,
        bracket,
        door,
        motion_limits=MotionLimits(effort=5.0, velocity=1.0, lower=0.0, upper=1.0),
        motion_properties=MotionProperties(damping=0.1, friction=0.05),
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(0.1),
    )
    model.articulation("handle_mount", ArticulationType.FIXED, door, handle)

    report = _baseline_report_for_model(model, tmp_path, name="moving_subassembly")
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK in failures
    details = failures[GENERATION_MOVABLE_BODY_PHYSICS_CHECK]
    assert "part='door'" in details
    assert "part='door_handle'" in details
    assert "inertial" in details
    assert "rigid_body.physics_material" in details
    assert "part='base'" not in details
    assert "part='fixed_bracket'" not in details
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in failures


def test_generation_physics_gate_requires_floating_articulated_root(tmp_path: Path) -> None:
    model = ArticulatedObject("floating_carton", base_mode=BaseMode.FLOATING)
    body = model.part("carton_body")
    body.visual(Box((0.4, 0.3, 0.2)))
    flap = model.part("flap")
    flap.visual(Box((0.4, 0.01, 0.1)))
    model.articulation(
        "flap_hinge",
        ArticulationType.REVOLUTE,
        body,
        flap,
        motion_limits=MotionLimits(effort=2.0, velocity=1.0, lower=-1.0, upper=1.0),
        motion_properties=MotionProperties(damping=0.05, friction=0.02),
    )

    report = _baseline_report_for_model(model, tmp_path, name="floating_carton")
    failures = {failure.name: failure.details for failure in report.failures}
    details = failures[GENERATION_MOVABLE_BODY_PHYSICS_CHECK]

    assert "part='carton_body'" in details
    assert "part='flap'" in details


def test_generation_physics_gate_requires_explicit_scalar_joint_dynamics(
    tmp_path: Path,
) -> None:
    model = _complete_prismatic_model(friction=None)

    report = _baseline_report_for_model(model, tmp_path, name="missing_joint_friction")
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK not in failures
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK in failures
    details = failures[GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK]
    assert "joint='slide'" in details
    assert "motion_properties.friction" in details
    assert "missing non-negative finite motion_properties.damping" not in details


def test_generation_physics_gate_requires_explicit_physx_resistance(tmp_path: Path) -> None:
    model = _complete_prismatic_model()
    model.articulations[0].physx_joint_resistance = None

    report = _baseline_report_for_model(model, tmp_path, name="missing_physx_resistance")
    failures = {failure.name: failure.details for failure in report.failures}

    details = failures[GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK]
    assert "explicit physx_joint_resistance" in details


def test_generation_physics_gate_accepts_complete_scalar_mechanism(tmp_path: Path) -> None:
    model = _complete_prismatic_model()

    report = _baseline_report_for_model(model, tmp_path, name="complete_mechanism")
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK in report.checks
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK in report.checks
    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK not in failures
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in failures


@pytest.mark.parametrize(
    ("armature", "warns"),
    [
        pytest.param(None, True, id="unmitigated"),
        pytest.param(7.98e-5, False, id="effective-ratio-at-provisional-boundary"),
    ],
)
def test_physx_armature_diagnostic_uses_existing_armature_in_angular_denominator(
    armature: float | None,
    warns: bool,
) -> None:
    model = ArticulatedObject("tiny_rocker", base_mode=BaseMode.FLOATING)
    body = model.part(
        "body",
        inertial=_source_inertial(mass=18.0, diagonal=(0.008, 0.008, 0.008)),
    )
    rocker = model.part(
        "rocker",
        inertial=_source_inertial(mass=0.0046, diagonal=(2e-7, 2e-7, 2e-7)),
    )
    model.articulation(
        "body_to_rocker",
        ArticulationType.REVOLUTE,
        body,
        rocker,
        axis=(0.0, 0.0, 1.0),
        physx_armature=armature,
    )

    report = _physx_armature_report(model)
    armature_warnings = [
        warning
        for warning in report.warnings
        if warning.startswith(GENERATION_PHYSX_ARMATURE_STABILITY_CHECK)
    ]

    assert report.passed
    assert GENERATION_PHYSX_ARMATURE_STABILITY_CHECK in report.checks
    assert bool(armature_warnings) is warns
    if warns:
        warning = armature_warnings[0]
        assert "adjusted source-proxy screening ratio=40000:1" in warning
        assert "max-principal/child-min-principal conditioning ratio=40000:1" in warning
        assert "total physx_armature>=7.98e-05 kg*m^2" in warning
        assert "additional>=7.98e-05 kg*m^2" in warning
        assert "not an NVIDIA threshold or runtime guarantee" in warning
        assert "It never authors armature automatically" in warning


def test_physx_armature_diagnostic_projects_inertial_origin_rotation() -> None:
    model = ArticulatedObject("rotated_inertia", base_mode=BaseMode.FLOATING)
    root = model.part(
        "root",
        inertial=_source_inertial(mass=1.0, diagonal=(0.01, 0.01, 0.01)),
    )
    child = model.part(
        "child",
        inertial=_source_inertial(
            mass=0.1,
            diagonal=(1e-6, 0.02, 0.02),
            origin_rpy=(0.0, 0.0, 0.5 * 3.141592653589793),
        ),
    )
    model.articulation(
        "rotated_axis",
        ArticulationType.REVOLUTE,
        root,
        child,
        axis=(1.0, 0.0, 0.0),
    )

    report = _physx_armature_report(model)

    assert report.passed
    assert not report.warnings


def test_physx_armature_diagnostic_does_not_double_rotate_joint_origin_axis() -> None:
    model = ArticulatedObject("rotated_joint", base_mode=BaseMode.FLOATING)
    root = model.part(
        "root",
        inertial=_source_inertial(mass=1.0, diagonal=(0.01, 0.01, 0.01)),
    )
    child = model.part(
        "child",
        inertial=_source_inertial(mass=0.1, diagonal=(0.02, 1e-6, 0.02)),
    )
    model.articulation(
        "rotated_joint_axis",
        ArticulationType.REVOLUTE,
        root,
        child,
        origin=Origin(rpy=(0.0, 0.0, 0.5 * 3.141592653589793)),
        axis=(1.0, 0.0, 0.0),
    )

    report = _physx_armature_report(model)

    assert report.passed
    assert not report.warnings


def test_physx_armature_diagnostic_keeps_floating_split_components_independent() -> None:
    model = ArticulatedObject("split_components", base_mode=BaseMode.FLOATING)
    root_a = model.part(
        "root_a",
        inertial=_source_inertial(mass=1.0, diagonal=(1e-4, 1e-4, 1e-4)),
    )
    child_a = model.part(
        "child_a",
        inertial=_source_inertial(mass=0.1, diagonal=(1e-6, 1e-6, 1e-6)),
    )
    root_b = model.part(
        "root_b",
        inertial=_source_inertial(mass=10.0, diagonal=(1.0, 1.0, 1.0)),
    )
    child_b = model.part(
        "child_b",
        inertial=_source_inertial(mass=10.0, diagonal=(1.0, 1.0, 1.0)),
    )
    model.articulation("joint_a", ArticulationType.REVOLUTE, root_a, child_a)
    model.articulation("place_b", ArticulationType.FLOATING, root_a, root_b)
    model.articulation("joint_b", ArticulationType.REVOLUTE, root_b, child_b)

    report = _physx_armature_report(model)

    assert report.passed
    assert not report.warnings


def test_physx_armature_diagnostic_keeps_rigid_fixed_edge_in_fixed_base_component() -> None:
    model = ArticulatedObject("fixed_root_component", base_mode=BaseMode.FIXED)
    root = model.part(
        "root",
        inertial=_source_inertial(mass=10.0, diagonal=(0.01, 0.01, 0.01)),
    )
    bracket = model.part(
        "bracket",
        inertial=_source_inertial(mass=0.1, diagonal=(1e-6, 1e-6, 1e-6)),
    )
    child = model.part(
        "child",
        inertial=_source_inertial(mass=0.1, diagonal=(1e-6, 1e-6, 1e-6)),
    )
    model.articulation("mount", ArticulationType.FIXED, root, bracket)
    model.articulation("tiny_hinge", ArticulationType.REVOLUTE, bracket, child)

    report = _physx_armature_report(model)

    assert report.passed
    assert len(report.warnings) == 1
    assert "joint='tiny_hinge'" in report.warnings[0]
    assert "adjusted source-proxy screening ratio=10000:1" in report.warnings[0]
    assert "component=['bracket', 'child', 'root']" in report.warnings[0]


def test_physx_armature_diagnostic_does_not_bridge_through_nonrigid_fixed_edges() -> None:
    model = ArticulatedObject("nonrigid_split", base_mode=BaseMode.FLOATING)
    root_a = model.part(
        "root_a",
        inertial=_source_inertial(mass=1.0, diagonal=(1e-4, 1e-4, 1e-4)),
    )
    child_a = model.part(
        "child_a",
        inertial=_source_inertial(mass=1.0, diagonal=(1e-4, 1e-4, 1e-4)),
    )
    flexible = model.part("flexible", body_model=BodyModel.FLEXIBLE_1D)
    root_b = model.part(
        "root_b",
        inertial=_source_inertial(mass=10.0, diagonal=(1.0, 1.0, 1.0)),
    )
    child_b = model.part(
        "child_b",
        inertial=_source_inertial(mass=10.0, diagonal=(1.0, 1.0, 1.0)),
    )
    model.articulation("joint_a", ArticulationType.REVOLUTE, root_a, child_a)
    model.articulation("attach_flexible", ArticulationType.FIXED, child_a, flexible)
    model.articulation("attach_b", ArticulationType.FIXED, flexible, root_b)
    model.articulation("joint_b", ArticulationType.REVOLUTE, root_b, child_b)

    report = _physx_armature_report(model)

    assert report.passed
    assert not report.warnings


@pytest.mark.parametrize(
    ("armature", "warns"),
    [
        pytest.param(None, True, id="unmitigated"),
        pytest.param(0.5, False, id="effective-ratio-at-provisional-boundary"),
    ],
)
def test_physx_armature_diagnostic_uses_mass_proxy_for_prismatic_joint(
    armature: float | None,
    warns: bool,
) -> None:
    model = ArticulatedObject("tiny_slider", base_mode=BaseMode.FLOATING)
    root = model.part(
        "root",
        inertial=_source_inertial(mass=100.0, diagonal=(1.0, 1.0, 1.0)),
    )
    slider = model.part(
        "slider",
        inertial=_source_inertial(mass=0.5, diagonal=(1e-3, 1e-3, 1e-3)),
    )
    model.articulation(
        "slide",
        ArticulationType.PRISMATIC,
        root,
        slider,
        physx_armature=armature,
    )

    report = _physx_armature_report(model)
    armature_warnings = [
        warning
        for warning in report.warnings
        if warning.startswith(GENERATION_PHYSX_ARMATURE_STABILITY_CHECK)
    ]

    assert report.passed
    assert bool(armature_warnings) is warns
    if warns:
        assert "adjusted source-proxy screening ratio=200:1" in armature_warnings[0]
        assert "total physx_armature>=0.5 kg" in armature_warnings[0]


def test_physx_armature_diagnostic_reports_skipped_incomplete_component() -> None:
    model = ArticulatedObject("incomplete_proxy", base_mode=BaseMode.FLOATING)
    root = model.part("root")
    child = model.part(
        "child",
        inertial=_source_inertial(mass=0.1, diagonal=(1e-6, 1e-6, 1e-6)),
    )
    model.articulation("hinge", ArticulationType.REVOLUTE, root, child)

    report = _physx_armature_report(model)

    assert report.passed
    assert len(report.warnings) == 1
    assert "result=skipped" in report.warnings[0]
    assert "part(s) 'root'" in report.warnings[0]
    assert "No partial inertia-ratio estimate was used" in report.warnings[0]


@pytest.mark.parametrize(
    ("name", "mass", "com", "static_effort", "limits", "origin_rpy", "intent", "fails"),
    [
        (
            "monitor_tilt",
            3.2292,
            (0.0, -0.004195, 0.193619),
            1.5125,
            (-0.20944, 0.20944),
            (-0.244346, 0.0, 0.0),
            None,
            False,
        ),
        (
            "frame_to_lever",
            3.0,
            (0.0, -0.208, 0.0),
            1.2,
            (-0.25, 0.25),
            (0.0, 0.0, 0.0),
            True,
            True,
        ),
    ],
    ids=("cash_register_warns", "citrus_press_explicit_intent_fails"),
)
def test_gravity_hold_diagnostic_reports_undercovered_reference_mechanisms(
    name: str,
    mass: float,
    com: tuple[float, float, float],
    static_effort: float,
    limits: tuple[float, float],
    origin_rpy: tuple[float, float, float],
    intent: bool | None,
    fails: bool,
) -> None:
    model, _ = _gravity_hold_hinge_model(
        name=name,
        mass=mass,
        com=com,
        static_effort=static_effort,
        limits=limits,
        origin_rpy=origin_rpy,
        gravity_hold=intent,
        newton_joint_solver_tuning=_newton_pose_hold_tuning(),
    )

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}
    messages = "\n".join((*report.warnings, *failures.values()))

    assert (GENERATION_GRAVITY_HOLD_COVERAGE_CHECK in failures) is fails
    assert f"joint='{name}'" in messages
    assert "estimated maximum residual gravity load" in messages
    assert "covers only" in messages


@pytest.mark.parametrize(
    ("tuning", "fails"),
    [
        pytest.param(None, True, id="missing"),
        pytest.param(
            NewtonJointSolverTuning(solref_friction=(0.02, 1.0)),
            True,
            id="solref-only",
        ),
        pytest.param(_newton_pose_hold_tuning(d0=0.98), True, id="soft-d0"),
        pytest.param(_newton_pose_hold_tuning(), False, id="pose-hold-d0"),
    ],
)
def test_gravity_hold_diagnostic_requires_explicit_newton_friction_impedance(
    tuning: NewtonJointSolverTuning | None,
    fails: bool,
) -> None:
    model, _ = _gravity_hold_hinge_model(
        name="mailbox_drop_door",
        mass=0.721344,
        com=(0.0, -0.1461, 0.0),
        static_effort=1.242610929,
        limits=(0.0, 1.4835298641951802),
        gravity_hold=True,
        newton_joint_solver_tuning=tuning,
    )

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK not in failures
    assert (GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK in failures) is fails
    if fails:
        details = failures[GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK]
        assert "mjc:frictionloss sets the absolute effort threshold" in details
        assert "NewtonJointSolverTuning(solimp_friction=" in details
        assert "d0 is at least 0.999" in details
    else:
        assert not report.warnings


@pytest.mark.parametrize(
    ("intent", "warns"),
    [(None, True), (False, False)],
    ids=("legacy-warns", "explicit-opt-out"),
)
def test_gravity_hold_newton_impedance_legacy_warning_and_explicit_opt_out(
    intent: bool | None,
    warns: bool,
) -> None:
    model, _ = _gravity_hold_hinge_model(
        name="legacy_loaded_hinge",
        mass=0.5,
        com=(0.0, -0.1, 0.0),
        static_effort=1.0,
        gravity_hold=intent,
    )

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}
    impedance_warnings = [
        warning
        for warning in report.warnings
        if warning.startswith(GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK)
    ]

    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK not in failures
    assert GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK not in failures
    assert bool(impedance_warnings) is warns


@pytest.mark.parametrize(
    ("intent", "fails"),
    [(None, False), (True, True)],
    ids=("legacy-warns", "explicit-intent-fails"),
)
def test_gravity_hold_diagnostic_reports_unmeasurable_downstream_load(
    intent: bool | None,
    fails: bool,
) -> None:
    model, _ = _gravity_hold_hinge_model(
        name="unknown_payload_hinge",
        mass=1.0,
        com=(0.0, -0.2, 0.0),
        static_effort=1.0,
        gravity_hold=intent,
    )
    model.get_part("body").inertial = None

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}
    messages = "\n".join((*report.warnings, *failures.values()))

    assert (GENERATION_GRAVITY_HOLD_COVERAGE_CHECK in failures) is fails
    assert "could not estimate the gravity load" in messages


@pytest.mark.parametrize(
    ("name", "axis", "mass", "com", "static_effort"),
    [
        ("pan_to_tilt", (1.0, 0.0, 0.0), 1.0, (0.0, -0.18, 0.0), 2.42),
        ("gate_hinge", (0.0, 0.0, 1.0), 5.0, (1.0, 0.0, 0.0), 0.01),
    ],
    ids=("covered_tripod", "vertical_axis_gate"),
)
def test_gravity_hold_diagnostic_avoids_covered_and_axis_orthogonal_false_positives(
    name: str,
    axis: tuple[float, float, float],
    mass: float,
    com: tuple[float, float, float],
    static_effort: float,
) -> None:
    model, _ = _gravity_hold_hinge_model(
        name=name,
        mass=mass,
        com=com,
        static_effort=static_effort,
        axis=axis,
        gravity_hold=True,
        newton_joint_solver_tuning=(_newton_pose_hold_tuning() if name == "pan_to_tilt" else None),
    )

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK in report.checks
    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK not in failures
    assert GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK in report.checks
    assert GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK not in failures
    assert not report.warnings


def test_gravity_hold_newton_impedance_still_required_after_partial_spring_support() -> None:
    model, _ = _gravity_hold_hinge_model(
        name="spring_assisted_door",
        mass=1.0,
        com=(0.0, 0.0, 1.0),
        static_effort=10.0,
        limits=(0.0, 1.5707963267948966),
        gravity_hold=True,
        passive_spring=PassiveJointSpring(stiffness=7.0, reference=0.0),
    )

    report = _gravity_hold_report(model)
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_GRAVITY_HOLD_COVERAGE_CHECK not in failures
    assert GENERATION_GRAVITY_HOLD_NEWTON_IMPEDANCE_CHECK in failures


def test_gravity_hold_estimate_subtracts_passive_spring_restoring_effort() -> None:
    unsprung_model, unsprung_joint = _gravity_hold_hinge_model(
        name="dishwasher_door",
        mass=1.0,
        com=(0.0, 0.0, 1.0),
        static_effort=0.1,
        limits=(0.0, 0.5 * 3.141592653589793),
    )
    sprung_model, sprung_joint = _gravity_hold_hinge_model(
        name="dishwasher_door",
        mass=1.0,
        com=(0.0, 0.0, 1.0),
        static_effort=0.1,
        limits=(0.0, 0.5 * 3.141592653589793),
        passive_spring=PassiveJointSpring(stiffness=5.0, reference=0.0),
    )

    unsprung = _gravity_hold_estimate(unsprung_model, unsprung_joint, "revolute")
    sprung = _gravity_hold_estimate(sprung_model, sprung_joint, "revolute")

    assert unsprung is not None
    assert sprung is not None
    assert unsprung[0] == pytest.approx(9.81, abs=1e-6)
    assert 3.0 < sprung[0] < 3.5
    assert sprung[0] < unsprung[0]


def test_gravity_hold_estimate_respects_force_mode_drive_saturation() -> None:
    model, joint = _gravity_hold_hinge_model(
        name="driven_door",
        mass=1.0,
        com=(0.0, 0.0, 1.0),
        static_effort=0.1,
        limits=(0.0, 0.5 * 3.141592653589793),
        drive=JointDrive(
            mode="force",
            target_position=0.0,
            stiffness=20.0,
            damping=0.1,
            max_force=1.0,
        ),
    )

    estimate = _gravity_hold_estimate(model, joint, "revolute")

    assert estimate is not None
    assert estimate[0] == pytest.approx(8.81, abs=1e-6)
    assert estimate[1] == pytest.approx(9.81, abs=1e-6)


def test_gravity_hold_estimate_samples_prismatic_spring_residual_at_limits() -> None:
    model = _complete_prismatic_model()
    joint = model.get_articulation("slide")
    joint.passive_spring = PassiveJointSpring(stiffness=30.0, reference=0.08)

    estimate = _gravity_hold_estimate(model, joint, "prismatic")

    assert estimate is not None
    assert estimate[0] == pytest.approx(1.962, abs=1e-6)
    assert estimate[2] == pytest.approx(0.08)


def test_gravity_hold_estimate_includes_movable_downstream_bodies_at_q_zero() -> None:
    model, joint = _gravity_hold_hinge_model(
        name="carrier_hinge",
        mass=0.001,
        com=(0.0, 0.0, 0.0),
        static_effort=0.1,
        limits=(0.0, 0.1),
    )
    carrier = model.get_part("body")
    payload_geometry = Box((0.02, 0.02, 0.02))
    payload = model.part(
        "payload",
        inertial=Inertial.from_geometry(payload_geometry, mass=1.0),
    )
    payload.visual(payload_geometry, origin=Origin(xyz=(0.0, -1.0, 0.0)))
    model.articulation(
        "payload_hinge",
        ArticulationType.REVOLUTE,
        carrier,
        payload,
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=10.0, velocity=1.0, lower=-0.5, upper=0.5),
        motion_properties=MotionProperties(damping=0.0, friction=0.1),
        physx_joint_resistance=PhysxJointFriction.from_breakaway_effort(0.1),
    )

    estimate = _gravity_hold_estimate(model, joint, "revolute")

    assert estimate is not None
    assert estimate[0] > 9.7


def test_gravity_hold_estimate_excludes_independent_floating_placement_branch() -> None:
    model, joint = _gravity_hold_hinge_model(
        name="rack_hinge",
        mass=0.001,
        com=(0.0, 0.0, 0.0),
        static_effort=0.1,
        limits=(0.0, 0.1),
    )
    carrier = model.get_part("body")
    stock_geometry = Box((0.1, 0.1, 0.1))
    stock = model.part(
        "independent_stock",
        inertial=Inertial.from_geometry(stock_geometry, mass=10.0),
    )
    stock.visual(stock_geometry, origin=Origin(xyz=(0.0, -1.0, 0.0)))
    model.articulation(
        "stock_placement",
        ArticulationType.FLOATING,
        carrier,
        stock,
    )

    estimate = _gravity_hold_estimate(model, joint, "revolute")

    assert estimate is not None
    assert estimate[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("max_force", [None, 10.0], ids=("effort_fallback", "effort_boundary"))
def test_generation_physics_gate_accepts_drive_damping_for_active_joint(
    tmp_path: Path,
    max_force: float | None,
) -> None:
    model = _complete_prismatic_model(
        damping=0.0,
        drive=JointDrive(
            target_position=0.04,
            stiffness=10.0,
            damping=0.2,
            max_force=max_force,
        ),
    )

    report = _baseline_report_for_model(
        model,
        tmp_path,
        name=f"active_drive_damping_{max_force}",
    )
    failures = {failure.name: failure.details for failure in report.failures}

    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK not in failures
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in failures


def test_generation_physics_gate_requires_explicit_zero_passive_damping_with_drive(
    tmp_path: Path,
) -> None:
    model = _complete_prismatic_model(
        damping=None,
        drive=JointDrive(target_position=0.04, stiffness=10.0, damping=0.2),
    )

    report = _baseline_report_for_model(model, tmp_path, name="active_drive_missing_zero")
    failures = {failure.name: failure.details for failure in report.failures}

    details = failures[GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK]
    assert "active drive requires motion_properties.damping explicitly set to 0" in details


def test_generation_physics_gate_rejects_nonzero_passive_damping_with_drive(
    tmp_path: Path,
) -> None:
    model = _complete_prismatic_model(
        damping=0.1,
        drive=JointDrive(target_position=0.04, stiffness=10.0, damping=0.2),
    )

    report = _baseline_report_for_model(model, tmp_path, name="active_drive_damping_conflict")
    failures = {failure.name: failure.details for failure in report.failures}

    details = failures[GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK]
    assert "active drive conflicts with non-zero motion_properties.damping" in details
    assert "motion_properties.damping=0 and use drive.damping" in details


@pytest.mark.parametrize(
    ("max_force", "expected"),
    [
        (-0.1, "drive.max_force must be finite and non-negative"),
        (10.01, "drive.max_force must not exceed motion_limits.effort"),
    ],
    ids=("negative", "above_effort"),
)
def test_generation_physics_gate_rejects_invalid_explicit_drive_max_force(
    tmp_path: Path,
    max_force: float,
    expected: str,
) -> None:
    drive = JointDrive(target_position=0.04, stiffness=10.0, damping=0.2)
    # JointDrive already rejects ordinary negative authoring. Mutate the frozen value here so the
    # The optional semantic baseline's defense-in-depth rule covers malformed authored data too.
    object.__setattr__(drive, "max_force", max_force)
    model = _complete_prismatic_model(damping=0.0, drive=drive)

    report = _baseline_report_for_model(model, tmp_path, name=f"invalid_max_force_{max_force}")
    failures = {failure.name: failure.details for failure in report.failures}

    details = failures[GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK]
    assert expected in details


@pytest.mark.parametrize("max_force", (float("nan"), float("inf")), ids=("nan", "infinite"))
def test_model_validity_gate_rejects_nonfinite_explicit_drive_max_force(
    tmp_path: Path,
    max_force: float,
) -> None:
    drive = JointDrive(target_position=0.04, stiffness=10.0, damping=0.2)
    object.__setattr__(drive, "max_force", max_force)
    model = _complete_prismatic_model(damping=0.0, drive=drive)

    report = _baseline_report_for_model(model, tmp_path, name=f"nonfinite_max_force_{max_force}")
    failures = {failure.name: failure.details for failure in report.failures}

    assert "drive.max_force must be finite" in failures["check_model_valid"]
    assert GENERATION_INTERACTIVE_JOINT_PHYSICS_CHECK not in failures


def test_compiler_baseline_accepts_contact_supported_independent_shelf_goods(
    tmp_path: Path,
) -> None:
    model = _contact_supported_shelf()

    report = _baseline_report_for_model(model, tmp_path, name="contact_supported_shelf")

    assert report.passed
    assert "fail_if_isolated_parts()" in report.checks
    assert GENERATION_MOVABLE_BODY_PHYSICS_CHECK in report.checks
    assert not any("Isolated parts detected" in warning for warning in report.warnings)


def test_compiler_baseline_rejects_suspended_independent_shelf_goods(
    tmp_path: Path,
) -> None:
    model = _contact_supported_shelf(gap=0.01)

    report = _baseline_report_for_model(model, tmp_path, name="suspended_shelf_goods")
    failures = {failure.name: failure.details for failure in report.failures}

    assert "fail_if_isolated_parts()" in failures
    assert "product_1" in failures["fail_if_isolated_parts()"]
    assert "product_2" in failures["fail_if_isolated_parts()"]


def test_compiler_baseline_accepts_freestanding_furniture_with_ground_allowances(
    tmp_path: Path,
) -> None:
    model = _independent_table_and_chairs()
    authored_ctx = SdkTestContext(model)
    for chair_name in ("left_chair", "right_chair"):
        authored_ctx.allow_isolated_part(
            chair_name,
            reason="independent furniture supported by the external simulator ground plane",
        )

    report = _baseline_report_for_model(
        model,
        tmp_path,
        name="freestanding_table_and_chairs",
        authored_report=authored_ctx.report(),
    )

    assert report.passed
    assert "fail_if_isolated_parts()" in report.checks
    assert any(
        "Isolated parts detected but allowed by justification" in warning
        for warning in report.warnings
    )


def test_compile_model_does_not_block_on_sampled_mechanical_self_collision(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(LEVER_COLLISION_MODEL, encoding="utf-8")

    result = compile_model(model_path, tmp_path / "build", render=False)

    assert result.usdc_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert "compiler_baseline_qc" not in report["checks_run"]


@pytest.mark.parametrize("texture_name", ["paper.png", "textures/paper.png"])
def test_compile_materializes_relative_texture_dependency(
    tmp_path: Path, texture_name: str
) -> None:
    from PIL import Image

    from storage.artifact_contract import validate_urdf_dependencies

    texture = tmp_path / texture_name
    texture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (130, 100, 60)).save(texture)
    source = MINIMAL_MODEL.replace(
        'root = model.part("body")',
        f'model.material("paper", texture={texture_name!r})\n    root = model.part("body")',
    ).replace('name="body")', 'name="body", material="paper")')
    model = tmp_path / "model.py"
    model.write_text(source)
    artifacts = compile_model(model, tmp_path / "build", render=False)
    hashes = json.loads(artifacts.report_path.read_text())["artifacts"]
    assert (artifacts.output_dir / texture_name).read_bytes() == texture.read_bytes()
    assert texture_name in hashes
    assert not validate_urdf_dependencies(artifacts.output_dir, hashes)
    (artifacts.output_dir / texture_name).unlink()
    assert any(
        issue.path == texture_name and issue.message == "missing URDF dependency"
        for issue in validate_urdf_dependencies(artifacts.output_dir, hashes)
    )


def test_copy_urdf_dependency_rejects_missing_escape_and_output_collision(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "model.urdf").write_text("a texture cannot replace the exported URDF")
    (tmp_path / "outside.png").write_bytes(b"outside")
    (source / "escape.png").symlink_to(tmp_path / "outside.png")
    for filename, message in (
        ("missing.png", "missing"),
        ("escape.png", "escapes"),
        ("model.urdf", "conflicts"),
        ("../outside.png", "unsafe"),
    ):
        with pytest.raises(RuntimeError, match=message):
            _copy_urdf_assets(
                f'<robot><material><texture filename="{filename}"/></material></robot>',
                source_root=source,
                output_dir=output,
            )


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, 300.0), ("", 300.0), ("  ", 300.0), ("42", 42.0), ("0.5", 0.5)]
)
def test_compile_timeout_defaults_when_unset_and_accepts_positive_seconds(
    raw: str | None, expected: float
) -> None:
    environ = {} if raw is None else {"USDCRAFT_COMPILE_TIMEOUT_SECONDS": raw}
    assert compile_timeout_seconds(environ) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "abc"])
def test_compile_timeout_rejects_invalid_values_without_fallback(
    raw: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="requires a positive finite"):
        compile_timeout_seconds({"USDCRAFT_COMPILE_TIMEOUT_SECONDS": raw})

    model_path = tmp_path / "model.py"
    model_path.write_text(MINIMAL_MODEL, encoding="utf-8")
    monkeypatch.setenv("USDCRAFT_COMPILE_TIMEOUT_SECONDS", raw)
    with pytest.raises(ValueError, match="requires a positive finite"):
        compile_urdf_report_maybe_timeout(model_path, run_checks=True)


def test_compile_worker_always_uses_spawn() -> None:
    from usdcraft.compile.mp_utils import get_mp_context

    # The credential holder is never forked.
    assert get_mp_context().get_start_method() == "spawn"
