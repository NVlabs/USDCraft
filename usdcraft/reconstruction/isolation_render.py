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

"""Strict fixed-view render evidence used only by the subject-isolation Agent."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from usdcraft.pipeline.hashing import sha256_file as _sha256
from usdcraft.reconstruction.isolation import normalized_scene_identity_sha256
from usdcraft.reconstruction.orientation import (
    BundleBuilder,
    OrientationRenderContractError,
    OrientationRenderUnavailableError,
    SessionFactory,
    _default_bundle_builder,
    _default_session_factory,
    _RenderSession,
    _safe_camera,
    _write_carrier,
)
from usdcraft.reconstruction.source import NormalizedSourceScene
from usdcraft.render.browser import DEFAULT_PROFILE, RenderRequest, RenderResult

ISOLATION_VIEWS: tuple[tuple[str, float, float], ...] = (
    ("azimuth_045", 45.0, 35.0),
    ("azimuth_135", 135.0, 35.0),
    ("azimuth_225", 225.0, 35.0),
    ("azimuth_315", 315.0, 35.0),
    ("top", 45.0, 82.0),
)
ISOLATION_RENDER_SIZE = 512
ISOLATION_RENDER_PROFILE = DEFAULT_PROFILE
ISOLATION_RENDER_MANIFEST_SCHEMA = "reconstruction_subject_isolation_render"


@dataclass(frozen=True, slots=True)
class IsolationRenderEvidence:
    root: Path
    native_contact_sheet: Path
    component_id_contact_sheet: Path
    native_render_manifest: Path
    component_id_render_manifest: Path
    native_view_paths: tuple[Path, ...]
    component_id_view_paths: tuple[Path, ...]


def _id_scene(
    scene: NormalizedSourceScene,
    inventory: Mapping[str, object],
) -> NormalizedSourceScene:
    raw_components = inventory.get("components")
    if not isinstance(raw_components, list):
        raise OrientationRenderContractError("isolation inventory has no component color key")
    colors: dict[str, np.ndarray] = {}
    for item in raw_components:
        if not isinstance(item, Mapping):
            raise OrientationRenderContractError("isolation component color entry is invalid")
        component_id = item.get("component_id")
        raw_rgba = item.get("display_rgba")
        if not isinstance(component_id, str) or not isinstance(raw_rgba, list):
            raise OrientationRenderContractError("isolation component color entry is incomplete")
        rgba = np.asarray(raw_rgba, dtype=np.float64)
        if rgba.shape != (4,) or not np.isfinite(rgba).all():
            raise OrientationRenderContractError("isolation component color entry is non-finite")
        colors[component_id] = np.clip(rgba / 255.0, 0.0, 1.0)
    components = []
    for component in scene.components:
        component_rgba = colors.get(component.component_id)
        if component_rgba is None:
            raise OrientationRenderContractError(
                f"isolation inventory omitted component {component.component_id!r}"
            )
        components.append(
            replace(
                component,
                colors_rgba=np.tile(component_rgba, (len(component.vertices), 1)),
                render_visual=None,
            )
        )
    return replace(scene, components=tuple(components))


def _model_hash(carrier_urdf: Path, carrier_glb: Path, *, scene_identity: str, variant: str) -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("carrier_urdf", _sha256(carrier_urdf)),
        ("carrier_glb", _sha256(carrier_glb)),
        ("scene_identity", scene_identity),
        ("variant", variant),
    ):
        digest.update(f"{label}:{value}\n".encode("utf-8"))
    return digest.hexdigest()


def _render_variant(
    *,
    scene: NormalizedSourceScene,
    source_identity: str,
    variant: str,
    output_dir: Path,
    session: _RenderSession,
) -> tuple[tuple[Path, ...], Path]:
    carrier_dir = output_dir / "carrier"
    carrier_urdf, carrier_glb, carrier_stats = _write_carrier(scene, carrier_dir)
    request = RenderRequest(
        asset_dir=carrier_dir,
        out_dir=output_dir,
        views=ISOLATION_VIEWS,
        size=ISOLATION_RENDER_SIZE,
        profile=ISOLATION_RENDER_PROFILE,
        model_hash=_model_hash(
            carrier_urdf,
            carrier_glb,
            scene_identity=source_identity,
            variant=variant,
        ),
    )
    try:
        result = session.render(request, allow_numpy_fallback=False)
    except Exception as exc:
        raise OrientationRenderUnavailableError(
            f"strict Chromium subject-isolation render failed: {type(exc).__name__}: {exc}"
        ) from exc
    if result is None:
        raise OrientationRenderUnavailableError(
            "strict Chromium subject-isolation renderer is unavailable"
        )
    if not isinstance(result, RenderResult):
        raise OrientationRenderContractError("subject-isolation renderer returned invalid output")
    if result.backend != "viewer_chromium" or result.fallback_reason is not None:
        raise OrientationRenderContractError(
            "subject-isolation evidence requires viewer_chromium and forbids renderer fallback"
        )
    if result.profile != ISOLATION_RENDER_PROFILE:
        raise OrientationRenderContractError(
            f"subject-isolation renderer returned unexpected profile {result.profile!r}"
        )
    expected_names = [view[0] for view in ISOLATION_VIEWS]
    observed_names = [item.get("name") for item in result.images]
    if observed_names != expected_names:
        raise OrientationRenderContractError(
            "subject-isolation renderer returned the wrong fixed view set"
        )
    image_paths: list[Path] = []
    image_payloads: list[dict[str, object]] = []
    resolved_output = output_dir.resolve()
    for expected_name, item in zip(expected_names, result.images):
        raw_path = item.get("path")
        if not isinstance(raw_path, (str, Path)):
            raise OrientationRenderContractError(
                f"subject-isolation render {expected_name!r} has no path"
            )
        path = Path(raw_path).resolve()
        try:
            path.relative_to(resolved_output)
        except ValueError as exc:
            raise OrientationRenderContractError(
                "subject-isolation renderer wrote outside its evidence directory"
            ) from exc
        if not path.is_file():
            raise OrientationRenderContractError(
                f"subject-isolation renderer omitted {expected_name}.png"
            )
        with Image.open(path) as image:
            image.verify()
        image_paths.append(path)
        image_payloads.append(
            {
                "name": expected_name,
                "path": path.relative_to(resolved_output).as_posix(),
                "sha256": _sha256(path),
                "camera": _safe_camera(item.get("camera")),
                "isolation_agent_input": False,
                "reconstruction_agent_input": False,
            }
        )
    manifest_path = output_dir / "render_manifest.json"
    manifest = {
        "schema": ISOLATION_RENDER_MANIFEST_SCHEMA,
        "evidence_role": "subject_isolation_only",
        "variant": variant,
        "backend": result.backend,
        "profile": result.profile,
        "size": ISOLATION_RENDER_SIZE,
        "source_identity_sha256": source_identity,
        "target_frame": {"up_axis": "+Z", "meters_per_unit": 1.0},
        "views": image_payloads,
        "carrier": {
            "urdf_sha256": _sha256(carrier_urdf),
            "glb_sha256": _sha256(carrier_glb),
            **carrier_stats,
            "isolation_agent_input": False,
            "reconstruction_agent_input": False,
        },
        "fallback_reason": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tuple(image_paths), manifest_path


def _sheet_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default()


def _compose_contact_sheet(
    *,
    image_paths: tuple[Path, ...],
    destination: Path,
    title: str,
    component_key: list[Mapping[str, object]] | None,
) -> None:
    if len(image_paths) != len(ISOLATION_VIEWS):
        raise OrientationRenderContractError("contact sheet requires the complete fixed view set")
    tile = ISOLATION_RENDER_SIZE
    columns = 3
    rows = 2
    header_height = 42
    label_height = 24
    legend_columns = 6
    legend_row_height = 18
    legend_rows = math.ceil(len(component_key or ()) / legend_columns)
    legend_height = 0 if component_key is None else 42 + legend_rows * legend_row_height
    canvas = Image.new(
        "RGB",
        (columns * tile, header_height + rows * (tile + label_height) + legend_height),
        (244, 246, 249),
    )
    draw = ImageDraw.Draw(canvas)
    font = _sheet_font()
    draw.rectangle((0, 0, canvas.width, header_height), fill=(20, 27, 39))
    draw.text((14, 13), f"{title} | SUBJECT ISOLATION ONLY", fill=(255, 255, 255), font=font)
    for index, (path, view) in enumerate(zip(image_paths, ISOLATION_VIEWS)):
        column = index % columns
        row = index // columns
        x = column * tile
        y = header_height + row * (tile + label_height)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if image.size != (tile, tile):
            image = image.resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + tile, x + tile, y + tile + label_height), fill=(31, 40, 55))
        draw.text(
            (x + 9, y + tile + 7),
            f"{view[0]}  az={view[1]:.0f}  el={view[2]:.0f}",
            fill=(237, 241, 247),
            font=font,
        )
    if component_key is not None:
        legend_top = header_height + rows * (tile + label_height)
        draw.rectangle((0, legend_top, canvas.width, canvas.height), fill=(232, 235, 241))
        draw.text(
            (14, legend_top + 10),
            "COMPONENT COLOR KEY (selection id = component:<id>)",
            fill=(23, 30, 43),
            font=font,
        )
        column_width = canvas.width // legend_columns
        for index, item in enumerate(component_key):
            column = index % legend_columns
            row = index // legend_columns
            x = column * column_width + 12
            y = legend_top + 34 + row * legend_row_height
            rgba = item.get("display_rgba")
            component_id = item.get("component_id")
            if not isinstance(rgba, list) or len(rgba) < 3 or not isinstance(component_id, str):
                raise OrientationRenderContractError("component color key entry is invalid")
            color = tuple(int(value) for value in rgba[:3])
            draw.rectangle((x, y, x + 13, y + 13), fill=color, outline=(40, 45, 55))
            draw.text((x + 18, y + 2), component_id, fill=(23, 30, 43), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def render_subject_isolation_evidence(
    scene: NormalizedSourceScene,
    inventory: Mapping[str, object],
    output_dir: Path,
    *,
    bundle_builder: BundleBuilder | None = None,
    session_factory: SessionFactory | None = None,
) -> IsolationRenderEvidence:
    """Render native-appearance and component-id contact sheets from the same raw scene."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    native_dir = root / "native"
    component_id_dir = root / "component_id"
    native_dir.mkdir(parents=True, exist_ok=False)
    component_id_dir.mkdir(parents=True, exist_ok=False)
    builder = bundle_builder or _default_bundle_builder
    make_session = session_factory or _default_session_factory
    source_identity = normalized_scene_identity_sha256(scene)
    with tempfile.TemporaryDirectory(prefix="usdcraft-recon-isolation-render-") as temporary:
        dist_dir = Path(temporary) / "dist"
        try:
            builder(dist_dir)
        except Exception as exc:
            if isinstance(exc, (OrientationRenderUnavailableError, OrientationRenderContractError)):
                raise
            raise OrientationRenderUnavailableError(
                f"temporary Viewer build failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not (dist_dir / "render.html").is_file():
            raise OrientationRenderUnavailableError(
                "temporary Viewer build did not produce render.html"
            )
        native_session = make_session(dist_dir)
        try:
            native_paths, native_manifest = _render_variant(
                scene=scene,
                source_identity=source_identity,
                variant="native_appearance",
                output_dir=native_dir,
                session=native_session,
            )
        finally:
            try:
                native_session.close()
            except Exception:
                pass
        component_id_session = make_session(dist_dir)
        try:
            id_paths, id_manifest = _render_variant(
                scene=_id_scene(scene, inventory),
                source_identity=source_identity,
                variant="component_id_colors",
                output_dir=component_id_dir,
                session=component_id_session,
            )
        finally:
            try:
                component_id_session.close()
            except Exception:
                pass

    raw_key = inventory.get("components")
    if not isinstance(raw_key, list) or not all(isinstance(item, Mapping) for item in raw_key):
        raise OrientationRenderContractError("isolation inventory component key is invalid")
    component_key = list(raw_key)
    native_sheet = root / "native_contact_sheet.png"
    id_sheet = root / "component_id_contact_sheet.png"
    _compose_contact_sheet(
        image_paths=native_paths,
        destination=native_sheet,
        title="NATIVE APPEARANCE FIXED MULTIVIEW",
        component_key=None,
    )
    _compose_contact_sheet(
        image_paths=id_paths,
        destination=id_sheet,
        title="COMPONENT-ID FIXED MULTIVIEW",
        component_key=component_key,
    )
    return IsolationRenderEvidence(
        root=root,
        native_contact_sheet=native_sheet,
        component_id_contact_sheet=id_sheet,
        native_render_manifest=native_manifest,
        component_id_render_manifest=id_manifest,
        native_view_paths=native_paths,
        component_id_view_paths=id_paths,
    )


__all__ = [
    "ISOLATION_RENDER_MANIFEST_SCHEMA",
    "ISOLATION_RENDER_PROFILE",
    "ISOLATION_RENDER_SIZE",
    "ISOLATION_VIEWS",
    "IsolationRenderEvidence",
    "render_subject_isolation_evidence",
]
