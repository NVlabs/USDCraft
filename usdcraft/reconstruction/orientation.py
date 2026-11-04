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

"""Strict Chromium render session and normalized-source carrier for Reconstruction evidence.

The source-overlay carrier (``source_overlay.py``) and the subject-isolation contact sheets
(``isolation_render.py``) share this module: the carrier writes an already-normalized source
(metres, Z-up) as a URDF plus GLB the Viewer render page can load, ``_ChromiumChannelLadder``
tries Playwright's bundled Chromium and then the installed stable Chrome, and the errors name
the strict contract — the numpy renderer is never an accepted fallback and every render must
come back from ``viewer_chromium`` without a fallback reason.

This module does not read or mutate ``viewer/web/dist``.  It builds the checked-out Viewer
source into a temporary directory and starts the existing headless render page from there.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from usdcraft.pipeline.hashing import sha256_file as _sha256_file
from usdcraft.render.browser import RenderRequest, RenderResult, ViewerRenderSession

if TYPE_CHECKING:
    from trimesh.visual import ColorVisuals, TextureVisuals

    from usdcraft.reconstruction.source import NormalizedSourceArtifacts, NormalizedSourceScene


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_WEB_ROOT = REPO_ROOT / "viewer" / "web"


class OrientationPreviewError(RuntimeError):
    """Base class for deterministic strict-render failures."""


class OrientationCarrierError(OrientationPreviewError):
    """The normalized source could not be represented by the render carrier."""


class OrientationRenderUnavailableError(OrientationPreviewError):
    """The required Chromium/PBR renderer could not produce the render."""


class OrientationRenderContractError(OrientationPreviewError):
    """A renderer returned output outside the strict render contract."""


class _RenderSession(Protocol):
    def render(
        self,
        request: RenderRequest,
        *,
        allow_numpy_fallback: bool = True,
    ) -> RenderResult | None: ...

    def close(self) -> None: ...


BundleBuilder = Callable[[Path], None]
SessionFactory = Callable[[Path], _RenderSession]


class _ChromiumChannelLadder:
    """Try Playwright's bundled Chromium, then the installed stable Chrome channel.

    Both routes run the same Chromium render page and satisfy the ``viewer_chromium`` backend
    contract.  Desktop source checkouts commonly have stable Chrome but not Playwright's separately
    downloaded browser; trying the second channel keeps the preview strict without adding a
    software-renderer fallback.
    """

    def __init__(self, dist_dir: Path):
        self._dist_dir = dist_dir
        self._sessions: list[ViewerRenderSession] = []

    def render(
        self,
        request: RenderRequest,
        *,
        allow_numpy_fallback: bool = True,
    ) -> RenderResult | None:
        if allow_numpy_fallback:
            raise OrientationRenderContractError(
                "Recon Chromium channel ladder must be called with fallback disabled"
            )
        for channel in ("chromium", "chrome"):
            session = ViewerRenderSession(
                dist_dir=self._dist_dir,
                browser_channel=channel,
            )
            self._sessions.append(session)
            result = session.render(request, allow_numpy_fallback=False)
            if result is not None:
                return result
        return None

    def close(self) -> None:
        for session in reversed(self._sessions):
            session.close()
        self._sessions.clear()


def _component_value(component: object, name: str, default: object = None) -> object:
    if isinstance(component, Mapping):
        return component.get(name, default)
    return getattr(component, name, default)


def _finite_vertices(component: object) -> np.ndarray:
    vertices = np.asarray(_component_value(component, "vertices"), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise OrientationCarrierError("normalized component vertices must have shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise OrientationCarrierError("normalized component contains non-finite vertices")
    return vertices


def _vertex_colors(component: object, vertex_count: int) -> np.ndarray | None:
    raw = _component_value(component, "colors_rgba")
    if raw is None:
        # Accept ``colors`` as an alternative spelling of ``colors_rgba``.
        raw = _component_value(component, "colors")
    if raw is None:
        return None
    colors = np.asarray(raw)
    if colors.ndim != 2 or len(colors) != vertex_count or colors.shape[1] not in {3, 4}:
        return None
    if np.issubdtype(colors.dtype, np.floating):
        if not np.isfinite(colors).all():
            return None
        if float(colors.max(initial=0.0)) <= 1.0:
            colors = np.rint(np.clip(colors, 0.0, 1.0) * 255.0)
        else:
            colors = np.rint(np.clip(colors, 0.0, 255.0))
    colors = colors.astype(np.uint8, copy=False)
    if colors.shape[1] == 3:
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.concatenate((colors, alpha), axis=1)
    return colors


def _native_render_visual(component: object, vertex_count: int) -> TextureVisuals | None:
    """Copy UV ownership while sharing the immutable source material and texture atlas."""

    visual: Any = _component_value(component, "render_visual")
    if visual is None or getattr(visual, "kind", None) != "texture":
        return None
    if getattr(visual, "material", None) is None:
        return None
    uv = getattr(visual, "uv", None)
    if uv is not None:
        values = np.asarray(uv)
        if values.ndim != 2 or len(values) != vertex_count or values.shape[1] < 2:
            return None
        if not np.isfinite(values[:, :2]).all():
            return None
    try:
        from trimesh.visual import TextureVisuals

        return TextureVisuals(
            uv=np.array(uv, copy=True) if uv is not None else None,
            material=visual.material,
            face_materials=(
                visual.face_materials.copy() if visual.face_materials is not None else None
            ),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _neutral_material() -> object:
    import trimesh

    return trimesh.visual.material.PBRMaterial(
        name="recon_source_neutral",
        baseColorFactor=np.asarray((184, 189, 199, 255), dtype=np.uint8),
        metallicFactor=0.05,
        roughnessFactor=0.52,
    )


def _material_has_texture(material: object) -> bool:
    return any(
        getattr(material, name, None) is not None
        for name in (
            "baseColorTexture",
            "emissiveTexture",
            "normalTexture",
            "occlusionTexture",
            "metallicRoughnessTexture",
        )
    )


def _write_carrier(
    scene: NormalizedSourceScene, carrier_dir: Path
) -> tuple[Path, Path, dict[str, int]]:
    """Write a one-link URDF whose visual is a normalized GLB scene.

    Meshes stay exact.  glTF can natively store point primitives, and Three's GLTFLoader displays
    them; a sub-pixel sentinel triangle is added only for a point-only scene because the current
    URDF viewer's visual-coverage assertion counts triangle meshes.  The sentinel is recorded in
    the manifest and is never used by AXIAL or evaluation.
    """

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - a declared project dependency
        raise OrientationCarrierError(f"trimesh is unavailable: {exc}") from exc

    components = tuple(getattr(scene, "components", ()))
    if not components:
        raise OrientationCarrierError("normalized source has no components to preview")

    carrier_scene = trimesh.Scene()
    mesh_count = 0
    point_count = 0
    native_material_count = 0
    native_texture_count = 0
    vertex_color_count = 0
    neutral_fallback_count = 0
    all_vertices: list[np.ndarray] = []
    for index, component in enumerate(components):
        vertices = _finite_vertices(component)
        all_vertices.append(vertices)
        component_id = str(_component_value(component, "component_id", f"component_{index:04d}"))
        geometry_name = f"source_{index:04d}_{component_id}"
        colors = _vertex_colors(component, len(vertices))
        raw_faces = _component_value(component, "faces")
        kind = str(
            _component_value(component, "kind", "mesh" if raw_faces is not None else "points")
        )

        if kind == "mesh" and raw_faces is not None:
            faces = np.asarray(raw_faces, dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
                raise OrientationCarrierError(
                    f"normalized mesh component {component_id!r} faces must have shape (M, 3)"
                )
            if np.any(faces < 0) or np.any(faces >= len(vertices)):
                raise OrientationCarrierError(
                    f"normalized mesh component {component_id!r} has an out-of-range face index"
                )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            native_visual = _native_render_visual(component, len(vertices))
            if native_visual is not None:
                mesh.visual = native_visual
                native_material_count += 1
                if _material_has_texture(native_visual.material):
                    native_texture_count += 1
            elif colors is not None:
                cast("ColorVisuals", mesh.visual).vertex_colors = colors
                vertex_color_count += 1
            else:
                mesh.visual = trimesh.visual.TextureVisuals(material=_neutral_material())
                neutral_fallback_count += 1
            raw_normals = _component_value(component, "normals")
            if raw_normals is not None:
                normals = np.asarray(raw_normals, dtype=np.float64)
                if normals.shape == vertices.shape and np.isfinite(normals).all():
                    mesh.vertex_normals = normals  # type: ignore[method-assign]
            carrier_scene.add_geometry(mesh, geom_name=geometry_name, node_name=geometry_name)
            mesh_count += 1
            continue

        if kind not in {"points", "point_cloud"}:
            raise OrientationCarrierError(
                f"normalized component {component_id!r} has unsupported kind {kind!r}"
            )
        if colors is None:
            colors = np.tile(np.asarray((184, 189, 199, 255), dtype=np.uint8), (len(vertices), 1))
            neutral_fallback_count += 1
        else:
            vertex_color_count += 1
        point_cloud = trimesh.points.PointCloud(vertices, colors=colors)
        carrier_scene.add_geometry(
            point_cloud,
            geom_name=geometry_name,
            node_name=geometry_name,
        )
        point_count += len(vertices)

    sentinel_count = 0
    if mesh_count == 0 and point_count > 0:
        joined = np.concatenate(all_vertices, axis=0)
        extent = float(np.max(np.ptp(joined, axis=0)))
        epsilon = max(extent * 1e-9, 1e-12)
        origin = joined[0]
        sentinel = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    origin,
                    origin + np.asarray([epsilon, 0.0, 0.0]),
                    origin + np.asarray([0.0, epsilon, 0.0]),
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            process=False,
        )
        carrier_scene.add_geometry(
            sentinel,
            geom_name="point_visual_coverage_sentinel",
            node_name="point_visual_coverage_sentinel",
        )
        sentinel_count = 1

    carrier_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = carrier_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    glb_path = assets_dir / "source.glb"
    urdf_path = carrier_dir / "model.urdf"
    try:
        payload = carrier_scene.export(file_type="glb")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("trimesh returned no GLB bytes")
        glb_path.write_bytes(payload)
    except Exception as exc:
        raise OrientationCarrierError(
            f"failed to export normalized preview carrier: {type(exc).__name__}: {exc}"
        ) from exc

    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="recon_normalized_source">
  <link name="source">
    <visual name="normalized_source">
      <geometry><mesh filename="assets/source.glb" scale="1 1 1"/></geometry>
    </visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return (
        urdf_path,
        glb_path,
        {
            "mesh_components": mesh_count,
            "point_count": point_count,
            "point_visual_coverage_sentinels": sentinel_count,
            "native_material_components": native_material_count,
            "native_texture_material_components": native_texture_count,
            "vertex_color_components": vertex_color_count,
            "neutral_fallback_components": neutral_fallback_count,
        },
    )


def _default_bundle_builder(output_dir: Path) -> None:
    package_json = VIEWER_WEB_ROOT / "package.json"
    if not package_json.is_file():
        raise OrientationRenderUnavailableError(
            f"Viewer source is unavailable at {VIEWER_WEB_ROOT}"
        )
    npm = shutil.which("npm")
    if npm is None:
        raise OrientationRenderUnavailableError("strict Chromium rendering requires npm")
    if not (VIEWER_WEB_ROOT / "node_modules").is_dir():
        raise OrientationRenderUnavailableError(
            "strict Chromium rendering requires viewer/web/node_modules; install Viewer dependencies"
        )
    subprocess.run(
        [
            npm,
            "--prefix",
            str(VIEWER_WEB_ROOT),
            "run",
            "build",
            "--",
            "--outDir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not (output_dir / "render.html").is_file():
        raise OrientationRenderUnavailableError(
            "temporary Viewer source build did not produce render.html"
        )


def _default_session_factory(dist_dir: Path) -> _RenderSession:
    return _ChromiumChannelLadder(dist_dir)


def _normalized_identity(artifacts: NormalizedSourceArtifacts) -> dict[str, str]:
    normalized_path = Path(getattr(artifacts, "normalized_usdc"))
    if not normalized_path.is_file():
        raise OrientationCarrierError(f"normalized source artifact is missing: {normalized_path}")
    normalized_sha = str(getattr(artifacts, "normalized_usdc_sha256", "") or "")
    observed_sha = _sha256_file(normalized_path)
    if normalized_sha and normalized_sha != observed_sha:
        raise OrientationCarrierError("normalized source artifact hash does not match its manifest")

    source_manifest_path = Path(getattr(artifacts, "source_manifest"))
    if not source_manifest_path.is_file():
        raise OrientationCarrierError(
            f"normalized source manifest is missing: {source_manifest_path}"
        )
    source_manifest_sha = str(getattr(artifacts, "source_manifest_sha256", "") or "")
    observed_manifest_sha = _sha256_file(source_manifest_path)
    if source_manifest_sha and source_manifest_sha != observed_manifest_sha:
        raise OrientationCarrierError("normalized source manifest hash does not match its artifact")
    return {
        "normalized_usdc_sha256": normalized_sha or observed_sha,
        "source_manifest_sha256": source_manifest_sha or observed_manifest_sha,
    }


def _safe_json(value: object) -> object:
    """Return a path-free, finite JSON projection of renderer diagnostics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json(item) for item in value]
    return str(value)


def _safe_camera(camera: object) -> dict[str, object]:
    if not isinstance(camera, Mapping):
        return {}
    allowed = (
        "azimuthDeg",
        "elevationDeg",
        "fovDeg",
        "distance",
        "center",
        "radius",
        "eye",
    )
    return {key: _safe_json(camera[key]) for key in allowed if key in camera}


__all__ = [
    "OrientationCarrierError",
    "OrientationPreviewError",
    "OrientationRenderContractError",
    "OrientationRenderUnavailableError",
]
