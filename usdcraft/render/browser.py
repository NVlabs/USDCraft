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

"""Viewer-parity headless renderer for Generation self-checks.

Renders a compiled model's viewer-parity views by driving the SAME scene/
loader/material code the interactive viewer uses (``viewer/web/render.html``,
built from ``src/render-entry.tsx``) in headless Chromium via Playwright.

Design contracts:

- **Profiles** (``review-pbr`` / ``review-shape``) pin every renderer
  state; the profile id, bundle hash, browser version, and WebGL backend are
  recorded in a ``render_manifest.json`` next to the outputs.
- **WebGL probe ladder** (no silent SwiftShader assumption): probe WebGL2 in
  the launched browser; when unavailable, relaunch once with the explicit
  ``--enable-unsafe-swiftshader`` policy and probe again; still unavailable →
  the backend declares itself unavailable and falls back.
- **Capabilities, not equivalence**: the numpy Lambert fallback is a DIFFERENT
  evidence class. Results carry a capability declaration so consumers can mark
  appearance diagnostics ``not_evaluable`` instead of judging gloss from a renderer
  that cannot draw it.
- **Browser reuse**: one browser/page per process (see
  ``render_generated_model_views``); per-view marginal cost is milliseconds.

The browser is owned by one dedicated worker thread so repeated render jobs can reuse it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from usdcraft.render.contract import STANDARD_RENDER_VIEWS

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_DIST_DIR = REPO_ROOT / "viewer" / "web" / "dist"
RENDER_PAGE_NAME = "render.html"

DEFAULT_PROFILE = "review-pbr"
SHAPE_PROFILE = "review-shape"
PARTS_PROFILE = "review-parts"
KNOWN_PROFILES = (DEFAULT_PROFILE, SHAPE_PROFILE, PARTS_PROFILE)

# Standard generated-model views use the shared numpy-renderer convention
# (Z-up robot frame azimuth / elevation); the render page owns the conversion
# to the viewer world.
DEFAULT_REVIEW_VIEWS: tuple[tuple[str, float, float], ...] = STANDARD_RENDER_VIEWS

VIEWER_CAPABILITIES = {
    "geometry": True,
    "textures": True,
    "metallic": True,
    "roughness": True,
    "transmission": True,
    "clearcoat": True,
    "environment_reflection": True,
}
NUMPY_CAPABILITIES = {
    "geometry": True,
    "textures": "partial",
    "metallic": False,
    "roughness": False,
    "transmission": False,
    "clearcoat": False,
    "environment_reflection": False,
}

RENDER_MANIFEST_NAME = "render_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class RenderRequest:
    """One Viewer render job over a compiled model's asset directory.

    ``asset_dir`` must contain ``model.urdf`` (and usually ``model.usdz`` plus
    its ``assets/``) — a canonical materialization or Generation staging directory.
    """

    asset_dir: Path
    out_dir: Path
    views: Sequence[tuple[str, float, float]] = DEFAULT_REVIEW_VIEWS
    size: int = 768
    profile: str = DEFAULT_PROFILE
    joint_positions: dict[str, float] = field(default_factory=dict)
    model_hash: str | None = None
    distance_factor: float | None = None


@dataclass(slots=True)
class RenderResult:
    backend: str  # "viewer_chromium" | "numpy_lambert"
    capabilities: dict[str, Any]
    profile: str
    images: list[dict[str, Any]]
    manifest_path: Path | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    parts_legend: dict[str, str] | None = None

    @property
    def image_paths(self) -> list[Path]:
        return [Path(item["path"]) for item in self.images]


class RenderBundleError(RuntimeError):
    """The built render page is missing or unusable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_render_bundle(dist_dir: Path = RENDER_DIST_DIR) -> dict[str, Any]:
    """Validate the built render page and return its identity for the manifest.

    Vite content-hashes every chunk filename, and ``render.html`` references
    the hashed entry — so hashing ``render.html`` alone identifies the bundle
    graph. Staleness against sources is a cheap mtime heuristic (warn, not
    fail: a stale bundle still renders, it just may lag viewer changes).
    """
    page = dist_dir / RENDER_PAGE_NAME
    if not page.is_file():
        raise RenderBundleError(
            f"{page} is missing; build the viewer first: npm --prefix viewer/web run build"
        )
    bundle_hash = _sha256_file(page)
    src_dir = dist_dir.parent / "src"
    try:
        newest_src = max(
            (p.stat().st_mtime for p in src_dir.rglob("*") if p.is_file()), default=0.0
        )
        if newest_src > page.stat().st_mtime:
            logger.warning(
                "viewer/web/src is newer than the built render bundle; "
                "run npm --prefix viewer/web run build for current-code renders"
            )
    except OSError:
        pass
    return {"bundle_hash": bundle_hash, "dist_dir": str(dist_dir)}


class RenderSetupError(RuntimeError):
    """The local render setup (bundle or Playwright Chromium) is not installed."""


_SETUP_HINT = "Run `uv run usdcraft setup` from the repository root, then rerun this command."


def require_render_setup(dist_dir: Path = RENDER_DIST_DIR) -> None:
    """Fail before a run authors when the Viewer renderer cannot start.

    Generation and Reconstruction candidate renders use the built render page in Playwright
    Chromium; this run-start check turns a missing setup into one actionable error instead of
    a run whose renders degrade to the numpy fallback. It does not launch a browser.
    """

    page = dist_dir / RENDER_PAGE_NAME
    if not page.is_file():
        raise RenderSetupError(f"The render bundle {page} is missing. {_SETUP_HINT}")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
    except Exception as exc:
        raise RenderSetupError(
            f"Playwright could not locate its Chromium browser ({type(exc).__name__}: {exc}). "
            f"{_SETUP_HINT}"
        ) from exc
    if not executable.exists():
        raise RenderSetupError(
            f"The Playwright Chromium browser is not installed ({executable} is missing). "
            f"{_SETUP_HINT}"
        )


class _AssetRequestHandler(SimpleHTTPRequestHandler):
    """Serves the render bundle at / and registered asset roots at /model-assets/<token>/."""

    server: "_AssetServer"

    def log_message(self, *_args) -> None:  # quiet
        pass

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/model-assets/"):
            parts = clean.split("/", 3)
            if len(parts) >= 4:
                token, rest = parts[2], parts[3]
                root = self.server.asset_roots.get(token)
                if root is not None:
                    candidate = (root / rest).resolve()
                    if str(candidate).startswith(str(root.resolve())):
                        return str(candidate)
            return str(self.server.dist_dir / "__not_found__")
        relative = clean.lstrip("/") or RENDER_PAGE_NAME
        return str(self.server.dist_dir / relative)


class _AssetServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, dist_dir: Path):
        super().__init__(("127.0.0.1", 0), partial(_AssetRequestHandler, directory=str(dist_dir)))
        self.dist_dir = dist_dir
        self.asset_roots: dict[str, Path] = {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def register_asset_dir(self, asset_dir: Path) -> str:
        token = hashlib.sha256(str(asset_dir.resolve()).encode("utf-8")).hexdigest()[:16]
        self.asset_roots[token] = asset_dir.resolve()
        return f"/model-assets/{token}"


mimetypes.add_type("model/vnd.usdz+zip", ".usdz")
mimetypes.add_type("model/gltf-binary", ".glb")


class ViewerRenderSession:
    """Owns one HTTP server + one headless Chromium page; reuse across renders."""

    def __init__(
        self,
        *,
        dist_dir: Path = RENDER_DIST_DIR,
        browser_channel: str = "chromium",
    ):
        self._dist_dir = dist_dir
        self._browser_channel = browser_channel
        self._lock = threading.Lock()
        self._server: _AssetServer | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._bundle: dict[str, Any] | None = None
        self._webgl: dict[str, Any] | None = None
        self._browser_version: str | None = None
        self._unavailable_reason: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def _start_locked(self) -> None:
        if self._page is not None or self._unavailable_reason is not None:
            return
        try:
            self._bundle = ensure_render_bundle(self._dist_dir)
        except RenderBundleError as exc:
            self._unavailable_reason = f"render bundle unavailable: {exc}"
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self._unavailable_reason = f"playwright unavailable: {exc}"
            return
        startup_stage = "local asset server"
        try:
            self._server = _AssetServer(self._dist_dir)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            startup_stage = "Playwright startup"
            self._playwright = sync_playwright().start()
            startup_stage = "Chromium/WebGL startup"
            probe = self._launch_and_probe(extra_args=[])
            if not probe["available"]:
                # Explicit software-WebGL retry (Chrome >=138 removed the automatic
                # SwiftShader fallback); this renderer process only loads localhost
                # content we serve ourselves.
                logger.warning("WebGL2 unavailable on default launch; retrying with SwiftShader")
                self._close_browser_locked()
                probe = self._launch_and_probe(extra_args=["--enable-unsafe-swiftshader"])
            if not probe["available"]:
                self._unavailable_reason = "webgl2_unavailable"
                self._close_browser_locked()
                return
            self._webgl = probe
        except Exception as exc:
            self._unavailable_reason = f"{startup_stage} unavailable: {type(exc).__name__}: {exc}"
            self._close_browser_locked()
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            if self._server is not None:
                self._server.shutdown()
                self._server = None
            logger.warning("%s", self._unavailable_reason)

    def _launch_and_probe(self, *, extra_args: list[str]) -> dict[str, Any]:
        assert self._playwright is not None and self._server is not None
        self._browser = self._playwright.chromium.launch(
            channel=self._browser_channel, headless=True, args=extra_args
        )
        self._browser_version = self._browser.version
        self._page = self._browser.new_page(viewport={"width": 768, "height": 768})
        self._page.goto(f"{self._server.base_url}/{RENDER_PAGE_NAME}")
        self._page.wait_for_function("() => window.usdcraftRendererReady === true")
        return self._page.evaluate("() => window.usdcraftRenderer.probeWebgl2()")

    def _close_browser_locked(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        self._browser = None
        self._page = None

    def close(self) -> None:
        with self._lock:
            self._close_browser_locked()
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            if self._server is not None:
                self._server.shutdown()
                self._server = None

    # -- rendering -----------------------------------------------------------

    def render(
        self, request: RenderRequest, *, allow_numpy_fallback: bool = True
    ) -> RenderResult | None:
        """Render the request; on unavailability fall back to numpy or return None.

        With ``allow_numpy_fallback=False``, unavailability returns ``None`` instead of producing
        the deterministic Lambert fallback.
        """
        if request.profile not in KNOWN_PROFILES:
            raise ValueError(f"unknown render profile {request.profile!r}")
        with self._lock:
            self._start_locked()
            if self._unavailable_reason is not None:
                if not allow_numpy_fallback:
                    return None
                return self._numpy_fallback(request, reason=self._unavailable_reason)
            try:
                return self._render_locked(request)
            except Exception as exc:
                logger.warning(
                    "viewer render failed (%s: %s)%s",
                    type(exc).__name__,
                    exc,
                    "; numpy fallback" if allow_numpy_fallback else "",
                )
                if not allow_numpy_fallback:
                    return None
                return self._numpy_fallback(request, reason=f"{type(exc).__name__}: {exc}")

    def _render_locked(self, request: RenderRequest) -> RenderResult:
        assert self._page is not None and self._server is not None
        page = self._page
        if page.viewport_size != {"width": request.size, "height": request.size}:
            page.set_viewport_size({"width": request.size, "height": request.size})
        base_path = self._server.register_asset_dir(request.asset_dir)
        asset_revision = self._asset_revision(request)

        page.evaluate(
            "([url, revision]) => window.usdcraftRenderer.load(url, revision)",
            [base_path, asset_revision],
        )
        deadline = time.monotonic() + 120.0
        state = "unknown"
        while time.monotonic() < deadline:
            state = page.evaluate("() => window.usdcraftRenderer.readyState()")
            if state == "ready" or state.startswith("error:"):
                break
            time.sleep(0.05)
        if state != "ready":
            raise RuntimeError(f"render page did not become ready: {state}")
        page.evaluate("(id) => window.usdcraftRenderer.setProfile(id)", request.profile)
        parts_legend = page.evaluate(
            "() => window.usdcraftRenderer.partsLegend ? window.usdcraftRenderer.partsLegend()"
            " : null"
        )
        if request.joint_positions:
            page.evaluate(
                "(pose) => window.usdcraftRenderer.setJointPose(pose)",
                dict(request.joint_positions),
            )
        # Settle frames: joint pose + textures + PMREM compile.
        page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )

        request.out_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        for name, azimuth, elevation in request.views:
            pose: dict[str, Any] = {"azimuthDeg": azimuth, "elevationDeg": elevation}
            if request.distance_factor is not None:
                pose["distanceFactor"] = request.distance_factor
            camera = page.evaluate("(pose) => window.usdcraftRenderer.setCamera(pose)", pose)
            data_url = page.evaluate("() => window.usdcraftRenderer.capturePng()")
            png = base64.b64decode(data_url.split(",", 1)[1])
            path = request.out_dir / f"{name}.png"
            path.write_bytes(png)
            images.append(
                {
                    "name": name,
                    "path": str(path),
                    "sha256": hashlib.sha256(png).hexdigest(),
                    "camera": camera,
                }
            )
        diagnostics = page.evaluate("() => window.usdcraftRenderer.diagnostics()")
        if diagnostics.get("assetRevisionKey") != asset_revision:
            raise RuntimeError(
                "render page reported a stale asset revision: "
                f"{diagnostics.get('assetRevisionKey')!r} != {asset_revision!r}"
            )

        result = RenderResult(
            backend="viewer_chromium",
            capabilities=dict(VIEWER_CAPABILITIES),
            profile=request.profile,
            images=images,
            manifest_path=None,
            diagnostics=diagnostics,
            parts_legend=parts_legend if isinstance(parts_legend, dict) else None,
        )
        result.manifest_path = self._write_manifest(request, result)
        return result

    @staticmethod
    def _asset_revision(request: RenderRequest) -> str:
        if request.model_hash:
            return request.model_hash
        hashes: list[str] = []
        for name in ("model.urdf", "model.usdz"):
            path = request.asset_dir / name
            if path.is_file():
                hashes.append(f"{name}:{_sha256_file(path)}")
        if not hashes:
            raise FileNotFoundError(f"no model.urdf or model.usdz under {request.asset_dir}")
        return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()

    # -- fallback + manifest ---------------------------------------------------

    def _numpy_fallback(self, request: RenderRequest, *, reason: str) -> RenderResult:
        from usdcraft.render.offscreen import render_urdf_views

        request.out_dir.mkdir(parents=True, exist_ok=True)
        paths = render_urdf_views(
            request.asset_dir,
            request.out_dir,
            views=[(name, az, el) for name, az, el in request.views],
            size=request.size,
            joint_positions=dict(request.joint_positions),
        )
        by_name = {path.stem: path for path in paths}
        images = [
            {
                "name": name,
                "path": str(by_name[name]),
                "sha256": _sha256_file(by_name[name]),
                "camera": {"azimuthDeg": az, "elevationDeg": el},
            }
            for name, az, el in request.views
            if name in by_name
        ]
        result = RenderResult(
            backend="numpy_lambert",
            capabilities=dict(NUMPY_CAPABILITIES),
            profile=request.profile,
            images=images,
            manifest_path=None,
            fallback_reason=reason,
        )
        result.manifest_path = self._write_manifest(request, result)
        return result

    def _write_manifest(self, request: RenderRequest, result: RenderResult) -> Path:
        model_urdf = request.asset_dir / "model.urdf"
        model_usdz = request.asset_dir / "model.usdz"
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "backend": result.backend,
            "capabilities": result.capabilities,
            "profile": result.profile,
            "size": request.size,
            "model_hash": request.model_hash,
            "asset_revision": self._asset_revision(request),
            "urdf_sha256": _sha256_file(model_urdf) if model_urdf.is_file() else None,
            "usdz_sha256": _sha256_file(model_usdz) if model_usdz.is_file() else None,
            "joint_positions": dict(request.joint_positions),
            "bundle": self._bundle,
            "browser_channel": self._browser_channel,
            "browser_version": self._browser_version,
            "webgl": self._webgl,
            "fallback_reason": result.fallback_reason,
            "parts_legend": result.parts_legend,
            "images": result.images,
            "diagnostics": result.diagnostics,
        }
        path = request.out_dir / RENDER_MANIFEST_NAME
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


_GLOBAL_SESSION: ViewerRenderSession | None = None
_GLOBAL_LOCK = threading.Lock()
_RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="usdcraft-viewer-render")


def _render_on_worker(request: RenderRequest, *, allow_numpy_fallback: bool) -> RenderResult | None:
    """Run process-global Playwright work on its single owning thread."""
    global _GLOBAL_SESSION
    with _GLOBAL_LOCK:
        if _GLOBAL_SESSION is None:
            _GLOBAL_SESSION = ViewerRenderSession()
        session = _GLOBAL_SESSION
    return session.render(request, allow_numpy_fallback=allow_numpy_fallback)


def render_generated_model_views(request: RenderRequest) -> RenderResult:
    """Render fixed Viewer views with the process-global browser session."""
    result = _RENDER_EXECUTOR.submit(_render_on_worker, request, allow_numpy_fallback=True).result()
    assert result is not None  # fallback-allowed renders always produce a result
    return result
