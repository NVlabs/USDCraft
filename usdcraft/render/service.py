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

"""Run-scoped Host renderer. The sandboxed Author exchanges bounded JSON files, not sockets.

The Host executes only the existing compiled-candidate renderer, never Author commands. Requests
and output stay in this run. This does not change Codex permissions or add a semantic reviewer.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUEUE_DIR = ".candidate_render"
MAX_REQUEST_BYTES = 16_384
REQUEST_TIMEOUT = 180.0


@dataclass(frozen=True, slots=True)
class RenderServiceConfig:
    workspace_environment_variable: str
    label: str
    execution: str
    require_build_directory: bool


def _write(path: Path, value: Any) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value), encoding="utf-8")
    temp.replace(path)


class CandidateRenderService:
    def __init__(
        self,
        root: Path,
        *,
        config: RenderServiceConfig,
        render_candidate: Callable[..., dict[str, Any]],
    ) -> None:
        self.config = config
        self.render_candidate = render_candidate
        self.root = root.resolve()
        self.queue = self.root / QUEUE_DIR
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> CandidateRenderService:
        self.queue.mkdir(exist_ok=True)
        _write(self.queue / "active.json", {"pid": os.getpid()})
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        (self.queue / "active.json").unlink(missing_ok=True)
        self.stop.set()
        self.thread.join(timeout=5)

    def _path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Render paths must stay inside this Author workspace")
        return path

    def _serve(self) -> None:
        from usdcraft.render.browser import ViewerRenderSession

        # Renderer objects are created, used and closed on their owning thread.
        session = None
        try:
            while not self.stop.wait(0.05):
                for request in sorted(self.queue.glob("*.request.json")):
                    response = request.with_name(request.name.replace(".request.", ".response."))
                    if response.exists():
                        continue
                    try:
                        if request.is_symlink() or request.stat().st_size > MAX_REQUEST_BYTES:
                            raise ValueError("Invalid render request file")
                        data = json.loads(request.read_text())
                        output = self._path(data["out_dir"])
                        if not output.is_relative_to(self.root / "author_feedback"):
                            raise ValueError("Render output must stay inside author_feedback")
                        asset_dir = self._path(data["asset_dir"])
                        if self.config.require_build_directory and asset_dir != self.root / "build":
                            raise ValueError("Render input must be this run's build directory")
                        if session is None:
                            session = ViewerRenderSession()
                        result = self.render_candidate(
                            asset_dir,
                            out_dir=output,
                            azimuth=float(data["azimuth"]),
                            elevation=float(data["elevation"]),
                            joint_positions=data["joint_positions"],
                            renderer=session.render,
                        )
                        result["execution"] = self.config.execution
                        if output.is_dir():
                            _write(output / "feedback.json", result)
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "error": f"Host renderer: {type(exc).__name__}: {exc}",
                        }
                    _write(response, result)
        finally:
            if session is not None:
                session.close()


def request_candidate_render(
    asset_dir: Path,
    *,
    out_dir: Path,
    config: RenderServiceConfig,
    azimuth: float,
    elevation: float,
    joint_positions: dict[str, float],
) -> dict[str, Any]:
    """CLI transport: a sandboxed Author needs no browser process or listening socket."""
    root = Path(os.environ[config.workspace_environment_variable]).resolve()
    queue = root / QUEUE_DIR
    if not (queue / "active.json").is_file():
        raise RuntimeError(f"{config.label} Host renderer is unavailable for this run")
    paths = [asset_dir.resolve(), out_dir.resolve()]
    if any(not path.is_relative_to(root) or path == root for path in paths):
        raise ValueError("Render paths must stay inside this Author workspace")
    token = uuid.uuid4().hex
    request = queue / f"{token}.request.json"
    response = queue / f"{token}.response.json"
    _write(
        request,
        {
            "asset_dir": str(paths[0].relative_to(root)),
            "out_dir": str(paths[1].relative_to(root)),
            "azimuth": azimuth,
            "elevation": elevation,
            "joint_positions": joint_positions,
        },
    )
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while time.monotonic() < deadline:
        if response.is_file():
            result = json.loads(response.read_text())
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Host renderer failed"))
            return result
        if not (queue / "active.json").is_file():
            raise RuntimeError(f"{config.label} Host renderer stopped")
        time.sleep(0.05)
    raise RuntimeError(f"{config.label} Host render request timed out")
