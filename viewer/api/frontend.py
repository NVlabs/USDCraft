# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file is derived from Articraft (https://github.com/mattzh72/articraft),
# copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
# and has been modified by NVIDIA.
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

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


def install_frontend_routes(app: FastAPI, *, dist_dir: Path | None = None) -> None:
    frontend_dist = dist_dir or Path(__file__).resolve().parents[1] / "web" / "dist"
    if frontend_dist.exists():
        resolved_dist_dir = frontend_dist.resolve()

        @app.get("/", include_in_schema=False)
        async def frontend_index() -> FileResponse:
            return FileResponse(
                resolved_dist_dir / "index.html",
                headers={"Cache-Control": "no-cache"},
            )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def frontend_fallback(full_path: str) -> FileResponse:
            if (
                full_path == "api"
                or full_path.startswith("api/")
                or full_path in {"health", "docs", "redoc", "openapi.json"}
            ):
                raise HTTPException(status_code=404)
            requested_path = Path(full_path)
            if requested_path.is_absolute() or ".." in requested_path.parts:
                raise HTTPException(status_code=404)
            target = (resolved_dist_dir / requested_path).resolve()
            try:
                target.relative_to(resolved_dist_dir)
            except ValueError as exc:
                raise HTTPException(status_code=404) from exc
            if target.exists() and target.is_file():
                cache_control = (
                    "public, max-age=31536000, immutable"
                    if target.parts[-2:-1] == ("assets",)
                    else "no-cache"
                )
                return FileResponse(target, headers={"Cache-Control": cache_control})
            if requested_path.parts[:1] == ("assets",):
                raise HTTPException(status_code=404)
            return FileResponse(
                resolved_dist_dir / "index.html",
                headers={"Cache-Control": "no-cache"},
            )

        return

    @app.get("/", include_in_schema=False)
    async def frontend_missing() -> dict[str, str]:
        return {"message": "Frontend build not found. Run `uv run usdcraft setup`."}
