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

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from storage.record_mutations import recover_pending_deletions, recover_pending_promotions
from viewer.api.file_resolver import ViewerFileResolver
from viewer.api.frontend import install_frontend_routes
from viewer.api.host_guard import HostGuardMiddleware, log_host_policy, resolve_allowed_hosts
from viewer.api.reconstruction_records import ReconstructionRecordStore
from viewer.api.routes import (
    collections_router,
    files_router,
    reconstruction_records_router,
    records_router,
    status_router,
)
from viewer.api.store import ViewerStore

logger = logging.getLogger(__name__)


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    configured = os.getenv("USDCRAFT_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path.cwd().resolve()


READ_ONLY_ENV = "USDCRAFT_VIEWER_READ_ONLY"


def _resolve_read_only(read_only: bool | None) -> bool:
    """`--read-only` reaches the uvicorn-imported app through the environment."""

    if read_only is not None:
        return read_only
    return os.getenv(READ_ONLY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _install_middleware(app: FastAPI, allowed_hosts: frozenset[str] | None) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8770",
            "http://localhost:8770",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    # Added last so it is the outermost layer: a foreign Host is refused before any route runs.
    app.add_middleware(HostGuardMiddleware, allowed_hosts=allowed_hosts)


def create_app(
    *,
    repo_root: Path | None = None,
    frontend_dist: Path | None = None,
    read_only: bool | None = None,
    allowed_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Build the Viewer API. ``read_only`` (or ``USDCRAFT_VIEWER_READ_ONLY=1``) refuses the two
    Workbench mutations — rating and explicitly confirmed deletion — with 403 and tells the
    frontend to hide their controls; every projection stays as it is.

    Requests are accepted only for a loopback ``Host`` (``localhost``, ``127.0.0.1``, ``[::1]``,
    any port) plus ``allowed_hosts`` (default: the comma-separated
    ``USDCRAFT_VIEWER_ALLOWED_HOSTS``, where ``usdcraft viewer`` puts its ``--host``); any other
    ``Host`` answers 400, which closes the DNS-rebinding path to the unauthenticated API."""

    app = FastAPI(title="USDCraft Viewer API")
    resolved_repo_root = _resolve_repo_root(repo_root)
    store = ViewerStore(resolved_repo_root)
    deletion_recovery = recover_pending_deletions(store.repo)
    promotion_recovery = recover_pending_promotions(store.repo)

    app.state.repo_root = resolved_repo_root
    app.state.read_only = _resolve_read_only(read_only)
    app.state.allowed_hosts = resolve_allowed_hosts(allowed_hosts)
    log_host_policy(app.state.allowed_hosts)
    app.state.viewer_store = store
    app.state.file_resolver = ViewerFileResolver(store)
    app.state.reconstruction_record_store = ReconstructionRecordStore(resolved_repo_root)
    app.state.deletion_recovery = deletion_recovery

    if deletion_recovery.completed_record_ids or deletion_recovery.rolled_back_record_ids:
        logger.info(
            "Recovered pending record deletions: completed=%s rolled_back=%s",
            deletion_recovery.completed_record_ids,
            deletion_recovery.rolled_back_record_ids,
        )
    if deletion_recovery.cleanup_pending_record_ids or deletion_recovery.errors:
        logger.warning(
            "Record deletion recovery remains pending: records=%s errors=%s",
            deletion_recovery.cleanup_pending_record_ids,
            deletion_recovery.errors,
        )
    if promotion_recovery.rolled_back_record_ids or promotion_recovery.cleaned_staging_dirs:
        logger.info(
            "Recovered interrupted record promotions: rolled_back=%s cleaned_staging=%s",
            promotion_recovery.rolled_back_record_ids,
            promotion_recovery.cleaned_staging_dirs,
        )
    if promotion_recovery.errors:
        logger.warning("Promotion recovery errors: %s", promotion_recovery.errors)

    _install_middleware(app, app.state.allowed_hosts)
    app.include_router(status_router)
    app.include_router(collections_router)
    app.include_router(records_router)
    app.include_router(files_router)
    app.include_router(reconstruction_records_router)
    install_frontend_routes(app, dist_dir=frontend_dist)
    return app


app = create_app()
