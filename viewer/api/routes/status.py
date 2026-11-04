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

import asyncio

from fastapi import APIRouter

from viewer.api.dependencies import ReadOnlyDep, ViewerStoreDep
from viewer.api.schemas import HealthResponse, ViewerBootstrapResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/api/status", response_model=HealthResponse)
async def status(store: ViewerStoreDep, read_only: ReadOnlyDep) -> HealthResponse:
    return HealthResponse(repo_root=store.repo_root.as_posix(), read_only=read_only)


@router.get("/api/bootstrap", response_model=ViewerBootstrapResponse)
async def bootstrap(store: ViewerStoreDep) -> ViewerBootstrapResponse:
    return await asyncio.to_thread(store.bootstrap)
