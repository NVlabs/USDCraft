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

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from viewer.api.dependencies import FileResolverDep
from viewer.api.media import (
    TEXT_MEDIA_TYPES,
    file_cache_control,
    file_media_type,
    read_text_file_payload,
)
from viewer.api.schemas import RecordTextFileResponse

router = APIRouter()


async def _text_response(
    *,
    record_id: str,
    file_path: str,
    target: Path,
    preview_bytes: int,
    full: bool,
) -> RecordTextFileResponse:
    if target.suffix.lower() not in TEXT_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Text preview is only supported for text files")
    content, truncated, byte_count = await asyncio.to_thread(
        read_text_file_payload,
        target,
        preview_bytes=preview_bytes,
        full=full,
    )
    return RecordTextFileResponse(
        record_id=record_id,
        file_path=file_path,
        content=content,
        truncated=truncated,
        byte_count=byte_count,
        preview_byte_limit=None if full else preview_bytes,
    )


@router.get("/api/records/{record_id}/text/{file_path:path}", response_model=RecordTextFileResponse)
async def record_text_file(
    record_id: str,
    file_path: str,
    resolver: FileResolverDep,
    preview_bytes: int = Query(default=131072, ge=4096, le=1048576),
    full: bool = False,
) -> RecordTextFileResponse:
    _, target = resolver.resolve_record_target(record_id, file_path)
    return await _text_response(
        record_id=record_id,
        file_path=file_path,
        target=target,
        preview_bytes=preview_bytes,
        full=full,
    )


@router.get("/api/records/{record_id}/files/{file_path:path}")
async def record_file(
    record_id: str,
    file_path: str,
    request: Request,
    resolver: FileResolverDep,
) -> FileResponse:
    _, target = resolver.resolve_record_target(record_id, file_path)
    return FileResponse(
        target,
        media_type=file_media_type(target),
        headers={
            "Cache-Control": file_cache_control(immutable=bool(request.query_params.get("rev")))
        },
    )


@router.get(
    "/api/records/{record_id}/revisions/{revision_id}/text/{file_path:path}",
    response_model=RecordTextFileResponse,
)
async def record_revision_text_file(
    record_id: str,
    revision_id: str,
    file_path: str,
    resolver: FileResolverDep,
    preview_bytes: int = Query(default=131072, ge=4096, le=1048576),
    full: bool = False,
) -> RecordTextFileResponse:
    _, target = resolver.resolve_record_revision_target(record_id, revision_id, file_path)
    return await _text_response(
        record_id=record_id,
        file_path=file_path,
        target=target,
        preview_bytes=preview_bytes,
        full=full,
    )


@router.get("/api/records/{record_id}/revisions/{revision_id}/files/{file_path:path}")
async def record_revision_file(
    record_id: str,
    revision_id: str,
    file_path: str,
    request: Request,
    resolver: FileResolverDep,
) -> FileResponse:
    _, target = resolver.resolve_record_revision_target(record_id, revision_id, file_path)
    return FileResponse(
        target,
        media_type=file_media_type(target),
        headers={
            "Cache-Control": file_cache_control(immutable=bool(request.query_params.get("rev")))
        },
    )


@router.get("/api/records/{record_id}/traces/{file_path:path}")
async def record_trace_file(
    record_id: str,
    file_path: str,
    resolver: FileResolverDep,
) -> FileResponse:
    target, media_type = resolver.resolve_record_trace_target(record_id, file_path)
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "no-store"})


@router.get("/api/records/{record_id}/revisions/{revision_id}/traces/{file_path:path}")
async def record_revision_trace_file(
    record_id: str,
    revision_id: str,
    file_path: str,
    resolver: FileResolverDep,
) -> FileResponse:
    target, media_type = resolver.resolve_record_revision_trace_target(
        record_id, revision_id, file_path
    )
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "no-store"})
