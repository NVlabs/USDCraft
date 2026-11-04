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

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from storage.record_mutations import RecordDeletionRecoveryError, RecordNotFoundError
from viewer.api.dependencies import ReconstructionRecordStoreDep, RecordMutationsDep
from viewer.api.media import (
    TEXT_MEDIA_TYPES,
    file_cache_control,
    file_media_type,
    read_text_file_payload,
)
from viewer.api.schemas import (
    AuthorTimelineResponse,
    AxialGeometryResponse,
    ReconstructionRecordDetailResponse,
    ReconstructionRecordSummaryResponse,
    RecordDeleteRequest,
    RecordDeleteResponse,
    RecordRatingRequest,
    RecordTextFileResponse,
)

router = APIRouter(prefix="/api/reconstruction-records")


@router.put("/{record_id}/rating", response_model=ReconstructionRecordSummaryResponse)
async def update_reconstruction_record_rating(
    record_id: str,
    payload: RecordRatingRequest,
    store: ReconstructionRecordStoreDep,
    mutations: RecordMutationsDep,
) -> ReconstructionRecordSummaryResponse:
    """The same storage-owned rating mutation as Generation, bound to Reconstruction records."""

    try:
        await asyncio.to_thread(
            mutations.update_rating, record_id, payload.rating, pipeline="reconstruction"
        )
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}") from exc
    summary = await asyncio.to_thread(store.summary, record_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return summary


@router.delete("/{record_id}", response_model=RecordDeleteResponse)
async def delete_reconstruction_record(
    record_id: str,
    payload: RecordDeleteRequest,
    mutations: RecordMutationsDep,
) -> RecordDeleteResponse:
    """Explicitly confirmed permanent deletion of one Reconstruction record."""

    if payload.confirm_record_id != record_id:
        raise HTTPException(
            status_code=409,
            detail="Record deletion confirmation does not match the URL record ID.",
        )
    try:
        result = await asyncio.to_thread(
            mutations.delete_record, record_id, pipeline="reconstruction"
        )
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}") from exc
    except RecordDeletionRecoveryError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Record deletion recovery is pending: {record_id}",
        ) from exc
    return RecordDeleteResponse(
        record_id=result.record_id,
        materialization_deleted=result.materialization_deleted,
        cleanup_pending=result.cleanup_pending,
    )


@router.get("", response_model=list[ReconstructionRecordSummaryResponse])
async def reconstruction_records(
    store: ReconstructionRecordStoreDep,
) -> list[ReconstructionRecordSummaryResponse]:
    """The fast list projection: never re-runs the full canonical contract per record.

    A row whose materialization was not already validated (via `detail`, a prior `validate`
    call, or a prior full check still covered by the signature cache) reports
    `materialization_status="not_validated"` rather than a guessed canonical status.
    """

    return await asyncio.to_thread(store.list_records)


@router.get("/{record_id}", response_model=ReconstructionRecordDetailResponse)
async def reconstruction_record_detail(
    record_id: str,
    store: ReconstructionRecordStoreDep,
) -> ReconstructionRecordDetailResponse:
    detail = await asyncio.to_thread(store.detail, record_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Reconstruction record not found: {record_id}")
    return detail


@router.get(
    "/{record_id}/text/{file_path:path}",
    response_model=RecordTextFileResponse,
)
async def reconstruction_record_text_file(
    record_id: str,
    file_path: str,
    store: ReconstructionRecordStoreDep,
    preview_bytes: int = Query(default=131072, ge=4096, le=1048576),
    full: bool = False,
) -> RecordTextFileResponse:
    target = await asyncio.to_thread(store.text_file, record_id, file_path)
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


@router.get("/{record_id}/author-timeline", response_model=AuthorTimelineResponse)
async def reconstruction_author_timeline(
    record_id: str, store: ReconstructionRecordStoreDep
) -> AuthorTimelineResponse:
    """Read-only per-turn tool timeline of a Provider run; 404 when no trace is retained."""

    timeline = await asyncio.to_thread(store.author_timeline, record_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"No Provider trace for record: {record_id}")
    return timeline


@router.get("/{record_id}/axial-geometry", response_model=AxialGeometryResponse)
async def reconstruction_axial_geometry(
    record_id: str, store: ReconstructionRecordStoreDep
) -> AxialGeometryResponse:
    """Read-only geometry of the last check_axial call; 404 when the record has none."""

    geometry = await asyncio.to_thread(store.axial_geometry, record_id)
    if geometry is None:
        raise HTTPException(
            status_code=404, detail=f"No check_axial geometry for record: {record_id}"
        )
    return geometry


@router.get("/{record_id}/files/{file_path:path}")
async def reconstruction_record_file(
    record_id: str,
    file_path: str,
    request: Request,
    store: ReconstructionRecordStoreDep,
) -> FileResponse:
    target = await asyncio.to_thread(store.declared_file, record_id, file_path)
    return FileResponse(
        target,
        media_type=file_media_type(target),
        headers={
            "Cache-Control": file_cache_control(immutable=bool(request.query_params.get("rev")))
        },
    )


@router.get("/{record_id}/materialized/{file_path:path}")
async def reconstruction_record_materialized_file(
    record_id: str,
    file_path: str,
    request: Request,
    store: ReconstructionRecordStoreDep,
) -> FileResponse:
    target = await asyncio.to_thread(store.materialized_file, record_id, file_path)
    return FileResponse(
        target,
        media_type=file_media_type(target),
        headers={
            "Cache-Control": file_cache_control(immutable=bool(request.query_params.get("rev")))
        },
    )
