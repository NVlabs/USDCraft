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

from fastapi import APIRouter, HTTPException

from storage.identifiers import validate_record_id
from storage.record_mutations import RecordDeletionRecoveryError, RecordNotFoundError
from storage.revisions import active_inputs_dir
from viewer.api.dependencies import RecordMutationsDep, ViewerStoreDep
from viewer.api.schemas import (
    RecordDeleteRequest,
    RecordDeleteResponse,
    RecordDetailResponse,
    RecordHistoryResponse,
    RecordInputImageResponse,
    RecordRatingRequest,
    RecordSummaryResponse,
)

_INPUT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

router = APIRouter()


@router.get("/api/records", response_model=list[RecordSummaryResponse])
async def records(store: ViewerStoreDep) -> list[RecordSummaryResponse]:
    return await asyncio.to_thread(store.list_records)


@router.put("/api/records/{record_id}/rating", response_model=RecordSummaryResponse)
async def update_record_rating(
    record_id: str,
    payload: RecordRatingRequest,
    store: ViewerStoreDep,
    mutations: RecordMutationsDep,
) -> RecordSummaryResponse:
    _validate_record_id(record_id)
    try:
        await asyncio.to_thread(mutations.update_rating, record_id, payload.rating)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}") from exc
    summary = await asyncio.to_thread(store.record_summary, record_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return summary


@router.get("/api/records/{record_id}", response_model=RecordDetailResponse)
async def record_detail(record_id: str, store: ViewerStoreDep) -> RecordDetailResponse:
    _validate_record_id(record_id)
    detail = await asyncio.to_thread(store.record_detail, record_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return detail


@router.delete("/api/records/{record_id}", response_model=RecordDeleteResponse)
async def delete_record(
    record_id: str,
    payload: RecordDeleteRequest,
    mutations: RecordMutationsDep,
) -> RecordDeleteResponse:
    _validate_record_id(record_id)
    if payload.confirm_record_id != record_id:
        raise HTTPException(
            status_code=409,
            detail="Record deletion confirmation does not match the URL record ID.",
        )
    try:
        result = await asyncio.to_thread(mutations.delete_record, record_id)
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


@router.get("/api/records/{record_id}/summary", response_model=RecordSummaryResponse)
async def record_summary(record_id: str, store: ViewerStoreDep) -> RecordSummaryResponse:
    _validate_record_id(record_id)
    summary = await asyncio.to_thread(store.record_summary, record_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return summary


@router.get("/api/records/{record_id}/history", response_model=RecordHistoryResponse)
async def record_history(record_id: str, store: ViewerStoreDep) -> RecordHistoryResponse:
    _validate_record_id(record_id)
    history = await asyncio.to_thread(store.record_history, record_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return history


@router.get("/api/records/{record_id}/input-image", response_model=RecordInputImageResponse)
async def record_input_image(record_id: str, store: ViewerStoreDep) -> RecordInputImageResponse:
    _validate_record_id(record_id)

    def _resolve() -> tuple[bool, str | None]:
        record = store.repo.read_json(store.repo.layout.record_metadata_path(record_id))
        if (
            not isinstance(record, dict)
            or str(record.get("pipeline") or "generation") != "generation"
        ):
            return False, None
        inputs_dir = active_inputs_dir(store.repo, record_id, record=record)
        if not inputs_dir.is_dir():
            return True, None
        images = [
            path
            for path in sorted(inputs_dir.iterdir())
            if path.is_file() and path.suffix.lower() in _INPUT_IMAGE_EXTENSIONS
        ]
        for preferred_stem in ("inspiration", "reference"):
            preferred = next((path for path in images if path.stem == preferred_stem), None)
            if preferred is not None:
                return True, f"inputs/{preferred.name}"
        return True, f"inputs/{images[0].name}" if images else None

    found, path = await asyncio.to_thread(_resolve)
    if not found:
        raise HTTPException(status_code=404, detail=f"Record not found: {record_id}")
    return RecordInputImageResponse(path=path)


def _validate_record_id(record_id: str) -> None:
    try:
        validate_record_id(record_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid record ID format") from exc
