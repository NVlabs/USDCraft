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

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from storage.record_mutations import RecordMutationStore
from viewer.api.file_resolver import ViewerFileResolver
from viewer.api.reconstruction_records import ReconstructionRecordStore
from viewer.api.store import ViewerStore


def get_viewer_store(request: Request) -> ViewerStore:
    return cast(ViewerStore, request.app.state.viewer_store)


def get_file_resolver(request: Request) -> ViewerFileResolver:
    return cast(ViewerFileResolver, request.app.state.file_resolver)


def is_read_only(request: Request) -> bool:
    return bool(getattr(request.app.state, "read_only", False))


def get_record_mutations(request: Request) -> RecordMutationStore:
    """The storage-owned mutation store; a read-only Viewer refuses every mutation route."""

    if is_read_only(request):
        raise HTTPException(
            status_code=403,
            detail="Viewer is read-only: rating and deletion are disabled on this deployment.",
        )
    store = get_viewer_store(request)
    return RecordMutationStore(store.repo)


def get_reconstruction_record_store(request: Request) -> ReconstructionRecordStore:
    return cast(ReconstructionRecordStore, request.app.state.reconstruction_record_store)


ViewerStoreDep = Annotated[ViewerStore, Depends(get_viewer_store)]
ReadOnlyDep = Annotated[bool, Depends(is_read_only)]
FileResolverDep = Annotated[ViewerFileResolver, Depends(get_file_resolver)]
RecordMutationsDep = Annotated[RecordMutationStore, Depends(get_record_mutations)]
ReconstructionRecordStoreDep = Annotated[
    ReconstructionRecordStore,
    Depends(get_reconstruction_record_store),
]
