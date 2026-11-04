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

import fcntl
import threading
import weakref
from contextlib import contextmanager
from typing import Iterator

from storage.identifiers import validate_record_id
from storage.repo import StorageRepo

_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: weakref.WeakValueDictionary[tuple[str, str], threading.RLock] = (
    weakref.WeakValueDictionary()
)
_THREAD_STATE = threading.local()


def _local_record_lock(repo: StorageRepo, record_id: str) -> threading.RLock:
    key = (repo.root.as_posix(), record_id)
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def record_lock(repo: StorageRepo, record_id: str) -> Iterator[None]:
    """Serialize every canonical transition for one record across threads and processes.

    Re-entrant within one thread: the RLock admits the nested holder, and a per-thread
    depth counter skips the second flock, which would otherwise self-deadlock because
    flock is scoped to the open file description, not the thread.
    """

    record_id = validate_record_id(record_id)
    key = (repo.root.as_posix(), record_id)
    local_lock = _local_record_lock(repo, record_id)
    with local_lock:
        depths: dict[tuple[str, str], int] = getattr(_THREAD_STATE, "depths", None) or {}
        _THREAD_STATE.depths = depths
        if depths.get(key, 0) > 0:
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return
        lock_path = repo.layout.record_lock_path(record_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            depths[key] = 1
            try:
                yield
            finally:
                del depths[key]
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["record_lock"]
