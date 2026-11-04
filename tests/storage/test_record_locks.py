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

import threading
from pathlib import Path

import storage.record_locks as record_locks
from storage.repo import StorageRepo


def test_waiting_thread_keeps_shared_local_record_lock_alive(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_waiting_lock"
    owner_lock = record_locks._local_record_lock(repo, record_id)
    waiter_has_lock_reference = threading.Event()
    waiter_acquired = threading.Event()
    release_waiter = threading.Event()
    observed_lock: list[threading.RLock] = []

    def wait_for_lock() -> None:
        waiter_lock = record_locks._local_record_lock(repo, record_id)
        observed_lock.append(waiter_lock)
        waiter_has_lock_reference.set()
        with waiter_lock:
            waiter_acquired.set()
            release_waiter.wait(timeout=2)

    with owner_lock:
        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()
        assert waiter_has_lock_reference.wait(timeout=2)
        assert observed_lock == [owner_lock]
        assert not waiter_acquired.is_set()

    assert waiter_acquired.wait(timeout=2)
    release_waiter.set()
    waiter.join(timeout=2)
    assert not waiter.is_alive()


def test_record_lock_is_reentrant_within_one_thread(tmp_path: Path) -> None:
    repo = StorageRepo(tmp_path)
    record_id = "rec_reentrant"
    entered: list[str] = []

    def nested_hold() -> None:
        with record_locks.record_lock(repo, record_id):
            entered.append("outer")
            with record_locks.record_lock(repo, record_id):
                entered.append("inner")

    worker = threading.Thread(target=nested_hold, daemon=True)
    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive(), "nested record_lock deadlocked in one thread"
    assert entered == ["outer", "inner"]

    # The flock must be fully released afterwards: a fresh acquisition succeeds.
    with record_locks.record_lock(repo, record_id):
        entered.append("again")
    assert entered[-1] == "again"
