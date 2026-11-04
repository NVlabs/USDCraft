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

from pathlib import Path

from storage.data_validation import validate_data_format
from storage.repo import StorageRepo


def test_explicit_empty_selection_does_not_expand_to_all(tmp_path: Path) -> None:
    bad = tmp_path / "data/records/rec_broken"
    bad.mkdir(parents=True)
    (bad / "record.json").write_text("{broken")
    repo = StorageRepo(tmp_path)
    empty = validate_data_format(repo, record_ids=[])
    assert empty.ok and empty.record_count == 0 and empty.record_errors == {}
    assert not validate_data_format(repo).ok
    assert not validate_data_format(repo, record_ids=[], require_records=True).ok
