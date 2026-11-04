# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

setup:
    uv sync
    uv run usdcraft setup

format:
    uv run ruff format .

lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy
    npm --prefix viewer/web run lint
    npm --prefix viewer/web run typecheck

test:
    uv run pytest -q

test-fast:
    uv run pytest -q -m "not slow"

viewer:
    uv run usdcraft viewer

