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

"""Host-header validation that keeps a loopback Viewer out of DNS-rebinding reach.

A page on an attacker-controlled domain can re-point that domain at 127.0.0.1 and then reach the
unauthenticated local Viewer same-origin; its requests still carry the attacker's ``Host``. The
guard therefore answers 400 unless the ``Host`` names a loopback address or a host the user bound
explicitly. Starlette's ``TrustedHostMiddleware`` is not used because it splits the header on the
first ``:`` and so cannot match a bracketed IPv6 host such as ``[::1]:8770``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Comma-separated extra hosts; `usdcraft viewer` adds its `--host` value here.
ALLOWED_HOSTS_ENV = "USDCRAFT_VIEWER_ALLOWED_HOSTS"
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
# A wildcard bind serves every interface under whatever name reaches it, so it cannot be reduced
# to a Host allowlist; the deployment then needs the external access-control boundary the README
# already requires for any non-loopback bind.
WILDCARD_BIND_HOSTS = frozenset({"*", "0.0.0.0", "::", "[::]"})


def normalize_host(value: str) -> str | None:
    """Lower-cased host of a ``Host`` header or bind address, without port; None if malformed."""

    host = value.strip().lower()
    if not host:
        return None
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return None
        rest = host[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return None
        return host[: end + 1]
    if host.count(":") > 1:
        # A bare IPv6 address (a bind value such as ``::1``); a Host header always brackets it.
        return f"[{host}]"
    name, _, port = host.partition(":")
    if port and not port.isdigit():
        return None
    name = name.rstrip(".")
    return name or None


def resolve_allowed_hosts(extra: Iterable[str] | None = None) -> frozenset[str] | None:
    """Loopback names plus the explicit extra hosts; None means every Host is accepted.

    ``extra`` defaults to ``USDCRAFT_VIEWER_ALLOWED_HOSTS``.
    """

    if extra is None:
        extra = os.getenv(ALLOWED_HOSTS_ENV, "").split(",")
    allowed = set(LOOPBACK_HOSTS)
    for raw in extra:
        if raw.strip().lower() in WILDCARD_BIND_HOSTS:
            return None
        host = normalize_host(raw)
        if host is not None:
            allowed.add(host)
    return frozenset(allowed)


class HostGuardMiddleware:
    """Reject HTTP/WebSocket requests whose ``Host`` is not in the allowlist (400)."""

    def __init__(self, app: ASGIApp, allowed_hosts: frozenset[str] | None) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.allowed_hosts is None or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = normalize_host(Headers(scope=scope).get("host", ""))
        if host is not None and host in self.allowed_hosts:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})
            return
        response = PlainTextResponse("Invalid host header", status_code=400)
        await response(scope, receive, send)


def log_host_policy(allowed_hosts: frozenset[str] | None) -> None:
    if allowed_hosts is None:
        logger.warning(
            "Viewer bound to a wildcard address: Host-header checking is off; put an external "
            "access-control boundary in front of it."
        )


__all__ = [
    "ALLOWED_HOSTS_ENV",
    "HostGuardMiddleware",
    "LOOPBACK_HOSTS",
    "log_host_policy",
    "normalize_host",
    "resolve_allowed_hosts",
]
