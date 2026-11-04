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

from usdcraft.pipeline.provider.client import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MAX_REQUEST_BYTES,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_REASONING_EFFORT,
    DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER_ENDPOINT,
    PROVIDER_ENDPOINTS,
    NvidiaHttpResponse,
    NvidiaHttpTransport,
    NvidiaProtocolError,
    NvidiaProviderConfig,
    NvidiaProviderError,
    NvidiaRequestError,
    NvidiaResponse,
    NvidiaResponsesClient,
    NvidiaRuntimeIdentity,
    ProviderEndpoint,
    resolve_nvidia_runtime_identity,
    resolve_provider_endpoint,
)
from usdcraft.pipeline.provider.harness import (
    DEFAULT_PROVIDER_MAX_TURNS,
    ProviderHarness,
    ProviderHarnessCancelledError,
    ProviderHarnessError,
    ProviderHarnessExhaustedError,
    ProviderHarnessResult,
    ProviderResponseClient,
    image_path_to_data_url,
)
from usdcraft.pipeline.provider.progress import ProviderProgressCallback
from usdcraft.pipeline.tools import (
    ProviderCheckRunner,
    ProviderCompileRunner,
    ProviderCompletionStatus,
    ProviderToolExecution,
    ProviderToolExecutor,
    ProviderToolResult,
    build_provider_tool_schemas,
)

__all__ = [
    "DEFAULT_NVIDIA_BASE_URL",
    "DEFAULT_NVIDIA_MAX_REQUEST_BYTES",
    "DEFAULT_NVIDIA_MODEL",
    "DEFAULT_NVIDIA_REASONING_EFFORT",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_PROVIDER_ENDPOINT",
    "DEFAULT_NVIDIA_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_PROVIDER_MAX_TURNS",
    "NvidiaHttpResponse",
    "NvidiaHttpTransport",
    "NvidiaProtocolError",
    "NvidiaProviderConfig",
    "NvidiaProviderError",
    "NvidiaRequestError",
    "NvidiaResponse",
    "NvidiaResponsesClient",
    "NvidiaRuntimeIdentity",
    "PROVIDER_ENDPOINTS",
    "ProviderCheckRunner",
    "ProviderCompileRunner",
    "ProviderCompletionStatus",
    "ProviderEndpoint",
    "ProviderHarness",
    "ProviderHarnessCancelledError",
    "ProviderHarnessError",
    "ProviderHarnessExhaustedError",
    "ProviderHarnessResult",
    "ProviderProgressCallback",
    "ProviderResponseClient",
    "ProviderToolExecution",
    "ProviderToolExecutor",
    "ProviderToolResult",
    "build_provider_tool_schemas",
    "image_path_to_data_url",
    "resolve_nvidia_runtime_identity",
    "resolve_provider_endpoint",
]
