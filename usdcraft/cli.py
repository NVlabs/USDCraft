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

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from usdcraft import __version__
from usdcraft.config import load_repo_env

if TYPE_CHECKING:
    from usdcraft.generation.contract import GenerationPreflight
    from usdcraft.reconstruction.contract import ReconstructionPreflight


def _repo_root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else Path.cwd().resolve()


def _checkout_root(value: str | None) -> Path:
    """Resolve the repository root for a command that needs a USDCraft source checkout."""

    root = _repo_root(value)
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        project = {}
    if not (root / "scaffold.py").is_file() or project.get("project", {}).get("name") != "usdcraft":
        raise SystemExit(
            f"{root} is not a USDCraft repository checkout. Run this command from the "
            "repository root or pass --repo-root <path>."
        )
    return root


def _require_render_setup() -> None:
    """Run-start check shared by every Generation and Reconstruction arm (single and batch)."""

    from usdcraft.render.browser import require_render_setup

    require_render_setup()


def _print_generation_approval(preflight: GenerationPreflight) -> None:
    """Print the one compact user-facing approval summary.

    The complete execution plan remains private to ``fingerprint_payload()``. The
    fingerprint is the only technical field exposed because the later confirmation
    command needs that exact stateless approval token.
    """

    request = preflight.request
    runtime = request.runtime
    print(f"prompt={json.dumps(request.prompt, ensure_ascii=False)}")
    print(f"model={runtime.model_id}")
    print(f"reasoning_effort={runtime.reasoning_effort}")
    print(f"input_mode={request.input_kind.value}")
    if request.image_path is not None:
        print(f"image_path={json.dumps(str(request.image_path), ensure_ascii=False)}")
    print(f"approval_fingerprint={preflight.fingerprint}")


def _print_reconstruction_approval(preflight: ReconstructionPreflight) -> None:
    """Print only the compact user-verifiable Codex-arm Reconstruction approval summary."""

    request = preflight.request
    runtime = request.runtime
    print(f"source_path={json.dumps(str(request.source_path), ensure_ascii=False)}")
    print("semantic_image_path=" + json.dumps(str(request.semantic_image_path), ensure_ascii=False))
    print(f"prompt={json.dumps(request.prompt, ensure_ascii=False)}")
    if request.label is not None:
        print(f"label={json.dumps(request.label, ensure_ascii=False)}")
    if request.guidance is not None:
        print(f"guidance={json.dumps(request.guidance, ensure_ascii=False)}")
    print(f"up_axis={request.up_axis or 'embedded_or_format_contract'}")
    print(f"front_axis={request.front_axis or 'unresolved_or_embedded'}")
    print(
        "meters_per_unit="
        + (
            str(request.meters_per_unit)
            if request.meters_per_unit is not None
            else "embedded_or_format_contract"
        )
    )
    print(
        "mesh_cleanup="
        + ("subject_isolation_agent" if request.mesh_cleanup else "skipped_retain_all")
    )
    print(f"model={runtime.model_id}")
    print(f"reasoning_effort={runtime.reasoning_effort}")
    print("harness=" + json.dumps(request.harness_policy.to_dict(), sort_keys=True))
    print(f"approval_fingerprint={preflight.fingerprint}")


def _run_init(args: argparse.Namespace) -> int:
    _create_data_layout(_checkout_root(args.repo_root))
    return 0


_NODE_REQUIREMENT = "Node.js ^20.19 or >=22.12 (required by Vite 7)"


def _node_npm() -> str:
    """Return the npm executable after checking that a Vite 7 compatible Node.js is on PATH."""

    node, npm = shutil.which("node"), shutil.which("npm")
    if node is None or npm is None:
        raise RuntimeError(f"npm/Node.js was not found on PATH; install {_NODE_REQUIREMENT}.")
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    major, minor = (int(part) for part in version.lstrip("v").split(".")[:2])
    if not ((major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major > 22):
        raise RuntimeError(f"Found Node.js {version}; USDCraft needs {_NODE_REQUIREMENT}.")
    return npm


def _run_setup(args: argparse.Namespace) -> int:
    """Install the render browser, build the Viewer/render bundle and create the data layout."""

    root = _checkout_root(args.repo_root)
    web_root = root / "viewer" / "web"
    try:
        npm = _node_npm()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    steps = (
        (
            "Installing the Playwright Chromium browser",
            [sys.executable, "-m", "playwright", "install", "chromium"],
            root,
        ),
        ("Installing viewer/web dependencies (npm ci)", [npm, "ci"], web_root),
        ("Building the Viewer and render bundle (npm run build)", [npm, "run", "build"], web_root),
    )
    total = len(steps) + 1
    for index, (label, command, cwd) in enumerate(steps, start=1):
        print(f"[{index}/{total}] {label}", flush=True)
        if subprocess.run(command, cwd=cwd).returncode != 0:
            print(f"Setup failed at step {index}: {' '.join(command)}", file=sys.stderr)
            return 1
    print(f"[{total}/{total}] Creating the local data layout", flush=True)
    _create_data_layout(root)
    print("setup=complete")
    return 0


def _create_data_layout(root: Path) -> None:
    for path in (
        root / "data" / "records",
        root / "data" / "local" / "generation_runs",
        root / "data" / "local" / "reconstruction_runs",
        root / "data" / "local" / "reconstruction_author_runs",
        root / "data" / "local" / "materialization_replacement_staging",
        root / "data" / "cache" / "record_materialization",
    ):
        path.mkdir(parents=True, exist_ok=True)
    print(f"initialized={root / 'data'}")


def _single_run_failed(
    args: argparse.Namespace,
    exc: BaseException,
    *,
    pipeline: str,
    provider: str,
    failure_label: str,
    failure_context: Callable[[BaseException], Any],
) -> int:
    """Report one failed single-item run with the same keys on every arm."""

    context = failure_context(exc)
    if args.json:
        payload: dict[str, object] = {
            "status": "failed",
            "pipeline": pipeline,
            "provider": provider,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if context is not None:
            payload["run_id"] = context.run_id
            payload["workspace_dir"] = str(context.workspace_dir)
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"{failure_label} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if context is not None:
            print(f"run_id={context.run_id}", file=sys.stderr)
            print(f"workspace_dir={context.workspace_dir}", file=sys.stderr)
    return 1


def _single_run_succeeded(
    args: argparse.Namespace,
    result: dict[str, object],
    *,
    root: Path,
    pipeline: str,
    provider: str,
) -> int:
    """Report one successful single-item run with the same keys on every arm."""

    payload: dict[str, object] = {"status": result["status"], "pipeline": pipeline}
    payload["provider"] = provider
    payload.update(
        (key, value)
        for key, value in result.items()
        if key not in {"status", "pipeline", "provider"}
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _print_run_result(payload, root=root)
    return 0


def _run_generation(args: argparse.Namespace) -> int:
    from usdcraft.generation.command import (
        execute_generation,
        generation_failure_context,
        preflight_generation,
    )

    root = _checkout_root(args.repo_root)
    load_repo_env(root)
    try:
        preflight = preflight_generation(
            repo_root=root,
            prompt=args.prompt,
            image=Path(args.image).expanduser() if args.image else None,
            model=args.model,
            reasoning=args.reasoning,
            label=args.label,
            keep_workspace=args.keep_workspace,
        )
        if args.confirm_generation is None and not args.direct:
            if args.json:
                print(json.dumps(preflight.to_dict(), ensure_ascii=False))
            else:
                _print_generation_approval(preflight)
            return 0
        _require_render_setup()
        result = execute_generation(
            preflight=preflight,
            approved_fingerprint=args.confirm_generation,
            direct=args.direct,
        )
    except Exception as exc:
        return _single_run_failed(
            args,
            exc,
            pipeline="generation",
            provider="codex-cli",
            failure_label="Generation",
            failure_context=generation_failure_context,
        )
    return _single_run_succeeded(
        args, result.to_dict(), root=root, pipeline="generation", provider="codex-cli"
    )


def _reconstruction_harness_options(args: argparse.Namespace) -> dict:
    options = {"image_observation": args.image_observation}
    if hasattr(args, "max_turns"):
        # Provider only: the Codex arm has no turn limit and no --max-turns.
        options["max_turns"] = args.max_turns
    return options


def _add_reconstruction_harness_arguments(
    parser: argparse.ArgumentParser, *, provider: bool
) -> None:
    parser.add_argument(
        "--image-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the independent advisory image inventory (default: on). "
            "Disabling it still attaches the semantic image to the Author."
        ),
    )
    if provider:
        parser.add_argument(
            "--max-turns",
            type=int,
            help="Provider Author turn budget, 1–200 (default: 80). The Codex arm has none.",
        )


def _add_mesh_cleanup_argument(parser: argparse.ArgumentParser) -> None:
    """The optional Subject-Isolation Agent switch, identical on both Reconstruction arms."""

    parser.add_argument(
        "--mesh-cleanup",
        action="store_true",
        help=(
            "Run the optional Subject-Isolation Agent to drop mesh artifacts such as background, "
            "scene dressing or detached flypoints. Default: skipped; every complete source "
            "component is retained by an explicit Host declaration."
        ),
    )


def _run_reconstruction(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.command import (
        execute_reconstruction,
        preflight_reconstruction,
        reconstruction_failure_context,
    )

    root = _checkout_root(args.repo_root)
    load_repo_env(root)
    try:
        preflight = preflight_reconstruction(
            repo_root=root,
            source=Path(args.source),
            semantic_image=Path(args.image),
            up_axis=args.up_axis,
            front_axis=args.front_axis,
            meters_per_unit=args.meters_per_unit,
            guidance=args.guidance,
            prompt=args.prompt,
            mesh_cleanup=args.mesh_cleanup,
            **_reconstruction_harness_options(args),
            model=args.model,
            reasoning=args.reasoning,
            label=args.label,
            keep_workspace=args.keep_workspace,
        )
        if args.confirm_reconstruction is None and not args.direct:
            if args.json:
                print(json.dumps(preflight.to_dict(), ensure_ascii=False))
            else:
                _print_reconstruction_approval(preflight)
            return 0
        _require_render_setup()
        result = execute_reconstruction(
            preflight=preflight,
            approved_fingerprint=args.confirm_reconstruction,
            direct=args.direct,
        )
    except Exception as exc:
        return _single_run_failed(
            args,
            exc,
            pipeline="reconstruction",
            provider="codex-cli",
            failure_label="Reconstruction",
            failure_context=reconstruction_failure_context,
        )
    return _single_run_succeeded(
        args, result.to_dict(), root=root, pipeline="reconstruction", provider="codex-cli"
    )


def _run_reconstruction_provider(args: argparse.Namespace) -> int:
    from usdcraft.pipeline.provider import resolve_provider_endpoint
    from usdcraft.pipeline.single_display import ProviderSingleDisplay
    from usdcraft.reconstruction.command import reconstruction_failure_context
    from usdcraft.reconstruction.provider_command import (
        PROGRESS_LABEL,
        execute_provider_reconstruction,
    )

    root = _checkout_root(args.repo_root)
    load_repo_env(root)
    # The Provider arm's identity is the endpoint its repository `.env` selects.
    provider = resolve_provider_endpoint(root)
    try:
        _require_render_setup()
        with ProviderSingleDisplay(
            stream=sys.stderr, force_plain=args.json, label=PROGRESS_LABEL
        ) as display:
            result = execute_provider_reconstruction(
                repo_root=root,
                source=Path(args.source),
                semantic_image=Path(args.image),
                up_axis=args.up_axis,
                front_axis=args.front_axis,
                meters_per_unit=args.meters_per_unit,
                guidance=args.guidance,
                prompt=args.prompt,
                label=args.label,
                model=args.model,
                reasoning=args.reasoning,
                timeout_seconds=args.timeout,
                keep_workspace=args.keep_workspace,
                mesh_cleanup=args.mesh_cleanup,
                **_reconstruction_harness_options(args),
                **_api_style_option(args),
                progress=display,
            )
    except Exception as exc:
        return _single_run_failed(
            args,
            exc,
            pipeline="reconstruction",
            provider=provider,
            failure_label="Provider Reconstruction",
            failure_context=reconstruction_failure_context,
        )
    return _single_run_succeeded(
        args, result.to_dict(), root=root, pipeline="reconstruction", provider=provider
    )


def _run_generation_provider(args: argparse.Namespace) -> int:
    from usdcraft.generation.command import generation_failure_context
    from usdcraft.generation.provider_command import execute_provider_generation
    from usdcraft.pipeline.provider import resolve_provider_endpoint
    from usdcraft.pipeline.single_display import ProviderSingleDisplay

    root = _checkout_root(args.repo_root)
    load_repo_env(root)
    # The Provider arm's identity is the endpoint its repository `.env` selects.
    provider = resolve_provider_endpoint(root)
    try:
        _require_render_setup()
        with ProviderSingleDisplay(stream=sys.stderr, force_plain=args.json) as display:
            result = execute_provider_generation(
                repo_root=root,
                prompt=args.prompt,
                image=Path(args.image).expanduser() if args.image else None,
                model=args.model,
                reasoning=args.reasoning,
                timeout_seconds=args.timeout,
                label=args.label,
                keep_workspace=args.keep_workspace,
                **_api_style_option(args),
                progress=display,
            )
    except Exception as exc:
        return _single_run_failed(
            args,
            exc,
            pipeline="generation",
            provider=provider,
            failure_label="Provider Generation",
            failure_context=generation_failure_context,
        )
    return _single_run_succeeded(
        args, result.to_dict(), root=root, pipeline="generation", provider=provider
    )


def _api_style_option(args: argparse.Namespace) -> dict[str, Any]:
    """Pass a non-default Provider wire protocol through; the default keeps call shapes stable."""

    api_style = getattr(args, "api_style", "responses")
    return {"api_style": api_style} if api_style != "responses" else {}


def _print_run_result(payload: dict[str, object], *, root: Path) -> None:
    """Print the one shared key=value result block of every single-item run (all four commands)."""

    print(f"status={payload['status']}")
    print(f"pipeline={payload['pipeline']}")
    print(f"provider={payload['provider']}")
    print(f"run_id={payload['run_id']}")
    print(f"record_id={payload['record_id']}")
    print(f"record_dir={payload['record_dir']}")
    print(f"model={payload['model_path']}")
    if payload.get("workspace_dir"):
        print(f"workspace_dir={payload['workspace_dir']}")
    warnings = payload.get("warnings") or []
    if warnings:
        # Exporter degradation notes, one compact line as the Codex arms print them.
        print(f"warnings={json.dumps(list(cast('list[object]', warnings)), ensure_ascii=False)}")
    print(f"viewer=uv run usdcraft viewer --repo-root {root}")


def _print_batch_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        line = f"row={row['row_id']} status={row['status']}"
        if row.get("record_id") is not None:
            line += f" record_id={row['record_id']}"
        if row.get("run_id") is not None:
            line += f" run_id={row['run_id']}"
        if row.get("workspace_dir") is not None:
            line += f" workspace_dir={row['workspace_dir']}"
        if row.get("warnings"):
            line += f" warnings={json.dumps(list(cast('list[object]', row['warnings'])), ensure_ascii=False)}"
        if row.get("error") is not None:
            line += f" error={json.dumps(row['error'], ensure_ascii=False)}"
        print(line)


def _run_provider_batch(
    args: argparse.Namespace,
    *,
    execute: Callable[..., Any],
    pipeline: str,
    display_label: str,
    failure_label: str,
    arm: Literal["provider", "codex"] = "provider",
) -> int:
    """Shared CLI shell of the Provider and Codex batch commands of both products."""

    from usdcraft.pipeline.batch import ProviderBatchInterrupted
    from usdcraft.pipeline.batch_display import ProviderBatchDisplay
    from usdcraft.pipeline.provider import resolve_provider_endpoint

    root = _checkout_root(args.repo_root)
    load_repo_env(root)
    # The Codex arm is `codex-cli`; the Provider arm is the endpoint its `.env` selects.
    provider = "codex-cli" if arm == "codex" else resolve_provider_endpoint(root)
    options: dict[str, Any] = {}
    try:
        if arm == "provider":
            # Time limits and the wire protocol are Provider-only; the Codex arm has neither.
            options.update(
                timeout_seconds=args.timeout, run_timeout_seconds=_batch_run_timeout(args)
            )
            options.update(_api_style_option(args))
        _require_render_setup()
        with ProviderBatchDisplay(
            stream=sys.stderr, force_plain=args.json, label=display_label
        ) as display:
            result = execute(
                csv_path=Path(args.csv),
                repo_root=root,
                model=args.model,
                reasoning=args.reasoning,
                concurrency=args.concurrency,
                keep_workspace=args.keep_workspace,
                progress=display,
                **options,
            )
    except ProviderBatchInterrupted as exc:
        payload = {"status": "interrupted", "pipeline": pipeline, **exc.to_dict()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("status=interrupted")
            print(f"provider={payload['provider']}")
            print(f"csv={payload['csv_path']}")
            print(f"total={payload['total']}")
            print(f"completed={payload['completed']}")
            print(f"succeeded={payload['succeeded']}")
            print(f"failed={payload['failed']}")
            _print_batch_rows(payload["rows"])
            print(
                "cancelled_row_ids=" + json.dumps(payload["cancelled_row_ids"], ensure_ascii=False)
            )
            print(
                "in_flight_row_ids=" + json.dumps(payload["in_flight_row_ids"], ensure_ascii=False)
            )
        return 130
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "pipeline": pipeline,
                        "provider": provider,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"{failure_label} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    result_payload = result.to_dict()
    payload = {"status": result_payload["status"], "pipeline": pipeline, **result_payload}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"status={payload['status']}")
        print(f"provider={payload['provider']}")
        print(f"csv={payload['csv_path']}")
        print(f"total={payload['total']}")
        print(f"succeeded={payload['succeeded']}")
        print(f"failed={payload['failed']}")
        _print_batch_rows(payload["rows"])
    return 0 if payload["failed"] == 0 else 1


def _batch_run_timeout(args: argparse.Namespace) -> float | None:
    """The validated per-row deadline of a Provider batch; ``--run-timeout 0`` means no limit."""

    from usdcraft.pipeline.deadline import validate_run_timeout

    if args.run_timeout == 0:
        return None
    validate_run_timeout(args.run_timeout)
    return args.run_timeout


def _validate_batch_csv(
    args: argparse.Namespace,
    *,
    load: Callable[[Path], tuple[Any, ...]],
    failure_label: str,
    check_rows: Callable[[tuple[Any, ...]], None] | None = None,
) -> int:
    """`--validate-only`: the CSV contract alone, with no runtime resolution, model calls or writes."""

    try:
        if hasattr(args, "run_timeout"):
            _batch_run_timeout(args)
        rows = load(Path(args.csv))
        if check_rows is not None:
            check_rows(rows)
    except Exception as exc:
        print(f"{failure_label} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    payload = {"status": "validated", "total": len(rows), "scope": "csv_contract"}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"status=validated total={len(rows)} scope=csv_contract")
    return 0


def _run_generation_batch(args: argparse.Namespace) -> int:
    from usdcraft.generation.provider_batch import execute_provider_batch, load_provider_batch

    if args.validate_only:
        return _validate_batch_csv(
            args, load=load_provider_batch, failure_label="Provider Generation Batch"
        )
    return _run_provider_batch(
        args,
        execute=execute_provider_batch,
        pipeline="generation",
        display_label="Provider Generation",
        failure_label="Provider Batch",
    )


def _run_reconstruction_batch(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.provider_batch import (
        execute_reconstruction_batch,
        load_reconstruction_batch,
    )

    if args.validate_only:
        return _validate_batch_csv(
            args,
            load=load_reconstruction_batch,
            failure_label="Provider Reconstruction Batch",
        )
    return _run_provider_batch(
        args,
        execute=execute_reconstruction_batch,
        pipeline="reconstruction",
        display_label="Provider Reconstruction",
        failure_label="Provider Reconstruction Batch",
    )


def _run_reconstruction_codex_batch(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.codex_batch import (
        execute_codex_reconstruction_batch,
        validate_codex_batch_rows,
    )
    from usdcraft.reconstruction.provider_batch import load_reconstruction_batch

    if args.validate_only:
        return _validate_batch_csv(
            args,
            load=load_reconstruction_batch,
            failure_label="Codex Reconstruction Batch",
            check_rows=validate_codex_batch_rows,
        )
    return _run_provider_batch(
        args,
        execute=execute_codex_reconstruction_batch,
        pipeline="reconstruction",
        display_label="Codex Reconstruction",
        failure_label="Codex Reconstruction Batch",
        arm="codex",
    )


def _run_generation_codex_batch(args: argparse.Namespace) -> int:
    from usdcraft.generation.codex_batch import (
        execute_codex_generation_batch,
        validate_codex_generation_batch_rows,
    )
    from usdcraft.generation.provider_batch import load_provider_batch

    if args.validate_only:
        return _validate_batch_csv(
            args,
            load=load_provider_batch,
            failure_label="Codex Generation Batch",
            check_rows=validate_codex_generation_batch_rows,
        )
    return _run_provider_batch(
        args,
        execute=execute_codex_generation_batch,
        pipeline="generation",
        display_label="Codex Generation",
        failure_label="Codex Generation Batch",
        arm="codex",
    )


def _run_compile(args: argparse.Namespace) -> int:
    from usdcraft.compile import compile_model

    # The same repository .env loading as every other command: only the whitelisted settings
    # (USDCRAFT_COMPILE_TIMEOUT_SECONDS, USDCRAFT_CODEX_BIN) are loaded, so the compile timeout
    # applies to a manual compile exactly as to a run.
    load_repo_env(_repo_root(args.repo_root))
    try:
        artifacts = compile_model(
            Path(args.model),
            Path(args.output),
            clean=not args.no_clean,
            render=not args.no_render,
        )
    except Exception as exc:
        print(f"Compile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("status=success")
    print(f"urdf={artifacts.urdf_path}")
    print(f"physx_usdc={artifacts.usdc_path}")
    print(f"newton_usdc={artifacts.newton_usdc_path}")
    print(f"asset_usda={artifacts.asset_usda_path}")
    print(f"usdz={artifacts.usdz_path}")
    print(f"report={artifacts.report_path}")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    from usdcraft.compile import check_model
    from usdcraft.compile.feedback import compile_signal_bundle_from_exception

    load_repo_env(_repo_root(args.repo_root))
    try:
        result = check_model(Path(args.model))
    except Exception as exc:
        bundle = compile_signal_bundle_from_exception(exc)
        print(f"Authoring checks failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(bundle.summary, file=sys.stderr)
        for signal in bundle.signals:
            print(
                f"{signal.severity}={signal.code}: {signal.summary}",
                file=sys.stderr,
            )
            if signal.details:
                print(signal.details, file=sys.stderr)
        return 1
    print("status=success")
    print("scope=authoring_only")
    print(f"model={result.model_path}")
    print(result.signal_bundle.summary)
    for warning in result.warnings:
        print(f"warning={warning}")
    return 0


def _run_recon_check_axial(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.diagnostic_cli import main as diagnostic_main

    argv = [args.axial, args.candidate_usd]
    if args.source is not None:
        argv.extend(("--source", args.source))
    if args.payload_dir is not None:
        argv.extend(("--payload-dir", args.payload_dir))
    if args.json:
        argv.append("--json")
    return diagnostic_main(argv)


def _run_recon_measure_face(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.diagnostic_cli import measure_main

    argv = [args.axial, f"--face={args.face}"]
    if args.z_band is not None:
        argv.extend(("--z-band", str(args.z_band[0]), str(args.z_band[1])))
    return measure_main(argv)


def _run_recon_author_feedback(args: argparse.Namespace) -> int:
    from usdcraft.reconstruction.author_tools import measure_mesh, render_candidate, render_source

    try:
        if args.command == "recon-measure-mesh":
            result = measure_mesh(
                Path(args.source),
                out_dir=Path(args.output),
                bounds=args.bounds,
                max_points=args.max_points,
            )
        elif args.command == "recon-compare-part":
            from usdcraft.reconstruction.part_comparison import compare_part

            result = compare_part(
                Path(args.source),
                Path(args.asset_dir),
                out_dir=Path(args.output),
                part=args.part,
                element=args.element,
                face=args.face,
                bounds=args.bounds,
            )
        else:
            positions = {}
            for item in getattr(args, "joint", []):
                name, value = item.split("=", 1)
                if name in positions:
                    raise ValueError("Joint names must be unique")
                positions[name] = float(value)
            import os

            if os.environ.get("USDCRAFT_RECONSTRUCTION_WORKSPACE"):
                from usdcraft.reconstruction.render_service import (
                    request_candidate_render,
                    request_source_render,
                )

                renderer = (
                    request_source_render
                    if args.command == "recon-render-source"
                    else request_candidate_render
                )
            else:
                renderer = (
                    render_source if args.command == "recon-render-source" else render_candidate
                )
            result = renderer(
                Path(args.asset_dir),
                out_dir=Path(args.output),
                azimuth=args.azimuth,
                elevation=args.elevation,
                **({"joint_positions": positions} if args.command != "recon-render-source" else {}),
            )
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


def _run_rematerialize(args: argparse.Namespace) -> int:
    from usdcraft.generation.rematerialize import rematerialize_record

    root = _repo_root(args.repo_root)
    load_repo_env(root)
    try:
        result = rematerialize_record(
            repo_root=root,
            record_id=args.record_id,
            render=args.render,
        )
    except Exception as exc:
        print(f"Rematerialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("status=success")
    print(f"record_id={result.record_id}")
    print(f"revision_id={result.revision_id}")
    print(f"model={result.model_path}")
    print(f"materialization={result.materialization_dir}")
    print(f"report={result.report_path}")
    print(f"transaction_id={result.transaction_id}")
    print(f"recovered={'true' if result.recovered else 'false'}")
    print(f"cleanup_pending={'true' if result.cleanup_pending else 'false'}")
    for warning in result.warnings:
        print(f"warning={warning}")
    return 0


def _run_record_validate(
    args: argparse.Namespace, *, pipeline: Literal["generation", "reconstruction"]
) -> int:
    """Explicit Storage validation, independent of Viewer filtering and caches."""

    from storage.data_validation import validate_data_format
    from storage.repo import StorageRepo

    if args.record_id is None and not args.all_records:
        print("Pass a record_id or --all.", file=sys.stderr)
        return 2
    if args.record_id is not None and args.all_records:
        print("Pass either a record_id or --all, not both.", file=sys.stderr)
        return 2
    root = _repo_root(args.repo_root)
    result = validate_data_format(
        StorageRepo(root),
        record_ids=[args.record_id] if args.record_id is not None else None,
        pipeline=pipeline,
    )
    for record_id, errors in result.record_errors.items():
        display = result.materializations.get(record_id)
        if display is None:
            print(f"record_id={record_id} status=invalid")
        else:
            print(
                f"record_id={record_id} "
                f"materialization_status={display.status} "
                f"materialization_displayable={'true' if display.displayable else 'false'} "
                f"record_status={'invalid' if errors else 'valid'}"
            )
        for error in errors:
            print(f"  issue: {error}")
    if args.all_records:
        status = "ok" if result.ok else "invalid"
        print(f"data_format={status} records={result.record_count} errors={len(result.errors)}")
    return 0 if result.ok else 1


def _run_reconstruction_validate(args: argparse.Namespace) -> int:
    return _run_record_validate(args, pipeline="reconstruction")


def _run_generation_validate(args: argparse.Namespace) -> int:
    return _run_record_validate(args, pipeline="generation")


def _viewer_web_root(root: Path) -> Path:
    import viewer

    packaged_web_root = Path(viewer.__file__).resolve().parent / "web"
    source_web_root = root / "viewer" / "web"
    return source_web_root if (source_web_root / "package.json").is_file() else packaged_web_root


def _viewer_npm(web_root: Path) -> str:
    if not (web_root / "package.json").is_file():
        raise RuntimeError(
            "Viewer frontend is not bundled and no source checkout is available for rebuilding."
        )
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Viewer build requires npm.")
    if not (web_root / "node_modules").is_dir():
        subprocess.run([npm, "--prefix", str(web_root), "ci"], check=True)
    return npm


def _ensure_viewer_build(root: Path, *, rebuild: bool) -> None:
    web_root = _viewer_web_root(root)
    dist = web_root / "dist" / "index.html"
    if dist.is_file() and not rebuild:
        return
    npm = _viewer_npm(web_root)
    subprocess.run([npm, "--prefix", str(web_root), "run", "build"], check=True)


def _run_viewer(args: argparse.Namespace) -> int:
    import uvicorn

    root = _repo_root(args.repo_root)
    port = args.port if args.port is not None else 8770
    try:
        _ensure_viewer_build(root, rebuild=args.rebuild)
    except Exception as exc:
        print(f"Viewer build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    os.environ["USDCRAFT_REPO_ROOT"] = str(root)
    os.environ["USDCRAFT_VIEWER_READ_ONLY"] = "1" if args.read_only else "0"
    # The Viewer rejects requests whose Host header is not a loopback name or the bound address.
    extra_hosts = [
        host.strip()
        for host in os.environ.get("USDCRAFT_VIEWER_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    os.environ["USDCRAFT_VIEWER_ALLOWED_HOSTS"] = ",".join(dict.fromkeys([args.host, *extra_hosts]))
    uvicorn.run("viewer.api.app:app", host=args.host, port=port, reload=False)
    return 0


_REPO_ROOT_HELP = "USDCraft repository root (default: the current directory); its .env is loaded."
_JSON_HELP = "Print one machine-readable JSON object instead of key=value lines."
_KEEP_WORKSPACE_HELP = (
    "Keep the working directory of each successful run (for a batch, of every row); "
    "a failed run always keeps it."
)
_REASONING_HELP = (
    "Provider (NVIDIA or OpenAI API) reasoning effort{scope}: typical values low (default), medium, high, xhigh; other "
    "values are passed verbatim. Use provider-default to omit the reasoning field for models that "
    "reject an explicit effort."
)


def _add_repo_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", help=_REPO_ROOT_HELP)


def _add_keep_workspace_argument(parser: argparse.ArgumentParser) -> None:
    """One ``--keep-workspace`` flag and wording on every single and batch run command."""

    parser.add_argument("--keep-workspace", action="store_true", help=_KEEP_WORKSPACE_HELP)


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """``--repo-root``, ``--keep-workspace`` and ``--json`` with one wording everywhere."""

    _add_repo_root_argument(parser)
    _add_keep_workspace_argument(parser)
    parser.add_argument("--json", action="store_true", help=_JSON_HELP)


def _add_provider_runtime_arguments(parser: argparse.ArgumentParser, *, scope: str) -> None:
    """The Provider runtime overrides shared by every direct-execution Provider command."""

    parser.add_argument("--model", help=f"Provider (NVIDIA or OpenAI API) model override{scope}.")
    parser.add_argument("--reasoning", help=_REASONING_HELP.format(scope=scope))
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Deadline in seconds for each Provider HTTP request, i.e. one model turn "
            "(default: 900). It is not a run clock: a timed-out request is retried from scratch, "
            "so keep it above the longest model turn."
        ),
    )


def _add_api_style_argument(parser: argparse.ArgumentParser, *, batch: bool = False) -> None:
    parser.add_argument(
        "--api-style",
        choices=("responses", "chat-completions"),
        default="responses",
        help=(
            "Provider wire protocol for the Author and image observation (default: responses)"
            + ("; a CSV api_style cell overrides it per row." if batch else ".")
        ),
    )


def _add_batch_arguments(
    parser: argparse.ArgumentParser, *, runtime_scope: str = " for every row"
) -> None:
    """The coordinator options shared by `generation-batch` and `reconstruction-batch`."""

    _add_provider_runtime_arguments(parser, scope=runtime_scope)
    _add_api_style_argument(parser, batch=True)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent rows (1-32; default: 1).",
    )
    from usdcraft.pipeline.deadline import DEFAULT_RUN_TIMEOUT_SECONDS

    parser.add_argument(
        "--run-timeout",
        type=float,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            f"Time limit for each row, from its start through saving the record "
            f"(default: {DEFAULT_RUN_TIMEOUT_SECONDS:g}; 0 = no limit). A row that runs out of "
            "time is stopped, keeps its working directory and is reported as failed. Needs "
            "macOS or Linux. Independent of --timeout."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the CSV contract only: no runtime resolution, model calls or workspace "
            "writes."
        ),
    )
    _add_output_arguments(parser)


def _add_codex_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """The coordinator options shared by both Codex batch commands (no time limits)."""

    parser.add_argument(
        "--model",
        help=(
            "Default Codex model for rows without a CSV runtime pair; pair with --reasoning. "
            "Without either, rows inherit the calling Codex task's runtime."
        ),
    )
    parser.add_argument(
        "--reasoning",
        help="Default Codex reasoning effort; pair with --model. CSV model/reasoning pairs override both.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Maximum concurrent rows (1-32; default: 1)."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the CSV contract only, without runtime resolution, Codex or workspace writes.",
    )
    _add_output_arguments(parser)


def _add_source_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--up-axis",
        help=(
            "The source axis that points up: +X, -X, +Y, -Y, +Z or -Z. Needed only when the "
            "source file does not declare it; write a negative value as --up-axis=-Y."
        ),
    )
    parser.add_argument(
        "--front-axis",
        help=(
            "Optional source axis the object's front faces: +X, -X, +Y, -Y, +Z or -Z; write a "
            "negative value as --front-axis=-Y."
        ),
    )
    parser.add_argument(
        "--meters-per-unit",
        type=float,
        help=(
            "Length of one source unit in metres, e.g. 0.01 for a file in centimetres. "
            "Default: the unit the file declares; a unit-less source whose longest side is 1 or "
            "2 units is scaled from an estimate based on the image."
        ),
    )


def _add_camera_arguments(parser: argparse.ArgumentParser, *, joints: bool) -> None:
    """Output and camera options shared by the authoring render commands."""

    parser.add_argument(
        "--output", required=True, help="New directory that receives the rendered image."
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=45,
        help="Camera angle around the vertical axis, in degrees (default: 45).",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=30,
        help="Camera angle above the horizon, in degrees from -90 to 90 (default: 30).",
    )
    if joints:
        parser.add_argument(
            "--joint",
            action="append",
            default=[],
            metavar="NAME=VALUE",
            help=(
                "Pose one joint before rendering, in radians or metres; repeat for more joints. "
                "Unlisted joints stay at their rest position."
            ),
        )


class _Parser(argparse.ArgumentParser):
    """Require exact long options so an unknown flag cannot silently match a longer one."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="usdcraft")
    parser.add_argument("--version", action="version", version=f"usdcraft {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_help = (
        "One-time checkout setup: install Playwright Chromium, run npm ci and npm run build "
        "in viewer/web (Viewer and render bundle), and create the local data layout. Needs "
        f"{_NODE_REQUIREMENT}."
    )
    setup = sub.add_parser("setup", help=setup_help, description=setup_help)
    setup.add_argument(
        "--repo-root", help="USDCraft repository root (default: the current directory)."
    )
    setup.set_defaults(func=_run_setup)

    init = sub.add_parser("init", help="Create the local USDCraft data layout.")
    _add_repo_root_argument(init)
    init.set_defaults(func=_run_init)

    generation = sub.add_parser(
        "generation",
        help="Create one new asset directly with the Provider (NVIDIA or OpenAI API).",
    )
    generation.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Optional asset/function specification. With --image and no text, Generation uses "
            "the standard complete simulator-ready image prompt."
        ),
    )
    generation.add_argument(
        "--image",
        help="Optional inspiration image; never ground truth. Its concrete path is executed.",
    )
    _add_provider_runtime_arguments(generation, scope="")
    _add_api_style_argument(generation)
    generation.add_argument(
        "--label",
        help="Optional concise object name used for the canonical record.",
    )
    _add_output_arguments(generation)
    generation.set_defaults(func=_run_generation_provider)

    generation_codex = sub.add_parser(
        "generation-codex",
        help=(
            "Codex Generation arm (--direct, or zero-write preflight followed by "
            "--confirm-generation); creates one canonical record from text and/or one "
            "inspiration image."
        ),
    )
    generation_codex.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Optional asset/function specification. With --image and no text, Generation uses "
            "the standard complete simulator-ready image prompt."
        ),
    )
    generation_codex.add_argument("--image", help="Optional inspiration image; never ground truth.")
    generation_codex.add_argument(
        "--model",
        help=(
            "Explicit Codex model. Must be paired with --reasoning; otherwise Generation "
            "inherits the root Codex task runtime."
        ),
    )
    generation_codex.add_argument(
        "--reasoning",
        default=None,
        help=("Explicit Codex reasoning effort, preserved verbatim. Must be paired with --model."),
    )
    generation_codex.add_argument(
        "--label",
        help="Optional concise object name used for the canonical record.",
    )
    _add_repo_root_argument(generation_codex)
    _add_keep_workspace_argument(generation_codex)
    generation_authorization = generation_codex.add_mutually_exclusive_group()
    generation_authorization.add_argument(
        "--direct",
        action="store_true",
        help="Execute immediately after input validation, without the preflight/confirmation exchange.",
    )
    generation_authorization.add_argument(
        "--confirm-generation",
        metavar="FINGERPRINT",
        help="Execute only when FINGERPRINT exactly matches the current zero-write preflight.",
    )
    generation_codex.add_argument("--json", action="store_true", help=_JSON_HELP)
    generation_codex.set_defaults(func=_run_generation)

    generation_batch = sub.add_parser(
        "generation-batch",
        help="Directly generate CSV rows with the Provider (NVIDIA or OpenAI API).",
    )
    generation_batch.add_argument(
        "csv",
        help=(
            "CSV with row_id, prompt, image_path; optional label, api_style and paired "
            "model/reasoning overrides. Unknown columns are rejected."
        ),
    )
    _add_batch_arguments(
        generation_batch,
        runtime_scope=" as a default (a CSV model/reasoning pair takes precedence)",
    )
    generation_batch.set_defaults(func=_run_generation_batch)

    generation_codex_batch = sub.add_parser(
        "generation-codex-batch",
        help="Directly generate CSV rows with Codex; no fingerprint confirmation exchange.",
    )
    generation_codex_batch.add_argument(
        "csv",
        help=(
            "Same Generation CSV schema as generation-batch, except that the Provider-only "
            "api_style cells must be empty."
        ),
    )
    _add_codex_batch_arguments(generation_codex_batch)
    generation_codex_batch.set_defaults(func=_run_generation_codex_batch)

    reconstruction = sub.add_parser(
        "reconstruction",
        help=(
            "Reconstruct one declared source instance directly with the Provider "
            "(NVIDIA or OpenAI API)."
        ),
    )
    reconstruction.add_argument("source", help="Source mesh, USD scene, or point cloud.")
    reconstruction.add_argument(
        "--image",
        required=True,
        help="The one semantic image of the subject being reconstructed.",
    )
    _add_source_frame_arguments(reconstruction)
    _add_reconstruction_harness_arguments(reconstruction, provider=True)
    _add_mesh_cleanup_argument(reconstruction)
    reconstruction.add_argument(
        "--prompt",
        help=(
            "Optional request text: what to reconstruct. Default: a fixed sentence naming the "
            "primary subject shown in the semantic image and source mesh."
        ),
    )
    reconstruction.add_argument(
        "--guidance",
        help="Optional user-authored reconstruction guidance bound into the request.",
    )
    _add_provider_runtime_arguments(reconstruction, scope="")
    _add_api_style_argument(reconstruction)
    reconstruction.add_argument("--label", help="Optional concise canonical display name.")
    _add_output_arguments(reconstruction)
    reconstruction.set_defaults(func=_run_reconstruction_provider)

    reconstruction_codex = sub.add_parser(
        "reconstruction-codex",
        help=(
            "Codex Reconstruction arm (--direct, or zero-write preflight followed by "
            "--confirm-reconstruction); creates one canonical record from one source mesh and one "
            "semantic image."
        ),
    )
    reconstruction_codex.add_argument("source", help="User-supplied source mesh/USD/point cloud.")
    reconstruction_codex.add_argument(
        "--image",
        required=True,
        help="The one semantic image of the subject being reconstructed.",
    )
    _add_source_frame_arguments(reconstruction_codex)
    _add_reconstruction_harness_arguments(reconstruction_codex, provider=False)
    reconstruction_codex.add_argument(
        "--prompt",
        help=(
            "Optional request text: what to reconstruct, bound into the approval fingerprint. "
            "Default: a fixed sentence naming the primary subject."
        ),
    )
    reconstruction_codex.add_argument(
        "--guidance",
        help=(
            "Optional user-authored reconstruction guidance, included in the approval "
            "fingerprint; no coordinator patch is added later."
        ),
    )
    _add_mesh_cleanup_argument(reconstruction_codex)
    reconstruction_codex.add_argument(
        "--model",
        help=(
            "Explicit Codex model. Must be paired with --reasoning; otherwise Reconstruction "
            "inherits the root Codex task runtime."
        ),
    )
    reconstruction_codex.add_argument(
        "--reasoning",
        help="Explicit Codex reasoning effort. Must be paired with --model.",
    )
    reconstruction_codex.add_argument("--label", help="Optional concise canonical display name.")
    _add_repo_root_argument(reconstruction_codex)
    _add_keep_workspace_argument(reconstruction_codex)
    reconstruction_authorization = reconstruction_codex.add_mutually_exclusive_group()
    reconstruction_authorization.add_argument(
        "--direct",
        action="store_true",
        help="Execute immediately after input validation, without the preflight/confirmation exchange.",
    )
    reconstruction_authorization.add_argument(
        "--confirm-reconstruction",
        metavar="FINGERPRINT",
        help="Execute only when FINGERPRINT exactly matches the current zero-write preflight.",
    )
    reconstruction_codex.add_argument("--json", action="store_true", help=_JSON_HELP)
    reconstruction_codex.set_defaults(func=_run_reconstruction)

    reconstruction_batch = sub.add_parser(
        "reconstruction-batch",
        help="Directly reconstruct CSV rows with the Provider (NVIDIA or OpenAI API).",
    )
    reconstruction_batch.add_argument(
        "csv",
        help=(
            "CSV with row_id, source_path, image_path; optional prompt, guidance, label, "
            "mesh_cleanup, image_observation, max_turns, api_style, frame columns "
            "and paired model/reasoning overrides."
        ),
    )
    _add_batch_arguments(
        reconstruction_batch,
        runtime_scope=" as a default (a CSV model/reasoning pair takes precedence)",
    )
    reconstruction_batch.set_defaults(func=_run_reconstruction_batch)

    codex_batch = sub.add_parser(
        "reconstruction-codex-batch",
        help="Directly reconstruct CSV rows with Codex; no fingerprint confirmation exchange.",
    )
    codex_batch.add_argument(
        "csv",
        help=(
            "Same Reconstruction CSV schema as reconstruction-batch (including the optional "
            "mesh_cleanup column), except that the Provider-only max_turns and api_style cells "
            "are rejected."
        ),
    )
    _add_codex_batch_arguments(codex_batch)
    codex_batch.set_defaults(func=_run_reconstruction_codex_batch)

    compile_one = sub.add_parser("compile", help="Strictly compile one trusted model.py.")
    compile_one.add_argument("model", help="Path of the model.py to compile.")
    compile_one.add_argument(
        "--output", required=True, help="Directory that receives the compiled asset files."
    )
    compile_one.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the optional preview images; only the asset files and report are written.",
    )
    compile_one.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Keep existing files in the output directory instead of deleting it before compiling."
        ),
    )
    _add_repo_root_argument(compile_one)
    compile_one.set_defaults(func=_run_compile)

    check_one = sub.add_parser(
        "check",
        help="Run trusted model.py authoring tests and diagnostics without promotion.",
    )
    check_one.add_argument("model", help="Path of the model.py to check.")
    _add_repo_root_argument(check_one)
    check_one.set_defaults(func=_run_check)

    recon_check_axial = sub.add_parser(
        "recon-check-axial",
        help=(
            "Authoring-only AXIAL re-encode feedback for one compiled Reconstruction "
            "candidate; advisory only, it never blocks a record."
        ),
    )
    recon_check_axial.add_argument("axial", help="The run's AXIAL text file (inputs/axial.txt).")
    recon_check_axial.add_argument(
        "candidate_usd", help="The compiled candidate USD file to compare with the source."
    )
    recon_check_axial.add_argument(
        "--source",
        help=(
            "Host-side normalized source USD; prints the mesh-to-mesh DEFECTS audit the Provider "
            "check_axial tool returns instead of the stack re-encode report."
        ),
    )
    recon_check_axial.add_argument(
        "--payload-dir",
        help="Directory receiving one structured payload per call for the record's axial summary.",
    )
    recon_check_axial.add_argument("--json", action="store_true", help=_JSON_HELP)
    recon_check_axial.set_defaults(func=_run_recon_check_axial)

    mesh_query = sub.add_parser(
        "recon-measure-mesh", help="Query frozen mesh regions for the Author."
    )
    mesh_query.add_argument("source", help="The frozen source mesh (inputs/source_mesh.usdc).")
    mesh_query.add_argument(
        "--output", required=True, help="New directory that receives this query's result files."
    )
    bounds_help = (
        "Source region in metres, six space-separated values. Example: "
        "--bounds -0.10 -0.20 0.00 0.10 0.20 0.30 (no commas or '=' after --bounds)."
    )
    bounds_metavar = ("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX")
    mesh_query.add_argument(
        "--bounds", nargs=6, type=float, metavar=bounds_metavar, help=bounds_help
    )
    mesh_query.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Maximum number of sample points to store (1-4096; default: 256).",
    )
    mesh_query.set_defaults(func=_run_recon_author_feedback)
    part_query = sub.add_parser(
        "recon-compare-part",
        help="Compare a named candidate part against a selected source region.",
    )
    part_query.add_argument("source", help="The frozen source mesh (inputs/source_mesh.usdc).")
    part_query.add_argument("asset_dir", help="The compiled candidate directory (build).")
    part_query.add_argument(
        "--output", required=True, help="New directory that receives the comparison files."
    )
    part_query.add_argument("--part", required=True, help="Name of the candidate part to compare.")
    part_query.add_argument(
        "--element", help="Optional name of one visual element of that part to compare alone."
    )
    part_query.add_argument(
        "--face",
        required=True,
        choices=["-X", "+X", "-Y", "+Y", "-Z", "+Z"],
        help="Side to compare from; write a negative value as --face=-Y.",
    )
    part_query.add_argument(
        "--bounds", nargs=6, type=float, required=True, metavar=bounds_metavar, help=bounds_help
    )
    part_query.set_defaults(func=_run_recon_author_feedback)
    generation_view = sub.add_parser(
        "generation-render", help="Render a compiled Generation candidate during authoring."
    )
    generation_view.add_argument("asset_dir", help="The compiled candidate directory (build).")
    _add_camera_arguments(generation_view, joints=True)
    from usdcraft.generation.candidate_render import run_render_cli

    generation_view.set_defaults(func=run_render_cli)

    candidate_view = sub.add_parser(
        "recon-render", help="Render a compiled Reconstruction candidate."
    )
    candidate_view.add_argument("asset_dir", help="The compiled candidate directory (build).")
    _add_camera_arguments(candidate_view, joints=True)
    candidate_view.set_defaults(func=_run_recon_author_feedback)

    source_view = sub.add_parser(
        "recon-render-source", help="Render the frozen source mesh at an Author-selected camera."
    )
    source_view.add_argument(
        "asset_dir", help="The run's staged source render input (inputs/source_render)."
    )
    _add_camera_arguments(source_view, joints=False)
    source_view.set_defaults(func=_run_recon_author_feedback)

    recon_measure_face = sub.add_parser(
        "recon-measure-face",
        help=(
            "Authoring-only per-face AXIAL readings (FACE_FEATURES protrusions, FACE_OPENINGS, "
            "optional body islands of a z band) for the Codex Reconstruction Author."
        ),
    )
    recon_measure_face.add_argument("axial", help="The run's AXIAL text file (inputs/axial.txt).")
    recon_measure_face.add_argument(
        "--face",
        required=True,
        choices=("-Y", "+Y", "-X", "+X", "+Z", "-Z"),
        help="Axis face; use the equals form (--face=-Y) for a negative token.",
    )
    recon_measure_face.add_argument(
        "--z-band",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="Optional height range in metres; also lists the separate solid pieces in that range.",
    )
    recon_measure_face.set_defaults(func=_run_recon_measure_face)

    reconstruction_validate = sub.add_parser(
        "reconstruction-validate",
        help=(
            "User-invoked full canonical contract check of one or all Reconstruction records; "
            "the Viewer record list never runs this automatically."
        ),
    )
    reconstruction_validate.add_argument(
        "record_id",
        nargs="?",
        help="Validate only this record. Omit together with --all to validate every record.",
    )
    reconstruction_validate.add_argument(
        "--all",
        dest="all_records",
        action="store_true",
        help="Validate every Reconstruction record (required when record_id is omitted).",
    )
    _add_repo_root_argument(reconstruction_validate)
    reconstruction_validate.set_defaults(func=_run_reconstruction_validate)

    generation_validate = sub.add_parser(
        "generation-validate",
        help=(
            "User-invoked full canonical contract check of one or all Generation records; "
            "the Viewer record list never runs this automatically."
        ),
    )
    generation_validate.add_argument(
        "record_id",
        nargs="?",
        help="Validate only this record. Omit together with --all to validate every record.",
    )
    generation_validate.add_argument(
        "--all",
        dest="all_records",
        action="store_true",
        help="Validate every Generation record (required when record_id is omitted).",
    )
    _add_repo_root_argument(generation_validate)
    generation_validate.set_defaults(func=_run_generation_validate)

    rematerialize = sub.add_parser(
        "rematerialize",
        help=(
            "User-invoked offline rebuild of one current Generation materialization; "
            "Viewer never runs this command."
        ),
    )
    rematerialize.add_argument("record_id", help="The Generation record to rebuild.")
    _add_repo_root_argument(rematerialize)
    rematerialize.add_argument(
        "--render",
        action="store_true",
        help="Also redraw the optional preview images (they are not part of the record itself).",
    )
    rematerialize.set_defaults(func=_run_rematerialize)

    viewer = sub.add_parser(
        "viewer",
        help="Build and serve the local Viewer (its only writes are rating and confirmed deletion).",
    )
    _add_repo_root_argument(viewer)
    viewer.add_argument(
        "--host",
        default="127.0.0.1",
        help="Network address to serve on (default: 127.0.0.1, this computer only).",
    )
    viewer.add_argument("--port", type=int, help="Port to serve on (default: 8770).")
    viewer.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Viewer web page before serving, even if a build already exists.",
    )
    viewer.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Serve a review copy: star rating and record deletion answer 403 and their controls "
            "are hidden; every read-only projection stays available."
        ),
    )
    viewer.set_defaults(func=_run_viewer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
