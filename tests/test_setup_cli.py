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

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from usdcraft import cli as usdcraft_cli
from usdcraft.render import browser


def _checkout(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "scaffold.py").write_text("object_model = object()\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "usdcraft"\n', encoding="utf-8")
    return root


def _fake_tools(monkeypatch: pytest.MonkeyPatch, *, node_version: str = "v22.12.0") -> list:
    calls: list[tuple[list[str], Path | None]] = []
    # Replace only the CLI module's view of shutil/subprocess, never the global modules.
    monkeypatch.setattr(usdcraft_cli, "shutil", SimpleNamespace(which=lambda name: f"/bin/{name}"))

    def run(command, cwd=None, **_kwargs):
        calls.append((list(command), cwd))
        stdout = node_version if command[-1] == "--version" else ""
        return SimpleNamespace(returncode=0, stdout=stdout)

    _fake_subprocess(monkeypatch, run)
    return calls


def _fake_subprocess(monkeypatch: pytest.MonkeyPatch, run) -> None:
    fake = SimpleNamespace(run=run, CalledProcessError=subprocess.CalledProcessError)
    monkeypatch.setattr(usdcraft_cli, "subprocess", fake)


def test_setup_installs_browser_builds_bundle_and_creates_data_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _checkout(tmp_path / "repo")
    calls = _fake_tools(monkeypatch)

    assert usdcraft_cli.main(["setup", "--repo-root", str(root)]) == 0

    web = root / "viewer" / "web"
    assert calls == [
        (["/bin/node", "--version"], None),
        ([sys.executable, "-m", "playwright", "install", "chromium"], root),
        (["/bin/npm", "ci"], web),
        (["/bin/npm", "run", "build"], web),
    ]
    assert (root / "data" / "records").is_dir()
    assert (root / "data" / "cache" / "record_materialization").is_dir()
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("[")] == [
        "[1/4] Installing the Playwright Chromium browser",
        "[2/4] Installing viewer/web dependencies (npm ci)",
        "[3/4] Building the Viewer and render bundle (npm run build)",
        "[4/4] Creating the local data layout",
    ]
    assert lines[-1] == "setup=complete"


def test_setup_stops_at_the_first_failed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _checkout(tmp_path / "repo")
    calls = _fake_tools(monkeypatch)

    def run(command, cwd=None, **_kwargs):
        calls.append((list(command), cwd))
        failed = command[-1] == "ci"
        return SimpleNamespace(returncode=1 if failed else 0, stdout="v24.1.0")

    _fake_subprocess(monkeypatch, run)

    assert usdcraft_cli.main(["setup", "--repo-root", str(root)]) == 1

    assert calls[-1][0] == ["/bin/npm", "ci"]
    assert "Setup failed at step 2: /bin/npm ci" in capsys.readouterr().err
    assert not (root / "data").exists()


@pytest.mark.parametrize("missing", ["node", "npm"])
def test_setup_requires_node_and_npm_before_installing_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    root = _checkout(tmp_path / "repo")
    calls = _fake_tools(monkeypatch)
    monkeypatch.setattr(
        usdcraft_cli,
        "shutil",
        SimpleNamespace(which=lambda name: None if name == missing else f"/bin/{name}"),
    )

    assert usdcraft_cli.main(["setup", "--repo-root", str(root)]) == 1

    assert calls == []
    assert "npm/Node.js was not found on PATH" in (err := capsys.readouterr().err)
    assert "Node.js ^20.19 or >=22.12" in err


@pytest.mark.parametrize("version", ["v18.20.4", "v20.18.1", "v21.7.0", "v22.11.0"])
def test_setup_rejects_node_versions_vite_does_not_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
) -> None:
    root = _checkout(tmp_path / "repo")
    calls = _fake_tools(monkeypatch, node_version=version)

    assert usdcraft_cli.main(["setup", "--repo-root", str(root)]) == 1

    assert [command for command, _cwd in calls] == [["/bin/node", "--version"]]
    assert f"Found Node.js {version}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["setup"],
        ["init"],
        ["generation", "a lamp"],
        ["generation-codex", "a lamp"],
        ["generation-batch", "jobs.csv"],
        ["reconstruction", "scan.glb", "--image", "photo.jpg"],
        ["reconstruction-codex", "scan.glb", "--image", "photo.jpg"],
        ["reconstruction-batch", "jobs.csv"],
        ["reconstruction-codex-batch", "jobs.csv"],
    ],
)
def test_checkout_commands_reject_a_directory_that_is_not_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n', encoding="utf-8")
    (tmp_path / "scaffold.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        usdcraft_cli.main(argv)

    assert str(exc_info.value) == (
        f"{tmp_path.resolve()} is not a USDCraft repository checkout. Run this command from the "
        "repository root or pass --repo-root <path>."
    )
    assert not (tmp_path / "data").exists()


def test_init_accepts_an_explicit_checkout_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _checkout(tmp_path / "repo")

    assert usdcraft_cli.main(["init", "--repo-root", str(root)]) == 0

    assert (root / "data" / "local" / "generation_runs").is_dir()
    assert capsys.readouterr().out == f"initialized={root.resolve() / 'data'}\n"


def test_render_setup_requires_the_built_render_page(tmp_path: Path) -> None:
    with pytest.raises(browser.RenderSetupError) as exc_info:
        browser.require_render_setup(tmp_path / "dist")

    message = str(exc_info.value)
    assert f"The render bundle {tmp_path / 'dist' / 'render.html'} is missing." in message
    assert "Run `uv run usdcraft setup` from the repository root" in message


def test_render_setup_requires_the_playwright_chromium_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "render.html").write_text("bundle", encoding="utf-8")
    missing = tmp_path / "ms-playwright" / "chromium"

    class FakePlaywright:
        chromium = SimpleNamespace(executable_path=str(missing))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr("playwright.sync_api.sync_playwright", FakePlaywright)
    with pytest.raises(browser.RenderSetupError, match="Chromium browser is not installed"):
        browser.require_render_setup(dist)

    missing.parent.mkdir()
    missing.write_text("", encoding="utf-8")
    browser.require_render_setup(dist)


def _raise_render_setup() -> None:
    raise browser.RenderSetupError("The render bundle is missing. Run `uv run usdcraft setup`.")


def test_provider_run_fails_before_authoring_without_render_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    executed: list[object] = []
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", _raise_render_setup)
    monkeypatch.setattr(
        "usdcraft.generation.provider_command.execute_provider_generation",
        lambda **kwargs: executed.append(kwargs),
    )
    repo = _checkout(tmp_path / "repo")

    assert usdcraft_cli.main(["generation", "a lamp", "--repo-root", str(repo)]) == 1

    assert executed == []
    assert (
        "Provider Generation failed: RenderSetupError: The render bundle is missing. "
        "Run `uv run usdcraft setup`." in capsys.readouterr().err
    )


def test_batch_run_fails_before_any_row_without_render_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    executed: list[object] = []
    monkeypatch.setattr(usdcraft_cli, "_require_render_setup", _raise_render_setup)
    monkeypatch.setattr(
        "usdcraft.reconstruction.provider_batch.execute_reconstruction_batch",
        lambda **kwargs: executed.append(kwargs),
    )
    root = _checkout(tmp_path / "repo")

    exit_code = usdcraft_cli.main(
        ["reconstruction-batch", str(tmp_path / "jobs.csv"), "--repo-root", str(root)]
    )

    assert exit_code == 1
    assert executed == []
    assert "RenderSetupError" in capsys.readouterr().err
