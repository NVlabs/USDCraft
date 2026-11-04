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

"""Generate the third-party dependency inventory and collect license texts.

Inputs are the two lockfiles (`uv.lock`, `viewer/web/package-lock.json`) and the packages they
resolve to as installed in `.venv` and `viewer/web/node_modules`. The installed distributions are
the evidence: each package's license expression, copyright line and license file come from its own
metadata and license files, not from a registry summary.

Outputs:

- `THIRD_PARTY_NOTICES.md`: the inventory tables (incorporated material, Python runtime, Viewer
  production dependencies, build/test tools, runtime downloads).
- `third_party/licenses/python/<name>-<version>/`: license files of every Python runtime package.
- `viewer/web/public/licenses/npm/<name>-<version>/`: license files of every Viewer production
  dependency (Vite copies `public/` into `dist/`, so the bundle carries them).

Run from the repository root after `uv sync` and `npm --prefix viewer/web ci`:

    uv run python tools/third_party_notices.py
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from importlib.metadata import Distribution, PathDistribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = next((ROOT / ".venv" / "lib").glob("python3.*")) / "site-packages"
NODE_MODULES = ROOT / "viewer" / "web" / "node_modules"
PY_LICENSE_DIR = ROOT / "third_party" / "licenses" / "python"
NPM_LICENSE_DIR = ROOT / "viewer" / "web" / "public" / "licenses" / "npm"
OUTPUT = ROOT / "THIRD_PARTY_NOTICES.md"

LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)([.\-_].*)?$", re.IGNORECASE)
COPYRIGHT_RE = re.compile(r"copyright\s*(\(c\)|©)?\s*\d{4}", re.IGNORECASE)

# Classifier text -> SPDX, used only when a distribution has no License-Expression and its License
# field is a full text rather than an identifier.
CLASSIFIER_SPDX = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": (
        "LGPL-2.1-or-later"
    ),
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
}

# Raw `License` metadata strings mapped to SPDX expressions. Each mapping was checked against the
# distribution's license file (2- vs 3-clause BSD, Apache version, dual-license terms).
LICENSE_NORMALIZE = {
    "Apache 2.0": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    "Apache Public License 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
}
# Per-distribution verified values where the metadata alone is not enough: (license, copyright,
# note). An empty license or copyright keeps the detected value.
PY_OVERRIDES: dict[str, tuple[str, str, str]] = {
    # `License: BSD` only; the license file is the 3-clause text.
    "numba": ("BSD-2-Clause", "Copyright (c) 2012, Anaconda, Inc.", "2-clause text in LICENSE"),
    "python-fcl": ("BSD-3-Clause", "", "3-clause text in LICENSE"),
    "vtk": ("BSD-3-Clause", "", "3-clause text in LICENSE"),
    "python-dateutil": (
        "Apache-2.0 OR BSD-3-Clause",
        "Copyright 2017- Paul Ganssle and dateutil contributors",
        "dual-licensed; both texts in LICENSE",
    ),
    "aiohttp": (
        "Apache-2.0 AND MIT",
        "Copyright aio-libs contributors",
        "MIT covers vendored llhttp",
    ),
    "aiohappyeyeballs": ("PSF-2.0", "Copyright 2023 J. Nick Koston", ""),
    "usd-core": (
        "LicenseRef-TOST-1.0",
        "Copyright 2016 Pixar",
        "Tomorrow Open Source Technology License 1.0",
    ),
    "cadquery": ("Apache-2.0", "Copyright CadQuery Development Team", ""),
    "cadquery-ocp": ("Apache-2.0", "Copyright 2020 OCP contributors", ""),
    "cadquery-ocp-proxy": ("Apache-2.0", "Copyright Bernhard Walter", ""),
    "cython": ("Apache-2.0", "Copyright the Cython contributors", ""),
    "casadi": (
        "LGPL-3.0-or-later",
        "Copyright 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven",
        "transitive dependency of cadquery; imported dynamically, unmodified",
    ),
    "colorama": (
        "BSD-3-Clause",
        "Copyright Jonathan Hartley 2013",
        "Windows-only (sys_platform == 'win32'); not installed here",
    ),
    "matplotlib": (
        "Matplotlib license (PSF-based)",
        "Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved. "
        "Copyright (c) 2002-2011 John D. Hunter; All Rights Reserved",
        "",
    ),
    "typing-extensions": (
        "PSF-2.0",
        "Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved. "
        "Copyright (c) 1995-2001 Corporation for National Research Initiatives; All Rights Reserved. "
        "Copyright (c) 1991-1995 Stichting Mathematisch Centrum Amsterdam, The Netherlands. "
        "All rights reserved",
        "full notices in the LICENSE",
    ),
    "manifold3d": ("Apache-2.0", "Copyright the Manifold contributors (see AUTHORS)", ""),
    "trame-common": ("Apache-2.0", "Copyright 2022 Kitware Inc.", ""),
}
# npm packages with no license file in the package root get the text from a sibling package of
# the same monorepo, which carries the identical license.
NPM_LICENSE_FALLBACK = {"@radix-ui/": "node_modules/@radix-ui/react-tabs/LICENSE"}
# Upstream license texts for packages in which no license file was found where this script looks
# (the wheel's .dist-info, the npm package root); see that directory's README for sources.
UPSTREAM_LICENSE_DIR = ROOT / "third_party" / "upstream_licenses"
# Verified facts for npm packages whose package.json is incomplete or disagrees with upstream.
NPM_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "highlightjs-vue": (
        "CC0-1.0",
        "Copyright (c) 2019, Sara Lissette Luis Ibáñez",
        "package.json declares CC0-1.0; the upstream repository LICENSE is BSD-3-Clause, and that "
        "text is shipped",
    ),
    "format": (
        "MIT",
        "Copyright 2010 - 2014 Sami Samhuri",
        "license and copyright from the package Readme",
    ),
    "@fontsource/ibm-plex-mono": ("OFL-1.1", "Copyright 2017 IBM Corp. All rights reserved.", ""),
    "lucide-react": (
        "ISC",
        "Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2026 as part of Feather "
        "(MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2026.",
        "",
    ),
    "argparse": (
        "Python-2.0",
        "Copyright (c) 2001-2020 Python Software Foundation; All Rights Reserved",
        "full notices in the LICENSE",
    ),
}


# A license file to copy: its source path and its destination path relative to the package's
# license folder. The relative path keeps upstream subdirectories so that same-named files
# (several `LICENSE.md` in numpy, for example) never collide.
LicenseFile = tuple[Path, str]


def _upstream_license_files(name: str) -> list[LicenseFile]:
    folder = UPSTREAM_LICENSE_DIR / name.replace("/", "__")
    if not folder.is_dir():
        return []
    return [(f, f.name) for f in sorted(folder.iterdir()) if f.is_file()]


# Sections of NOTICE that are incorporated (copied/adapted) material rather than dependencies.
INCORPORATED = [
    (
        "Articraft (portions of the SDK, Viewer, compile feedback, storage, `scaffold.py`)",
        "Apache-2.0",
        "Copyright the Articraft contributors",
        "`sdk/`, `tests/sdk/`, `viewer/api/`, `viewer/web/`, `usdcraft/compile/`, `storage/`, `scaffold.py`",
        "[`NOTICE`](NOTICE); upstream [github.com/articraftresearch/Articraft](https://github.com/articraftresearch/Articraft)",
    ),
    (
        "cq_gears (adapted gear generator)",
        "Apache-2.0",
        "Copyright 2021 meadiode@github",
        "`sdk/_extensions/cadquery/gears.py`",
        "File header; [`NOTICE`](NOTICE); upstream [github.com/meadiode/cq_gears](https://github.com/meadiode/cq_gears)",
    ),
    (
        "CadQuery documentation and examples (adapted)",
        "Apache-2.0",
        "Copyright CadQuery Development Team",
        "`sdk/_docs/cadquery/`, `sdk/_examples/cadquery/`",
        "[`NOTICE`](NOTICE); upstream [github.com/CadQuery/cadquery](https://github.com/CadQuery/cadquery)",
    ),
    (
        "DejaVu Sans and DejaVu Sans Bold",
        "Bitstream-Vera AND Arev terms (DejaVu changes public domain)",
        "Copyright 2003 Bitstream, Inc.; Arev glyphs Copyright 2006 Tavmjong Bah",
        "`sdk/_core/assets/fonts/`",
        "[`sdk/_core/assets/fonts/LICENSE.txt`](sdk/_core/assets/fonts/LICENSE.txt)",
    ),
    (
        "Sora (bundled through `@fontsource/sora`)",
        "OFL-1.1",
        "Copyright 2019 The Sora Project Authors",
        "`viewer/web/` (Viewer bundle)",
        "[`viewer/web/public/licenses/OFL-Sora.txt`](viewer/web/public/licenses/OFL-Sora.txt)",
    ),
    (
        "IBM Plex Mono (bundled through `@fontsource/ibm-plex-mono`)",
        "OFL-1.1",
        "Copyright 2017 IBM Corp. All rights reserved.",
        "`viewer/web/` (Viewer bundle)",
        "[`viewer/web/public/licenses/OFL-IBM-Plex-Mono.txt`](viewer/web/public/licenses/OFL-IBM-Plex-Mono.txt)",
    ),
    (
        "shadcn/ui resizable and scroll-area components (adapted)",
        "MIT",
        "Copyright (c) 2023 shadcn",
        "`viewer/web/src/components/ui/resizable.tsx`, `viewer/web/src/components/ui/scroll-area.tsx`",
        "[`viewer/web/public/licenses/LICENSE-shadcn-ui.txt`](viewer/web/public/licenses/LICENSE-shadcn-ui.txt)",
    ),
]


@dataclass
class Entry:
    name: str
    version: str
    license: str
    copyright: str
    homepage: str
    license_files: list[str] = field(default_factory=list)
    direct: bool = False
    notes: list[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _first_copyright_line(text: str) -> str:
    # A real copyright line names a year or carries a (c)/© mark; license boilerplate that merely
    # mentions "copyright" (owner, notice, holder, license to reproduce) does not.
    strict = re.compile(r"copyright\s*(\(c\)|©)?\s*(\(c\)|©|\d{4})", re.IGNORECASE)
    for line in text.splitlines():
        lowered = line.lower()
        if strict.search(line) and not any(
            bad in lowered for bad in ("copyright owner", "copyright notice", "notice of copyright")
        ):
            return re.sub(r"^[\s#*/\-]+", "", line).strip()
    return ""


def _check_copyright(name: str, line: str) -> None:
    # An extracted line that is cut off or quotes a notice from license prose would be a wrong
    # summary; such packages need an explicit entry in PY_OVERRIDES or NPM_OVERRIDES.
    if len(line) > 160 or line.lower().startswith("i.e."):
        raise SystemExit(f"{name}: copyright line needs an override: {line[:80]!r}...")


def _copy_license_files(files: list[LicenseFile], dest: Path) -> list[str]:
    """Copy each license file to ``dest/<relative path>`` and verify the copy byte for byte.

    Two different files mapping to one destination is an error, never a silent overwrite.
    """

    out = []
    for src, rel in files:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise SystemExit(f"unsafe license path {rel!r} for {src}")
        target = dest / rel_path
        if target.exists():
            if filecmp.cmp(src, target, shallow=False):
                continue
            raise SystemExit(
                f"license file collision: {src} and an earlier file both map to {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        if not filecmp.cmp(src, target, shallow=False):
            raise SystemExit(f"license copy does not match its original: {src} -> {target}")
        out.append(str(target.relative_to(ROOT)))
    return out


# --- Python -----------------------------------------------------------------------------------


def _load_uv_lock() -> tuple[dict[str, dict], set[str], set[str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {_norm(p["name"]): p for p in lock["package"]}
    project = packages["usdcraft"]
    direct_runtime = {_norm(d["name"]) for d in project.get("dependencies", [])}
    dev_groups = project.get("dev-dependencies", {})
    direct_dev = {_norm(d["name"]) for group in dev_groups.values() for d in group}

    def closure(seeds: set[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            name = stack.pop()
            if name in seen or name == "usdcraft":
                continue
            seen.add(name)
            for dep in packages[name].get("dependencies", []):
                stack.append(_norm(dep["name"]))
            for extra_deps in packages[name].get("optional-dependencies", {}).values():
                # Only follow extras the lock actually resolved for this project: uv records them
                # as plain dependencies on the consumer side, so nothing to add here.
                del extra_deps
        return seen

    runtime = closure(direct_runtime)
    dev_only = closure(direct_dev) - runtime
    all_locked = set(packages) - {"usdcraft"}
    unaccounted = all_locked - runtime - dev_only
    if unaccounted:
        print(f"WARNING: locked packages outside both closures: {sorted(unaccounted)}")
    return packages, runtime, dev_only


def _py_distribution(name: str) -> Distribution | None:
    for info in SITE_PACKAGES.glob("*.dist-info"):
        dist = PathDistribution(info)
        if _norm(dist.metadata["Name"]) == name:
            return dist
    return None


def _py_license_expression(dist: Distribution) -> tuple[str, list[str]]:
    md = dist.metadata
    notes: list[str] = []
    expr = md.get("License-Expression")
    if expr:
        return expr.strip(), notes
    classifiers = [c for c in md.get_all("Classifier", []) if c.startswith("License ::")]
    spdx_from_classifiers = sorted(
        {CLASSIFIER_SPDX[c] for c in classifiers if c in CLASSIFIER_SPDX}
    )
    raw = (md.get("License") or "").strip()
    if raw and "\n" not in raw and len(raw) < 80:
        return LICENSE_NORMALIZE.get(raw, raw), notes
    if spdx_from_classifiers:
        return " AND ".join(spdx_from_classifiers), notes
    notes.append("NEEDS MANUAL REVIEW: no expression, no classifier")
    return "UNKNOWN", notes


def _py_license_files(dist: Distribution) -> list[LicenseFile]:
    base = Path(str(dist._path))  # type: ignore[attr-defined]
    files: list[LicenseFile] = []
    declared = dist.metadata.get_all("License-File", [])
    for rel in declared:
        for candidate in (base / "licenses" / rel, base / rel, base / Path(rel).name):
            if candidate.is_file():
                files.append((candidate, rel))
                break
        else:
            raise SystemExit(f"{dist.metadata['Name']}: declared License-File {rel!r} not found")
    if not files:
        for candidate in sorted(base.rglob("*")):
            if candidate.is_file() and LICENSE_FILE_RE.match(candidate.name):
                rel_path = candidate.relative_to(base)
                if rel_path.parts[0] == "licenses":
                    rel_path = Path(*rel_path.parts[1:])
                files.append((candidate, str(rel_path)))
    return files


def _py_homepage(dist: Distribution) -> str:
    md = dist.metadata
    urls = md.get_all("Project-URL", [])
    preferred = ("Homepage", "Home", "Source", "Repository", "Source Code", "GitHub")
    for key in preferred:
        for u in urls:
            label, _, url = u.partition(",")
            if label.strip().lower() == key.lower():
                return url.strip()
    if md.get("Home-page"):
        return md["Home-page"]
    if urls:
        return urls[0].partition(",")[2].strip()
    return f"https://pypi.org/project/{md['Name']}/"


def collect_python() -> tuple[list[Entry], list[Entry]]:
    packages, runtime, dev_only = _load_uv_lock()
    project = packages["usdcraft"]
    direct = {_norm(d["name"]) for d in project.get("dependencies", [])}
    direct |= {_norm(d["name"]) for g in project.get("dev-dependencies", {}).values() for d in g}
    if PY_LICENSE_DIR.exists():
        shutil.rmtree(PY_LICENSE_DIR)

    def build(name: str, copy_licenses: bool) -> Entry:
        locked = packages[name]
        dist = _py_distribution(name)
        if dist is None:
            if name in PY_OVERRIDES:
                lic, cp, note = PY_OVERRIDES[name]
                upstream = _upstream_license_files(name)
                dest = PY_LICENSE_DIR / f"{locked['name']}-{locked['version']}"
                return Entry(
                    name=locked["name"],
                    version=locked["version"],
                    license=lic,
                    copyright=cp,
                    homepage=f"https://pypi.org/project/{locked['name']}/{locked['version']}/",
                    license_files=_copy_license_files(upstream, dest)
                    if copy_licenses and upstream
                    else [],
                    notes=[note] if note else [],
                    direct=name in direct,
                )
            return Entry(
                name=locked["name"],
                version=locked["version"],
                license="UNKNOWN",
                copyright="",
                homepage="",
                notes=["NOT INSTALLED in .venv; run `uv sync --group dev`"],
                direct=name in direct,
            )
        if dist.version != locked["version"]:
            raise SystemExit(f"{name}: installed {dist.version} != locked {locked['version']}")
        license_expr, notes = _py_license_expression(dist)
        files = _py_license_files(dist)
        if not files and _upstream_license_files(name):
            files = _upstream_license_files(name)
            notes.append(
                "license text from upstream; no license file was found in the inspected "
                ".dist-info location"
            )
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f, _ in files)
        copyright_line = _first_copyright_line(text)
        if not copyright_line:
            author = dist.metadata.get("Author") or dist.metadata.get("Author-email") or ""
            copyright_line = f"Copyright {author}".strip() if author else ""
        if name not in PY_OVERRIDES:
            _check_copyright(name, copyright_line)
        if name in PY_OVERRIDES:
            lic, cp, note = PY_OVERRIDES[name]
            license_expr = lic or license_expr
            copyright_line = cp or copyright_line
            upstream_note = [n for n in notes if "from upstream" in n]
            notes = ([note] if note else []) + upstream_note
        copied = (
            _copy_license_files(files, PY_LICENSE_DIR / f"{locked['name']}-{locked['version']}")
            if copy_licenses and files
            else []
        )
        if not files and not any("no license file" in n for n in notes):
            notes.append("no license file found in the inspected .dist-info location")
        return Entry(
            name=locked["name"],
            version=locked["version"],
            license=license_expr,
            copyright=copyright_line,
            homepage=_py_homepage(dist),
            license_files=copied,
            direct=name in direct,
            notes=notes,
        )

    runtime_entries = sorted((build(n, True) for n in runtime), key=lambda e: e.name.lower())
    dev_entries = sorted((build(n, False) for n in dev_only), key=lambda e: e.name.lower())
    return runtime_entries, dev_entries


# --- npm ---------------------------------------------------------------------------------------


def _npm_license_files(pkg_dir: Path) -> list[LicenseFile]:
    return [
        (p, p.name)
        for p in sorted(pkg_dir.iterdir())
        if p.is_file() and LICENSE_FILE_RE.match(p.name)
    ]


def _npm_type_only(pkg_dir: Path) -> bool:
    # Declaration-only packages (`@types/*`, `csstype`) carry `.d.ts` files and no JavaScript, CSS
    # or other assets, so nothing of them can end up in the built bundle.
    if not pkg_dir.is_dir():
        return False
    files = [
        p
        for p in pkg_dir.rglob("*")
        if p.is_file() and "node_modules" not in p.relative_to(pkg_dir).parts
    ]
    declarations = [p for p in files if p.name.endswith(".d.ts")]
    other = [
        p
        for p in files
        if not p.name.endswith(".d.ts")
        and p.suffix not in {".json", ".md", ".flow"}
        and not LICENSE_FILE_RE.match(p.name)
    ]
    return bool(declarations) and not other


def collect_npm() -> tuple[list[Entry], list[Entry]]:
    lock = json.loads((ROOT / "viewer" / "web" / "package-lock.json").read_text(encoding="utf-8"))
    pk = lock["packages"]
    root = pk[""]
    direct = set(root.get("dependencies", {})) | set(root.get("devDependencies", {}))
    if NPM_LICENSE_DIR.exists():
        shutil.rmtree(NPM_LICENSE_DIR)

    runtime: list[Entry] = []
    dev: list[Entry] = []
    for path, meta in pk.items():
        if not path:
            continue
        name = path.rsplit("node_modules/", 1)[1]
        pkg_dir = ROOT / "viewer" / "web" / path
        notes: list[str] = []
        pj = {}
        if (pkg_dir / "package.json").is_file():
            pj = json.loads((pkg_dir / "package.json").read_text(encoding="utf-8"))
        else:
            notes.append("NOT INSTALLED in node_modules; run `npm ci`")
        license_expr = meta.get("license") or pj.get("license") or "UNKNOWN"
        if isinstance(license_expr, dict):
            license_expr = license_expr.get("type", "UNKNOWN")
        if license_expr == "UNKNOWN" and isinstance(pj.get("licenses"), list) and pj["licenses"]:
            license_expr = " OR ".join(
                str(entry.get("type", "UNKNOWN")) for entry in pj["licenses"]
            )
            notes.append("from the legacy `licenses` field of package.json")
        if license_expr == "UNKNOWN":
            notes.append("NEEDS MANUAL REVIEW: no license field")
        files = _npm_license_files(pkg_dir) if pkg_dir.is_dir() else []
        for prefix, fallback in NPM_LICENSE_FALLBACK.items():
            if not files and name.startswith(prefix):
                files = [(ROOT / "viewer" / "web" / fallback, "LICENSE")]
                notes.append(
                    "no license file found in the package root; identical monorepo LICENSE copied from "
                    + fallback.split("node_modules/")[1]
                )
        if not files and _upstream_license_files(name):
            files = _upstream_license_files(name)
            notes.append(
                "license text from upstream; no license file was found in the package root"
            )
        text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f, _ in files)
        copyright_line = _first_copyright_line(text)
        author = pj.get("author")
        if not copyright_line and author:
            copyright_line = author if isinstance(author, str) else author.get("name", "")
        homepage = pj.get("homepage") or ""
        if not homepage:
            repo = pj.get("repository")
            if isinstance(repo, dict):
                homepage = repo.get("url", "")
            elif isinstance(repo, str):
                homepage = repo
        homepage = re.sub(r"^git\+", "", homepage)
        homepage = re.sub(r"\.git$", "", homepage) or f"https://www.npmjs.com/package/{name}"
        if name in NPM_OVERRIDES:
            license_expr, copyright_line, note = NPM_OVERRIDES[name]
            notes = [note] if note else []
        else:
            _check_copyright(name, copyright_line)
        is_dev = bool(meta.get("dev"))
        if not is_dev and _npm_type_only(pkg_dir):
            notes.append("type declarations only; no runtime code")
        copied = []
        if not is_dev and files:
            copied = _copy_license_files(files, NPM_LICENSE_DIR / f"{name}-{meta['version']}")
        if not files and not is_dev:
            notes.append("no license file found in the package root")
        if any("NOT INSTALLED" in n for n in notes) and is_dev:
            # Platform-specific optional binaries that npm skips on this machine; the lockfile
            # still records their license.
            notes = ["optional platform package, not installed here"]
        entry = Entry(
            name=name,
            version=meta["version"],
            license=license_expr,
            copyright=copyright_line,
            homepage=homepage,
            license_files=copied,
            direct=name in direct,
            notes=notes,
        )
        (dev if is_dev else runtime).append(entry)
    key = lambda e: e.name.lower()  # noqa: E731
    return sorted(runtime, key=key), sorted(dev, key=key)


# --- rendering ---------------------------------------------------------------------------------


def _label(path: str) -> str:
    """The license file's path inside its package folder, e.g. `numpy/random/LICENSE.md`."""

    parts = Path(path).parts
    for root in (PY_LICENSE_DIR, NPM_LICENSE_DIR):
        root_parts = root.relative_to(ROOT).parts
        if parts[: len(root_parts)] == root_parts:
            rest = parts[len(root_parts) :]
            # npm scoped packages occupy two path components (`@scope/name-version`).
            skip = 2 if rest and rest[0].startswith("@") else 1
            return "/".join(rest[skip:])
    return Path(path).name


def _md(s: str) -> str:
    return s.replace("|", "\\|")


def _table(entries: list[Entry], *, with_files: bool) -> str:
    head = "| Package | Version | License | Copyright | License text | Upstream |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    if not with_files:
        head = "| Package | Version | License | Copyright | Upstream |"
        sep = "| --- | --- | --- | --- | --- |"
    rows = [head, sep]
    for e in entries:
        name = f"**{_md(e.name)}**" if e.direct else _md(e.name)
        notes = f" ({'; '.join(e.notes)})" if e.notes else ""
        copyright_ = _md(e.copyright) or "see license text"
        if with_files:
            files = (
                ", ".join(f"[{_label(f)}]({f})" for f in e.license_files)
                if e.license_files
                else "see upstream"
            )
            rows.append(
                f"| {name} | {e.version} | {_md(e.license)}{notes} | {copyright_} | {files} | <{e.homepage}> |"
            )
        else:
            rows.append(
                f"| {name} | {e.version} | {_md(e.license)}{notes} | {copyright_} | <{e.homepage}> |"
            )
    return "\n".join(rows)


def render(
    py_rt: list[Entry], py_dev: list[Entry], npm_rt: list[Entry], npm_dev: list[Entry]
) -> str:
    inc_rows = [
        "| Material | License | Copyright | Location in this repository | License text |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m, lic, cp, loc, txt in INCORPORATED:
        inc_rows.append(f"| {m} | {lic} | {cp} | {loc} | {txt} |")
    summary = {
        "python_runtime": len(py_rt),
        "python_dev": len(py_dev),
        "npm_runtime": len(npm_rt),
        "npm_dev": len(npm_dev),
    }
    type_only = sum("type declarations only; no runtime code" in e.notes for e in npm_rt)
    return f"""# Third-party notices

USDCraft is licensed under the Apache License, Version 2.0 ([`LICENSE`](LICENSE)). This document
lists the third-party material it incorporates or depends on, with each component's resolved
version, license, copyright holder and the location of its license text. [`NOTICE`](NOTICE) carries
the attribution statements for incorporated material; this file is the complete inventory.

The inventory is generated by [`tools/third_party_notices.py`](tools/third_party_notices.py) from
the two lockfiles (`uv.lock`, `viewer/web/package-lock.json`) and verified against the installed
distributions' own metadata and license files. Package names in **bold** are direct dependencies
declared in `pyproject.toml` or `viewer/web/package.json`; the others are transitive.

What is distributed: USDCraft is distributed as source code through its Git repository. The
material in Section 1 is part of that source. Section 2 lists the Python packages the program
imports at runtime; they are installed by `uv sync` from PyPI and are not redistributed in this
repository, and their license texts are kept under [`third_party/licenses/python/`](third_party/licenses/python/)
for reference. Section 3 lists the Viewer's production dependencies from its lockfile, the
superset of what the Viewer bundle (`viewer/web/dist/`, built by `usdcraft setup` and included in
the Python wheel) can contain; their license texts are under
[`viewer/web/public/licenses/npm/`](viewer/web/public/licenses/npm/), which Vite copies into the
bundle. Section 4 lists tools used to build, lint or test. Section 5 lists binaries that
`usdcraft setup` downloads at install time.

Summary: {summary["python_runtime"]} Python runtime packages, {summary["npm_runtime"]} Viewer
production dependencies ({type_only} of them type declarations only), {summary["python_dev"]} Python and {summary["npm_dev"]} JavaScript build/test
packages.

## 1. Incorporated material

Code, documentation and fonts copied into or adapted for this repository. Each source file derived
from another project states so in its license header; `NOTICE` explains the one exception
(`scaffold.py`).

{chr(10).join(inc_rows)}

## 2. Python runtime dependencies

Resolved from `uv.lock` as the dependency closure of the `[project] dependencies` in
`pyproject.toml`. Installed by `uv sync`; not redistributed by this repository.

{_table(py_rt, with_files=True)}

Notes:

- **scipy** is licensed under BSD-3-Clause. Its binary wheels additionally bundle compiled-language
  runtime libraries (the GCC runtime `libgcc`/`libgfortran`, licensed GPL-3.0-or-later WITH
  GCC-exception-3.1, and `libquadmath`, licensed LGPL-2.1-or-later), as declared in the wheel's own
  `LICENSES_bundled.txt`. The GCC runtime library exception permits linking them into programs under
  any license; USDCraft does not redistribute the wheels. The same applies to **numpy**'s bundled
  OpenBLAS (BSD-3-Clause) and GCC runtime.
- **usd-core** (OpenUSD) is licensed under the Tomorrow Open Source Technology License 1.0
  (`LicenseRef-TOST-1.0`), a permissive license derived from Apache-2.0.
- **pillow** is licensed under the MIT-CMU (HPND) license.

## 3. Viewer production dependency inventory

Resolved from `viewer/web/package-lock.json` as the packages not marked development-only. This is
the lockfile's production dependency list, including packages that provide type declarations only
(marked in the License column); which code actually ends up in the Viewer bundle is determined by
the build output. Vite builds the bundle into `viewer/web/dist/`, which `usdcraft setup` builds
locally and which the Python wheel includes as a packaging artifact. The license texts of all
packages listed here ship with the bundle under `licenses/npm/`.

{_table(npm_rt, with_files=True)}

## 4. Build, lint and test tools

Python packages from the `dev` dependency group and JavaScript packages marked development-only in
the lockfile. They run on the developer's machine or in CI and are not themselves included in the
source distribution or the wheel. Code a build tool generates into the Viewer bundle (for example
small runtime helpers emitted by the bundler) is governed by that tool's license listed here.

### Python

{_table(py_dev, with_files=False)}

The build backend declared in `[build-system]`, **hatchling** (MIT, Copyright (c) Ofek Lev,
<https://github.com/pypa/hatch>), is resolved by the build frontend in an isolated environment and
is not part of the lockfile.

### JavaScript

{_table(npm_dev, with_files=False)}

## 5. Binaries downloaded at install time

`usdcraft setup` runs `playwright install chromium`, which downloads platform-specific binaries from
Playwright's CDN into the user's Playwright cache. None of these binaries is in this repository, in
the Viewer bundle or in the wheel.

| Component | Version (Playwright 1.61.0) | License | Use in USDCraft | License text |
| --- | --- | --- | --- | --- |
| Chromium (Chrome for Testing / headless shell) | 149.0.7827.55, Playwright build 1228 | BSD-3-Clause (Chromium) plus the licenses of its bundled components, platform-dependent | Headless rendering of candidate assets to PNG still images | <https://chromium.googlesource.com/chromium/src/+/main/LICENSE> and the `LICENSE`/`about:credits` files in the downloaded build |
| FFmpeg (Playwright build) | 7.0.1, Playwright build 1011 | LGPL-2.1-or-later (Playwright's build enables VP8 encode/decode and MJPEG decode; no GPL components) | Not used. Playwright downloads it with the browser for its own video-recording feature; USDCraft never records video and never invokes it. | <https://github.com/microsoft/playwright/blob/main/browser_patches/ffmpeg/> and the `LICENSE` in the downloaded archive |

## Regenerating this file

```bash
uv sync --group dev
npm --prefix viewer/web ci
uv run python tools/third_party_notices.py
```

Review the `NEEDS MANUAL REVIEW` markers in the output, resolve them in `PY_OVERRIDES` or upstream,
and commit the regenerated file together with the copied license texts.
"""


def main() -> int:
    py_rt, py_dev = collect_python()
    npm_rt, npm_dev = collect_npm()
    OUTPUT.write_text(render(py_rt, py_dev, npm_rt, npm_dev), encoding="utf-8")
    problems = [
        f"{e.name} {e.version}: {'; '.join(e.notes)}"
        for e in [*py_rt, *py_dev, *npm_rt, *npm_dev]
        if any("NEEDS MANUAL REVIEW" in n or "NOT INSTALLED" in n for n in e.notes)
    ]
    for p in problems:
        print("REVIEW:", p)
    print(
        f"wrote {OUTPUT.relative_to(ROOT)}: python runtime={len(py_rt)} dev={len(py_dev)}, "
        f"npm runtime={len(npm_rt)} dev={len(npm_dev)}; {len(problems)} to review"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
