# Contributing to USDCraft

Thank you for helping improve USDCraft. This guide covers the local setup, the checks a change must
pass, and how to submit it.

## Set up

Follow the install steps in the [README](README.md) (`uv sync`, then `uv run usdcraft setup`).
Contributions only need the source checkout; no API key is required to run the test suite.

## Checks

Run these from the repository root before opening a pull request. CI runs the same
commands.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv lock --check
uv run pytest -q
```

If you change the Viewer (`viewer/web`):

```bash
npm --prefix viewer/web run lint
npm --prefix viewer/web run typecheck
npm --prefix viewer/web test
npm --prefix viewer/web run build
```

`just lint` and `just test` run most of these in one step if you use [`just`](https://github.com/casey/just).

For a quick local run while iterating, skip the tests marked `slow` (heavy geometry builds, real
compiles and real subprocesses): `uv run pytest -q -m 'not slow'` (or `just test-fast`). CI always
runs the full suite, so run `uv run pytest -q` before opening a pull request.

## Tests

- The test suite is offline. Tests must not call a remote model or require an API key; use the fake
  Provider clients and fake Codex runners that the existing tests use.
- The task text sent to the model is covered by snapshot files in `tests/golden/prompts/`. When you
  change task text on purpose, regenerate them with `USDCRAFT_UPDATE_GOLDENS=1 uv run pytest -q`
  and review the diff before committing.
- Changes to the compiler, the SDK exporters or the Reconstruction evidence code can change every
  generated asset. Describe how you checked that existing outputs stay the same (or why they change).

## Where things live

[`AGENTS.md`](AGENTS.md) maps each component to its documentation and source directory. Each
component README describes its contract; update it in the same change when behavior changes.

## Pull requests

- Keep each pull request focused on one change and describe what it changes and how you tested it.
- Add or update tests for the behavior you change.
- Add a line to the `Unreleased` section at the top of [`CHANGELOG.md`](CHANGELOG.md) (create it if missing) for user-visible changes.
- Do not commit API keys, `.env` files, records under `data/`, or generated build output.

By contributing you agree that your contributions are licensed under the
[Apache License, Version 2.0](LICENSE).

#### Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

Link to DCO: <https://developercertificate.org/>
