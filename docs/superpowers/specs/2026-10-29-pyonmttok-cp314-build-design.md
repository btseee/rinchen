# Build pyonmttok from source for CPython 3.14 — design

Date: 2026-10-29

## Context

`eole` (core Rinchen dependency) cannot be imported on Python 3.14: its
`eole/bin/__init__.py` unconditionally eager-imports modules with unguarded
`import pyonmttok` statements, and `pyonmttok` 1.38.1 ships no sdist and no
cp314 wheel — only cp310/cp311/cp312 wheels for win_amd64/macosx/manylinux
(confirmed via `uv lock` failure and PyPI JSON API). No dependency-resolution
workaround avoids this (see `STATE.md`); the only remaining option is
building `pyonmttok` from source for cp314.

`pyonmttok` is `OpenNMT/Tokenizer`'s Python binding: C++11, CMake, ICU
dependency, single-file pybind11 binding (`bindings/python/pyonmttok/Python.cc`).
Its `pyproject.toml` pins `pybind11>=2.10.4` with no upper bound, so a modern
pybind11 (2.13.6) should bind against cp314 without upstream code changes.
Upstream's own CI (`cibuildwheel`) builds `cp310-* cp311-* cp312-*` only;
cp314 requires running their build scripts manually with no upstream
precedent, so this is building a new target, not reproducing a known-good one.

This machine has: CMake 4.4.4, git, full MSVC toolchain (Visual Studio
Community 2026, `cl.exe` confirmed), Docker 29.9.2 (CLI works; Desktop's
Linux container backend is not currently running — will need to be started
before the Linux task runs). The user has chosen to scope both platforms
(Windows win_amd64 and Linux manylinux x86_64) in this pass, since both are
needed: Windows for this dev machine, Linux for CI and the eventual GPU
training box.

## Scope

In:
- Clone `OpenNMT/Tokenizer` at the git tag matching PyPI's published
  `1.38.1`, with submodules.
- **Windows build:** download prebuilt ICU4C 73.2 (MSVC2019 build, matching
  upstream's own `prepare_build_environment_windows.sh` pin), CMake-build the
  Tokenizer C++ library and DLLs with MSVC, build the Python wheel with this
  machine's CPython 3.14.7 and pybind11 2.13.6 via `python -m build --wheel`.
  Output: `pyonmttok-1.38.1-cp314-cp314-win_amd64.whl`.
- **Linux build:** start Docker Desktop if not running; run the build inside
  `quay.io/pypa/manylinux_2_28_x86_64`. First runtime check: does that image
  provide a cp314 interpreter under `/opt/python/`? If yes, build ICU from
  source statically (per upstream's `prepare_build_environment_linux.sh`),
  CMake-build, build the wheel, `auditwheel repair` it to manylinux
  compliance. If no cp314 interpreter is available in that image, stop and
  report back — do not substitute a different base image or Python build
  without asking, since that changes the manylinux compliance tag's meaning.
- **Verification (both platforms):** install the built wheel into a fresh
  scratch venv (Linux: inside the same container) and run the tokenize/
  detokenize round-trip from pyonmttok's own README:
  ```python
  import pyonmttok
  t = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
  tokens, _ = t.tokenize("Hello World!")
  assert t.detokenize(tokens) == "Hello World!"
  ```
  This proves the binary actually works (imports, runs its C++ core via the
  binding) — not just that the wheel builds.
- Commit both wheels under `vendor/wheels/`.
- Add `[tool.uv.sources]` path-based overrides in `pyproject.toml`, keyed by
  `sys_platform` marker, pointing `pyonmttok` at the matching local wheel.
- Re-run real `uv lock` (no dry-run shortcuts) and confirm it succeeds with
  `pyonmttok` resolved from the local wheel.
- Update `STATE.md`/`DECISIONS.md`: replace the open pyonmttok blocker entry
  with a resolved entry citing the build method, ICU/pybind11 versions used,
  and the `uv lock` success.

Out of scope:
- Actually running `eole` end-to-end (training, conversion, CLI commands
  beyond the import-level check `uv lock` gives us). That's D0's
  `D-EOLE-PY`/`D-EOLE-CLI` doctor checks, not this task.
- Publishing the wheel anywhere public (PyPI, GitHub Releases).
- macOS build (not a target platform for this project; dev machine is
  Windows, CI/GPU box is Linux).
- Any D0 implementation.
- If the manylinux image lacks cp314: building Python itself from source
  inside the container. That's a meaningfully bigger task (building CPython
  with the right manylinux-compliant toolchain) — if the runtime check hits
  this, stop and bring it back as a decision point, don't silently absorb it
  into this plan.

## Testing

No unit tests in the traditional sense — this produces binary artifacts, not
project source code. The acceptance gate is the tokenize/detokenize
round-trip smoke test above, run against each built wheel in a fresh venv,
plus a final real `uv lock` succeeding on this machine. Both are concrete,
scriptable pass/fail checks — written as a script under `scripts/` so the
plan can treat "run script, assert exit 0" as a normal TDD-style step.

## Risks / open questions carried into the plan

- Linux manylinux image's cp314 availability is unverified (Docker wasn't
  running during brainstorming). The plan's first Linux step must check this
  and treat "no cp314 interpreter" as a stop-and-report condition, not a
  silent fallback.
- ICU version pin (73.2) and pybind11 version pin (2.13.6) are taken from
  upstream's current build scripts / this project's own prior cp314
  validation elsewhere; if either fails to build against cp314, the plan
  should surface the actual compiler/linker error rather than guessing at a
  different version.
