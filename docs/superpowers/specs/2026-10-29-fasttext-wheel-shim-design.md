# fasttext-wheel cp314 shim — design

Date: 2026-10-29

## Context

Closing the `pyonmttok` cp314 blocker surfaced a second, separate blocker: the
first-ever real `uv sync` on this project fails building `fasttext-wheel==0.9.2`
(a transitive dependency of `eole==0.6.4`, declared in `eole`'s `setup.py` as
`"fasttext-wheel"` with no version constraint). It has no cp314 wheel on PyPI
(wheels only through cp312) but does have an sdist, so `uv` tries to build it
from source and fails.

Investigation (live builds, not assumption) found three independent,
well-understood bugs in the bundled C++/Python source, all confirmed by
actually reproducing and then fixing each on real hardware:

1. Windows/MSVC: `fasttext_pybind.cc:171,185` use the POSIX type `ssize_t`
   unqualified, which MSVC never defines. Fix verified: a
   `#if defined(_MSC_VER)` guarded `typedef SSIZE_T ssize_t;` (via
   `<BaseTsd.h>`) before first use. Confirmed: a full clean rebuild with only
   this patch produces a working `cp314-cp314-win_amd64` wheel that
   successfully imports, trains, and predicts.
2. Linux/gcc14: `src/args.cc` uses `uint64_t` without `<cstdint>` ever being
   included (older glibc/libstdc++ pulled it in transitively; gcc14's headers
   no longer do). Fix verified: adding `#include <cstdint>` to `src/args.h`
   (so every translation unit including it is covered). Confirmed inside the
   pinned `quay.io/pypa/manylinux_2_28_x86_64@sha256:e0adf18521ad35192df733a6c62d35df7801957d24e0470238aacecfab2188a3`
   image with its `cp314-cp314` interpreter: clean rebuild + `auditwheel
   repair` + real train/predict smoke test all passed.
3. Both platforms: `python/fasttext_module/fasttext/FastText.py` calls
   `np.array(x, copy=False)` in three functions (five call sites) — valid
   under NumPy 1.x, but NumPy 2.x raises `ValueError` there instead of
   silently copying. Fix verified: replacing each with `np.asarray(x)` (the
   exact migration NumPy's own 2.0 guide recommends) resolved the error; the
   patched wheel's `train_supervised`/`predict` round-trip succeeded on both
   platforms.

This proves building the real `fasttext-wheel` from source for cp314 is
possible, mirroring the `pyonmttok` approach exactly (small source patch,
build native wheel per platform, commit to `vendor/wheels/`).

However, a second investigation found a materially better option.
`eole` imports `fasttext` in exactly one place —
`eole/transforms/clean.py`'s `warm_up()`, inside the optional, lazily-loaded
`clean` data-transform — and uses only one call:
`fasttext.load_model(lid.176.ftz)` followed by `.predict(...)`, for language
identification. This project's `pyproject.toml` already pins
`fasttext-new==0.1.0`, a maintained, pure-Rust (PyO3) "drop-in replacement"
for the archived `fasttext` package, with a cp314-native wheel (no C++
toolchain needed at all) and a built-in `install_as_fasttext()` compatibility
shim. Live verification: installed `fasttext-new`, downloaded the real
`lid.176.ftz` eole fetches, and ran the exact call `clean.py` makes — it
correctly identified a Mongolian Cyrillic test sentence as `__label__mn`
(0.806 confidence) and an English sentence as `__label__en` (0.923
confidence). `fasttext-new`'s own documentation claims bit-identical parity
with the original C++ package specifically on the `lid.176` models, which is
exactly the feature `eole` uses.

The user chose the shim approach over building the real C++ package from
source: a first-party pure-Python package, distributed under the exact name
`eole` requires (`fasttext-wheel`), that re-exports `fasttext-new`. This
avoids native compilation entirely (one `py3-none-any` wheel, not two
platform-specific ones), has no per-platform or future-platform maintenance
burden, and does not touch an archived, Python-2-era C++ codebase. The
accepted tradeoff: `fasttext-new` is a small, "alpha"-status third-party
package (not from Meta, not yet widely adopted), verified correct for the
specific feature `eole` needs but less battle-tested overall than the
original.

## Scope

In:
- Create `vendor/fasttext-wheel-shim/` as a minimal `hatchling`-built,
  pure-Python source package (not part of the main `uv` workspace — it's
  built once into a wheel, like `vendor/wheels/` already holds for
  `pyonmttok`, not synced as a live dependency during Rinchen's own
  development).
  - `pyproject.toml`: `name = "fasttext-wheel"`, `version = "0.9.2"`,
    `dependencies = ["fasttext-new==0.1.0"]`, `requires-python = ">=3.10"`.
  - `src/fasttext/__init__.py`: re-exports `load_model`, `train_supervised`,
    `train_unsupervised`, `tokenize`, `FastText` from `fasttext_new`, with a
    module docstring explaining what this replaces and why (the three bugs
    above, and the scope of `eole`'s actual usage).
  - `src/fasttext/FastText.py`: re-exports `fasttext_new.FastText`'s names
    under `fasttext.FastText`, for any code that imports that submodule path
    directly (the real package's own internal layout), even though the one
    confirmed caller (`eole`) doesn't need it.
- Build the wheel (`python -m build --wheel`), commit
  `vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl`.
- Record it in a new `manifests/fasttext_wheel_build.json`. Confirmed:
  `scripts/build_pyonmttok/record_manifest.py`'s `MANIFEST_PATH` is currently
  a hardcoded module constant pointing at `manifests/pyonmttok_build.json`
  (not parameterized), so extend its CLI with an optional `--manifest PATH`
  argument (defaulting to the current hardcoded path, so the two existing
  call sites in `build_windows.sh`/`build_linux_inner.sh` keep working
  unchanged), and pass `--manifest manifests/fasttext_wheel_build.json` from
  this shim's own build step. Add a test for the new flag. Keep the fasttext
  manifest a separate file from `manifests/pyonmttok_build.json` — different
  package, different build story — with `platform = "universal"` as its one
  entry.
- Wire into `pyproject.toml`: add `fasttext-wheel==0.9.2` to
  `[project.dependencies]` (required for the `[tool.uv.sources]` override to
  apply at all — confirmed necessary for `pyonmttok` for the same reason:
  path overrides only apply to direct project dependencies, not arbitrary
  transitive ones) and a `[tool.uv.sources]` entry pointing at the committed
  wheel (no platform marker needed — the wheel is `py3-none-any`).
- Re-run real `uv lock`, then for the first time ever on this project, real
  `uv sync --dev`, then real `uv run pytest -q` and `uv run ruff check`.
  This is the actual closing proof: CI's own `uv sync --dev` step should now
  succeed.
- A smoke test (new, under `tests/`, marked `@pytest.mark.network` per this
  repo's existing marker convention) that installs the shim package plus
  `fasttext-new` in a fresh venv, downloads the real `lid.176.ftz` file
  exactly as `eole/transforms/clean.py` does, and asserts
  `fasttext.load_model(...).predict(...)` returns `("__label__mn",)` for a
  Mongolian Cyrillic test string and `("__label__en",)` for an English one —
  verifying the actual behavior `eole`'s `clean` transform depends on, not
  just that the import succeeds.
- `STATE.md`/`DECISIONS.md` updates: replace the open fasttext-wheel blocker
  entry with a resolved entry explaining the shim rationale, explicitly
  naming the three confirmed upstream bugs (as evidence that building from
  source was a real, validated alternative, not a skipped option), and
  naming `fasttext-new`'s alpha status as an accepted tradeoff.
- `CLAUDE.md` repo layout: add `vendor/fasttext-wheel-shim/` alongside the
  existing `vendor/wheels/` entry.

Out of scope:
- Building the real `facebookresearch/fastText` C++ package from source for
  cp314. This was fully investigated and three fixes were validated live
  (see Context), but the user chose the shim instead. The validated patches
  are documented here as evidence but not applied to any committed code.
- Actually exercising `eole`'s `clean` transform in a real Rinchen pipeline
  run (that's D0+/data-pipeline work, not this fix).
- Vetting or auditing `fasttext-new` beyond the functional verification
  already done (no supply-chain/security audit of its Rust internals).
- Any D0 implementation.

## Testing

The new smoke test (described above) is the primary acceptance gate — it
exercises the actual feature `eole` depends on, with real input and a real
downloaded model, not a mock. The broader acceptance gate for the whole fix
is the real `uv sync --dev` succeeding for the first time on this project,
followed by the existing `uv run pytest -q` / `uv run ruff check` finally
running against the project's real locked environment instead of the
ephemeral `--no-project` workaround used throughout the `pyonmttok` fix.

## Risks / open questions carried into the plan

- `fasttext-new`'s PyPI releases could in principle stop shipping a cp314
  wheel in some future version; since this project pins `==0.1.0` exactly,
  that's not an immediate risk, but worth a one-line note in `DECISIONS.md`
  for whoever revisits this later.
- The shim's own `pyproject.toml` lives outside the main `uv` workspace (by
  design, matching how `pyonmttok`'s source checkout is never part of the
  workspace either) — the plan must make sure building it doesn't
  accidentally get swept into `uv`'s workspace member discovery.
