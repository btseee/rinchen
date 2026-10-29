# fasttext-wheel cp314 shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the `fasttext-wheel` cp314 blocker (eole's hard, unversioned
`fasttext-wheel` dependency has no cp314 wheel) by authoring a pure-Python
compatibility package, distributed under the exact name `fasttext-wheel`,
that re-exports the already-pinned `fasttext-new==0.1.0`. Get a real `uv
sync --dev` to succeed for the first time on this project.

**Architecture:** A tiny `hatchling`-built package at
`vendor/fasttext-wheel-shim/` whose `fasttext/__init__.py` re-exports
`fasttext_new`'s public API. Build it into one `py3-none-any` wheel, commit
it to `vendor/wheels/`, wire it into `pyproject.toml` the same two-part way
`pyonmttok` needed (`[project.dependencies]` + `[tool.uv.sources]`), then
verify with a real pytest test that loads the actual `lid.176.ftz` model
`eole`'s `clean` transform uses and checks real predictions.

**Tech Stack:** `hatchling`, `fasttext-new` (PyO3/Rust), `uv`, Python 3.14.7.

## Global Constraints

- Every Python invocation runs on CPython 3.14.x
  (`assert sys.version_info[:2]==(3,14)`), per `CLAUDE.md`. No 3.12/3.13
  fallback environment.
- The shim's distribution name is exactly `fasttext-wheel`, version `0.9.2`
  (matches the real package's PyPI version, for clarity in logs/manifests —
  it is not a claim of behavioral identity). Its one dependency is
  `fasttext-new==0.1.0`, matching the version already pinned in the main
  `pyproject.toml`.
- `vendor/fasttext-wheel-shim/` must NOT become a `uv` workspace member of
  the main project — it is built once into a wheel and consumed like any
  other vendored binary artifact, the same relationship `pyonmttok`'s
  external source checkout has to this repo.
- Building the real `facebookresearch/fastText` C++ package from source is
  explicitly out of scope for this plan (investigated and validated as an
  alternative during brainstorming, but the user chose the shim).
- `[tool.uv.sources]` path overrides only apply to direct project
  dependencies, not arbitrary transitive ones — confirmed the hard way while
  closing the `pyonmttok` blocker. `fasttext-wheel` must be added to
  `[project.dependencies]` directly for the override to take effect.

---

### Task 1: Add a `--manifest` option to `record_manifest.py`

**Files:**
- Modify: `scripts/build_pyonmttok/record_manifest.py`
- Modify: `tests/test_record_manifest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `main(platform, wheel_path_str, extra=None, manifest_path=None)`
  — when `manifest_path` is `None` (the default, used by every existing
  call site), behavior is unchanged (writes to the hardcoded
  `MANIFEST_PATH`, i.e. `manifests/pyonmttok_build.json`). When given a
  `Path`, writes there instead. Also produces a new `_parse_cli(argv)`
  helper returning `(manifest_path: Path | None, platform: str, wheel: str,
  extra: dict[str, str])`, consumed only by this file's own `__main__`
  block. Task 2 invokes the script as
  `record_manifest.py --manifest manifests/fasttext_wheel_build.json <platform> <wheel> [key=value ...]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_record_manifest.py` (after the existing tests, same file,
same imports already present — no new imports needed):

```python
def test_main_writes_to_custom_manifest_path(tmp_path, monkeypatch):
    default_manifest_path = tmp_path / "pyonmttok_build.json"
    _write_manifest(default_manifest_path)
    monkeypatch.setattr(record_manifest, "MANIFEST_PATH", default_manifest_path)
    monkeypatch.setattr(record_manifest, "REPO_ROOT", tmp_path)

    custom_manifest_path = tmp_path / "fasttext_wheel_build.json"
    _write_manifest(custom_manifest_path)

    wheel = tmp_path / "fasttext_wheel-0.9.2-py3-none-any.whl"
    wheel.write_bytes(b"fake shim wheel")

    assert (
        record_manifest.main(
            "universal", str(wheel), manifest_path=custom_manifest_path
        )
        == 0
    )

    # The custom manifest got the record...
    custom_manifest = json.loads(custom_manifest_path.read_text(encoding="utf-8"))
    assert len(custom_manifest["builds"]) == 1
    assert custom_manifest["builds"][0]["platform"] == "universal"

    # ...and the default manifest was left untouched.
    default_manifest = json.loads(default_manifest_path.read_text(encoding="utf-8"))
    assert default_manifest["builds"] == []


def test_parse_cli_extracts_manifest_flag(tmp_path):
    manifest_path, platform, wheel, extra = record_manifest._parse_cli(
        ["--manifest", str(tmp_path / "custom.json"), "universal", "x.whl", "k=v"]
    )
    assert manifest_path == tmp_path / "custom.json"
    assert platform == "universal"
    assert wheel == "x.whl"
    assert extra == {"k": "v"}


def test_parse_cli_without_manifest_flag_defaults_to_none():
    manifest_path, platform, wheel, extra = record_manifest._parse_cli(
        ["win_amd64", "x.whl"]
    )
    assert manifest_path is None
    assert platform == "win_amd64"
    assert wheel == "x.whl"
    assert extra == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with pytest pytest tests/test_record_manifest.py -v`
Expected: the three new tests FAIL — `test_main_writes_to_custom_manifest_path`
with a `TypeError` (`main()` doesn't accept `manifest_path` yet), the other
two with `AttributeError: module 'record_manifest' has no attribute '_parse_cli'`.

- [ ] **Step 3: Implement the `manifest_path` parameter and `_parse_cli`**

In `scripts/build_pyonmttok/record_manifest.py`, change the `main` signature
and its two uses of `MANIFEST_PATH`:

```python
def main(
    platform: str,
    wheel_path_str: str,
    extra: dict[str, str] | None = None,
    manifest_path: Path | None = None,
) -> int:
    target_manifest = manifest_path if manifest_path is not None else MANIFEST_PATH
    wheel_path = Path(wheel_path_str).resolve()
    if not wheel_path.is_file():
        print(f"wheel not found: {wheel_path}", file=sys.stderr)
        return 1

    if extra:
        for key in extra:
            if not key:
                print("invalid extra field: empty key", file=sys.stderr)
                return 1
            if key in RESERVED_FIELDS:
                print(
                    f"invalid extra field: {key!r} is a reserved core field "
                    "and cannot be overridden",
                    file=sys.stderr,
                )
                return 1

    manifest = json.loads(target_manifest.read_text(encoding="utf-8"))
    try:
        rel_wheel = wheel_path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel_wheel = wheel_path.name
    record = {
        "platform": platform,
        "wheel": rel_wheel,
        "sha256": sha256_of(wheel_path),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        record.update(extra)
    builds = [b for b in manifest.get("builds", []) if b["platform"] != platform]
    builds.append(record)
    manifest["builds"] = builds
    target_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {platform} build: {record['wheel']} ({record['sha256'][:12]}...)")
    return 0
```

Add `_parse_cli` and update the `__main__` block (replace the existing
`if __name__ == "__main__":` block entirely):

```python
def _parse_cli(argv: list[str]) -> tuple[Path | None, str, str, dict[str, str]]:
    manifest_path: Path | None = None
    if argv[:1] == ["--manifest"]:
        if len(argv) < 2:
            print("--manifest requires a path argument", file=sys.stderr)
            raise SystemExit(2)
        manifest_path = Path(argv[1])
        argv = argv[2:]
    if len(argv) < 2:
        print(
            "usage: record_manifest.py [--manifest PATH] <platform> <wheel_path> [key=value ...]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return manifest_path, argv[0], argv[1], _parse_extra(argv[2:])


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    _manifest_path, _platform, _wheel, _extra = _parse_cli(sys.argv[1:])
    raise SystemExit(main(_platform, _wheel, _extra, _manifest_path))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with pytest pytest tests/test_record_manifest.py -v`
Expected: all tests pass (9 existing + 3 new = 12 passed).

- [ ] **Step 5: Run ruff**

Run: `uv run --no-project --with ruff ruff check scripts/build_pyonmttok tests/test_record_manifest.py`
Expected: all checks passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_pyonmttok/record_manifest.py tests/test_record_manifest.py
git commit -m "build: add --manifest option to record_manifest.py"
```

---

### Task 2: Build the fasttext-wheel shim package

**Files:**
- Create: `vendor/fasttext-wheel-shim/pyproject.toml`
- Create: `vendor/fasttext-wheel-shim/src/fasttext/__init__.py`
- Create: `vendor/fasttext-wheel-shim/src/fasttext/FastText.py`
- Create: `tests/test_fasttext_wheel_shim.py`
- Create: `manifests/fasttext_wheel_build.json`
- Modify: `.gitignore` if needed (check first — `vendor/wheels/*.whl` is
  already un-ignored from the `pyonmttok` work; this task only adds a new
  file under that same already-un-ignored path, so no change should be
  needed, but verify with `git check-ignore -v` before assuming)

**Interfaces:**
- Consumes: `scripts/build_pyonmttok/record_manifest.py`'s `--manifest` flag
  from Task 1.
- Produces: `vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl`.

- [ ] **Step 1: Write the shim package**

Create `vendor/fasttext-wheel-shim/pyproject.toml`:

```toml
[project]
name = "fasttext-wheel"
version = "0.9.2"
description = "Compatibility shim satisfying eole's fasttext-wheel dependency via fasttext-new"
requires-python = ">=3.10"
dependencies = ["fasttext-new==0.1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fasttext"]
```

Create `vendor/fasttext-wheel-shim/src/fasttext/__init__.py`:

```python
"""Compatibility shim for eole's hard dependency on the PyPI distribution
"fasttext-wheel". Rinchen does not build the original facebookresearch/fastText
C++ package from source: it has no cp314 wheel, and building it from source hits
several pre-existing C++ portability bugs (MSVC-only `ssize_t`, gcc14 missing
`<cstdint>`, and a NumPy-1-only `copy=False` call in its Python wrapper) --
see docs/superpowers/specs/2026-10-29-fasttext-wheel-shim-design.md.

eole uses `fasttext` in exactly one place (`eole/transforms/clean.py`'s lazy,
optional `clean` transform), only for `fasttext.load_model(...)` to run
language identification via the `lid.176` model. `fasttext-new` (already a
direct Rinchen dependency) is a maintained, cp314-native, NumPy-2-compatible
Rust/PyO3 reimplementation verified bit-identical to the original on the
lid.176 models. This package re-exports it under the `fasttext` name so
`import fasttext` resolves correctly wherever eole (or anything else) expects
the real package -- no native compilation involved.
"""

from fasttext_new import (
    FastText,
    load_model,
    tokenize,
    train_supervised,
    train_unsupervised,
)

__version__ = "0.9.2"

__all__ = [
    "FastText",
    "__version__",
    "load_model",
    "tokenize",
    "train_supervised",
    "train_unsupervised",
]
```

Create `vendor/fasttext-wheel-shim/src/fasttext/FastText.py`:

```python
"""Re-export of fasttext_new's FastText submodule under the fasttext.FastText
name, matching the real fasttext-wheel package's internal module layout, for
any code that imports this submodule path directly rather than the package's
top level."""

from fasttext_new.FastText import (
    _FastText,
    cbow,
    load_model,
    skipgram,
    supervised,
    tokenize,
    train_supervised,
    train_unsupervised,
    unsupervised_default,
)

__all__ = [
    "_FastText",
    "cbow",
    "load_model",
    "skipgram",
    "supervised",
    "tokenize",
    "train_supervised",
    "train_unsupervised",
    "unsupervised_default",
]
```

- [ ] **Step 2: Build the wheel**

Run:
```bash
python -c "import sys; assert sys.version_info[:2] == (3, 14), sys.version"
rm -rf build/fasttext_wheel_shim
mkdir -p build/fasttext_wheel_shim
cd vendor/fasttext-wheel-shim
python -m pip install --quiet build
python -m build --wheel --outdir ../../build/fasttext_wheel_shim/dist
cd ../..
ls build/fasttext_wheel_shim/dist
```

Expected: ends with `fasttext_wheel-0.9.2-py3-none-any.whl` listed. If the
build fails, read the actual error -- this is a pure-Python package with one
dependency declared by name only; a failure here means something is wrong
with the `pyproject.toml` itself (e.g. a typo), not a toolchain issue.

- [ ] **Step 3: Write the verification test**

Create `tests/test_fasttext_wheel_shim.py`:

```python
"""Verifies the fasttext-wheel shim (vendor/fasttext-wheel-shim/) actually
performs the one real-world task eole's `clean` transform depends on:
loading the lid.176 language-identification model and predicting with it.
Requires network access to download the real model file, exactly as
eole/transforms/clean.py does at runtime.
"""
import urllib.request

import pytest


@pytest.mark.network
def test_shim_loads_lid176_and_identifies_languages(tmp_path):
    fasttext = pytest.importorskip("fasttext")

    model_path = tmp_path / "lid.176.ftz"
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        model_path,
    )

    model = fasttext.load_model(str(model_path))

    mongolian_labels, _ = model.predict(
        "Сайн байна уу, энэ бол монгол хэл дээрх өгүүлбэр юм"
    )
    assert mongolian_labels == ("__label__mn",)

    english_labels, _ = model.predict(
        "This is an English sentence for language identification."
    )
    assert english_labels == ("__label__en",)
```

- [ ] **Step 4: Run the test against the built wheel (ephemeral env, no project sync needed yet)**

Run:
```bash
uv run --no-project \
  --with ./vendor/fasttext-wheel-shim \
  --with pytest \
  pytest tests/test_fasttext_wheel_shim.py -v -m network
```
Expected: `1 passed`. (This uses `uv run --with <local-path>` to build and
install the shim package on the fly, pulling in `fasttext-new==0.1.0`
transitively per the shim's own `pyproject.toml` — no need for the main
project's `uv sync` to work yet, which is Task 3's job.)

If this fails, check the actual error before assuming the shim is wrong —
it could be a network failure fetching the real model file, which is a
legitimate test environment issue, not a shim defect.

- [ ] **Step 5: Commit the wheel and record it in the new manifest**

```bash
mkdir -p vendor/wheels
cp build/fasttext_wheel_shim/dist/fasttext_wheel-0.9.2-py3-none-any.whl vendor/wheels/
echo '{"source": "vendor/fasttext-wheel-shim/", "builds": []}' > manifests/fasttext_wheel_build.json
python -m json.tool manifests/fasttext_wheel_build.json
python scripts/build_pyonmttok/record_manifest.py \
  --manifest manifests/fasttext_wheel_build.json \
  universal vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl
cat manifests/fasttext_wheel_build.json
git check-ignore -v vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl || echo "not ignored (expected)"
git add vendor/fasttext-wheel-shim/ tests/test_fasttext_wheel_shim.py \
  vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl manifests/fasttext_wheel_build.json
git commit -m "build: add fasttext-wheel shim package backed by fasttext-new"
```

(If `git check-ignore -v` reports the wheel IS ignored, stop and fix
`.gitignore` before committing -- do not force-add with `-f`.)

---

### Task 3: Wire the shim into the project and close the blocker

**Files:**
- Modify: `pyproject.toml`
- Modify: `STATE.md`
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl` from Task 2.

- [ ] **Step 1: Add the direct dependency and source override**

In `pyproject.toml`, add `"fasttext-wheel==0.9.2",` to the `dependencies`
list in `[project]` (anywhere in the list is fine; place it near the other
`fasttext-new==0.1.0` entry for readability).

Add to `[tool.uv.sources]` (alongside the existing `torch` and `pyonmttok`
entries):

```toml
fasttext-wheel = { path = "vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl" }
```

No `marker` is needed — this wheel is `py3-none-any`, valid for every
platform already covered by `[tool.uv.environments]`.

- [ ] **Step 2: Run the real `uv lock` and verify it succeeds, and that the shim is not a workspace member**

Run: `uv lock`
Expected: exits 0.

Then confirm `vendor/fasttext-wheel-shim` was NOT pulled in as a `uv`
workspace member (per Global Constraints): check that `uv.lock` does not
list it as an editable/workspace source -- e.g.
`grep -n "fasttext-wheel-shim" uv.lock` should show it only referenced via
the `vendor/wheels/*.whl` path, not as a separate workspace package entry.
If it DOES appear as a workspace member, stop and report -- do not add
workaround exclusions without understanding why `uv` picked it up.

- [ ] **Step 3: Run the real `uv sync --dev` for the first time on this project**

Run: `uv sync --dev`
Expected: exits 0. This is the actual close of both blockers (`pyonmttok`
and `fasttext-wheel`) at once -- the project's real `.venv` has never
successfully synced before this.

If it fails, read the exact error. A new, different package failing to
install here would be a third, separate blocker -- report it rather than
guessing at a fix, the same discipline applied throughout this project's
`pyonmttok` and `fasttext-wheel` investigations.

- [ ] **Step 4: Run the real test suite and lint for the first time**

Run: `uv run pytest -q`
Expected: all tests pass, including `tests/test_fasttext_wheel_shim.py`'s
network test (now running against the real synced environment, not the
`--no-project --with` workaround from Task 2).

Run: `uv run ruff check`
Expected: no findings, or only pre-existing findings unrelated to this
plan's changes (check `git blame` on any flagged line before assuming it's
pre-existing).

- [ ] **Step 5: Update `STATE.md`**

Replace the fasttext-wheel blocker bullet (the one starting "Open: `uv
sync`... now fails building `fasttext-wheel==0.9.2`" or equivalent, check
the file's current exact wording) with:

```markdown
- Resolved: `fasttext-wheel==0.9.2` has no cp314 wheel and fails to build
  from source (confirmed: MSVC lacks `ssize_t`, gcc14 needs `<cstdint>`, and
  its Python wrapper uses NumPy-1-only `copy=False`). Rather than building
  the archived upstream C++ package from source, a first-party pure-Python
  shim (`vendor/fasttext-wheel-shim/`) re-exports the already-pinned
  `fasttext-new==0.1.0` under the `fasttext` name. `eole` uses `fasttext`
  only in its optional `clean` transform's language-identification step
  (`eole/transforms/clean.py`), which this shim was verified against
  directly: it correctly loads the real `lid.176.ftz` model and identifies
  both Mongolian and English test strings. [EVIDENCE:
  `docs/superpowers/plans/2026-10-29-fasttext-wheel-shim.md`,
  `tests/test_fasttext_wheel_shim.py`, 2026-10-29.] `uv sync --dev` succeeds
  for the first time on this project as of this change; `uv run pytest -q`
  and `uv run ruff check` both run against the real synced environment.
  Accepted tradeoff: `fasttext-new` is a small, alpha-status third-party
  package, not from Meta and not yet widely adopted.
```

Update the top banner's "Current next step" line to point at D0 (both
known Phase 0 blockers are now resolved).

- [ ] **Step 6: Update `DECISIONS.md`**

Add a new dated entry:

```markdown
## 2026-10-29: fasttext-wheel cp314 shim via fasttext-new

[EVIDENCE: live verification against the real `lid.176.ftz` model,
`tests/test_fasttext_wheel_shim.py`, 2026-10-29.] `fasttext-wheel==0.9.2`
(a transitive, unversioned `eole` dependency) has no cp314 wheel. Building
it from source was investigated and validated as feasible (three confirmed
C++/Python portability bugs, all fixed and build-tested live on both
Windows and Linux during brainstorming), but not applied. [DESIGN: chose
instead to ship a first-party pure-Python shim package
(`vendor/fasttext-wheel-shim/`, distributed as `fasttext-wheel` to satisfy
`eole`'s dependency name) that re-exports the already-pinned, maintained,
cp314-native `fasttext-new==0.1.0`. `eole` only uses `fasttext` for
`load_model`/`predict` in its optional language-ID `clean` transform, which
`fasttext-new` is independently documented and here independently verified
to handle correctly for the `lid.176` models eole actually downloads. This
avoids native compilation entirely and needs no per-platform maintenance,
at the accepted cost of depending on a smaller, alpha-status third-party
package instead of the archived original.]
```

- [ ] **Step 7: Update `CLAUDE.md`'s repo layout**

Add `vendor/fasttext-wheel-shim/` on its own line near the existing
`vendor/wheels/` entry.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock STATE.md DECISIONS.md CLAUDE.md
git commit -m "build: wire fasttext-wheel shim into pyproject, close fasttext-wheel blocker"
```
