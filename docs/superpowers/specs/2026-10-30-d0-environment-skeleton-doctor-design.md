# D0 — Environment, skeleton, doctor — design

Date: 2026-10-30

## Context

The master plan (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`,
`<milestones>` section) defines D0 as: pyproject restructure, justfile,
CI, pre-commit, the config loader, logging, seed utility, the manifest
model/store/stage runner, and `rinchen doctor`. Its own STOP gate is
narrower than the document's overall project-wide "definition of done"
(8 items under `<mission>`, checked by a later `just accept`): D0 only
needs `just gate` green locally and in CI, doctor CPU checks passing, and
`docs/VERIFY.md` populated. The broader definition-of-done items (85%
coverage, `mypy --strict` across the whole `src/rinchen/` tree, every
module existing, `just tiny` actually working) mature across D1-D15 as
those modules get built — D0 does not attempt to satisfy them early.

Both prerequisite blockers (pyonmttok cp314, fasttext-wheel cp314) are
resolved; `uv sync --dev` succeeds and CI's exact command sequence passes
locally. This plan is the first real content under `src/rinchen/`.

Decisions already made this session, carried into this plan without
re-litigation:
- **Full pyproject.toml rewrite now**, matching the master doc's toolchain
  section exactly (not incremental).
- **vLLM stays dropped** entirely (not even as an extra) — already
  recorded in `DECISIONS.md`; this plan documents it as a deviation from
  the master doc's toolchain section rather than re-deciding it.
- **One plan, not four sub-plans**, despite D0's size.

New facts gathered for this design (verified live, not assumed):
- Every new package this restructure adds has a cp314-safe release: either
  a pure-Python `py3`/`py2.py3` wheel (`typer`, `structlog`,
  `pydantic-settings`, `rich`, `jinja2`, `python-multipart`,
  `prometheus-client`, `pytest-xdist`, `pytest-cov`, `pytest-timeout`,
  `pre-commit`) or an explicit `cp314` wheel (`regex`, `hypothesis`), or a
  platform-tagged-but-CPython-ABI-independent wheel (`ty`, a Rust binary
  shipped as `py3-none-<platform>`, covering `win_amd64` and
  `manylinux_2_17_x86_64`). No new native build is needed for D0.
- The existing `src/rinchen/` tree (from the superseded Phase 0-14 plan)
  has empty stub packages: `align/`, `convert/`, `data/`, `eval/`,
  `filter/`, `metrics/`, `serve/`, `synth/`, `text/`, plus `cli/`. Of
  these, `align/`, `convert/`, and `data/` have no equivalent in the new
  `<repo_layout>` at all (`align` becomes `mine/align.py` in D4,
  `convert` becomes `script_conv/` in D5, `data` becomes `acquire/` +
  `corpus/` in D1) — this plan removes those three as orphaned. The
  others (`text/`, `filter/`, `metrics/`, `synth/`, `eval/`, `serve/`) do
  have a future home in the new layout and are left as untouched empty
  placeholder packages for their respective later milestones. `cli/` is
  the one directory D0 itself fills in.

## Scope

In:

**1. `pyproject.toml` restructure.** Keep every already-verified, already
solved core-ML pin exactly as-is (`torch`, `eole`, `pyonmttok`,
`ctranslate2`, `transformers`, `tokenizers`, `sentencepiece`, `sacrebleu`,
`kiwipiepy`, `fasttext-new`, `fasttext-wheel`, `datasketch`, and the
existing `[tool.uv.sources]`/`[tool.uv.environments]`/`[tool.uv.index]`
entries). Add, each verified above: `pydantic-settings`, `typer`, `rich`,
`structlog`, `jinja2`, `python-multipart`, `prometheus-client`, `regex`,
`PyYAML` to `[project.dependencies]`; `hypothesis`, `pytest-xdist`,
`pytest-cov`, `pytest-timeout`, `pre-commit`, `ty` to the `dev`
dependency group. Run the real `uv lock` / `uv sync --dev` and record the
resolved versions (the master doc's own instruction: "Resolve latest at
D0 ... and record"). vLLM is NOT added (documented deviation, already
decided).

**2. Tooling files.**
- `justfile` with recipes `gate` (ruff format --check + ruff check +
  mypy + pytest -m "not network"), `test` (full pytest), `doctor`
  (`uv run rinchen doctor`), `tiny` and `accept` as explicit stub
  recipes that print "not implemented until D15"/"not implemented until
  the project-wide definition of done is met" and exit 0 — real CI
  structure now, honest about what's not built yet.
- `.github/workflows/ci.yml`: `ubuntu-latest`, `astral-sh/setup-uv`
  with Python 3.14, `uv sync --frozen --group dev`, `just gate`,
  `just tiny` (the stub — exercises the CI step exists, not that the
  pipeline works), cache uv, timeout, artifact upload for
  `coverage.xml` (coverage itself is not yet gated at 85% — that's the
  project-wide definition of done, not D0's).
- `.pre-commit-config.yaml`: ruff format + ruff check hooks.
- Update `CLAUDE.md`'s repo layout block and "Current phase" section;
  update `STATE.md`'s banner.

**3. Old `src/rinchen/` stub cleanup.** Remove `align/`, `convert/`,
`data/` (orphaned, no home in the new layout, see Context). Leave
`text/`, `filter/`, `metrics/`, `synth/`, `eval/`, `serve/` untouched.

**4. Core library modules** (only what D0's own "Implement:" line names):
- `src/rinchen/__init__.py` (version string), `__main__.py` (invokes the
  typer CLI), `py.typed` (marker file, empty).
- `src/rinchen/config/{settings.py,schema.py,loader.py}`: a pydantic
  `RinchenConfig` schema (minimal at D0 — just enough fields to prove the
  merge mechanism: e.g. `profile: str`, `seed: int`, `paths: dict[str,
  str]`), `load(profile: str, *overlays: Path, cli_overrides:
  dict[str, str] | None = None) -> RinchenConfig` that merges base →
  profile → overlay YAML files → CLI `--set a.b=c` dotted overrides,
  reads `RINCHEN_*` env vars via `pydantic-settings`, and can dump the
  resolved config + its sha256. `configs/profiles/tiny.yaml` gets created
  as a minimal real file (not just referenced) so the loader has
  something real to merge against; `full.yaml` is a stub with a comment
  that later milestones populate it.
- `src/rinchen/utils/seed.py`: `set_all(seed: int) -> None` (stdlib
  `random`, `numpy.random`, `torch.manual_seed` — guarded imports since
  not every later module needs torch).
- `src/rinchen/utils/logging.py`: `get(name: str) -> structlog.BoundLogger`
  wired to JSON output on stderr, one-time configuration.
- `src/rinchen/manifest/{model.py,store.py,stage.py}`: `Manifest`
  pydantic model matching the `<contracts>` section's schema exactly
  (`schema`, `stage`, `created`, `code_version`, `config_sha256`, `seed`,
  `inputs`, `outputs`, `metrics`, `tool_versions`); `store.py` reads/
  writes `<stage_dir>/manifest.json`; `stage.py`'s stage runner skips a
  stage iff inputs+config sha match an existing manifest and `--force`
  is absent (the actual mechanism `test_manifest_skip` checks).
- `src/rinchen/errors.py`: the domain exceptions named in
  `<coding_standards>` (`ConfigError`, `DataError`, `UpstreamError`,
  `ParityError`, `DoctorCheckFailed`), each a plain subclass of a common
  `RinchenError` — D0 only needs enough of these for `config/` and
  `manifest/` to raise correctly, and `DoctorCheckFailed` for the doctor
  command; the others are declared now (cheap, one file) since
  `<coding_standards>` specifies the full set, but most stay unused
  until their owning module exists.

**5. CLI skeleton + `rinchen doctor`.**
- `src/rinchen/cli/main.py`: a `typer.Typer()` app named `rinchen`,
  registering one sub-`Typer` per module listed in the repo layout's
  `cli/` directory: `data`, `spm`, `filter`, `mine`, `convert_script`,
  `synth`, `train`, `export`, `eval`, `serve`, `annotate`, `doctor`,
  `tiny`. Every sub-app except `doctor` is a stub `typer.Typer()` with no
  commands registered yet (an empty sub-app still responds to `--help`
  with exit 0, which is what `test_cli_help` checks — confirmed this is
  typer's actual behavior, not assumed, before committing to this
  design).
- `src/rinchen/cli/doctor.py`: the real implementation. Six checks:
  - `D-PY`: `sys.version_info[:2] == (3, 14)`.
  - `D-WHEELS`: import every package in `[project.dependencies]`,
    print name + version (reuses the parsing approach from
    `scripts/probe_env.py`'s `parse_pinned_packages`, adapted to import
    rather than query PyPI — this check runs against the real installed
    environment, not the network).
  - `D-EOLE-PY`: `python -c "import eole"` succeeds (this is the exact
    thing the pyonmttok/fasttext-wheel work spent two whole plans
    proving works — this check is the permanent regression guard for
    that work).
  - `D-EOLE-CLI`: `eole --help`, and `build_vocab`/`train`/`predict`/
    `convert`/`model average` each parse `--help` without error.
  - `D-CT2-CONV`: `python -m ctranslate2.converters.eole_ct2 --help`
    succeeds, or (fallback) the `eole` console-script entry point is
    found via `importlib.metadata.entry_points`.
  - `D-CT2-IMPORT`: import `ctranslate2`, note (informational, not a
    failure) whether `torch` got imported as a side effect (tracking
    CTranslate2 issue #2079 the master doc flags).
  - `D-TORCH-CUDA`: GPU check — in CI this reports "skipped (no GPU)"
    rather than failing; on this dev machine it reports whatever
    `torch.cuda.is_available()` says.
  Each check returns a structured result (id, status: pass/fail/skip,
  message); `rinchen doctor` prints a summary table and exits non-zero
  if any non-skipped check failed. Every check's result also gets a row
  in `docs/VERIFY.md` (per the master doc's state-file convention) —
  D0's STOP gate requires every row present with a result, not just the
  doctor command existing.

**6. Tests first** (written before the implementation, per this
project's existing TDD discipline): `test_config_merge` (3-layer merge,
CLI override wins), `test_manifest_skip` (identical inputs+config →
skipped; changed config → rerun), `test_doctor_py` (fails when
`sys.version_info` is faked to `(3, 13)` via monkeypatch — not by
actually running on 3.13), `test_cli_help` (every one of the 13
subcommands exits 0 on `--help`, parametrized).

**7. Runbook.** `docs/runbooks/00_environment.md`: GPU host setup, CUDA
13 driver check, `rinchen doctor --gpu` usage — this is operator
documentation, not code; written last, after the doctor command exists
to document accurately.

Out of scope (explicitly deferred to their own later milestones, per the
master doc's own phase boundaries — not omissions):
- Every other `src/rinchen/` package named in `<repo_layout>` but not in
  D0's own "Implement:" line (`lang/`, `split/`, `tok/`, `acquire/`,
  `corpus/`, `mine/`, `spm/`, `script_conv/`, `mbr/`, `train/`,
  `export/`, `infer/`, `annotate/`, `_vendor/`, and the real contents of
  `text/`, `filter/`, `metrics/`, `synth/`, `eval/`, `serve/`).
- Real logic behind any CLI subcommand other than `doctor`.
- `just tiny` actually running a pipeline (D15), `just accept` actually
  checking the full definition-of-done (continuous, matures over time).
- 85% coverage / `mypy --strict` enforcement across the whole tree (not
  meaningful yet — most of the tree doesn't exist).
- Re-litigating the vLLM decision or the pyonmttok/fasttext-wheel direct-
  dependency exceptions — both already decided and documented.

## Testing

TDD as already practiced this session: each of the four test functions
above is written and confirmed failing before its implementation exists,
then made to pass. `D-PY`/`D-WHEELS`/`D-EOLE-PY`/`D-EOLE-CLI`/
`D-CT2-CONV`/`D-CT2-IMPORT` are also exercised for real (not just unit-
tested in isolation) by actually running `rinchen doctor` against the
real synced environment — this is the acceptance gate for the whole
plan, mirroring how `uv lock`/`uv sync --dev` were the real acceptance
gates for the pyonmttok and fasttext-wheel plans.

## Risks / open questions carried into the plan

- Resolved during design: verified live that a truly empty
  `typer.Typer()` sub-app, registered via `app.add_typer(sub,
  name="data")` with zero commands, responds to `rinchen data --help`
  with exit code 0 (`typer.testing.CliRunner`, confirmed directly). The
  plan can commit to `test_cli_help`'s exact assertions without further
  verification.
- The minimal `RinchenConfig` schema and `configs/profiles/tiny.yaml`
  content are deliberately small at D0 (just enough to prove the merge
  mechanism); later milestones will add real fields as they need
  profile-driven config, not this plan.
