# Phase 0 / P0.1 — Environment probe design

Date: 2026-10-28

## Context

The repo scaffold (see `docs/superpowers/specs/2026-10-28-repo-scaffold-design.md`)
is complete. `STATE.md` lists Phase 0 as not started, with two known blockers
recorded in `DECISIONS.md`: whether `eole` 0.6.4's `pyonmttok>=1.38.1` dependency
has a cp314 wheel, and the deliberate exclusion of vLLM from the main workspace.

The research doc (`docs/research/2026-10-28-python314-stack-and-prompt.md`)
defines Phase 0 as seven tasks (P0.1–P0.7): environment probe, FLORES+ loader,
metrics, CT2 baselines, off-target rate, speed benchmarks, and an Eole→CT2
parity smoke test. These are independent subsystems and each needs its own
spec. This document covers only P0.1, the environment probe — the task every
other Phase 0 task depends on, since none of them can run until we know what
installs on Python 3.14.

This machine: Python 3.14.7, uv 0.13.5 installed, no NVIDIA GPU (`nvidia-smi`
not found) — a CPU-only dev environment. The probe must handle that
gracefully; GPU-dependent checks are not expected to produce data here.

## Scope

In:
- `scripts/probe_env.py` — a script that checks the Python version, dry-resolves
  the pinned dependency stack via `uv lock`, and checks PyPI for per-package
  cp314/cp314t/abi3 wheel availability.
- `manifests/phase0_env_probe.json` — machine-readable output.
- `reports/phase_0.md` — human-readable STOP-gate summary.
- Unit tests for the deterministic parts (wheel-tag classification, report
  formatting) and integration tests for the IO wrappers (PyPI fetch, `uv lock`
  subprocess, `nvidia-smi` subprocess).
- A `[tool.pytest.ini_options]` addition to `pyproject.toml` registering
  `network` and `subprocess` markers, so these tests can be excluded in
  constrained environments without deleting them.

Out of scope (future specs):
- P0.2 (FLORES+ loader), P0.3 (metrics incl. COMET-22/MetricX-24), P0.4 (CT2
  baselines), P0.5 (off-target rate), P0.6 (speed benchmarks), P0.7 (Eole→CT2
  parity smoke test).
- Actually installing any package (`uv sync`) — this is a dry resolve only.
- Resolving the `pyonmttok` cp314 gap or the vLLM exclusion — this task
  detects and reports; follow-up action is a separate decision once the data
  is in.

## Behavior

1. **Python version gate.** If `sys.version_info[:2] != (3, 14)`, the script
   exits non-zero immediately with a clear error before doing anything else.
   This mirrors the hard constraint ("Check ... at the start of every script
   entrypoint"), not just a line in the report.
2. **GPU/CUDA detection.** Run `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`.
   If the command is not found or exits non-zero, record `gpu: null` and
   continue — this is expected on a CPU-only dev machine, not a failure.
3. **Dry dependency resolution.** Run `uv lock` against the repo's real
   `pyproject.toml` (no `--upgrade` flag — resolve against the pins as
   written). Capture exit code and, on failure, uv's stderr output verbatim
   (truncated to a reasonable length for the report) so the specific
   unresolvable package/constraint is visible.
4. **Per-package wheel-tag check.** For every dependency listed in
   `pyproject.toml`'s `[project.dependencies]` and `[dependency-groups] dev`,
   fetch `https://pypi.org/pypi/<name>/<pinned-version-or-latest>/json` and
   classify the `urls` file list into one of: `cp314`, `cp314t`, `abi3`
   (filename contains `-abi3-`), `sdist-only` (no wheel files at all), or
   `version-not-found` (the exact pinned version has no PyPI release — this
   is expected for unpinned packages, where "latest" is queried instead).
   **Special case — `torch`:** it's pinned from a custom index
   (`download.pytorch.org/whl/cu130`), not plain PyPI. The script queries
   PyPI's own `torch` release as a general signal (does the vanilla PyPI
   package ship cp314 at all) AND separately fetches
   `https://download.pytorch.org/whl/cu130/torch/` (the index's simple HTML
   listing) and checks that listing's filenames for a cp314 cu130 wheel. Both
   results are recorded distinctly, labeled `torch_pypi` and
   `torch_cu130_index`. [DESIGN: torch's real install source is the custom
   index, so that's the result that actually matters for this project, but
   the PyPI check is cheap and gives a secondary signal if the custom index
   is unreachable.]
5. **Manifest output.** Write `manifests/phase0_env_probe.json`:
   ```json
   {
     "python_version": "3.14.7",
     "platform": "win32",
     "cpu_count": 8,
     "gpu": null,
     "uv_version": "0.13.5",
     "uv_lock": {"success": true, "stderr_tail": null},
     "packages": {
       "torch": {"requested": "2.14.1", "torch_pypi": ["cp314", "..."], "torch_cu130_index": ["cp314"], "status": "ok"},
       "eole": {"requested": "0.6.4", "wheel_tags": ["none-any"], "status": "ok"},
       "...": "..."
     }
   }
   ```
   (Exact key names are illustrative; the implementation plan fixes them
   precisely.)
6. **Report output.** Write `reports/phase_0.md`: a short STOP-gate summary
   (per the research doc's phase convention — goal, result table, open
   questions) covering Python version check result, `uv lock` pass/fail (and
   why, if it failed), and a per-package wheel-tag table with a flag column
   for anything that looks like a real Phase-0 blocker (sdist-only or
   version-not-found on a package the stack needs a wheel for).

## Testing strategy (TDD)

Deterministic, unit-tested (no network/subprocess, fixture-fed):
- `classify_wheel_tags(filenames: list[str]) -> set[str]` — given PyPI file
  listing filenames, returns which of `{cp314, cp314t, abi3, sdist}` are
  present.
- `build_manifest(...) -> dict` — pure function assembling the JSON structure
  from already-gathered data.
- `render_report(...) -> str` — pure function formatting the Markdown report
  from already-gathered data.

Integration-tested, one test each, marked `@pytest.mark.network` or
`@pytest.mark.subprocess` (registered in `pyproject.toml` so `pytest -m "not network"`
works without deleting the test):
- `fetch_pypi_json(name, version)` — real HTTP call to pypi.org.
- `fetch_torch_cu130_index()` — real HTTP call to download.pytorch.org.
- `run_uv_lock()` — real subprocess call to `uv lock` in a temp copy of the
  repo (so it doesn't mutate the real `uv.lock` if one doesn't exist yet, and
  is safe to re-run).
- `detect_gpu()` — real subprocess call to `nvidia-smi` (or its absence).

One end-to-end smoke test, unmarked (runs by default since this machine has
Python 3.14 and uv): runs `scripts/probe_env.py` for real and asserts
`manifests/phase0_env_probe.json` and `reports/phase_0.md` exist and parse
(valid JSON / non-empty Markdown with the expected section headers).

## Out of scope / explicitly deferred

- Installing any package.
- Deciding what to do about the `pyonmttok` cp314 gap or the vLLM exclusion —
  this task surfaces the facts; the decision is a follow-up once the probe
  has run for real and we know whether the gap is confirmed.
- P0.2 through P0.7.
