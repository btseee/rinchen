# DECISIONS.md

Each entry is tagged `[EVIDENCE: source]` (verified fact) or `[DESIGN: rationale]`
(a choice we made). Full backing detail is in
`docs/research/2026-10-28-python314-stack-and-prompt.md`.

- Pin `eole==0.6.4`, not the GitHub-tagged 0.6.5. [EVIDENCE: 0.6.4 is on PyPI;
  0.6.5 was only tagged on GitHub as of 2026-10-28]. Upgrade only after Phase 0's
  parity smoke test passes on 0.6.5.
- Build our own COMET-22 inference instead of using `unbabel-comet`. [EVIDENCE:
  unbabel-comet 2.2.7 pins `transformers<5`, incompatible with our transformers
  5.21.0 pin].
- Build our own MetricX-24 inference instead of the official `metricx` repo.
  [EVIDENCE: official code pins `transformers==4.30.2`].
- Build Korean `ko-mecab` tokenizer from source (python-mecab-ko sdist) for BLEU;
  fall back to `kiwipiepy`-based `BLEU[ko-kiwi]` labeled non-comparable if the
  build fails. [DESIGN: no cp314 wheel found for python-mecab-ko as of 2026-10-28].
- Use `position_encoding_type: SinusoidalInterleaved` in every Eole model, never
  Relative/rotary/ALiBi. [EVIDENCE: CTranslate2 issue #2134 (open as of
  2026-10-27) shows the Eole→CT2 converter mis-scales relative position keys;
  CT2 also rejects rotary for encoder-decoder].
- Do not include vLLM in the main uv workspace. [EVIDENCE: vllm==0.32.0 requires
  torch==2.13.0 and transformers<5.20.0, which conflicts with the base
  torch==2.14.1/transformers==5.21.0 pins; uv shares one lockfile across all
  dependency groups, so an in-workspace group cannot isolate this]. If a
  generator model is needed (Phase 6), use a separate non-workspace uv project
  with its own lockfile, or the llama.cpp path the research doc calls
  3.14-safe.
- Track manifests/*.parquet in git rather than ignoring it. [DESIGN: the hard
  constraints require recording resolved versions and checkpoint hashes in
  manifests/, and later phases (release.json) depend on those hashes being
  present in a fresh clone].
- **[SUPERSEDED 2026-10-29 — see the "Build pyonmttok from source for cp314"
  entry below, which resolves this directly; the `eole --no-deps` fallback
  described below was never attempted. Kept as-is, unedited, as the
  historical record of the original finding.]** Confirmed: pyonmttok has no
  cp314 wheel as of 2026-10-28. [EVIDENCE: Phase 0
  P0.1 probe, see reports/phase_0_env_probe.md]. `uv lock` fails outright
  because `eole==0.6.4` requires `pyonmttok>=1.38.1`, and `pyonmttok` only
  ships wheels for cp310/cp311/cp312. Fallback to try first: `eole --no-deps`
  plus only the dependencies `import eole.bin.main` and `eole train --help`
  actually need (see STATE.md). Never work around this by running any part
  of the project on cp310/cp311/cp312 — the hard constraints forbid any
  fallback interpreter.
- All 30 direct dependencies resolve to wheel-tag status "ok" on Python
  3.14.7, including `warcio`. [EVIDENCE: Phase 0 P0.1 probe, 2026-10-29, see
  reports/phase_0_env_probe.md]. An earlier run of the probe had a
  wheel-tag-classifier bug that didn't recognize compressed tags like
  `py2.py3` (warcio's real tag), so it was wrongly reported as sdist-only
  with no wheel at all; the classifier is fixed and this bullet replaces
  that earlier, incorrect finding.

## 2026-10-29: Build pyonmttok from source for cp314

[EVIDENCE: live build, `manifests/pyonmttok_build.json`, tag `v1.38.1`
commit `ae4b83b0b2af2172ed306c372835e84b51d5d77e`.] No cp314 wheel or sdist
exists for `pyonmttok` 1.38.1 on PyPI (only cp310/cp311/cp312). Built it
from source for both win_amd64 and manylinux_2_28_x86_64 using ICU 73.2 and
pybind11 3.1.0 (pyonmttok's own `pyproject.toml` requires `pybind11>=2.10.4`
with no upper bound; pybind11 3.1.0 explicitly classifies Python 3.14
support). Wheels committed under `vendor/wheels/`, wired in via
`[tool.uv.sources]` path overrides. Wiring the override in required two
more `pyproject.toml` changes, confirmed by direct experiment rather than
assumed: (1) `pyonmttok==1.38.1` added to `[project.dependencies]` directly
— `[tool.uv.sources]` path overrides only apply to a project's direct or
workspace dependencies, and `pyonmttok` is otherwise only a transitive
dependency of `eole`, so without this the override had no effect and
`uv lock` failed identically to before; (2) `[tool.uv] environments =
["sys_platform == 'win32'", "sys_platform == 'linux'"]` added — `uv lock`
resolves universally across all platforms by default, and this project
targets only Windows (dev) and Linux (CI/GPU) with no macOS target
anywhere in its docs or plan, so without restricting the locked
environments `uv lock` tried to also solve for other platforms and failed
there since no wheel or source exists for them. This unblocks `uv lock`,
which now succeeds for the first time. It does not by itself unblock D0:
the first real `uv sync` against this lockfile surfaced a new, separate,
previously-unreached blocker in `fasttext-wheel==0.9.2` (a transitive
`eole` dependency, distinct from the already-pinned `fasttext-new`
package) — no cp314 wheel upstream, and its C++ extension fails to build
from source under MSVC (`fasttext_pybind.cc(171): error C2065: 'ssize_t':
undeclared identifier`, cascading into pybind11 `C2672` overload-resolution
errors). Same category of problem as this pyonmttok gap (native extension,
no cp314 wheel), but not yet investigated or fixed; see STATE.md's current
next step.

**Note on the direct `pyonmttok` dependency vs. the master plan.** The
master plan (`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`) lists
`pyonmttok` under "Do NOT depend on: ... (we replace them)" — that line is
about Rinchen's own text-processing pipeline, which must not import
`pyonmttok` directly. The `pyonmttok==1.38.1` entry added to
`[project.dependencies]` above is unrelated to that intent: it exists
*only* as a dependency-resolution mechanism, because `uv`'s
`[tool.uv.sources]` path overrides apply only to a project's direct or
workspace dependencies, not to a package that is otherwise pulled in
transitively via `eole`. Without the direct pin, the locally-built wheel
override is silently ignored and `uv lock` fails exactly as before. Adding
this pin does not reverse or weaken the master plan's "we replace them"
design decision — Rinchen's own code still may not `import pyonmttok`
directly; `eole` is the only thing in this workspace allowed to depend on
it.
