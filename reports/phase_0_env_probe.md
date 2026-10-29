# Phase 0 Report: Environment Probe (P0.1)

**Goal:** determine whether the pinned Python 3.14 dependency stack resolves cleanly, and record per-package wheel availability, before P0.2-P0.7 build on top of it.

**Python version:** 3.14.7 (OK)
**Platform:** win32
**CPU count:** 12
**GPU:** none detected
**uv version:** uv 0.13.5 (46776a95d 2026-10-21 x86_64-pc-windows-msvc)
**Probed at:** 2026-10-29T08:37:16.599211+00:00
**uv lock:** FAILED

## Blockers

- `uv lock` fails to resolve the pinned stack on this Python version. Raw resolver output:

```
Using CPython 3.14.7 interpreter at: ~\AppData\Local\Programs\Python\Python314\python.exe
error: No solution found when resolving dependencies for split (markers: sys_platform == 'win32')
  cause: Because pyonmttok==1.38.1 has no wheels with a matching Python version tag (e.g., `cp314`) and only pyonmttok<=1.38.1 is available, we can conclude that pyonmttok>=1.38.1 cannot be used.
         And because eole==0.6.4 depends on pyonmttok>=1.38.1 and your project depends on eole==0.6.4, we can conclude that your project's requirements are unsatisfiable.

hint: Wheels are available for `pyonmttok` (v1.38.1) with the following Python ABI tags: `cp310`, `cp311`, `cp312`

```

## Package wheel-tag check

| package | requested | checked version | wheel tags | status |
|---|---|---|---|---|
| beautifulsoup4 | latest | 4.15.0 | py3-none, sdist | ok |
| ctranslate2 | 4.8.4 | 4.8.4 | cp314, cp314t | ok |
| datasets | latest | 5.1.0 | py3-none, sdist | ok |
| datasketch | 2.0.0 | 2.0.0 | py3-none, sdist | ok |
| duckdb | latest | 2.0.0 | abi3, sdist | ok |
| eole | 0.6.4 | 0.6.4 | py3-none, sdist | ok |
| fastapi | latest | 0.162.0 | py3-none, sdist | ok |
| fasttext-new | 0.1.0 | 0.1.0 | abi3, cp314t, sdist | ok |
| httpx | latest | 0.28.1 | py3-none, sdist | ok |
| huggingface_hub | latest | 2.3.0 | py3-none, sdist | ok |
| kiwipiepy | 0.24.0 | 0.24.0 | abi3, cp314t, sdist | ok |
| mypy | latest | 2.4.1 | cp314, cp314t, py3-none, sdist | ok |
| numpy | latest | 2.5.4 | cp314, cp314t, sdist | ok |
| pandas | latest | 3.1.0 | cp314, cp314t, sdist | ok |
| polars | latest | 2.1.0 | py3-none, sdist | ok |
| pyarrow | latest | 26.0.0 | cp314, cp314t, sdist | ok |
| pydantic | latest | 2.14.1 | py3-none, sdist | ok |
| pytest | latest | 9.2.0 | py3-none, sdist | ok |
| ruff | latest | 0.17.3 | py3-none, sdist | ok |
| sacrebleu | 2.6.0 | 2.6.0 | py3-none, sdist | ok |
| scipy | latest | 1.18.1 | cp314, cp314t, sdist | ok |
| selectolax | latest | 1.0.2 | cp314, cp314t, sdist | ok |
| sentence-transformers | latest | 6.2.0 | py3-none, sdist | ok |
| sentencepiece | 0.2.1 | 0.2.1 | cp314, cp314t, sdist | ok |
| tokenizers | 0.23.3 | 0.23.3 | abi3, cp314t, sdist | ok |
| torch | 2.14.1 | 2.14.1 | pypi=cp314, cp314t; cu130=cp314, cp314t | ok |
| trafilatura | latest | 2.3.1 | py3-none, sdist | ok |
| transformers | 5.21.0 | 5.21.0 | py3-none, sdist | ok |
| uvicorn | latest | 0.54.0 | py3-none, sdist | ok |
| warcio | latest | 1.8.2 | py3-none, sdist | ok |

## Open questions

- Does `eole --no-deps` (installing only the dependencies that `import eole.bin.main` and `eole train --help` require) avoid the `pyonmttok` cp314 gap, or does eole's own code import pyonmttok eagerly? See DECISIONS.md.
- Should this probe be re-run once `pyonmttok` or `eole` ship a cp314 wheel, to confirm the gap has closed?
