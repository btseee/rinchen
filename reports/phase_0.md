# Phase 0 Report: Environment Probe

**Python version:** 3.14.7 (OK)
**Platform:** win32
**CPU count:** 12
**GPU:** none detected
**uv version:** uv 0.13.5 (46776a95d 2026-10-21 x86_64-pc-windows-msvc)
**uv lock:** FAILED

```
Using CPython 3.14.7 interpreter at: C:\Users\BBadral\AppData\Local\Programs\Python\Python314\python.exe
error: No solution found when resolving dependencies for split (markers: sys_platform == 'win32')
  cause: Because pyonmttok==1.38.1 has no wheels with a matching Python version tag (e.g., `cp314`) and only pyonmttok<=1.38.1 is available, we can conclude that pyonmttok>=1.38.1 cannot be used.
         And because eole==0.6.4 depends on pyonmttok>=1.38.1 and your project depends on eole==0.6.4, we can conclude that your project's requirements are unsatisfiable.

hint: Wheels are available for `pyonmttok` (v1.38.1) with the following Python ABI tags: `cp310`, `cp311`, `cp312`

```

## Package wheel-tag check

| package | requested | wheel tags | status |
|---|---|---|---|
| beautifulsoup4 | latest | py3-none, sdist | ok |
| ctranslate2 | 4.8.4 | cp314, cp314t | ok |
| datasets | latest | py3-none, sdist | ok |
| datasketch | 2.0.0 | py3-none, sdist | ok |
| duckdb | latest | abi3, sdist | ok |
| eole | 0.6.4 | py3-none, sdist | ok |
| fastapi | latest | py3-none, sdist | ok |
| fasttext-new | 0.1.0 | abi3, cp314t, sdist | ok |
| httpx | latest | py3-none, sdist | ok |
| huggingface_hub | latest | py3-none, sdist | ok |
| kiwipiepy | 0.24.0 | abi3, cp314t, sdist | ok |
| mypy | latest | cp314, cp314t, py3-none, sdist | ok |
| numpy | latest | cp314, cp314t, sdist | ok |
| pandas | latest | cp314, cp314t, sdist | ok |
| polars | latest | py3-none, sdist | ok |
| pyarrow | latest | cp314, cp314t, sdist | ok |
| pydantic | latest | py3-none, sdist | ok |
| pytest | latest | py3-none, sdist | ok |
| ruff | latest | py3-none, sdist | ok |
| sacrebleu | 2.6.0 | py3-none, sdist | ok |
| scipy | latest | cp314, cp314t, sdist | ok |
| selectolax | latest | cp314, cp314t, sdist | ok |
| sentence-transformers | latest | py3-none, sdist | ok |
| sentencepiece | 0.2.1 | cp314, cp314t, sdist | ok |
| tokenizers | 0.23.3 | abi3, cp314t, sdist | ok |
| torch | 2.14.1 | pypi=['cp314', 'cp314t'], cu130=['cp314', 'cp314t'] | ok |
| trafilatura | latest | py3-none, sdist | ok |
| transformers | 5.21.0 | py3-none, sdist | ok |
| uvicorn | latest | py3-none, sdist | ok |
| warcio | latest | sdist | sdist-only ⚠️ |
