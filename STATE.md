# STATE.md

Phase 0 (environment and benchmark) — not started.

Repo scaffold is complete: uv workspace, directory tree, CLAUDE.md, STATE.md,
DECISIONS.md, .gitignore, and CI stub. No package has been installed or
verified on Python 3.14 yet.

Known blockers for Phase 0 to resolve (see DECISIONS.md for the vLLM entry):
- eole 0.6.4 depends on pyonmttok>=1.38.1, which (as of this scaffold) has no
  published wheel for cp314; this may block `uv sync` until checked.
- A generator model (vLLM, or an alternative) for Phase 6 synthetic data has
  been deliberately left out of the main workspace; Phase 0 or Phase 6 must
  decide between a separate non-workspace uv project and the llama.cpp path.
