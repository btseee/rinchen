# Runbook: 00 — Environment

## Dev machine setup (CPU)

1. Install Python 3.14.x and `uv` (https://docs.astral.sh/uv/).
2. Install `just` (https://github.com/casey/just) — e.g. `winget install
   Casey.Just` on Windows, or your platform's package manager.
3. Clone the repo, then: `uv sync --dev`
4. Verify: `uv run rinchen doctor` — every check should print `[OK]` or
   `[SKIP]` (GPU checks skip on a CPU-only machine). Any `[FAIL]` means
   something in the pinned stack isn't installed correctly; see
   `docs/VERIFY.md` for what each check id verifies and the known
   `D-EOLE-CLI` discrepancy (eole 0.6.4 has no `model average`
   subcommand, contrary to the master plan's original description).

## GPU host setup

1. Confirm a CUDA 13-compatible driver is installed:
   `nvidia-smi` should report a driver version supporting CUDA 13.x.
2. Follow the CPU setup steps above (same `uv sync --dev`; `torch` is
   pinned to a CUDA 13.0 wheel via `[tool.uv.sources]`'s
   `pytorch-cu130` index).
3. Verify: `uv run rinchen doctor --gpu` is NOT yet a real flag at D0
   (the `D-TORCH-CUDA` check always runs as part of the normal `rinchen
   doctor` invocation and reports `pass` with the detected GPU name, or
   `skip` if none is found) — a `--gpu`-specific flag that fails loudly
   instead of skipping is left for a later milestone if it proves
   useful; for now, read the `D-TORCH-CUDA` line in the normal output.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on `ubuntu-24.04`
(pinned, not `ubuntu-latest`, to avoid the runner silently migrating to
Ubuntu 26 mid-project)
with CPython 3.14, syncs with `uv sync --frozen --group dev`, then runs
`just gate` (ruff format check, ruff check, mypy, fast tests) and `just
tiny` (currently a stub that always succeeds — real from D15 onward).
GPU checks are not run in CI (no GPU runner); `D-TORCH-CUDA` reports
`skip` there, which is expected and does not fail the gate.
