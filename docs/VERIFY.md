# VERIFY.md

Every unverified fact from the master plan gets a row here once its
owning milestone implements a doctor check for it, per
`docs/research/2026-10-29-rinchen-dev-agent-prompt.md`'s state-file
convention. This file grows across D0-D15; it is not meant to be
complete until the project is.

## D0 checks

| Check ID | Result | Notes |
|---|---|---|
| D-PY | pass | CPython 3.14.x confirmed via `sys.version_info`. |
| D-WHEELS | pass | Every `[project.dependencies]` entry resolves via `importlib.metadata.version` in the synced environment. |
| D-EOLE-PY | pass | `import eole` succeeds (closes the pyonmttok cp314 blocker's real acceptance test). |
| D-EOLE-CLI | pass (with discrepancy noted) | `build_vocab`/`train`/`predict`/`convert` all parse `--help`. The master plan's doctor-check description also named "model average" (citing eole issue #136/PR #137); the installed eole 0.6.4 CLI has no `model` subcommand group at all (`eole --help` lists only `{convert,build_vocab,predict,serve,train,tools}`), and no file matching `*average*` exists anywhere in the installed package. Fallback: Rinchen's own `train/average.py` checkpoint averager (`build_ourselves_specs`, D8/D10 scope) is the path forward when that milestone needs averaging -- not re-attempted here. |
| D-CT2-CONV | pass | `python -m ctranslate2.converters.eole_ct2 --help` succeeds directly. |
| D-CT2-IMPORT | pass | `import ctranslate2` succeeds; confirmed CTranslate2 issue #2079 (`import ctranslate2` eagerly imports torch) does NOT reproduce on the installed ctranslate2 4.8.4 -- in a fresh process, `torch` is absent from `sys.modules` both before and after `import ctranslate2` (4.8.4 lazy-loads `converters`/`specs` specifically to avoid this; the doctor's own real output shows no side-effect note for this check) -- separately, `eole --help` itself does still import torch eagerly, confirmed independently. |
| D-TORCH-CUDA | skip | No GPU on this dev machine; CI also skips (ubuntu-latest has no GPU). |
