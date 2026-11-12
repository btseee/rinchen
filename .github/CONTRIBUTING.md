# Contributing to Rinchen

Thank you for helping. Rinchen is a CPU-first machine translation pipeline with Mongolian at its center. It is written for Python 3.14 and run with uv and just.

This guide says where to ask, how to set up, how to change the code and how to propose it. If something here is unclear, that is a fault of the guide, and an issue about it is welcome. Everyone who takes part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions and problems

- To ask a question, report a problem or suggest something, [open an issue and choose a form](https://github.com/btseee/rinchen/issues/new/choose): bug report, feature request, documentation problem or accessibility problem.
- Never report a security problem in public. Follow the [security policy](SECURITY.md) instead.
- To report behavior that breaks the rules of the community, follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

You need CPython 3.14 (the project accepts no other version), [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

1. Clone the repository and enter it.
2. Install the locked dependencies with `uv sync --frozen --group dev`. This is the only setup step that needs a network.
3. Check the environment with `uv run rinchen doctor`. It prints one line per check, `[OK]`, `[FAIL]` or `[SKIP]`, and exits with a non-zero code when a check fails. A skipped check is fine; for example, the GPU check is skipped on a machine without a GPU.
4. Run the gate with `just gate` (see the next section).
5. Run `just tiny` to see the whole pipeline work. It takes the committed fixtures through every stage on the CPU, with no network and no GPU, and writes a report to `artifacts/tiny/report.md`.

## The gate

`just gate` is what CI runs on every change, and you should run it before you push. It runs these four commands in order and stops at the first that fails:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not network" -q
```

To format a file you changed, run `uv run ruff format path/to/file.py`. Do not reformat files you did not change.

## Tests

The tests live in four directories under `tests`:

- `tests/unit`: small tests of one module or one rule.
- `tests/property`: property-based tests that try many generated inputs.
- `tests/golden`: tests that compare output with a recorded result.
- `tests/integration`: tests that run several parts together, including the real command line.

Helpers are in `tests/support` and recorded inputs are in `tests/fixtures`.

- Write the test before the code. For a fix, show the failing run of the new test in your pull request, then the passing run.
- A numeric threshold is a gate. When a measurement misses its bound, improve the result; the bound is never lowered, and a failing test is never weakened, skipped or deleted.
- A fixture is a committed file. A fixture that `.gitignore` hides passes on your machine and breaks a clean checkout. `tests/unit/test_fixture_files_are_tracked.py` lists files of that kind.
- A test makes no network request. A test that must use a network carries the `network` marker, and `just gate` leaves it out.

## Branches and commits

- Branch from `main`. Name the branch with lowercase words joined by hyphens, such as `d15-tiny-pipeline` or `d6-synthetic-data`.
- Make one logical change per commit.
- Write the subject as a prefix, a colon and a short description in plain words, with no full stop at the end. Milestone work uses `D<N>:`, as in `D15: fix a recording that .gitignore hid from the repository`. Other work uses a type word: `docs:`, `fix:`, `chore:`, `ci:`, `build:`, `feat:` or `test:`, as in `ci: add lint/type/test workflow for Python 3.14`.
- Use the body to say why the change is needed, what failed before it, and how you checked it: the commands and what they printed.

## Pull requests

1. Open the pull request against `main` and fill in the template.
2. CI runs on every pull request, in the workflow `ci.yml`: the job `gate` runs `just gate` and `just doctor`, and the job `tiny` runs `just tiny`. Both should pass.
3. A reviewer checks that:
   - the change is focused and does one thing;
   - the tests came first and cover the behavior, with the failing run shown for a fix;
   - no threshold was lowered and no test was weakened, skipped or deleted;
   - every fixture is committed, and every new file is wired in (imported, registered or referenced);
   - the project rules below are respected;
   - the documentation is updated where behavior changed.
4. Answer review comments with new commits on the same branch.

## Project rules

These rules bind every contribution. They are copied word for word from [CLAUDE.md](../CLAUDE.md), the file the maintainers and their tools follow.

Always:

- Run everything on CPython 3.14.x. Check `python -c "import sys; assert sys.version_info[:2]==(3,14)"` at the start of every script entrypoint.
- Use uv; one workspace; extra envs only for real dependency conflicts, all on 3.14.
- Train every Rinchen model from random init.
- Use position_encoding_type: SinusoidalInterleaved in every Eole model.
- Record every resolved package version and checkpoint hash in manifests/.
- Write tests before code; numeric acceptance thresholds are gates.

Never:

- Use Python 3.12/3.13 or any fallback env, container, or conda env with another Python.
- Fine-tune, LoRA, or continue-train any pretrained model. External models only as evaluation baselines or synthetic-data generators whose outputs become training text.
- Use Inner Mongolian data, varieties, datasets, tools, or FLORES+ khk_Mong.
- Use galig/Latin informal Mongolian.
- Use Relative, rotary, ALiBi, sliding window, or GQA in Eole models meant for CT2.
- Tune on FLORES+ devtest. Use FLORES+ dev for model selection only; report devtest.

The GPU is optional and the CPU is the default. The gate and the tiny pipeline need no GPU, and commands that take a device, such as `rinchen translate` and `rinchen serve`, use the CPU unless you ask for another.

## Accessibility

Documentation, messages and output are for everyone, including people who use assistive technology. Write plain language, keep headings in order, give links text that says where they go, give images a text alternative, and do not carry meaning by color alone.

The [accessibility statement](../ACCESSIBILITY.md) lists what the project does and the limits it knows about, and says how to report a barrier.

## Documentation

The long-form documentation lives on the [project Wiki](https://github.com/btseee/rinchen/wiki). It describes the current state of the code only, and its home page links every page, so start there.

When a change alters behavior, update the matching Wiki page, or name the page in your pull request if you cannot edit it. To report a page that is missing or wrong, use the documentation form when you open an issue.
