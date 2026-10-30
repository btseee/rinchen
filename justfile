# Rinchen development commands. Run `just --list` to see all recipes.

gate:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy src
    uv run pytest -m "not network" -q

test:
    uv run pytest -q

doctor:
    uv run rinchen doctor

tiny:
    echo "just tiny: not implemented until D15 (tiny end-to-end pipeline). See docs/research/2026-10-29-rinchen-dev-agent-prompt.md milestone D15."

accept:
    echo "just accept: not implemented until the full project-wide definition of done is met. See <mission> in docs/research/2026-10-29-rinchen-dev-agent-prompt.md."
