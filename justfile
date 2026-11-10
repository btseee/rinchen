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

# The CPU container of the tiny export (D11). Documentation: nothing runs these two recipes in CI
# and the image was never built (UNMEASURED). The model directory is RINCHEN_MODEL_DIR (default
# models/tiny).

# Builds and starts the service in its container (never run here).
serve-tiny:
    docker compose -f docker/compose.serve.yaml up --build

# Validates the compose file only (read only, builds and starts nothing, needs no daemon).
docker-check:
    docker compose -f docker/compose.serve.yaml config -q
