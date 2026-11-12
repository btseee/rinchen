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
    uv run --frozen --no-sync rinchen tiny run --profile tiny

accept:
    echo "just accept: not implemented until the full project-wide definition of done is met. See <mission> in https://github.com/btseee/rinchen/wiki/Research-Master-Plan-D0-D15."

# Checks a clone of the Wiki (https://github.com/btseee/rinchen.wiki.git), where the documents live:
# the content rules, internal links, characters, code fences, page names and the Verification-Log
# rows that configs/maintain/maintain.yaml names. Example: just wiki-check ../rinchen.wiki
wiki-check wiki_dir:
    uv run --frozen --no-sync python scripts/check_wiki.py "{{wiki_dir}}" --verify-ids-from configs/maintain/maintain.yaml

# The CPU container of the tiny export (D11). Documentation: nothing runs these two recipes in CI
# and the image was never built (UNMEASURED). The model directory is RINCHEN_MODEL_DIR (default
# models/tiny).

# Builds and starts the service in its container (never run here).
serve-tiny:
    docker compose -f docker/compose.serve.yaml up --build

# Validates the compose file only (read only, builds and starts nothing, needs no daemon).
docker-check:
    docker compose -f docker/compose.serve.yaml config -q
