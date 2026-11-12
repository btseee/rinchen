"""The broken copies of the files of D15's Task 8, one rule each (spec P16): the table `BREAKS` of named
changes, each a text replaced in one file of a set (`Files`) and the sentence `findings` must say
about the copy. The change, the sentences and the anchors are in `tests/support/ci_workflow_edits.py`.
"""

from __future__ import annotations

from typing import Final

from tests.support.ci_workflow import SUMMARY_IF, TINY_RECIPE
from tests.support.ci_workflow_edits import (
    BELOW,
    CI,
    COVERAGE_STEP,
    DOCTOR_STEP,
    GATE_HEAD,
    GATE_KEYS,
    GATE_STEPS_SENTENCE,
    IGNORE_PAIR,
    JOBS,
    JSON_PATH,
    MARKDOWN_PATH,
    NAMES_STUB,
    NIGHTLY_BELOW,
    NO_TIMEOUT,
    NOT_45,
    NOT_IGNORED,
    NOT_ONCE,
    NOT_PINNED,
    NOT_THE_RECIPE,
    ONE_RECIPE,
    ONE_UPLOAD,
    ONE_VERSION,
    PERMISSIONS,
    PIPELINE_STEP,
    RECIPE_LINE,
    RENAMED,
    RUN_TINY,
    STUB_STEP,
    SUMMARY,
    SUMMARY_LINE,
    TINY_HEAD,
    TINY_KEYS,
    TINY_PERMISSIONS,
    TINY_STEPS_SENTENCE,
    TINY_TIMEOUT,
    TOP_KEYS,
    TRIGGER,
    UPLOAD,
    UPLOAD_HEAD,
    UPLOAD_NAME_LINE,
    Case,
    after_pipeline,
    ci,
    ignore,
    nightly,
    numbered,
    recipe,
    release,
)

TRIGGER_CASES: Final = [
    (
        "a pull_request_target trigger",
        ci("  pull_request:\n", "  pull_request_target:\n", TRIGGER),
    ),
    ("a push of every branch", ci("[main]", "[main, dev]", TRIGGER)),
    (
        "a schedule",
        ci(
            "on:\n  push:",
            'on:\n  schedule:\n    - cron: "0 0 * * *"\n  push:',
            TRIGGER,
        ),
    ),
    (
        "a manual dispatch",
        ci("on:\n  push:", "on:\n  workflow_dispatch:\n  push:", TRIGGER),
    ),
    (
        "an environment at the top",
        ci("jobs:\n  gate:", "env:\n  X: y\n\njobs:\n  gate:", TOP_KEYS),
    ),
    (
        "write-all permissions at the top",
        ci("jobs:\n  gate:", "permissions: write-all\n\njobs:\n  gate:", TOP_KEYS),
    ),
    (
        "a file that is not YAML",
        ci(
            "name: CI\n",
            "name: CI\nfoo: [unclosed\n",
            CI + "the file is not a YAML mapping",
        ),
    ),
    ("the job tiny renamed", ci("  tiny:\n", "  pipeline:\n", JOBS)),
    (
        "a third job",
        ci(
            "  tiny:\n    runs-on",
            "  other:\n    runs-on: ubuntu-24.04\n    steps:\n      - run: just doctor\n\n  tiny:\n    runs-on",
            JOBS,
        ),
    ),
]

GATE_CASES: Final = [
    (
        "the stub put back in the gate: it runs the pipeline",
        ci(
            COVERAGE_STEP,
            STUB_STEP + COVERAGE_STEP,
            CI + "job gate runs the tiny pipeline",
        ),
    ),
    (
        "the stub put back in the gate: it names the stub",
        ci(COVERAGE_STEP, STUB_STEP + COVERAGE_STEP, "ci.yml" + NAMES_STUB),
    ),
    (
        "the pipeline in the gate under another name",
        ci(
            COVERAGE_STEP,
            "      - name: Pipeline\n        run: just tiny\n\n" + COVERAGE_STEP,
            CI + "job gate runs the tiny pipeline",
        ),
    ),
    ("the gate without the doctor", ci(DOCTOR_STEP, "", GATE_STEPS_SENTENCE)),
    (
        "the gate without its coverage upload",
        ci(
            "      - name: Upload coverage\n        if: always()\n        uses: actions/upload-artifact@v7\n"
            "        with:\n          name: coverage\n          path: coverage.xml\n          if-no-files-found: ignore\n",
            "",
            GATE_STEPS_SENTENCE,
        ),
    ),
    (
        "the gate on another runner",
        ci(
            GATE_HEAD,
            GATE_HEAD.replace("24.04", "latest"),
            CI + "job gate is not on ubuntu-24.04",
        ),
    ),
    (
        "the gate with a timeout",
        ci(GATE_HEAD, GATE_HEAD + "    timeout-minutes: 10\n", GATE_KEYS),
    ),
    (
        "the gate waiting for the pipeline",
        ci(GATE_HEAD, GATE_HEAD + "    needs: tiny\n", GATE_KEYS),
    ),
    (
        "the gate on another Python",
        ci('python-version: "3.14"', 'python-version: "3.13"', GATE_STEPS_SENTENCE, 0),
    ),
    (
        "a gate that syncs without --frozen",
        ci(
            "run: uv sync --frozen --group dev",
            "run: uv sync --group dev",
            GATE_STEPS_SENTENCE,
            0,
        ),
    ),
]

TINY_JOB_CASES: Final = [
    (
        "the job tiny on another runner",
        ci(
            TINY_HEAD,
            TINY_HEAD.replace("24.04", "latest"),
            CI + "job tiny is not on ubuntu-24.04",
        ),
    ),
    (
        "a self-hosted runner",
        ci(
            TINY_HEAD,
            TINY_HEAD.replace("ubuntu-24.04", "[self-hosted]"),
            CI + "it mentions self-hosted",
        ),
    ),
    ("no timeout", ci(TINY_TIMEOUT, "", NO_TIMEOUT)),
    ("a timeout of 20 minutes", ci(TINY_TIMEOUT, "    timeout-minutes: 20\n", BELOW)),
    ("a timeout of 44 minutes", ci(TINY_TIMEOUT, "    timeout-minutes: 44\n", NOT_45)),
    ("a timeout of 90 minutes", ci(TINY_TIMEOUT, "    timeout-minutes: 90\n", NOT_45)),
    (
        "a timeout that is not a number",
        ci(TINY_TIMEOUT, "    timeout-minutes: true\n", NO_TIMEOUT),
    ),
    (
        "a timeout in quotes",
        ci(TINY_TIMEOUT, '    timeout-minutes: "45"\n', NO_TIMEOUT),
    ),
    (
        "write permission",
        ci(TINY_PERMISSIONS, "    permissions:\n      contents: write\n", PERMISSIONS),
    ),
    (
        "write-all permissions",
        ci(TINY_PERMISSIONS, "    permissions: write-all\n", PERMISSIONS),
    ),
    ("no permissions", ci(TINY_PERMISSIONS, "", PERMISSIONS)),
    (
        "a token for the identity provider",
        ci(TINY_PERMISSIONS, TINY_PERMISSIONS + "      id-token: write\n", PERMISSIONS),
    ),
    (
        "a job that needs the gate",
        ci(
            TINY_HEAD,
            TINY_HEAD + "    needs: gate\n",
            CI + "job tiny waits for another job",
        ),
    ),
    (
        "a job that ignores a failure",
        ci(
            TINY_HEAD,
            TINY_HEAD + "    continue-on-error: true\n",
            CI + "job tiny ignores a failure",
        ),
    ),
    (
        "an environment on the job",
        ci(
            TINY_HEAD,
            TINY_HEAD + "    env:\n      X: y\n",
            CI + "job tiny sets an environment",
        ),
    ),
    (
        "a container for the job",
        ci(TINY_HEAD, TINY_HEAD + "    container: ubuntu:24.04\n", TINY_KEYS),
    ),
    (
        "a condition on the job",
        ci(TINY_HEAD, TINY_HEAD + "    if: github.event_name == 'push'\n", TINY_KEYS),
    ),
    (
        "a matrix on the job",
        ci(
            TINY_HEAD,
            TINY_HEAD + "    strategy:\n      matrix:\n        x: [1]\n",
            TINY_KEYS,
        ),
    ),
]

STEP_CASES: Final = [
    ("the pipeline step removed: it is not run", ci(PIPELINE_STEP, "", NOT_ONCE)),
    (
        "the pipeline step removed: the steps differ",
        ci(PIPELINE_STEP, "", TINY_STEPS_SENTENCE),
    ),
    (
        "the pipeline run twice",
        ci(PIPELINE_STEP, PIPELINE_STEP + PIPELINE_STEP, NOT_ONCE),
    ),
    (
        "the pipeline run with an option",
        ci(RUN_TINY, "        run: just tiny --dry-run\n", NOT_ONCE),
    ),
    (
        "the pipeline run without just",
        ci(RUN_TINY, f"        run: {TINY_RECIPE}\n", NOT_ONCE),
    ),
    (
        "the pipeline step renamed",
        ci(
            PIPELINE_STEP,
            PIPELINE_STEP.replace("Tiny pipeline", "Run it"),
            TINY_STEPS_SENTENCE,
        ),
    ),
    (
        "a step that ignores a failure",
        ci(
            RUN_TINY,
            RUN_TINY + "        continue-on-error: true\n",
            CI + "a step of job tiny ignores a failure",
        ),
    ),
    (
        "an environment on a step",
        ci(
            RUN_TINY,
            RUN_TINY + "        env:\n          X: y\n",
            CI + "a step of job tiny sets an environment",
        ),
    ),
    (
        "another Python in the job tiny",
        ci('python-version: "3.14"', 'python-version: "3.13"', TINY_STEPS_SENTENCE, 1),
    ),
    (
        "a sync without --frozen in the job tiny",
        ci(
            "run: uv sync --frozen --group dev",
            "run: uv sync --group dev",
            CI + "a uv sync is not uv sync --frozen --group dev",
            1,
        ),
    ),
    (
        "an upgrading sync in the job tiny",
        ci(
            "run: uv sync --frozen --group dev",
            "run: uv sync --frozen --group dev --upgrade",
            CI + "a run holds '--upgrade'",
            1,
        ),
    ),
    (
        "an action nobody else uses",
        ci(
            "uses: actions/checkout@v7", "uses: someone/else@v1", TINY_STEPS_SENTENCE, 1
        ),
    ),
    (
        "an extra step",
        ci(RUN_TINY, after_pipeline("Doctor", "run: just doctor"), TINY_STEPS_SENTENCE),
    ),
    (
        "a step that downloads",
        ci(
            RUN_TINY,
            after_pipeline("Fetch", "run: curl -O http://127.0.0.1/x"),
            CI + "a step of job tiny downloads something",
        ),
    ),
    (
        "a step that gets a file",
        ci(
            RUN_TINY,
            after_pipeline("Fetch", "run: wget http://127.0.0.1/x"),
            CI + "a step of job tiny downloads something",
        ),
    ),
]

VERSION_CASES: Final = [
    (
        "checkout at another version in the job tiny",
        ci(
            "uses: actions/checkout@v7",
            "uses: actions/checkout@v6",
            ONE_VERSION + "actions/checkout",
            1,
        ),
    ),
    (
        "setup-uv at another version in the job tiny",
        ci(
            "uses: astral-sh/setup-uv@v10.3.3",
            "uses: astral-sh/setup-uv@v10.3.2",
            ONE_VERSION + "astral-sh/setup-uv",
            1,
        ),
    ),
    (
        "setup-python at another version in the job tiny",
        ci(
            "uses: actions/setup-python@v7",
            "uses: actions/setup-python@v6",
            ONE_VERSION + "actions/setup-python",
            1,
        ),
    ),
    (
        "setup-just at another version in the job tiny",
        ci(
            "uses: extractions/setup-just@v4",
            "uses: extractions/setup-just@v3",
            ONE_VERSION + "extractions/setup-just",
            1,
        ),
    ),
    (
        "the upload of the report at another version",
        ci(
            UPLOAD_HEAD,
            UPLOAD_HEAD.replace("v7", "v6"),
            ONE_VERSION + "actions/upload-artifact",
        ),
    ),
    (
        "the upload of the coverage at another version",
        ci(
            "uses: actions/upload-artifact@v7",
            "uses: actions/upload-artifact@v6",
            ONE_VERSION + "actions/upload-artifact",
            0,
        ),
    ),
    (
        "checkout at another version in the gate",
        ci(
            "uses: actions/checkout@v7",
            "uses: actions/checkout@v6",
            ONE_VERSION + "actions/checkout",
            0,
        ),
    ),
    (
        "an action on a branch in the job tiny",
        ci(
            "uses: actions/checkout@v7",
            "uses: actions/checkout@main",
            CI + NOT_PINNED,
            1,
        ),
    ),
    (
        "checkout at another version in nightly",
        nightly(
            "uses: actions/checkout@v7",
            "uses: actions/checkout@v6",
            ONE_VERSION + "actions/checkout",
        ),
    ),
    (
        "setup-uv at another version in nightly",
        nightly(
            "uses: astral-sh/setup-uv@v10.3.3",
            "uses: astral-sh/setup-uv@v10",
            ONE_VERSION + "astral-sh/setup-uv",
        ),
    ),
    (
        "checkout at another version in release",
        release(
            "uses: actions/checkout@v7",
            "uses: actions/checkout@v6",
            ONE_VERSION + "actions/checkout",
        ),
    ),
    (
        "setup-just at another version in release",
        release(
            "uses: extractions/setup-just@v4",
            "uses: extractions/setup-just@v3",
            ONE_VERSION + "extractions/setup-just",
        ),
    ),
]

UPLOAD_CASES: Final = [
    (
        "a second upload",
        ci(
            RUN_TINY,
            after_pipeline("Upload more", "uses: actions/upload-artifact@v7"),
            ONE_UPLOAD,
        ),
    ),
    (
        "no upload",
        ci(
            UPLOAD_HEAD,
            "      - name: Nothing\n        if: always()\n        run: just doctor\n",
            ONE_UPLOAD,
        ),
    ),
    (
        "an upload of the whole directory",
        ci(
            JSON_PATH + MARKDOWN_PATH,
            "            artifacts/tiny/\n",
            CI + UPLOAD + "is not the two files of the report",
        ),
    ),
    (
        "an upload of the manifest too",
        ci(
            MARKDOWN_PATH,
            MARKDOWN_PATH + "            artifacts/tiny/manifest.json\n",
            CI + UPLOAD + "is not the two files of the report",
        ),
    ),
    (
        "an upload of the logs too",
        ci(
            MARKDOWN_PATH,
            MARKDOWN_PATH + "            artifacts/tiny/logs/\n",
            CI + UPLOAD + "is not the two files of the report",
        ),
    ),
    (
        "an upload of one file",
        ci(JSON_PATH, "", CI + UPLOAD + "is not the two files of the report"),
    ),
    (
        "an upload by a wildcard",
        ci(
            JSON_PATH + MARKDOWN_PATH,
            "            artifacts/tiny/*\n",
            CI + UPLOAD + "is not the two files of the report",
        ),
    ),
    (
        "an upload that is no condition",
        ci(
            UPLOAD_HEAD,
            UPLOAD_HEAD.replace("        if: always()\n", ""),
            CI + UPLOAD + "does not run after a failure",
        ),
    ),
    (
        "an upload that waits for success",
        ci(
            UPLOAD_HEAD,
            UPLOAD_HEAD.replace("always()", "success()"),
            CI + UPLOAD + "does not run after a failure",
        ),
    ),
    (
        "an upload under another name",
        ci(
            UPLOAD_NAME_LINE,
            "          name: report\n",
            CI + UPLOAD + "is not named tiny-report",
        ),
    ),
    (
        "an upload that fails when a file is missing",
        ci(
            "if-no-files-found: ignore",
            "if-no-files-found: error",
            CI + UPLOAD + "does not ignore missing files",
            1,
        ),
    ),
    (
        "an upload with a retention",
        ci(
            UPLOAD_NAME_LINE,
            UPLOAD_NAME_LINE + "          retention-days: 1\n",
            CI + UPLOAD + "sets options other than name, path and if-no-files-found",
        ),
    ),
    (
        "a summary without its condition on the file",
        ci(
            SUMMARY_IF, "${{ always() }}", CI + SUMMARY + "does not wait for the report"
        ),
    ),
    (
        "a summary skipped after a failure",
        ci(
            "${{ always() && hashFiles",
            "${{ hashFiles",
            CI + SUMMARY + "does not run when an earlier step failed",
        ),
    ),
    (
        "a summary waiting for another file",
        ci(
            "hashFiles('artifacts/tiny/report.md')",
            "hashFiles('artifacts/tiny/report.json')",
            CI + SUMMARY + "does not wait for the report",
        ),
    ),
    (
        "a summary that copies another file",
        ci(
            SUMMARY_LINE,
            SUMMARY_LINE.replace("report.md", "manifest.json"),
            CI + SUMMARY + "does not copy report.md",
        ),
    ),
    (
        "no summary",
        ci(
            SUMMARY_LINE,
            "        run: just doctor\n",
            CI + "job tiny does not hold exactly one step that writes the run page",
        ),
    ),
]

FORBIDDEN_CASES: Final = [
    (
        "a secret in a step",
        ci(
            RUN_TINY,
            after_pipeline("Echo", "run: echo ${{ secrets.HF_TOKEN }}"),
            CI + "a step names a secret",
        ),
    ),
    (
        "a secret in an environment",
        ci(
            RUN_TINY,
            RUN_TINY + "        env:\n          TOKEN: ${{ secrets.TOKEN }}\n",
            CI + "a step names a secret",
        ),
    ),
    (
        "a docker step",
        ci(
            RUN_TINY,
            after_pipeline("Build the docker image", "run: just doctor"),
            CI + "it mentions docker",
        ),
    ),
    (
        "a cache step",
        ci(
            RUN_TINY,
            after_pipeline("Cache", "uses: actions/cache@v4"),
            CI + "a step of job tiny caches",
        ),
    ),
    (
        "a cache of the Hugging Face directory",
        ci(
            RUN_TINY,
            after_pipeline(
                "Cache",
                "uses: actions/cache@v4\n        with:\n          path: ~/.cache/huggingface",
            ),
            CI + "the file reaches the Hugging Face hub or downloads a model",
        ),
    ),
    (
        "a cache of the virtual environment by the set-up",
        ci(
            "uses: astral-sh/setup-uv@v10.3.3",
            "uses: astral-sh/setup-uv@v10.3.3\n        with:\n          enable-cache: true",
            CI + "a step of job tiny caches",
            1,
        ),
    ),
    (
        "a model download",
        ci(
            RUN_TINY,
            after_pipeline("Model", "run: uv run hf download org/model"),
            CI + "the file reaches the Hugging Face hub or downloads a model",
        ),
    ),
    (
        "a token of the hub",
        ci(
            RUN_TINY,
            RUN_TINY + "        env:\n          HF_TOKEN: x\n",
            CI + "the file reaches the Hugging Face hub or downloads a model",
        ),
    ),
]

OTHER_CASES: Final = [
    (
        "the stub name put back in nightly: the names",
        nightly(PIPELINE_STEP, STUB_STEP, RENAMED),
    ),
    (
        "the stub name put back in nightly: the stub",
        nightly(PIPELINE_STEP, STUB_STEP, "nightly.yml" + NAMES_STUB),
    ),
    (
        "the pipeline step of nightly renamed",
        nightly(
            PIPELINE_STEP,
            PIPELINE_STEP.replace("Tiny pipeline", "Run the pipeline"),
            RENAMED,
        ),
    ),
    (
        "a nightly timeout of 120 minutes",
        nightly("timeout-minutes: 150", "timeout-minutes: 120", NIGHTLY_BELOW),
    ),
    (
        "a nightly timeout one minute under the bound",
        nightly("timeout-minutes: 150", "timeout-minutes: 123", NIGHTLY_BELOW),
    ),
    (
        "a nightly timeout that is not a number",
        nightly("timeout-minutes: 150", "timeout-minutes: true", NIGHTLY_BELOW),
    ),
    (
        "the stub named in release",
        release(
            "- name: Install just",
            "- name: Install just (stub until D15)",
            "release.yml" + NAMES_STUB,
        ),
    ),
    (
        "the stub named in a comment of ci",
        ci("# The tiny pipeline", "# The stub until D15", "ci.yml" + NAMES_STUB),
    ),
]

JUSTFILE_CASES: Final = [
    recipe(
        "the stub echo back: it is not the recipe",
        '    echo "just tiny: not implemented until D15 (tiny end-to-end pipeline)."\n',
        NOT_THE_RECIPE,
    ),
    recipe(
        "the stub echo back: it names the stub",
        '    echo "just tiny: not implemented until D15 (tiny end-to-end pipeline)."\n',
        "justfile" + NAMES_STUB,
    ),
    recipe(
        "a plain uv run", "    uv run rinchen tiny run --profile tiny\n", NOT_THE_RECIPE
    ),
    recipe(
        "a uv run that syncs",
        "    uv run --frozen rinchen tiny run --profile tiny\n",
        NOT_THE_RECIPE,
    ),
    recipe(
        "another profile",
        "    uv run --frozen --no-sync rinchen tiny run --profile full\n",
        NOT_THE_RECIPE,
    ),
    recipe("a dry run", f"    {TINY_RECIPE} --dry-run\n", NOT_THE_RECIPE),
    recipe("a second command", RECIPE_LINE + "    echo done\n", NOT_THE_RECIPE),
    recipe(
        "an upload: the word",
        f"    {TINY_RECIPE} --upload\n",
        "the tiny recipe holds '--upload'",
    ),
    recipe(
        "an upload: the option",
        f"    {TINY_RECIPE} --upload\n",
        "the tiny recipe is given an option the command lacks: --upload",
    ),
    recipe(
        "a container",
        "    docker run rinchen tiny run --profile tiny\n",
        "the tiny recipe holds 'docker'",
    ),
    recipe(
        "a graphics card",
        f"    nvidia-smi && {TINY_RECIPE}\n",
        "the tiny recipe holds 'nvidia'",
    ),
    recipe(
        "a lock", f"    uv lock && {TINY_RECIPE}\n", "the tiny recipe holds 'uv lock'"
    ),
    recipe(
        "an upgrade",
        f"    uv sync --upgrade && {TINY_RECIPE}\n",
        "the tiny recipe holds '--upgrade'",
    ),
    recipe(
        "an add",
        f"    uv add requests && {TINY_RECIPE}\n",
        "the tiny recipe holds 'uv add'",
    ),
    recipe(
        "a pip install",
        f"    uv pip install x && {TINY_RECIPE}\n",
        "the tiny recipe holds 'uv pip'",
    ),
    recipe(
        "an option the command lacks",
        "    uv run --frozen --no-sync rinchen tiny run --profil tiny\n",
        "the tiny recipe is given an option the command lacks: --profil",
    ),
    ("the recipe renamed", Case("justfile", "\ntiny:\n", "\ntiny-run:\n", ONE_RECIPE)),
    (
        "a second recipe of the same name",
        Case("justfile", "\ntiny:\n", "\ntiny:\n    echo first\n\ntiny:\n", ONE_RECIPE),
    ),
]

GITIGNORE_CASES: Final = [
    ("the ignore rule removed", ignore(IGNORE_PAIR, "", NOT_IGNORED)),
    ("an unanchored rule", ignore("\n/artifacts/\n", "\nartifacts/\n", NOT_IGNORED)),
    (
        "a rule for the tiny directory only",
        ignore("\n/artifacts/\n", "\n/artifacts/tiny/\n", NOT_IGNORED),
    ),
    (
        "the comment removed",
        ignore(
            "# The output directory of the tiny pipeline (just tiny)\n",
            "",
            "the ignore rule of the output directory has no comment above it",
        ),
    ),
    (
        "another comment",
        ignore(
            "(just tiny)\n",
            "(the pipeline)\n",
            "the ignore rule of the output directory has no comment above it",
        ),
    ),
    (
        "a rule that takes a report back",
        ignore(
            "/artifacts/\n",
            "/artifacts/\n!/artifacts/tiny/report.json\n",
            "a rule of .gitignore takes a path of the output directory back",
        ),
    ),
]

BREAKS: Final[dict[str, Case]] = numbered(
    [
        *TRIGGER_CASES,
        *GATE_CASES,
        *TINY_JOB_CASES,
        *STEP_CASES,
        *VERSION_CASES,
        *UPLOAD_CASES,
        *FORBIDDEN_CASES,
        *OTHER_CASES,
        *JUSTFILE_CASES,
        *GITIGNORE_CASES,
    ]
)
