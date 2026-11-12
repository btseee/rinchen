"""The community files of the repository: the documents GitHub's community profile asks for and the
statement about accessibility. They are text for people, so the test pins what makes them usable and
true: the files exist, are plain ASCII, hold nothing the project's content rules exclude, keep their
headings in order, use descriptive links that lead somewhere, carry only the address the owner approved
and only in the files the owner approved, and say nothing about the repository that the repository
contradicts (the gate recipe, the rule texts of `CLAUDE.md`, how a token is handled).

Nothing here starts a process or opens a socket: the files are read, parsed with `re`, `ast` and `yaml`,
and compared. Where a claim of a document is about code (the upload needs `HF_TOKEN`, a child process
gets no token), a small test reads that code, so the claim cannot go stale unnoticed. The helpers are in
`tests/support/community_files.py`, which also says how the needles of the content rules and the address
of the owner are spelled (as code points, so that no file of the test holds them).
"""

from __future__ import annotations

import ast
import re
import tomllib
from itertools import pairwise
from pathlib import Path
from typing import Final

import pytest
import yaml

from rinchen.config.serve import ServeConfig
from rinchen.maintain.commands import child_environment
from tests.support import community_files
from tests.support.community_files import (
    ADVISORY_URL,
    CODE_OF_CONDUCT,
    CONTRIBUTING,
    COVENANT_URL,
    EXTRA_NEEDLE_BASE,
    EXTRA_NEEDLE_CODE_POINTS,
    MAX_WORDS_IN_A_PARAGRAPH,
    OWNER_ADDRESS,
    PULL_REQUEST_TEMPLATE,
    ROOT,
    SECURITY,
    URL,
    VAGUE_LINK_TEXTS,
    WIKI_URL,
    excluded_words,
    headings,
    holders_of_the_address,
    images,
    links,
    non_ascii_lines,
    paragraphs,
    parameter_defaults,
    read_text,
    repository_paths,
    rules_of_claude,
    section,
    slug,
    squeezed,
)
from tests.unit.test_d13_content_rules import NEEDLE_CODE_POINTS

MARKDOWN_FILES: Final = (
    CODE_OF_CONDUCT,
    CONTRIBUTING,
    SECURITY,
    PULL_REQUEST_TEMPLATE,
)
NEW_FILES: Final = MARKDOWN_FILES
# The files that name the address of the owner, and the only ones that may.
HOLDERS_OF_THE_ADDRESS: Final = (CODE_OF_CONDUCT, SECURITY)


# ---- every file ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", NEW_FILES)
def test_the_file_exists_and_says_something(path: str) -> None:
    assert (ROOT / path).is_file(), path
    assert len(read_text(path).strip()) > 200, path


@pytest.mark.parametrize("path", NEW_FILES)
def test_the_file_is_plain_ascii(path: str) -> None:
    assert non_ascii_lines((ROOT / path).read_bytes()) == [], (
        "line numbers of the lines with a byte above 127"
    )


@pytest.mark.parametrize("path", NEW_FILES)
def test_the_file_holds_no_word_the_content_rules_exclude(path: str) -> None:
    assert excluded_words(read_text(path)) == [], "(line, needle number) pairs"


@pytest.mark.parametrize("path", NEW_FILES)
def test_the_file_ends_in_one_newline_and_no_line_ends_in_a_blank(path: str) -> None:
    text = read_text(path)

    assert text.endswith("\n") and not text.endswith("\n\n")
    assert [n for n, line in enumerate(text.split("\n")) if line != line.rstrip()] == []


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_the_headings_come_in_order_and_do_not_skip_a_level(path: str) -> None:
    levels = [level for level, _ in headings(read_text(path))]
    first = 2 if path == PULL_REQUEST_TEMPLATE else 1

    assert levels and levels[0] == first
    assert all(after <= before + 1 for before, after in pairwise(levels))
    if first == 1:
        assert levels.count(1) == 1


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_the_links_say_where_they_go_and_lead_there(path: str) -> None:
    text = read_text(path)
    anchors = {slug(title) for _, title in headings(text)}
    directory = (ROOT / path).parent

    for label, target in links(text):
        assert label.strip().lower() not in VAGUE_LINK_TEXTS, label
        assert not label.lower().startswith("http"), label
        if target.startswith("#"):
            assert target[1:] in anchors, target
        elif target.startswith("mailto:"):
            continue
        elif "://" in target:
            assert target.startswith("https://"), target
        else:
            assert (directory / target.split("#")[0]).exists(), target


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_an_image_has_a_text_alternative(path: str) -> None:
    for alternative, _ in images(read_text(path)):
        assert alternative.strip(), path


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_the_paragraphs_are_short(path: str) -> None:
    too_long = [
        paragraph[:40]
        for paragraph in paragraphs(read_text(path))
        if len(paragraph.split()) > MAX_WORDS_IN_A_PARAGRAPH
    ]

    assert too_long == []


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_every_path_the_file_names_exists(path: str) -> None:
    for span in repository_paths(read_text(path)):
        assert (ROOT / span.rstrip("/")).exists(), span


@pytest.mark.parametrize("path", MARKDOWN_FILES)
def test_the_file_points_at_the_wiki_home_and_at_no_page_that_will_move(
    path: str,
) -> None:
    text = read_text(path)

    assert {url.rstrip(".,;:") for url in URL.findall(text) if "/wiki" in url} <= {
        WIKI_URL
    }
    for moved in ("STATE.md", "DECISIONS.md", "docs/"):
        assert moved not in text, moved


# ---- the scan and the address -----------------------------------------------------------------------------


def test_the_needles_of_the_scan_are_the_content_rules_and_are_many() -> None:
    assert len(NEEDLE_CODE_POINTS) >= 12
    assert excluded_words("a sentence with nothing in it") == []
    for source in (Path(__file__), Path(community_files.__file__)):
        text = source.read_text(encoding="utf-8")
        assert excluded_words(text) == [], source.name
        assert OWNER_ADDRESS.split("@")[0] not in text.lower(), source.name


@pytest.mark.parametrize("number", range(len(EXTRA_NEEDLE_CODE_POINTS)))
def test_the_scan_finds_each_extra_needle_in_any_case(number: int) -> None:
    word = "".join(map(chr, EXTRA_NEEDLE_CODE_POINTS[number]))

    for planted in (word, word.upper(), word.capitalize()):
        expected = [(2, EXTRA_NEEDLE_BASE + number)]
        assert excluded_words(f"fine\nsays {planted} here\n") == expected


def test_the_address_of_the_owner_is_in_the_approved_files_and_nowhere_else() -> None:
    assert holders_of_the_address() == sorted(HOLDERS_OF_THE_ADDRESS)


@pytest.mark.parametrize("path", HOLDERS_OF_THE_ADDRESS)
def test_each_approved_file_gives_the_address_as_the_way_to_write(path: str) -> None:
    assert OWNER_ADDRESS in read_text(path)


# ---- the code of conduct ---------------------------------------------------------------------------------


def test_the_code_of_conduct_has_the_parts_of_the_covenant_it_follows_in_spirit() -> (
    None
):
    titles = [title for _, title in headings(read_text(CODE_OF_CONDUCT))]

    for wanted in (
        "Our pledge",
        "Our standards",
        "Responsibilities of the maintainers",
        "Scope",
        "Enforcement",
        "Reporting",
        "Enforcement guidelines",
        "Correction",
        "Warning",
        "Temporary ban",
        "Permanent ban",
        "Attribution",
    ):
        assert wanted in titles, wanted


def test_the_code_of_conduct_ends_with_one_line_naming_the_covenant_and_its_url() -> (
    None
):
    text = read_text(CODE_OF_CONDUCT)
    last = text.strip().splitlines()[-1]

    assert "Contributor Covenant" in last and "version 2.1" in last
    assert COVENANT_URL in last
    assert text.count(COVENANT_URL) == 1


def test_the_code_of_conduct_gives_the_address_and_githubs_own_tools_for_a_report() -> (
    None
):
    reporting = section(read_text(CODE_OF_CONDUCT), "Reporting")

    assert OWNER_ADDRESS in reporting
    assert "GitHub" in reporting and "abuse" in reporting
    assert "maintainer" in reporting


# ---- the contributing guide -----------------------------------------------------------------------------


def test_the_contributing_guide_has_its_sections() -> None:
    titles = [title for _, title in headings(read_text(CONTRIBUTING))]

    for wanted in (
        "Questions and problems",
        "Development setup",
        "The gate",
        "Tests",
        "Branches and commits",
        "Pull requests",
        "Project rules",
        "Documentation",
    ):
        assert wanted in titles, wanted


def test_the_setup_says_python_3_14_uv_the_sync_command_of_ci_and_the_doctor() -> None:
    setup = section(read_text(CONTRIBUTING), "Development setup")
    ci = yaml.safe_load(read_text(".github/workflows/ci.yml"))
    syncs = [
        step["run"]
        for step in ci["jobs"]["gate"]["steps"]
        if step.get("name") == "Sync dependencies"
    ]

    assert syncs == ["uv sync --frozen --group dev"]
    assert "uv sync --frozen --group dev" in setup
    assert "uv run rinchen doctor" in setup
    assert "just tiny" in setup
    assert "3.14" in setup and "uv" in setup


def test_the_gate_section_lists_the_commands_of_the_gate_recipe() -> None:
    recipe: list[str] = []
    lines = read_text("justfile").splitlines()
    for line in lines[lines.index("gate:") + 1 :]:
        if not line.startswith("    "):
            break
        recipe.append(line.strip())
    gate = section(read_text(CONTRIBUTING), "The gate")

    assert len(recipe) == 4
    for command in recipe:
        assert command in gate, command
    assert "just gate" in gate


def test_the_tests_section_names_the_four_kinds_and_the_rules_on_thresholds_and_fixtures() -> (
    None
):
    tests = section(read_text(CONTRIBUTING), "Tests")

    for directory in (
        "tests/unit",
        "tests/property",
        "tests/golden",
        "tests/integration",
    ):
        assert directory in tests and (ROOT / directory).is_dir(), directory
    for wanted in ("before the code", "never lowered", ".gitignore", "committed"):
        assert wanted in tests, wanted
    assert "tests/unit/test_fixture_files_are_tracked.py" in tests


def test_the_branch_and_commit_conventions_are_the_ones_the_history_uses() -> None:
    text = section(read_text(CONTRIBUTING), "Branches and commits")

    for wanted in ("main", "D<N>:", "docs:", "fix:", "chore:", "ci:", "feat:"):
        assert wanted in text, wanted


def test_the_pull_request_section_names_what_runs_and_what_a_reviewer_checks() -> None:
    text = section(read_text(CONTRIBUTING), "Pull requests")
    ci = yaml.safe_load(read_text(".github/workflows/ci.yml"))

    assert set(ci["jobs"]) == {"gate", "tiny"}
    assert "pull_request" in ci[True]  # PyYAML reads the key `on` as the boolean True
    for wanted in ("gate", "tiny", "reviewer"):
        assert wanted in text, wanted


def test_the_project_rules_copy_the_rule_texts_of_claude_md_word_for_word() -> None:
    rules = [*rules_of_claude("ALWAYS:"), *rules_of_claude("NEVER:")]
    section_text = squeezed(section(read_text(CONTRIBUTING), "Project rules"))
    markers = (
        "Run everything on CPython 3.14.x",
        "Use uv;",
        "Train every Rinchen model from random init",
        "position_encoding_type: SinusoidalInterleaved",
        "Record every resolved package version",
        "Write tests before code",
        "Use Python 3.12/3.13",
        "Fine-tune, LoRA",
        "Use Inner Mongolian data",
        "Use galig/Latin",
        "Use Relative, rotary, ALiBi",
        "Tune on FLORES+ devtest",
    )

    for marker in markers:
        matching = [rule for rule in rules if marker in rule]
        assert len(matching) == 1, marker
        assert squeezed(matching[0]) in section_text, marker


def test_the_project_rules_say_the_gpu_is_optional_and_the_cpu_the_default() -> None:
    text = section(read_text(CONTRIBUTING), "Project rules")

    assert "GPU is optional" in text and "CPU is the default" in text


def test_the_documentation_section_links_the_wiki_home() -> None:
    documentation = section(read_text(CONTRIBUTING), "Documentation")

    assert WIKI_URL in [target for _, target in links(documentation)]


def test_the_contributing_guide_points_at_the_conduct_and_security_files() -> None:
    targets = [target for _, target in links(read_text(CONTRIBUTING))]

    assert "CODE_OF_CONDUCT.md" in targets and "SECURITY.md" in targets


# ---- the security policy ----------------------------------------------------------------------------------


def test_the_security_policy_has_its_sections() -> None:
    titles = [title for _, title in headings(read_text(SECURITY))]

    for wanted in (
        "Supported versions",
        "Reporting a vulnerability",
        "What to include",
        "What to expect",
        "Scope",
        "Safe practices of the project",
    ):
        assert wanted in titles, wanted


def test_the_policy_says_where_to_report_privately_and_what_to_expect() -> None:
    text = read_text(SECURITY)
    reporting = section(text, "Reporting a vulnerability")
    expect = section(text, "What to expect")

    assert reporting.index(ADVISORY_URL) < reporting.index(OWNER_ADDRESS)
    assert "7 days" in expect and "14 days" in expect
    assert "coordinated" in expect and "no bounty program" in expect
    assert "`main`" in section(text, "Supported versions")


def test_the_policy_calls_the_project_unreleased_while_it_is_version_zero() -> None:
    project = tomllib.loads(read_text("pyproject.toml"))["project"]

    assert project["version"] == "0.0.0"
    assert "0.0.0" in section(read_text(SECURITY), "Supported versions")


def test_the_scope_names_what_is_in_and_what_is_out() -> None:
    scope = section(read_text(SECURITY), "Scope")

    for wanted in ("rinchen serve", "workflows", "rinchen release", "token"):
        assert wanted in scope, wanted
    for wanted in ("translations", "report those upstream", "GPU"):
        assert wanted in scope, wanted


def test_the_service_listens_on_the_loopback_by_default_as_the_scope_says() -> None:
    assert ServeConfig.model_fields["host"].default == "127.0.0.1"
    assert "127.0.0.1" in section(read_text(SECURITY), "Scope")
    assert "no authentication" in read_text("src/rinchen/cli/serve.py")


def test_no_workflow_uses_a_repository_secret_as_the_policy_says() -> None:
    names = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

    assert len(names) >= 3
    assert [p.name for p in names if "secrets." in p.read_text(encoding="utf-8")] == []


def test_the_release_nightly_and_tiny_jobs_ask_for_read_only_contents() -> None:
    workflows = ROOT / ".github" / "workflows"
    release = yaml.safe_load((workflows / "release.yml").read_text(encoding="utf-8"))
    nightly = yaml.safe_load((workflows / "nightly.yml").read_text(encoding="utf-8"))
    ci = yaml.safe_load((workflows / "ci.yml").read_text(encoding="utf-8"))

    assert release["permissions"] == {"contents": "read"}
    assert nightly["permissions"] == {"contents": "read"}
    assert ci["jobs"]["tiny"]["permissions"] == {"contents": "read"}


def test_the_upload_is_a_dry_run_unless_asked_and_reads_the_token_from_the_environment() -> (
    None
):
    tree = ast.parse(read_text("src/rinchen/cli/release_hub.py"))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    token_reads = [
        node
        for node in ast.walk(functions["token_from_environment"])
        if isinstance(node, ast.Constant) and node.value == "HF_TOKEN"
    ]
    environment_uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ]
    commands = [f for f in functions.values() if "upload_it" in parameter_defaults(f)]
    default = parameter_defaults(commands[0])["upload_it"]

    assert len(commands) == 1
    assert isinstance(default, ast.Constant) and default.value is False
    assert len(token_reads) == 1
    assert len(environment_uses) == 1


def test_the_upload_scans_the_staged_text_for_a_token_and_a_local_path() -> None:
    scan = read_text("src/rinchen/export/hub_scan.py")
    reading = read_text("src/rinchen/export/hub_files.py")

    assert "KIND_TOKEN" in scan and "KIND_PATH" in scan
    assert "kinds_in(read.data)" in reading
    assert "read_stage" in read_text("src/rinchen/export/hub.py")


def test_no_module_of_the_library_looks_for_a_token_file() -> None:
    names = (
        "get_token",
        "HfFolder",
        ".netrc",
        "huggingface/token",
        ".git-credentials",
    )
    found = [
        (path.name, name)
        for path in sorted((ROOT / "src" / "rinchen").rglob("*.py"))
        for name in names
        if name in path.read_text(encoding="utf-8")
    ]

    assert found == []


def test_a_child_process_of_the_pipeline_gets_no_token_and_stays_offline() -> None:
    parent = {"HF_TOKEN": "x", "SOME_API_KEY": "y", "PATH": "/usr/bin"}

    child = child_environment(parent)

    assert "HF_TOKEN" not in child and "SOME_API_KEY" not in child
    assert child["PATH"] == "/usr/bin"
    assert child["HF_HUB_OFFLINE"] == "1" and child["CUDA_VISIBLE_DEVICES"] == ""
    assert "child_environment(os.environ)" in read_text(
        "src/rinchen/tiny/step_runner.py"
    )


def test_the_tests_that_use_a_network_carry_a_marker_that_the_gate_leaves_out() -> None:
    pyproject = read_text("pyproject.toml")

    assert '"network: tests that make real network calls"' in pyproject
    assert 'pytest -m "not network" -q' in read_text("justfile")


# ---- the pull request template ---------------------------------------------------------------------------


def test_the_pull_request_template_has_the_sections_and_unchecked_boxes() -> None:
    text = read_text(PULL_REQUEST_TEMPLATE)
    titles = [title for _, title in headings(text)]
    kinds = section(text, "Type of change")
    checklist = section(text, "Checklist")

    assert titles == ["Summary", "Type of change", "What was verified", "Checklist"]
    assert len(re.findall(r"^- \[ \] ", kinds, re.MULTILINE)) >= 5
    assert len(re.findall(r"^- \[ \] ", checklist, re.MULTILINE)) >= 6
    assert "- [x]" not in text.lower()


def test_the_pull_request_template_asks_for_the_checks_the_project_relies_on() -> None:
    text = read_text(PULL_REQUEST_TEMPLATE)
    verified = section(text, "What was verified")
    checklist = section(text, "Checklist")

    assert "just gate" in verified
    for wanted in (
        "written first",
        "failing test",
        "threshold",
        "fixtures",
        "Wiki",
        "project rules",
        "accessibility",
    ):
        assert wanted in checklist, wanted
