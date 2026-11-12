"""`scripts/check_wiki.py`, the checker of a clone of the project's Wiki (the documents moved there).

Every test builds a small Wiki directory in `tmp_path` and asserts what the checker reports: the words
the content rules exclude (as needle numbers, never the word) on a page that is not a moved record,
internal links that point at no page or no heading, a character outside ASCII on such a page, a carriage
return or an unclosed code fence on any page, an illegal or duplicate page name, a missing `Home`, and a
row of the Verification-Log that a repository file names but the page does not hold. The needles are
spelled as code points here as in the content-rules test, and the checker's own are pinned to them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from check_wiki import (
    NEEDLE_CODE_POINTS,
    Problem,
    check_wiki,
    is_record_page,
    main,
    slug,
)

from rinchen.config.loader import REPO_ROOT
from tests.unit.test_d13_content_rules import NEEDLE_CODE_POINTS as CONTENT_RULES
from tests.unit.test_d13_content_rules import needles

# The id of a Verification-Log row, as the earlier repository test of the shipped table read it.
ROW_ID = re.compile(r"\bD(?:[0-9]+)?-[A-Z0-9]+(?:-[A-Z0-9]+)*")
GOOD: dict[str, str] = {
    "Home.md": "# Home\n\nSee [the guide](Getting-Started) and [a step](Getting-Started#first-steps).\n",
    "Getting-Started.md": (
        "# Getting started\n\n## First steps\n\nRun the gate.\n\n"
        "```sh\nuv run pytest  # [not a link](Nowhere)\n```\n\n"
        "Back to [Home](https://github.com/btseee/rinchen/wiki/Home) or "
        "[uv](https://docs.astral.sh/uv/). `[code](Nowhere)` is not a link.\n"
    ),
    "_Sidebar.md": "- [Home](Home)\n- [Getting started](Getting-Started)\n",
    "_Footer.md": "Pages are kept in a git repository.\n",
    "Verification-Log.md": "| id | result |\n| --- | --- |\n| D-ONE | pass |\n| D13-TWO-PARTS | pass |\n",
}


def build(tmp_path: Path, pages: dict[str, str], name: str = "wiki") -> Path:
    wiki = tmp_path / name
    wiki.mkdir()
    for page, text in pages.items():
        (wiki / page).write_bytes(text.encode("utf-8"))
    return wiki


def kinds(problems: list[Problem]) -> list[tuple[str, int, str]]:
    return [(problem.page, problem.line, problem.kind) for problem in problems]


# ---- the needles ---------------------------------------------------------------------------------------


def test_the_needles_of_the_checker_are_the_needles_of_the_content_rules() -> None:
    assert NEEDLE_CODE_POINTS == CONTENT_RULES


def test_a_good_wiki_has_no_problem(tmp_path: Path) -> None:
    assert check_wiki(build(tmp_path, GOOD)) == []


@pytest.mark.parametrize(
    "page", ["Home.md", "Getting-Started.md", "_Sidebar.md", "_Footer.md"]
)
def test_a_needle_on_a_guide_page_is_named_by_number_and_line_never_by_the_word(
    tmp_path: Path, page: str
) -> None:
    word = needles()[3]
    pages = {**GOOD, page: GOOD[page] + f"\nA line with {word.upper()} in it.\n"}

    problems = check_wiki(build(tmp_path, pages))

    line = len(pages[page].splitlines())
    assert kinds(problems) == [(page, line, "needle")]
    assert problems[0].detail == "needle 3"
    assert word not in str(problems[0]).lower()


@pytest.mark.parametrize(
    "page",
    [
        "Runbook-01-Acquisition.md",
        "Spec-D1-Data-Acquisition.md",
        "Plan-D1-Data-Acquisition.md",
        "Research-Master-Plan-D0-D15.md",
        "State-History-D3.md",
        "Decision-2026-10-30-D1-Candidate.md",
        "Verification-Log.md",
        "Stack.md",
    ],
)
def test_a_moved_record_may_hold_a_needle_and_other_characters(
    tmp_path: Path, page: str
) -> None:
    text = (
        GOOD.get(page, "# A record\n")
        + f"\nOriginal text: {needles()[0]} {chr(0x0430)}\n"
    )

    assert check_wiki(build(tmp_path, {**GOOD, page: text})) == []


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("Runbook-15-Tiny-Pipeline", True),
        ("Verification-Log", True),
        ("Stack", True),
        ("Decisions", True),
        ("Home", False),
        ("State", False),
        ("Decisions-Draft", False),
        ("Roadmap", False),
        ("_Sidebar", False),
        ("Stack-Notes", False),
        ("Verification-Log-2", False),
    ],
)
def test_which_pages_are_moved_records(name: str, record: bool) -> None:
    assert is_record_page(name) is record


# ---- links ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        "[x](Missing-Page)",
        "[x](Getting-Started#no-such-heading)",
        "[x](https://github.com/btseee/rinchen/wiki/Missing-Page)",
        "[x](https://github.com/btseee/rinchen/wiki/Home#nowhere)",
        "[x](#not-a-heading-here)",
        "[x](docs/runbooks/00_environment.md)",
    ],
)
def test_a_dangling_internal_link_is_a_problem(tmp_path: Path, link: str) -> None:
    pages = {**GOOD, "Home.md": GOOD["Home.md"] + f"\nAlso {link}.\n"}

    problems = check_wiki(build(tmp_path, pages))

    assert kinds(problems) == [("Home.md", 5, "link")]


@pytest.mark.parametrize(
    "link",
    [
        "[x](Home)",
        "[x](Home#home)",
        "[x](#home)",
        "[x](Getting-Started#first-steps)",
        "[x](https://github.com/btseee/rinchen/wiki)",
        "[x](https://github.com/btseee/rinchen/wiki/Getting-Started#first-steps)",
        "[x](https://github.com/btseee/rinchen/blob/main/README.md)",
        "[x](mailto:someone@example.invalid)",
    ],
)
def test_a_link_to_a_page_and_heading_that_exist_or_outside_the_wiki_is_fine(
    tmp_path: Path, link: str
) -> None:
    pages = {**GOOD, "Home.md": GOOD["Home.md"] + f"\nAlso {link}.\n"}

    assert check_wiki(build(tmp_path, pages)) == []


def test_an_anchor_alone_names_a_heading_of_its_own_page(tmp_path: Path) -> None:
    text = GOOD["Getting-Started.md"] + "\nUp to [the steps](#first-steps).\n"

    assert check_wiki(build(tmp_path, {**GOOD, "Getting-Started.md": text})) == []


def test_the_heading_slugs_follow_the_rules_of_the_rendered_page() -> None:
    assert slug("First steps") == "first-steps"
    assert slug("failure-lock_consistent") == "failure-lock_consistent"
    assert slug("What `uv sync` does (D0)!") == "what-uv-sync-does-d0"


def test_a_repeated_heading_gets_a_numbered_anchor(tmp_path: Path) -> None:
    pages = {
        **GOOD,
        "Home.md": "# Home\n\n## Notes\n\n## Notes\n\n[a](#notes-1) [b](#notes-2)\n",
    }

    problems = check_wiki(build(tmp_path, pages))

    assert kinds(problems) == [("Home.md", 7, "link")]
    assert "notes-2" in problems[0].detail


def test_a_double_bracket_link_to_a_missing_page_is_a_problem(tmp_path: Path) -> None:
    pages = {**GOOD, "Home.md": GOOD["Home.md"] + "\n[[Home]] [[Text|Missing-Page]]\n"}

    assert kinds(check_wiki(build(tmp_path, pages))) == [("Home.md", 5, "link")]


# ---- characters, line ends and fences ------------------------------------------------------------------


def test_a_character_outside_ascii_on_a_guide_page_is_named_by_line(
    tmp_path: Path,
) -> None:
    pages = {
        **GOOD,
        "Getting-Started.md": GOOD["Getting-Started.md"] + "caf" + chr(0xE9) + "\n",
    }

    problems = check_wiki(build(tmp_path, pages))

    assert kinds(problems) == [("Getting-Started.md", 12, "ascii")]


def test_a_carriage_return_is_a_problem_on_any_page(tmp_path: Path) -> None:
    pages = {**GOOD, "Stack.md": "# Stack\r\n\r\nfine\r\n"}

    assert kinds(check_wiki(build(tmp_path, pages))) == [("Stack.md", 1, "crlf")]


@pytest.mark.parametrize(
    "text",
    [
        "# A\n\n```py\nx = 1\n",
        "# A\n\n~~~\nx\n```\n",
        "# A\n\n````\nx\n```\n",
        "# A\n\n```\nx\n``` not a close\n",
    ],
)
def test_an_unclosed_code_fence_is_a_problem(tmp_path: Path, text: str) -> None:
    pages = {**GOOD, "Stack.md": text}

    assert kinds(check_wiki(build(tmp_path, pages))) == [("Stack.md", 3, "fence")]


def test_the_fence_left_open_is_named_after_closed_ones(tmp_path: Path) -> None:
    pages = {**GOOD, "Stack.md": "# A\n```\nx\n```\n```\ny\n```\n~~~\nz\n"}

    assert kinds(check_wiki(build(tmp_path, pages))) == [("Stack.md", 8, "fence")]


@pytest.mark.parametrize("text", ["# A\n```\nx\n```", "# A\n```\nx\n```\n```\ny\n```"])
def test_a_page_that_ends_on_its_closing_fence_is_fine(
    tmp_path: Path, text: str
) -> None:
    assert check_wiki(build(tmp_path, {**GOOD, "Stack.md": text})) == []


def test_a_longer_closing_fence_closes_and_an_indented_block_is_no_fence(
    tmp_path: Path,
) -> None:
    pages = {**GOOD, "Stack.md": "# A\n\n```\nx\n`````\n\n    ```\n"}

    assert check_wiki(build(tmp_path, pages)) == []


# ---- page names ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Bad Name.md",
        "bad_name.md",
        "-Leading.md",
        "Trailing-.md",
        "Double--Dash.md",
        "notes.txt",
    ],
)
def test_an_illegal_page_name_is_a_problem(tmp_path: Path, name: str) -> None:
    pages = {**GOOD, name: "# x\n"}

    assert kinds(check_wiki(build(tmp_path, pages))) == [(name, 0, "page-name")]


def test_two_names_that_differ_only_in_case_are_a_problem(tmp_path: Path) -> None:
    pages = {**GOOD, "HOME.md": "# Home again\n"}

    assert kinds(check_wiki(build(tmp_path, pages))) == [
        ("HOME.md", 0, "page-name"),
        ("Home.md", 0, "page-name"),
    ]


def test_the_git_directory_of_a_clone_is_not_a_page(tmp_path: Path) -> None:
    wiki = build(tmp_path, GOOD)
    (wiki / ".git").mkdir()
    (wiki / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

    assert check_wiki(wiki) == []


def test_a_wiki_without_a_home_page_is_a_problem(tmp_path: Path) -> None:
    pages = {name: text for name, text in GOOD.items() if name != "Home.md"}
    pages["_Sidebar.md"] = "- [Getting started](Getting-Started)\n"

    # The link of Getting-Started to the Wiki's root dangles too: the root is the Home page.
    assert kinds(check_wiki(build(tmp_path, pages))) == [
        ("Getting-Started.md", 11, "link"),
        ("Home.md", 0, "page-name"),
    ]


# ---- the rows a repository file names ------------------------------------------------------------------


def test_a_row_a_decision_names_must_be_on_the_verification_log(tmp_path: Path) -> None:
    wiki = build(tmp_path, GOOD)
    config = tmp_path / "maintain.yaml"
    config.write_text(
        "skipped:\n  D-NOT-A-DECISION: x\n"
        '    decision: "Wiki page Verification-Log rows D-ONE, D13-TWO-PARTS, D-MISSING"\n',
        encoding="utf-8",
    )

    problems = check_wiki(wiki, verify_ids_from=[config])

    assert kinds(problems) == [("maintain.yaml", 3, "row")]
    assert problems[0].detail == "D-MISSING is not a row of Verification-Log"


def test_a_file_that_names_no_row_is_a_problem(tmp_path: Path) -> None:
    config = tmp_path / "maintain.yaml"
    config.write_text('decision: "Wiki page Decisions: a pin"\n', encoding="utf-8")

    problems = check_wiki(build(tmp_path, GOOD), verify_ids_from=[config])

    assert kinds(problems) == [("maintain.yaml", 0, "row")]


# ---- the command ---------------------------------------------------------------------------------------


def test_the_command_exits_0_on_a_good_wiki_and_says_how_many_pages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(build(tmp_path, GOOD))]) == 0
    assert capsys.readouterr().out == "5 pages, 0 problems\n"


def test_the_command_exits_1_and_prints_one_line_per_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = {**GOOD, "Home.md": GOOD["Home.md"] + "[x](Missing)\r\n"}

    assert main([str(build(tmp_path, pages))]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "Home.md:4: crlf: a carriage return ends the line",
        "Home.md:4: link: Missing is not a page",
        "5 pages, 2 problems",
    ]


def test_the_command_exits_2_for_a_directory_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "absent")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_the_recipe_takes_the_wiki_directory_and_checks_the_rows_of_the_maintenance_config() -> (
    None
):
    lines = (REPO_ROOT / "justfile").read_text(encoding="utf-8").splitlines()
    start = lines.index("wiki-check wiki_dir:")
    body = []
    for line in lines[start + 1 :]:
        if not line.startswith("    "):
            break
        body.append(line.strip())

    command = (
        'uv run --frozen --no-sync python scripts/check_wiki.py "{{wiki_dir}}" '
        "--verify-ids-from configs/maintain/maintain.yaml"
    )
    assert body == [command]
    assert (REPO_ROOT / "scripts" / "check_wiki.py").is_file()
    assert (REPO_ROOT / "configs" / "maintain" / "maintain.yaml").is_file()


def test_the_shipped_maintenance_config_names_rows_the_checker_reads(
    tmp_path: Path,
) -> None:
    config = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
    rows = sorted(
        {
            row
            for line in config.read_text(encoding="utf-8").splitlines()
            if "decision:" in line
            for row in ROW_ID.findall(line)
        }
    )
    log = "| id | result |\n| --- | --- |\n" + "".join(
        f"| {row} | pass |\n" for row in rows
    )
    wiki = build(tmp_path, {**GOOD, "Verification-Log.md": log})

    assert len(rows) >= 10
    assert check_wiki(wiki, verify_ids_from=[config]) == []
    shorter = build(
        tmp_path,
        {**GOOD, "Verification-Log.md": log.replace(f"| {rows[0]} ", "| X ")},
        "w2",
    )
    assert kinds(check_wiki(shorter, verify_ids_from=[config]))[0][2] == "row"


def test_the_command_reads_the_rows_of_a_file_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "maintain.yaml"
    config.write_text(
        'decision: "Wiki page Verification-Log row D-GONE"\n', encoding="utf-8"
    )

    assert main([str(build(tmp_path, GOOD)), "--verify-ids-from", str(config)]) == 1
    assert (
        "maintain.yaml:1: row: D-GONE is not a row of Verification-Log"
        in capsys.readouterr().out
    )
