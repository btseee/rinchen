"""The issue forms and the configuration of the new-issue page (`.github/ISSUE_TEMPLATE`).

A form is YAML in the schema of GitHub's issue forms: `name`, `description`, `title`, `labels` and a
`body` of elements of the types `markdown`, `input`, `textarea`, `dropdown` and `checkboxes`, each with
the `attributes` its type allows and, for a field, an `id` and a `label`. GitHub refuses a form that
breaks the schema and shows nothing in its place, so the schema is pinned here from its documentation
(the keys, the types, unique ids and labels, options that are not empty). The files are also covered
by the checks of `test_github_community_files.py` that apply to every new file (ASCII, the content
rules, the newline at the end); this module pins what only a form has: which questions it asks, which
of them are required, that none asks for a secret, and that the accessibility form never requires a
reporter to say why they use an assistive tool. Nothing here starts a process or opens a socket.
"""

from __future__ import annotations

import re
from typing import Final

import pytest

from tests.support.community_files import (
    ACCESSIBILITY_FORM,
    ADVISORY_URL,
    BUG_REPORT,
    DOCUMENTATION,
    FEATURE_REQUEST,
    ISSUE_CONFIG,
    ISSUE_FORMS,
    URL,
    WIKI_URL,
    fields,
    form_items,
    load_yaml,
    read_text,
)

FIELD_TYPES: Final = frozenset({"input", "textarea", "dropdown", "checkboxes"})
# The keys GitHub's schema allows in `attributes` of each type of form element.
ATTRIBUTE_KEYS: Final = {
    "markdown": {"value"},
    "input": {"label", "description", "placeholder", "value"},
    "textarea": {"label", "description", "placeholder", "value", "render"},
    "dropdown": {"label", "description", "multiple", "options", "default"},
    "checkboxes": {"label", "description", "options"},
}
# What a field of a form must never ask a reporter for.
SECRET_WORDS: Final = re.compile(
    r"token|password|passphrase|secret|credential|api[ _-]?key|private key|\bssh\b",
    re.IGNORECASE,
)
# Nor, in the accessibility form, a reason for the use of an assistive tool.
DISCLOSURE_WORDS: Final = re.compile(
    r"disabilit|diagnos|condition|impairment", re.IGNORECASE
)


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_a_form_has_the_keys_of_github_issue_forms_schema(path: str) -> None:
    form = load_yaml(path)

    assert set(form) <= {"name", "description", "title", "labels", "body", "type"}
    for key in ("name", "description", "title"):
        assert isinstance(form[key], str) and form[key].strip(), key
    assert form["labels"] == ISSUE_FORMS[path]
    assert isinstance(form["body"], list) and form["body"]


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_every_element_of_a_form_has_what_the_schema_needs(path: str) -> None:
    items = form_items(path)

    assert any(item["type"] in FIELD_TYPES for item in items)
    for item in items:
        kind = item["type"]
        assert kind in ATTRIBUTE_KEYS, kind
        attributes = item["attributes"]
        assert set(attributes) <= ATTRIBUTE_KEYS[kind], (kind, set(attributes))
        if kind == "markdown":
            assert str(attributes["value"]).strip()
            assert "validations" not in item
            continue
        assert re.fullmatch(r"[A-Za-z0-9_-]+", str(item["id"])), item.get("id")
        assert str(attributes["label"]).strip(), item["id"]
        if "validations" in item:
            assert kind != "checkboxes"
            assert set(item["validations"]) == {"required"}
            assert isinstance(item["validations"]["required"], bool)


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_the_ids_and_the_labels_of_a_form_are_each_unique(path: str) -> None:
    asking = [item for item in form_items(path) if item["type"] != "markdown"]
    ids = [item["id"] for item in asking]
    labels = [item["attributes"]["label"] for item in asking]

    assert len(set(ids)) == len(ids)
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_the_options_of_a_dropdown_and_of_checkboxes_are_not_empty(path: str) -> None:
    for item in form_items(path):
        if item["type"] == "dropdown":
            options = item["attributes"]["options"]
            assert isinstance(options, list) and len(options) >= 2
            assert all(isinstance(o, str) and o.strip() for o in options)
            assert len(set(options)) == len(options)
        if item["type"] == "checkboxes":
            options = item["attributes"]["options"]
            assert isinstance(options, list) and options
            for option in options:
                assert str(option["label"]).strip()
                assert isinstance(option.get("required", False), bool)


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_no_field_of_a_form_asks_for_a_secret(path: str) -> None:
    for item in form_items(path):
        if item["type"] != "markdown":
            asked = f"{item['id']} {item['attributes']['label']}"
            assert SECRET_WORDS.search(asked) is None, item["id"]


@pytest.mark.parametrize("path", ISSUE_FORMS)
def test_a_form_links_only_over_https_and_to_the_wiki_home(path: str) -> None:
    urls = URL.findall(read_text(path))

    assert all(url.startswith("https://") for url in urls)
    assert {url.rstrip(".,;:") for url in urls if "/wiki" in url} <= {WIKI_URL}


def test_the_configuration_closes_blank_issues_and_points_to_the_wiki_and_the_advisory() -> (
    None
):
    configuration = load_yaml(ISSUE_CONFIG)
    contacts = configuration["contact_links"]

    assert configuration["blank_issues_enabled"] is False
    assert set(configuration) == {"blank_issues_enabled", "contact_links"}
    assert isinstance(contacts, list) and len(contacts) >= 2
    for contact in contacts:
        assert set(contact) == {"name", "url", "about"}
        assert all(str(value).strip() for value in contact.values())
        assert contact["url"].startswith("https://")
    assert {WIKI_URL, ADVISORY_URL} <= {contact["url"] for contact in contacts}


def test_the_bug_form_asks_what_the_maintainers_need_to_reproduce_a_problem() -> None:
    asked = fields(BUG_REPORT)
    needed = {"what-happened", "expected", "steps", "doctor", "versions", "os"}

    assert needed <= set(asked)
    for name in ("what-happened", "expected", "steps", "doctor", "os"):
        assert asked[name]["validations"]["required"] is True, name
    assert "uv run rinchen doctor" in asked["doctor"]["attributes"]["label"]
    assert asked["os"]["type"] == "dropdown"


def test_the_bug_form_tells_reporters_to_keep_secrets_out_and_where_security_goes() -> (
    None
):
    first = form_items(BUG_REPORT)[0]

    assert first["type"] == "markdown"
    assert ADVISORY_URL in first["attributes"]["value"]
    assert "secret" in first["attributes"]["value"].lower()


def test_the_feature_form_asks_for_the_problem_a_solution_and_the_area() -> None:
    asked = fields(FEATURE_REQUEST)

    assert {"problem", "solution", "alternatives", "area"} <= set(asked)
    assert asked["problem"]["validations"]["required"] is True
    assert asked["solution"]["validations"]["required"] is True
    assert asked["area"]["type"] == "dropdown"
    areas = " ".join(asked["area"]["attributes"]["options"]).lower()
    for wanted in (
        "filtering",
        "tokenizer",
        "script converter",
        "serving",
        "evaluation",
    ):
        assert wanted in areas, wanted


def test_the_documentation_form_asks_for_the_page_and_what_is_missing_or_wrong() -> (
    None
):
    asked = fields(DOCUMENTATION)

    assert {"location", "problem"} <= set(asked)
    assert asked["location"]["validations"]["required"] is True
    assert asked["problem"]["validations"]["required"] is True
    assert "missing or wrong" in asked["problem"]["attributes"]["label"]


def test_the_accessibility_form_asks_for_the_barrier_where_it_is_and_the_tool_used() -> (
    None
):
    asked = fields(ACCESSIBILITY_FORM)

    assert {"barrier", "where", "assistive-technology", "fix"} <= set(asked)
    assert asked["barrier"]["validations"]["required"] is True
    assert asked["where"]["validations"]["required"] is True
    places = " ".join(asked["where"]["attributes"]["options"])
    for wanted in ("CLI output", "report.md", "Wiki", "README", "HTTP service"):
        assert wanted in places, wanted


def test_the_accessibility_form_does_not_ask_for_a_diagnosis_or_require_the_tool() -> (
    None
):
    asked = fields(ACCESSIBILITY_FORM)

    validations = asked["assistive-technology"].get("validations", {})
    assert validations.get("required", False) is False
    for item in asked.values():
        label = str(item["attributes"]["label"])
        assert DISCLOSURE_WORDS.search(label) is None, label
    first = form_items(ACCESSIBILITY_FORM)[0]
    assert "do not have to" in first["attributes"]["value"].lower()
