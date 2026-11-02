# tests/unit/test_annotate_templates.py
"""Regression for Finding 5: the vertical-script toggle button must
actually change the rendered element's own inline writing-mode style (no
browser available in this environment, so this is verified by rendering
each template via Jinja2 directly and checking the emitted markup/JS, not
by driving a real browser -- see the finding's note on not claiming
browser-level UI verification that wasn't actually performed)."""

from __future__ import annotations

from rinchen.annotate.routes import templates

_CHOICES = [
    {"task_id": 10, "text": "candidate A"},
    {"task_id": 11, "text": "candidate B"},
]


def _render(name: str, is_vertical: bool, **extra: object) -> str:
    template = templates.env.get_template(name)
    return template.render(
        task_id=1, src="some source text", is_vertical=is_vertical, **extra
    )


def test_translate_task_toggle_flips_writing_mode_when_vertical():
    html = _render("partials/translate_task.html", is_vertical=True)
    assert "onclick=" in html
    assert "writingMode" in html
    assert "vertical-lr" in html
    assert "horizontal-tb" in html


def test_translate_task_has_no_toggle_when_not_vertical():
    html = _render("partials/translate_task.html", is_vertical=False)
    assert "onclick=" not in html


def test_review_task_toggles_source_and_each_candidate():
    html = _render("partials/review_task.html", is_vertical=True, choices=_CHOICES)
    # One toggle for the source text, plus one per translation candidate.
    assert html.count("onclick=") == 1 + len(_CHOICES)
    assert "writingMode" in html


def test_adjudicate_task_toggles_source_and_each_candidate():
    html = _render("partials/adjudicate_task.html", is_vertical=True, choices=_CHOICES)
    assert html.count("onclick=") == 1 + len(_CHOICES)
    assert "writingMode" in html
