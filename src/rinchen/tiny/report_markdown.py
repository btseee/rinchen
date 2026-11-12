"""The Markdown of a tiny run's report (D15, spec P10), rendered from the `Report` alone.

Blocks separated by one blank line, line feeds only, one final line feed:

- `# Tiny pipeline: <verdict> (profile <profile>)`;
- `## Stages`: a row per stage and, indented by its name (`filter.length`), a row per part of the filter
  and of the evaluation, each with its state and its closed reason; the `length` part carries the note of
  the token-ratio check that was skipped (`detail.token_ratio_check` of the filter), which is an `ok`
  part with a note and not a state of its own;
- `## Thresholds`: id, bound, value, verdict and detail of the eight, and the sentence that says where the
  wall-time seconds are (the report has none);
- `## Held-out`: the line that says these figures are reported and never gated, a row per direction and
  system, and the neural score or the closed reason it was not measured;
- `## Robustness`: the six values beside D12's reference floors (references of that measurement, not gates
  of this run) with the counts each rests on;
- `## Not measured`: the fixed list.

A float beside a bound or a floor (the thresholds and the robustness tables) is written with the fewest
significant digits, four or more, that keep it on its bound's side (`figure_beside_bound`: `1`, `99.36`,
`1.622e-06`, and `49.996` beside `>= 50.0`); any other float (the held-out scores) is written with two
decimals; a null is a hyphen, and every cell is escaped (a line break becomes a space and a backslash, a
pipe and a backtick are escaped) so that nothing can break a table. Nothing comes from the clock, the
environment or a file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.tiny.report import Report
from rinchen.tiny.report_schema import ENGINE_STATUSES, ROBUSTNESS_KEYS, UNMEASURED
from rinchen.tiny.states import PartRecord, StageRecord

HELD_OUT_LINE: Final = (
    "Held-out rows: generalisation on templated sentences, reported and never gated."
)
WALL_TIME_LINE: Final = (
    "The wall-time figure is in manifest.json: "
    "this report holds no duration and no timestamp."
)
ROBUSTNESS_LINE: Final = (
    "The six values of the robustness measurement beside the reference floors of that measurement. "
    "The floors are references and not gates of this run."
)
# D12's floors, as texts: a reference is shown as it was set, not rounded to two decimals.
REFERENCE_FLOORS: Final = {
    "entity_acc": "0.90",
    "mong_source_retention": "0.95",
    "numbers_acc": "0.90",
    "placeholder_acc": "0.99",
    "script_purity": "0.999",
    "term_acc": "0.99",
}
NOTE_KEY: Final = "token_ratio_check"
NOTED_STAGE: Final = "filter"
NOTED_PART: Final = "length"
NEURAL_STAGE: Final = "evaluate"
NEURAL_PART: Final = "neural_seam"
_NONE: Final = "-"
# The digits a figure beside a bound is first written with, and the count at which `repr` takes over
# (a float needs up to 17 digits to read back as itself and `repr` is the shortest text that does).
_SHORT_DIGITS: Final = 4
_LONG_DIGITS: Final = 17
_LINE_BREAKS: Final = re.compile(r"[\r\n]+")


def _cell(text: object) -> str:
    flat = _LINE_BREAKS.sub(" ", str(text))
    return flat.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [header, ["---"] * len(header), *rows]
    return ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in lines]


def _number(value: object) -> str:
    """A figure with no bound beside it (a held-out score): two decimals, a null as a hyphen."""
    if value is None:
        return _NONE
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def figure_beside_bound(
    value: object, bound: str, *, digits: int = _SHORT_DIGITS
) -> str:
    """A figure as its cell writes it beside its `bound` (`>= 50.0`, `<= 0.001`).

    A null is a hyphen and a whole number is itself. A float is the text with the fewest significant
    digits, `digits` or more (four for the Markdown, six for the line the command prints), that read
    back as a number is on the same side of the bound as the float: so a value that fails is never
    written as one that meets its bound, nor the other way round (two decimals wrote 49.996 as `50.00`
    and 237/238 as `1.00` beside `>= 1.0`, both `fail`; six digits wrote 49.99999996 as `50`). A figure
    far from its bound has `digits` of them (`99.36`, `1`, `1.622e-06` at four) and one a hair from it
    as many as show the side, up to `repr`, which reads back as the float itself.

    Raises:
        ValueError: `digits` is not from 1 to 17.
    """
    if not 1 <= digits <= _LONG_DIGITS:
        raise ValueError(f"digits is from 1 to {_LONG_DIGITS}")
    if value is None:
        return _NONE
    if not isinstance(value, float):
        return str(value)
    comparison, _, limit_text = bound.partition(" ")
    limit = float(limit_text)

    def meets(number: float) -> bool:
        return number >= limit if comparison == ">=" else number <= limit

    for count in range(digits, _LONG_DIGITS):
        text = f"{value:.{count}g}"
        if meets(float(text)) == meets(value):
            return text
    return repr(value)


# ---- stages ------------------------------------------------------------------------------------------


def _token_ratio_note(record: StageRecord) -> str:
    note = record.detail.get(NOTE_KEY) if record.id == NOTED_STAGE else None
    return f"{NOTE_KEY} skipped: {note}" if isinstance(note, str) else ""


def _part_row(record: StageRecord, part: PartRecord) -> list[object]:
    note = _token_ratio_note(record) if part.id == NOTED_PART else ""
    return [f"{record.id}.{part.id}", part.state, part.reason or note]


def _stage_rows(report: Report) -> list[list[object]]:
    rows: list[list[object]] = []
    for record in report.stages:
        has_noted_part = any(part.id == NOTED_PART for part in record.parts)
        stage_note = "" if has_noted_part else _token_ratio_note(record)
        rows.append([record.id, record.state, record.reason or stage_note])
        rows.extend(_part_row(record, part) for part in record.parts)
    return rows


def _stages(report: Report) -> str:
    table = _table(["stage", "state", "reason"], _stage_rows(report))
    return "\n".join(["## Stages", "", *table])


# ---- thresholds --------------------------------------------------------------------------------------


def _thresholds(report: Report) -> str:
    rows = [
        [
            result.id,
            result.bound,
            figure_beside_bound(result.value, result.bound),
            result.verdict,
            result.detail,
        ]
        for result in report.thresholds
    ]
    table = _table(["threshold", "bound", "value", "verdict", "detail"], rows)
    return "\n".join(["## Thresholds", "", *table, "", WALL_TIME_LINE])


# ---- held-out ----------------------------------------------------------------------------------------


def _statuses(statuses: Mapping[str, int]) -> str:
    return ", ".join(
        f"{name} {statuses[name]}" for name in ENGINE_STATUSES if name in statuses
    )


def _held_out_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[object]]:
    return [
        [
            run["direction"],
            run["system"],
            run["rows"],
            _number(run["chrf"]),
            _number(run["chrf_pp"]),
            _number(run["bleu"]),
            _statuses(run["statuses"]),
        ]
        for run in runs
    ]


def _neural_reason(report: Report) -> str | None:
    for record in report.stages:
        if record.id == NEURAL_STAGE:
            for part in record.parts:
                if part.id == NEURAL_PART:
                    return part.reason
    return None


def _neural_line(report: Report) -> str:
    block = report.blocks.get("neural_seam")
    if block is None:
        reason = _neural_reason(report)
        return (
            "Neural score: not measured."
            if reason is None
            else f"Neural score: not measured ({reason})."
        )
    rows = block["rows"]
    scores = ", ".join(
        f"{direction} {_number(score)}"
        + (f" over {rows[direction]} rows" if direction in rows else "")
        for direction, score in sorted(block["scores"].items())
    )
    return f"Neural score ({block['metric']} of {block['system']}, planted scores): {scores}."


def _held_out(report: Report) -> str:
    runs = report.blocks.get("heldout")
    parts = ["## Held-out", "", HELD_OUT_LINE, ""]
    if runs is None:
        parts += ["Not measured: no held-out run finished."]
    else:
        header = ["direction", "system", "rows", "chrF", "chrF++", "BLEU", "statuses"]
        parts += _table(header, _held_out_rows(runs))
    return "\n".join([*parts, "", _neural_line(report)])


# ---- robustness --------------------------------------------------------------------------------------


def _robustness(report: Report) -> str:
    block = report.blocks.get("robustness")
    heading = ["## Robustness", ""]
    if block is None:
        return "\n".join([*heading, "Not measured: the robustness run did not finish."])
    rows = [
        [
            key,
            figure_beside_bound(block["values"][key], f">= {REFERENCE_FLOORS[key]}"),
            REFERENCE_FLOORS[key],
            block["coverage"][key]["scored"],
            block["coverage"][key]["total"],
            block["coverage"][key]["excluded"],
        ]
        for key in ROBUSTNESS_KEYS
    ]
    header = ["measure", "value", "reference floor", "scored", "total", "excluded"]
    return "\n".join([*heading, ROBUSTNESS_LINE, "", *_table(header, rows)])


def _not_measured() -> str:
    return "\n".join(["## Not measured", "", *[f"- {item}" for item in UNMEASURED]])


def render(report: Report) -> str:
    """The Markdown of `report`: line feeds only and one final line feed."""
    blocks = [
        f"# Tiny pipeline: {report.verdict} (profile {report.profile})",
        _stages(report),
        _thresholds(report),
        _held_out(report),
        _robustness(report),
        _not_measured(),
    ]
    return "\n\n".join(blocks) + "\n"
