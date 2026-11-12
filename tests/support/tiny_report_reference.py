"""An independent statement of the report's bytes and Markdown (D15 Task 6, the differential check).

`reference_bytes` builds the document of a `Report` from its fields by hand and writes it with
`json.dumps` of the standard library; `reference_markdown` writes the Markdown line by line. Neither uses
`report_document`, `to_bytes`, `render` or any helper of `rinchen.tiny.report*`: they are the layout of
spec P10 and of the heading list of the task stated a second time, with their own constants, and the
test compares them with the code on a thousand random reports.
"""

from __future__ import annotations

import json
from typing import Any

from rinchen.tiny.report import Report

SIX = (
    "entity_acc",
    "mong_source_retention",
    "numbers_acc",
    "placeholder_acc",
    "script_purity",
    "term_acc",
)
FLOOR_TEXT = ("0.90", "0.95", "0.90", "0.99", "0.999", "0.99")
ENGINE = ("ok", "empty", "too_long", "truncated", "no_output")
NOT_MEASURED = [
    "any run on a GPU: no stage touched a device",
    "a real corpus: the fixtures are templated sentences",
    "a real generator behind the synthetic data stage",
    "neural metric scorers: only the seam of score files ran, with planted scores",
    "GitHub Actions: nothing ran there",
    "the time on a 4-core CPU runner",
]


def _round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def reference_document(report: Report) -> dict[str, Any]:
    failing = [s for s in report.stages if s.state == "fail"]
    not_passing = [t for t in report.thresholds if t.verdict != "pass"]
    stages = []
    for stage in report.stages:
        parts = [
            {"id": p.id, "state": p.state, "reason": p.reason} for p in stage.parts
        ]
        stages.append(
            {
                "id": stage.id,
                "state": stage.state,
                "reason": stage.reason,
                "detail": _round_trip(dict(stage.detail)),
                "parts": parts,
            }
        )
    thresholds = [
        {
            "id": t.id,
            "bound": t.bound,
            "value": t.value,
            "verdict": t.verdict,
            "detail": t.detail,
        }
        for t in report.thresholds
    ]
    return {
        "schema": "rinchen.tiny/1",
        "profile": report.profile,
        "qe_fake": report.qe_fake,
        "verdict": "fail" if failing or not_passing else "pass",
        "inputs": {
            "config_sha256": report.inputs.config_sha256,
            "fixtures": dict(report.inputs.fixtures),
            "lock_sha256": report.inputs.lock_sha256,
        },
        "stages": stages,
        "thresholds": thresholds,
        "blocks": _round_trip(dict(report.blocks)),
        "unmeasured": list(NOT_MEASURED),
    }


def reference_bytes(report: Report) -> bytes:
    return (
        json.dumps(reference_document(report), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _two(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "%.2f" % value  # noqa: UP031 - a second way of writing two decimals
    return str(value)


def _beside(value: Any, bound: str) -> str:
    """A figure beside its bound (`>= 50.0`), by its own statement of the rule: take the first of the
    texts with 4, 5 ... 16 significant digits that reads back on the same side of the bound as the
    figure, else the text that reads back as the figure itself. A null is a hyphen and a whole number
    is itself."""
    if value is None:
        return "-"
    if not isinstance(value, float):
        return str(value)
    sign, limit_text = bound.split(" ")
    edge = float(limit_text)

    def meets(number: float) -> bool:
        return number >= edge if sign == ">=" else number <= edge

    for digits in range(4, 17):
        text = format(value, "." + str(digits) + "g")
        if meets(float(text)) == meets(value):
            return text
    return repr(value)


def _row(cells: list[Any]) -> str:
    line = "|"
    for cell in cells:
        text = str(cell).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")
        line += " " + text + " |"
    return line


def _table(header: list[str], body: list[list[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header))] + [_row(cells) for cells in body]


def reference_markdown(report: Report) -> str:
    document = reference_document(report)
    blocks = document["blocks"]
    out: list[list[str]] = []
    out.append(
        [
            "# Tiny pipeline: "
            + document["verdict"]
            + " (profile "
            + document["profile"]
            + ")"
        ]
    )

    rows: list[list[Any]] = []
    for stage in document["stages"]:
        note = ""
        has_length = any(p["id"] == "length" for p in stage["parts"])
        token_note = (
            stage["detail"].get("token_ratio_check")
            if stage["id"] == "filter"
            else None
        )
        if isinstance(token_note, str):
            note = "token_ratio_check skipped: " + token_note
        rows.append(
            [
                stage["id"],
                stage["state"],
                stage["reason"] or ("" if has_length else note),
            ]
        )
        for part in stage["parts"]:
            cell = part["reason"] or (note if part["id"] == "length" else "")
            rows.append([stage["id"] + "." + part["id"], part["state"], cell])
    out.append(["## Stages", ""] + _table(["stage", "state", "reason"], rows))

    rows = [
        [
            t["id"],
            t["bound"],
            _beside(t["value"], t["bound"]),
            t["verdict"],
            t["detail"],
        ]
        for t in document["thresholds"]
    ]
    out.append(
        ["## Thresholds", ""]
        + _table(["threshold", "bound", "value", "verdict", "detail"], rows)
        + [
            "",
            "The wall-time figure is in manifest.json: this report holds no duration and no timestamp.",
        ]
    )

    held = [
        "## Held-out",
        "",
        "Held-out rows: generalisation on templated sentences, reported and never gated.",
        "",
    ]
    if blocks["heldout"] is None:
        held.append("Not measured: no held-out run finished.")
    else:
        body = []
        for run in blocks["heldout"]:
            statuses = ", ".join(
                s + " " + str(run["statuses"][s])
                for s in ENGINE
                if s in run["statuses"]
            )
            body.append(
                [
                    run["direction"],
                    run["system"],
                    run["rows"],
                    _two(run["chrf"]),
                    _two(run["chrf_pp"]),
                    _two(run["bleu"]),
                    statuses,
                ]
            )
        held += _table(
            ["direction", "system", "rows", "chrF", "chrF++", "BLEU", "statuses"], body
        )
    held.append("")
    seam = blocks["neural_seam"]
    if seam is None:
        why = None
        for stage in document["stages"]:
            if stage["id"] == "evaluate":
                for part in stage["parts"]:
                    if part["id"] == "neural_seam":
                        why = part["reason"]
        held.append(
            "Neural score: not measured."
            if why is None
            else "Neural score: not measured (" + why + ")."
        )
    else:
        pieces = []
        for direction in sorted(seam["scores"]):
            piece = direction + " " + _two(seam["scores"][direction])
            if direction in seam["rows"]:
                piece += " over " + str(seam["rows"][direction]) + " rows"
            pieces.append(piece)
        held.append(
            "Neural score ("
            + seam["metric"]
            + " of "
            + seam["system"]
            + ", planted scores): "
            + ", ".join(pieces)
            + "."
        )
    out.append(held)

    robust = ["## Robustness", ""]
    if blocks["robustness"] is None:
        robust.append("Not measured: the robustness run did not finish.")
    else:
        robust.append(
            "The six values of the robustness measurement beside the reference floors of that measurement. "
            "The floors are references and not gates of this run."
        )
        robust.append("")
        body = []
        for key, floor in zip(SIX, FLOOR_TEXT, strict=True):
            count = blocks["robustness"]["coverage"][key]
            body.append(
                [
                    key,
                    _beside(blocks["robustness"]["values"][key], ">= " + floor),
                    floor,
                    count["scored"],
                    count["total"],
                    count["excluded"],
                ]
            )
        robust += _table(
            ["measure", "value", "reference floor", "scored", "total", "excluded"], body
        )
    out.append(robust)

    out.append(["## Not measured", ""] + ["- " + item for item in NOT_MEASURED])
    return "\n\n".join("\n".join(block) for block in out) + "\n"
