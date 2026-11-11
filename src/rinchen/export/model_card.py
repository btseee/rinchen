"""The model card of D13 (spec T16): `render_card` turns the record of an export, the parsed report, the
facts of the staged files and the repository id into the `README.md` of a release, UTF-8 Markdown with
a YAML front matter. It is a pure function: the same inputs give the same bytes, with LF line ends,
no timestamp, no path and no address.

The card is rendered by our own code and never from the Hub's template: that template has a field that
the project's content rules exclude, and this card has none. The front matter holds four keys,
`language` (the ISO 639-3 codes of the model's languages; the traditional script `mn_Mong` is a script
of Khalkha and has none of its own), `library_name`, `pipeline_tag` and `tags` (with `toy` when every
figure comes from a generated fixture suite), and no `model-index`, `datasets` or `base_model`: the
results are in the body, labelled, and the model is trained from random initialisation, which the card
says. Whether the Hub accepts these values is UNVERIFIED (nothing was run against it).

The honesty rules are those of the sections that carry figures (`rinchen.export.model_card_evaluation`):
the banner is a function of the evidence of the system the card describes (the kinds of the suites in
which that system has a result), every missing value is `n/a (<reason>)`, no ranking is
claimed without its p-value, and the card describes the one system it was asked for. The limits and
the provenance are `rinchen.export.model_card_notes`'.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from rinchen.config.languages import known_language_codes
from rinchen.eval.report_markdown import FIXTURE_BANNER
from rinchen.eval.report_markdown_cells import plural, table
from rinchen.eval.report_read import ReportView
from rinchen.eval.report_view import EvidenceView
from rinchen.eval.result import DirectionResult
from rinchen.export.model_card_evaluation import (
    evaluation_section,
    robustness_section,
    targets_section,
)
from rinchen.export.model_card_notes import limits_section, provenance_section
from rinchen.export.record import ModelRecord

__all__ = ["HUB_LANGUAGE_CODES", "FileFacts", "render_card", "system_evidence"]

_TRADITIONAL_SCRIPT = "mn_Mong"
# The ISO 639-3 code of each registry language but the traditional script of Mongolian: the registry
# code is `<ISO 639-3>_<script>`.
HUB_LANGUAGE_CODES: Final = MappingProxyType(
    {
        code: code.partition("_")[0]
        for code in sorted(known_language_codes())
        if code != _TRADITIONAL_SCRIPT
    }
)
_TAGS: Final = ("rinchen", "translation", "ctranslate2", "mongolian")
_PLACEMENT: Final = {
    "src": "The tag of the target language goes in the source (`src`).",
    "tgt": (
        "The tag of the target language is forced as the first piece of the target (`tgt`)."
    ),
}
_FIXTURE_TAIL: Final = (
    "The suites of the kind fixture are generated: they test the tooling and say nothing "
    "about translation quality."
)
_MEASURED_TAIL: Final = (
    "Each figure is a measurement of the system named, on the suite named, and says nothing "
    "about other text."
)
_FILES_NOTE: Final = (
    "`README.md` is this card and `release.json` lists the digest of every other file, this "
    "card included."
)
_USAGE_TARGET: Final = (
    "khk_Cyrl"  # the example asks for Khalkha when the model writes it
)


@dataclass(frozen=True, slots=True)
class FileFacts:
    """A staged file: its `name`, its size in `bytes` and its `sha256`."""

    name: str
    bytes: int
    sha256: str


def render_card(
    record: ModelRecord,
    report: ReportView,
    system: str,
    files: Sequence[FileFacts],
    repo_id: str,
    *,
    report_sha256: str,
    export_manifest_sha256: str,
) -> str:
    """The card of the model in `record` and of `system` of `report`: `files` are the staged files
    it lists (the card cannot list itself or the release record, which lists its digest), `repo_id`
    the repository it is for, `report_sha256` the digest of `report.json` and
    `export_manifest_sha256` that of the export's manifest.

    Raises:
        ValueError: the report holds no result of `system`.
    """
    results = [result for result in report.results if result.system == system]
    if not results:
        raise ValueError(f"the report holds no result of the system {system!r}")
    evidence = system_evidence(report, system)
    sections = [
        _front_matter(record, evidence),
        [f"# {record.name}", "", f"> {_banner(evidence, results)}", ""],
        _what_it_is(record, repo_id),
        _directions(record),
        _files(files),
        _usage(record),
        evaluation_section(record, report, system, results),
        robustness_section(report, system),
        targets_section(report),
        limits_section(report, system, results),
        provenance_section(
            record,
            report_sha256=report_sha256,
            export_manifest_sha256=export_manifest_sha256,
        ),
    ]
    return "\n".join(line for section in sections for line in section) + "\n"


def system_evidence(report: ReportView, system: str) -> EvidenceView:
    """The evidence of `system`: the kinds of the suites in which it has a result (a direction
    that ran), sorted, and whether every one of them is a fixture. The report's own evidence also
    counts the suites of the other systems, and a card must not borrow from them: a system whose
    every table is from a fixture suite stays a fixture card when another system of the report
    was measured on a benchmark."""
    ran_in = {result.suite for result in report.results if result.system == system}
    kinds = tuple(
        sorted({suite.kind for suite in report.suites if suite.name in ran_in})
    )
    return EvidenceView(kinds, kinds == ("fixture",))


def _front_matter(record: ModelRecord, evidence: EvidenceView) -> list[str]:
    languages = sorted(
        {
            HUB_LANGUAGE_CODES[code]
            for code in record.languages
            if code in HUB_LANGUAGE_CODES
        }
    )
    tags = [*_TAGS, *(["toy"] if evidence.fixture_only else [])]
    return [
        "---",
        "language:",
        *(f"- {code}" for code in languages),
        "library_name: ctranslate2",
        "pipeline_tag: translation",
        "tags:",
        *(f"- {tag}" for tag in tags),
        "---",
        "",
    ]


def _banner(evidence: EvidenceView, results: Sequence[DirectionResult]) -> str:
    if evidence.fixture_only:
        return FIXTURE_BANNER
    kinds = evidence.kinds
    measured = len({result.direction for result in results})
    were = "was" if measured == 1 else "were"
    opening = (
        f"Evidence: suites of the {'kind' if len(kinds) == 1 else 'kinds'} "
        f"{', '.join(kinds)}. {plural(measured, 'direction')} {were} measured for this system."
    )
    return f"{opening} {_FIXTURE_TAIL if 'fixture' in kinds else _MEASURED_TAIL}"


def _what_it_is(record: ModelRecord, repo_id: str) -> list[str]:
    tags = ", ".join(f"{code} `{tag}`" for code, tag in sorted(record.tags.items()))
    converter = (
        ""
        if _TRADITIONAL_SCRIPT in record.tags
        else (
            " The traditional script (`Mong`) is not written by this export alone: it needs "
            "the script converter."
        )
    )
    return [
        "## What it is",
        "",
        (
            f"{record.name} is a translation model of the Rinchen project, exported for "
            f"CTranslate2 ({record.quantization}). Its transformer has "
            f"{plural(record.encoder_layers, 'encoder layer')}, "
            f"{plural(record.decoder_layers, 'decoder layer')}, hidden size "
            f"{record.hidden_size} and {record.vocabulary_entries} vocabulary entries. It was "
            "trained from random initialisation: no pretrained model is its base. That is the "
            "project's rule; the export's record cannot prove it, and the checkpoint's "
            "manifest stage is named under Provenance."
        ),
        "",
        f"Document level: {'yes' if record.document_level else 'no'}.",
        "",
        (
            f"{_PLACEMENT[record.tag_placement]} Targets and their tags: {tags}. The default "
            f"script of a Mongolian target is Cyrillic (`Cyrl`).{converter}"
        ),
        "",
        f"Repository: `{repo_id}`.",
        "",
    ]


def _directions(record: ModelRecord) -> list[str]:
    count = plural(len(record.directions), "direction")
    return [
        "## Directions",
        "",
        f"Trained for {count}: {', '.join(record.directions)}.",
        "",
    ]


def _files(files: Sequence[FileFacts]) -> list[str]:
    rows = [[fact.name, str(fact.bytes), fact.sha256] for fact in files]
    return ["## Files", "", *table(["file", "bytes", "sha256"], rows), _FILES_NOTE, ""]


def _usage(record: ModelRecord) -> list[str]:
    target = _USAGE_TARGET if _USAGE_TARGET in record.tags else record.targets[0]
    return [
        "## Usage",
        "",
        "```python",
        "import rinchen",
        "",
        'with rinchen.Translator("path/to/this/directory") as translator:',
        f'    result = translator.translate("text to translate", tgt="{target}")',
        "print(result.text)",
        "```",
        "",
    ]
