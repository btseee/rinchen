"""The model card of D13 (Task 9, spec T16): `render_card` is a pure function of the export's record,
the parsed report, the facts of the staged files and the repository id, rendered by our own code (the
Hub's template is never used: it has a field the project's content rules exclude).

The card of the fabricated release (`alpha` of the golden report as a Rinchen system, the six
directions of D12's toy) is `tests/fixtures/eval/golden_card.md` byte for byte; the file was written
from an independent stdlib-only rendering of the layout, after the literals below held. Its digests
and sizes are the fixed ones of `tests.support.release_documents` (the digests of a fabricated export
hold the directory it was made in). What the card claims is limited to what a table shows; the
honesty rules, the numbers and the variants of the report are in `test_export_model_card_honesty.py`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
import yaml

from rinchen.config.languages import known_language_codes
from rinchen.eval.report_markdown import FIXTURE_BANNER
from rinchen.export.model_card import HUB_LANGUAGE_CODES, FileFacts, render_card
from rinchen.export.record import ModelRecord
from tests.support.eval_report_golden import FIXTURES
from tests.support.model_exports import checkpoint_record, write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    GOLDEN_REPORT_SHA256,
    REPO_ID,
    SYSTEM,
    as_benchmark,
    bound_document,
    report_bytes,
    view_of,
    with_a_run_in_another_suite,
)
from tests.support.release_fixture import (
    NO_SCORER,
    SIX_DIRECTIONS,
    card_of,
    golden_facts,
)

pytestmark = pytest.mark.usefixtures("no_network")

GOLDEN_CARD: Final = FIXTURES / "golden_card.md"
GOLDEN_SIGNATURES: Final = {
    "BLEU": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0",
    "chrF": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0",
    "chrF++": "nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0",
}
FRONT_MATTER: Final = (
    "---\n"
    "language:\n"
    "- eng\n"
    "- khk\n"
    "- rus\n"
    "library_name: ctranslate2\n"
    "pipeline_tag: translation\n"
    "tags:\n"
    "- rinchen\n"
    "- translation\n"
    "- ctranslate2\n"
    "- mongolian\n"
    "- toy\n"
    "---\n"
)
SECTIONS: Final = (
    "## What it is",
    "## Directions",
    "## Files",
    "## Usage",
    "## Evaluation",
    "## Robustness",
    "## Release targets",
    "## Limits",
    "## Provenance",
)


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    """The record of a fabricated export of the six directions of D12's toy."""
    export = write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    )
    return export.record


def front_matter_of(card: str) -> tuple[str, str]:
    assert card.startswith("---\n")
    head, separator, body = card[len("---\n") :].partition("\n---\n")
    assert separator
    return head, body


# ---- the golden card ----------------------------------------------------------------------------


def test_the_card_of_the_fabricated_release_is_the_golden_card_byte_for_byte(
    record: ModelRecord,
) -> None:
    assert card_of(record).encode("utf-8") == GOLDEN_CARD.read_bytes()


def test_the_golden_card_is_under_eight_kilobytes_and_has_line_feeds_only() -> None:
    data = GOLDEN_CARD.read_bytes()

    assert len(data) < 8 * 1024
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")


def test_the_fixtures_of_the_evaluation_stay_under_fifty_kilobytes_together() -> None:
    # The golden plan of Task 10 (about 3 KB) is still to come.
    total = sum(path.stat().st_size for path in FIXTURES.iterdir() if path.is_file())

    assert total < 47 * 1024


# ---- the front matter ---------------------------------------------------------------------------


def test_the_front_matter_is_exactly_these_lines(record: ModelRecord) -> None:
    assert card_of(record).startswith(FRONT_MATTER)


def test_the_front_matter_has_four_keys_and_no_other(record: ModelRecord) -> None:
    head, _ = front_matter_of(card_of(record))

    assert yaml.safe_load(head) == {
        "language": ["eng", "khk", "rus"],
        "library_name": "ctranslate2",
        "pipeline_tag": "translation",
        "tags": ["rinchen", "translation", "ctranslate2", "mongolian", "toy"],
    }


def test_the_card_has_no_model_index_datasets_or_base_model(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    for key in ("model-index", "datasets", "base_model"):
        assert key not in card


def test_the_languages_are_the_iso_639_3_prefixes_of_the_languages_of_the_record_and_never_mn_mong(
    record: ModelRecord,
) -> None:
    assert "mn_Mong" in record.languages

    head, _ = front_matter_of(card_of(record))

    assert yaml.safe_load(head)["language"] == ["eng", "khk", "rus"]


def test_the_language_table_is_the_registry_without_the_traditional_script() -> None:
    assert dict(HUB_LANGUAGE_CODES) == {
        "khk_Cyrl": "khk",
        "eng_Latn": "eng",
        "rus_Cyrl": "rus",
        "zho_Hans": "zho",
        "kor_Hang": "kor",
        "jpn_Jpan": "jpn",
        "deu_Latn": "deu",
        "kaz_Cyrl": "kaz",
        "tur_Latn": "tur",
    }
    assert set(HUB_LANGUAGE_CODES) == known_language_codes() - {"mn_Mong"}


def test_the_language_list_of_another_model_is_sorted_and_without_repeats(
    tmp_path: Path,
) -> None:
    other = write_model_export(
        tmp_path / "export", directions=("khk_Cyrl-zho_Hans", "zho_Hans-khk_Cyrl")
    ).record

    head, _ = front_matter_of(card_of(other))

    assert yaml.safe_load(head)["language"] == ["khk", "zho"]


def test_the_toy_tag_is_there_only_when_every_suite_is_a_fixture(
    record: ModelRecord,
) -> None:
    measured = bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)

    head, _ = front_matter_of(card_of(record, as_benchmark(measured)))

    assert yaml.safe_load(head)["tags"] == [
        "rinchen",
        "translation",
        "ctranslate2",
        "mongolian",
    ]


# ---- the title and the banner -------------------------------------------------------------------


def test_the_title_is_the_name_of_the_model_and_the_banner_follows_it(
    record: ModelRecord,
) -> None:
    _, body = front_matter_of(card_of(record))

    assert body.startswith(
        f"\n# {record.name}\n\n> {FIXTURE_BANNER}\n\n## What it is\n"
    )


def test_the_fixture_banner_says_what_the_report_says() -> None:
    assert FIXTURE_BANNER == (
        "These figures come from generated fixture suites. They test the tooling and "
        "say nothing about translation quality."
    )


def test_the_banner_of_a_measured_report_names_the_kinds_and_the_directions_measured(
    record: ModelRecord,
) -> None:
    card = card_of(
        record,
        as_benchmark(
            bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)
        ),
    )

    assert (
        "> Evidence: suites of the kind benchmark. 2 directions were measured for this "
        "system. Each figure is a measurement of the system named, on the suite named, "
        "and says nothing about other text.\n"
    ) in card
    assert FIXTURE_BANNER not in card


def test_the_banner_of_a_mixed_report_says_that_the_fixture_suites_are_generated(
    record: ModelRecord,
) -> None:
    card = card_of(
        record,
        with_a_run_in_another_suite(
            bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256),
            SYSTEM,
            name="golden-human",
            kind="human",
        ),
    )

    assert (
        "> Evidence: suites of the kinds fixture, human. 2 directions were measured for "
        "this system. The suites of the kind fixture are generated: they test the "
        "tooling and say nothing about translation quality.\n"
    ) in card


# ---- the sections -------------------------------------------------------------------------------


def test_the_sections_come_in_this_order(record: ModelRecord) -> None:
    card = card_of(record)

    positions = [card.index(f"\n{heading}\n") for heading in SECTIONS]

    assert positions == sorted(positions)


def test_the_card_says_what_the_model_is_and_that_it_was_trained_from_random_initialisation(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        f"{record.name} is a translation model of the Rinchen project, exported for "
        "CTranslate2 (float32). Its transformer has 2 encoder layers, 1 decoder layer, "
        f"hidden size 64 and {record.vocabulary_entries} vocabulary entries."
    ) in card
    assert "trained from random initialisation" in card
    assert "the export's record cannot prove it" in card
    assert "Document level: no." in card


def test_the_card_names_the_tag_placement_the_targets_and_the_default_script(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert "The tag of the target language goes in the source (`src`)." in card
    assert (
        "Targets and their tags: eng_Latn `<2eng_Latn>`, khk_Cyrl `<2khk_Cyrl>`, "
        "rus_Cyrl `<2rus_Cyrl>`."
    ) in card
    assert "The default script of a Mongolian target is Cyrillic (`Cyrl`)." in card
    assert f"Repository: `{REPO_ID}`." in card


def test_the_directions_section_lists_what_the_model_was_trained_for(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "## Directions\n\n"
        "Trained for 6 directions: eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn, "
        "khk_Cyrl-rus_Cyrl, mn_Mong-eng_Latn, mn_Mong-rus_Cyrl, rus_Cyrl-khk_Cyrl.\n"
    ) in card


def test_the_files_section_lists_seven_files_with_their_full_digests(
    record: ModelRecord,
) -> None:
    card = card_of(record)
    rows = [
        f"| {fact.name} | {fact.bytes} | {fact.sha256} |" for fact in golden_facts()
    ]

    assert (
        "| file | bytes | sha256 |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"
        in card
    )
    assert len(rows) == 7
    assert all(len(fact.sha256) == 64 for fact in golden_facts())
    assert "`README.md` is this card" in card
    assert "`release.json` lists the digest of every other file" in card


def test_the_usage_section_is_the_translator_and_invents_no_address(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "## Usage\n\n"
        "```python\n"
        "import rinchen\n\n"
        'with rinchen.Translator("path/to/this/directory") as translator:\n'
        '    result = translator.translate("text to translate", tgt="khk_Cyrl")\n'
        "print(result.text)\n"
        "```\n"
    ) in card
    assert "http" not in card


# ---- the evaluation tables ----------------------------------------------------------------------


def test_the_evaluation_section_names_the_suite_its_kind_split_and_rows(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    for line in (
        "### Suite golden",
        "- kind: fixture",
        "- split: devtest",
        "- rows: 10",
    ):
        assert f"\n{line}\n" in card


def test_the_evaluation_section_names_the_bootstrap_and_the_three_signatures(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert "1,000 resamples, seed 20261110, percentile 95 %" in card
    for metric, signature in GOLDEN_SIGNATURES.items():
        assert f"\n- {metric} signature: `{signature}`\n" in card


def test_the_table_has_a_row_for_each_direction_that_was_measured_and_nothing_else(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "| direction | segments | outputs ok | resampling units | chrF++ | chrF | BLEU |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| khk_Cyrl-eng_Latn | 10 | 10 of 10 | 5 document units | 73.08 [54.91, 85.48] | "
        "72.44 [53.76, 85.63] | 59.40 [36.34, 72.14] |\n"
        "| eng_Latn-khk_Cyrl | 10 | 10 of 10 | 5 document units | 68.06 [50.67, 80.98] | "
        "67.41 [49.84, 80.59] | 50.87 [25.54, 67.88] |\n"
    ) in card


def test_a_direction_the_model_was_trained_for_that_the_suite_lacks_is_n_a_not_in_the_suite(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "\nn/a (not in the suite): khk_Cyrl-rus_Cyrl, rus_Cyrl-khk_Cyrl, "
        "mn_Mong-eng_Latn, mn_Mong-rus_Cyrl.\n"
    ) in card


def test_a_direction_the_model_was_not_trained_for_is_n_a_not_trained(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "\nn/a (not trained): khk_Cyrl-deu_Latn, deu_Latn-khk_Cyrl, khk_Cyrl-jpn_Jpan, "
        "jpn_Jpan-khk_Cyrl, khk_Cyrl-kaz_Cyrl, kaz_Cyrl-khk_Cyrl, khk_Cyrl-kor_Hang, "
        "kor_Hang-khk_Cyrl, khk_Cyrl-tur_Latn, tur_Latn-khk_Cyrl, khk_Cyrl-zho_Hans, "
        "zho_Hans-khk_Cyrl.\n"
    ) in card


def test_the_metrics_that_need_segment_scores_are_n_a_with_the_reason(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (f"\nCOMET-22, MetricX-24, MetricX-24-QE: n/a ({NO_SCORER}).\n") in card


def test_a_system_that_is_the_baseline_of_every_comparison_has_none_as_a_candidate(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "### Paired comparisons\n\n"
        "n/a (no comparison of the report has this system as the candidate)\n"
    ) in card


def test_the_robustness_section_of_a_report_without_a_run_says_so(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert "## Robustness\n\nn/a (no robustness run was attached)\n" in card


def test_the_release_targets_are_the_reports_with_their_status_and_basis(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    assert (
        "| target | status | threshold | value |\n"
        "| --- | --- | --- | --- |\n"
        "| beats_baseline | unmeasured | 12 | n/a (0 of 16 directions measured) |\n"
        "| off_target | unmeasured | 0.01 | n/a (no language identification is run) |\n"
        "| numbers | unmeasured | 0.99 | "
        "n/a (no robustness run of a rinchen system was given) |\n"
        "| placeholders | unmeasured | 1 | "
        "n/a (no robustness run of a rinchen system was given) |\n"
    ) in card


def test_the_limits_are_a_list_of_facts(record: ModelRecord) -> None:
    card = card_of(record)

    assert (
        "## Limits\n\n"
        "- The model is trained for the directions listed above and no others. Only the "
        "directions in the tables above are measured; a pair the model was not trained for "
        "(zero-shot) is never evaluated, and nothing here says that its output is in the "
        "language asked for.\n"
        "- A dev split is used for model selection only: the report refuses a dev run, so "
        "every figure here is from a devtest or test split.\n"
        "- Each interval resamples the unit its table names: a document when every row of a "
        "direction has a document id, a segment otherwise.\n"
        "- 16 intervals and 4 paired comparisons are reported with no correction for "
        "multiple comparisons. Read a difference with its p-value.\n"
        "- No run on a GPU is recorded in this report.\n"
        "- Not measured: COMET-22, MetricX-24 and MetricX-24-QE (no scorer is built), the "
        "share of outputs in the wrong language (no language identification is run) and "
        "real-world input beyond the suites above.\n"
    ) in card


def test_the_provenance_holds_the_checkpoint_the_export_manifest_and_the_report_in_full(
    record: ModelRecord,
) -> None:
    card = card_of(record)
    checkpoint = record.checkpoint

    assert (
        "## Provenance\n\n"
        "- rinchen_version: 0.0.0\n"
        "- checkpoint kind: step\n"
        "- checkpoint step: 200\n"
        "- checkpoint manifest stage: train_launch\n"
        f"- checkpoint weights sha256: {checkpoint.weights_sha256}\n"
        f"- checkpoint config sha256: {checkpoint.config_sha256}\n"
        f"- checkpoint vocabulary sha256: {checkpoint.vocab_sha256}\n"
        f"- checkpoint manifest sha256: {checkpoint.manifest_sha256}\n"
        f"- export manifest sha256: {GOLDEN_EXPORT_MANIFEST_SHA256}\n"
        f"- report.json sha256: {GOLDEN_REPORT_SHA256}\n"
    ) in card


def test_the_provenance_of_an_averaged_checkpoint_has_no_step(
    tmp_path: Path,
) -> None:
    averaged = write_model_export(
        tmp_path / "export",
        directions=SIX_DIRECTIONS,
        checkpoint=checkpoint_record("averaged"),
    ).record

    card = card_of(averaged)

    assert "- checkpoint kind: averaged\n" in card
    assert "- checkpoint step: n/a (an averaged model has no step)\n" in card
    assert "- checkpoint manifest stage: train_average\n" in card


# ---- no path, no timestamp, no address, the same bytes --------------------------------------------


def test_the_card_holds_no_address_no_path_and_no_timestamp(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    for forbidden in ("http", "/home", "/tmp", "/Users", "\\\\", "\r"):
        assert forbidden not in card
    assert not re.search(r"(?:^|[\s(\"'=,;])/[A-Za-z0-9._~-]", card, re.MULTILINE)
    assert not re.search(r"[A-Za-z]:[\\/]", card)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", card)
    assert "created" not in card


def test_rendering_twice_gives_equal_bytes(record: ModelRecord) -> None:
    assert card_of(record) == card_of(record)


def test_the_card_is_the_same_in_another_process_with_another_hash_seed_and_locale(
    record: ModelRecord, tmp_path: Path
) -> None:
    (tmp_path / "rinchen.json").write_bytes(record.to_bytes())
    report = bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)
    (tmp_path / "report.json").write_bytes(report_bytes(report))
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "from rinchen.eval.report_read import report_from_bytes\n"
        "from rinchen.export.model_card import FileFacts, render_card\n"
        "from rinchen.export.record import ModelRecord\n"
        "base = Path(sys.argv[2])\n"
        "record = ModelRecord.from_bytes((base / 'rinchen.json').read_bytes())\n"
        "report = report_from_bytes((base / 'report.json').read_bytes())\n"
        "files = [FileFacts(n, 1, '1' * 64) for n in ('model.bin', 'report.md')]\n"
        "sys.stdout.buffer.write(render_card(record, report, 'alpha', files, 'o/n',\n"
        "    report_sha256='2' * 64, export_manifest_sha256='3' * 64).encode('utf-8'))\n"
    )
    outputs = []
    for seed, locale in (("1", "C"), ("2", "C.UTF-8")):
        done = subprocess.run(
            [sys.executable, "-c", code, source_root, str(tmp_path)],
            capture_output=True,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": seed,
                "LC_ALL": locale,
                "TZ": "Asia/Ulaanbaatar",
                "CUDA_VISIBLE_DEVICES": "",
            },
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        outputs.append(done.stdout)

    assert outputs[0] == outputs[1]
    expected = render_card(
        record,
        view_of(report),
        "alpha",
        [FileFacts(n, 1, "1" * 64) for n in ("model.bin", "report.md")],
        "o/n",
        report_sha256="2" * 64,
        export_manifest_sha256="3" * 64,
    )
    assert outputs[0] == expected.encode("utf-8")


def test_a_system_with_no_result_in_the_report_is_a_value_error_naming_only_the_system(
    record: ModelRecord,
) -> None:
    with pytest.raises(ValueError) as caught:
        card_of(record, system="gamma")

    assert "gamma" in str(caught.value)
    assert "SECRET" not in str(caught.value)
