"""The D11 engine gates: script masking on real outputs, the three surfaces (`Engine`,
`rinchen.Translator`, `rinchen translate`) and document translation. Everything runs through
the real `rinchen` binary or the library with the real CTranslate2 on the CPU; nothing here
touches a GPU (`CUDA_VISIBLE_DEVICES` is the empty string for the session and every CTranslate2
translator of the tests is built on `device="cpu"`).

The data are the toy corpora and documents of `tests/support/tiny_parallel.py`: templated
sentences, so every number below tests machinery and ranks nothing. The toy TEST documents are
read, on purpose, by the document scenarios alone, to be translated: nothing trains on them or
chooses by them (the document corpus handed to `docify` holds the train and dev documents).

The models come from the session fixtures of `tests/integration/conftest.py` (the exports of
`tests/integration/test_export_parity.py`'s stages: the source-tag teacher at float32, int8 and
int8_float16, the target-tag teacher at float32, and a document model continued from the
source-tag teacher for 2,500 steps), and the D5 tiny converter of `tiny_run` and `tiny_ct2`.

Gates, all thresholds literals of this file and never lowered (a miss is answered with more
steps of the tiny config):

    G4  script masking, 500 outputs of requests the toy never trained (eng to eng, eng to rus,
        khk to khk, rus to rus, rus to eng; 100 dev sources each, `src=None`): 0 outputs with a
        letter of a script the registry does not give to the target with `always`, at least 10
        with `off` (so the gate cannot pass vacuously), 0 either way for the 400 outputs of the
        four trained directions; `retry` leaves every output at a script purity of 0.98 or
        more, retries at least 10 outputs and improves each; the D5 converter, asked for the
        traditional script, writes it in at least 400 of its words without a suppression list
        and in none with it
    G5  the surfaces: `Translator` equals `Engine` on 50 sentences, `rinchen translate` writes
        exactly `Engine`'s 50 lines (one line per input line, exit 0, `ok 50` on stderr), exit 2
        and nothing on stdout for an unknown target (and an output file that held something
        stays as it was, an empty input exits 2 as well), exit 1 and one `error:` line, with no
        "Exception ignored", when standard output is a pipe that was closed after the first line
        (`| head -1`), the traditional script through the D5
        converter has Mongolian letters and at least one conversion, `Cyrl` none
    G6  documents: every document has one translation per sentence, at least 90 % of the windows
        (of each size 2 to 4, and over the whole test set) are aligned, a 20-piece cap makes at
        least one window fall back and so does every window cut a piece or two before its end
        token, a sentence-only export refuses documents

Measured on this machine (CPU, two threads, Eole 0.6.4, CTranslate2 4.8.4, 2026-11-09; the
machine was shared with other work, so a time moves by a third from one run to the next, a count
or a score never does; the D11 spike measured the same counts and scores):

    G4  500 untrained requests, greedy: with `off` 96 outputs hold a letter of a script their
        target does not write (731 letters), with `always` none (all 500 ok), with `retry` 96
        are decoded again, 96 improved, none holds such a letter; the 400 trained requests: none
        either way, the same texts; the D5 converter: 448 words, 448 with the Mongolian script
        without a list, 0 with it (42 entries listed)
    G5  50 sentences: `rinchen translate` 1.2 s, the same 50 lines as `Engine` and `Translator`;
        `--script Mong`: 50 of 50 lines hold Mongolian letters, 50 conversions, `--script Cyrl`
        none; through the target-tag export the engine writes what CTranslate2 writes, word for
        word, with the forced tag taken off (400 dev rows)
    G6  the document model: `docify` 1.5 s, the 2,500-step continuation 128 s (loaded machine),
        export 4 s, `document_level` derived true; 66 test documents (400 sentences) give 106
        windows, 101 aligned (95.3 %) and 18 single sentences; windows of 2 sentences 325 of 334
        aligned (97.3 %), of 3 sentences 260 of 268 (97.0 %), of 4 sentences 191 of 202 (94.6 %);
        every one of the 400 sentences `ok`, chrF 84.8 over them (re-measured on 2026-11-10, same
        counts of windows and singles). The 36 sentences decoded alone (the 18 singles and the
        sentences of the 5 windows that did not line up) used to come back `no_output`, chrF
        78.5: the model was continued on windows of 2 to 4 sentences only, so after one sentence
        it writes `<sep>`, which the engine dropped as a reserved piece and now reads as the end
        of the sentence (it cuts the hypothesis there). The same 400 sentences as plain rows of
        the document model: 400 `ok`, chrF 88.4 (400 `no_output` before). With a cap of 20
        pieces 94 of the 106 windows fall back; each of
        the 334 windows of 2 sentences, decoded with a cap two pieces below its own full
        translation (21 distinct lengths), falls back although it has one segment per sentence

The whole of this module and `test_export_parity.py` takes 8 min 0 s on a machine with a load
average of 12 to 30 (the teachers 3.5 min, the document model 2.2 min, the D5 converter 1.7
min).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import pytest

import rinchen
from rinchen.errors import ConfigError
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import DecodeOptions, DocumentResult, Row, Translation
from rinchen.metrics.bleu_chrf import corpus_scores
from tests.support.d11_gate import (
    UNTRAINED_PAIRS,
    ConverterLeak,
    Exports,
    converter_leak,
    delta,
    dev_rows,
    has_mongolian_letter,
    open_engine,
    outputs_with_forbidden_letters,
    raw_greedy_texts,
    run_translate,
    run_translate_into_a_closed_pipe,
    trained_requests,
    untrained_requests,
)
from tests.support.d11_gate_doc import (
    WINDOW_SIZES,
    Case,
    DocumentModel,
    cases_of,
    window_lengths,
    windows_of_size,
)
from tests.support.distill_gate_stages import Workspace
from tests.support.tiny_converter import TO_SCRIPT_TAG, TinyRun

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden", "d11_quiet_engine_log"),
]

# The gates. They are never lowered: a miss is answered with more steps.
MIN_UNMASKED_LEAKS: Final = 10
MIN_RETRIES: Final = 10
PURITY_FLOOR: Final = 0.98
MIN_CONVERTER_WORDS_WITH_SCRIPT: Final = 400
MIN_ALIGNMENT: Final = 0.9
SURFACE_SENTENCES: Final = 50
UNTRAINED_OUTPUTS: Final = 500
TRAINED_OUTPUTS: Final = 400
# A cap far below the length of a window's translation: the model is cut before its end token.
SHORT_CAP: Final = 20
GREEDY_128: Final = DecodeOptions(beam_size=1, max_decoding_length=128)
# What `rinchen train docify` cuts windows to: the base config's 128 pieces less the two
# boundary tokens Eole's length filter counts on a target.
WINDOW_PIECES: Final = 126


# =================================================================================
# G4: script masking.


@dataclass(frozen=True)
class MaskedRun:
    """The outputs of one set of requests in one script mode, and what the engine counted."""

    rows: list[Row]
    translations: list[Translation]
    counters: dict[str, int]


@pytest.fixture(scope="module")
def masking(
    d11_workspace: Workspace, d11_exports: Exports
) -> dict[tuple[str, str], MaskedRun]:
    """The untrained and the trained requests, each in the three script modes (greedy)."""
    requests = {
        "untrained": untrained_requests(d11_workspace),
        "trained": trained_requests(d11_workspace),
    }
    runs: dict[tuple[str, str], MaskedRun] = {}
    engine = open_engine(d11_exports.float32.directory)
    try:
        for name, rows in requests.items():
            for mode in ("off", "always", "retry"):
                options = DecodeOptions(
                    beam_size=1,
                    max_decoding_length=128,
                    script_mask=mode,  # type: ignore[arg-type]
                )
                before = engine.counters()
                translations = engine.translate(rows, options)
                runs[(name, mode)] = MaskedRun(
                    rows, translations, delta(before, engine.counters())
                )
    finally:
        engine.close()
    return runs


def test_the_untrained_requests_are_500_and_the_trained_400(
    masking: dict[tuple[str, str], MaskedRun],
) -> None:
    assert len(UNTRAINED_PAIRS) * 100 == UNTRAINED_OUTPUTS
    for mode in ("off", "always", "retry"):
        assert len(masking[("untrained", mode)].translations) == UNTRAINED_OUTPUTS
        assert len(masking[("trained", mode)].translations) == TRAINED_OUTPUTS
    assert [row.tgt for row in masking[("untrained", "off")].rows] == [
        target for _, target in UNTRAINED_PAIRS for _ in range(100)
    ]


def test_with_the_mask_always_on_no_output_holds_a_letter_of_a_script_its_target_does_not_write(
    masking: dict[tuple[str, str], MaskedRun], record_property: Any
) -> None:
    run = masking[("untrained", "always")]
    leaked, letters = outputs_with_forbidden_letters(run.rows, run.translations)
    record_property("always_outputs_with_forbidden_letters", leaked)
    record_property(
        "always_status_counts", dict(Counter(t.status for t in run.translations))
    )

    assert (leaked, letters) == (0, 0)
    # Every output ends with the end token and has text: a list that forbade the end token
    # would leave no forbidden letter either, in outputs that never stop.
    assert all(t.status == "ok" and t.text for t in run.translations)


def test_without_the_mask_the_requests_the_toy_never_trained_leak_another_script(
    masking: dict[tuple[str, str], MaskedRun], record_property: Any
) -> None:
    run = masking[("untrained", "off")]
    leaked, letters = outputs_with_forbidden_letters(run.rows, run.translations)
    record_property("off_outputs_with_forbidden_letters", leaked)
    record_property("off_forbidden_letters", letters)
    record_property(
        "off_status_counts", dict(Counter(t.status for t in run.translations))
    )

    # The gate cannot pass vacuously: the same 500 requests leak when nothing is suppressed.
    assert leaked >= MIN_UNMASKED_LEAKS
    assert letters >= leaked
    assert run.counters.get("script_retries", 0) == 0


def test_the_four_trained_directions_do_not_leak_with_the_mask_or_without_it(
    masking: dict[tuple[str, str], MaskedRun], record_property: Any
) -> None:
    for mode in ("off", "always"):
        run = masking[("trained", mode)]
        leaked, letters = outputs_with_forbidden_letters(run.rows, run.translations)
        record_property(f"trained_{mode}_outputs_with_forbidden_letters", leaked)
        assert (leaked, letters) == (0, 0), mode
    # The mask changes nothing the model was right about.
    assert [t.text for t in masking[("trained", "always")].translations] == [
        t.text for t in masking[("trained", "off")].translations
    ]


def test_retry_decodes_the_impure_outputs_again_and_every_final_output_is_pure_enough(
    masking: dict[tuple[str, str], MaskedRun], record_property: Any
) -> None:
    run = masking[("untrained", "retry")]
    counters = run.counters
    record_property("retry_script_retries", counters.get("script_retries", 0))
    record_property(
        "retry_script_retry_improved", counters.get("script_retry_improved", 0)
    )
    record_property("retry_script_impure", counters.get("script_impure", 0))

    assert counters["script_retries"] >= MIN_RETRIES
    assert counters["script_retry_improved"] == counters["script_retries"]
    assert sum(t.retried for t in run.translations) == counters["script_retries"]
    for row, translation in zip(run.rows, run.translations, strict=True):
        if translation.script_purity is not None:
            assert translation.script_purity >= PURITY_FLOOR
        if translation.retried:
            # Decoded again with every other script suppressed: pure by construction.
            assert outputs_with_forbidden_letters([row], [translation]) == (0, 0)
    # `retry` is a floor, not a guarantee: only `always` promises no forbidden letter at all.
    leaked, _ = outputs_with_forbidden_letters(run.rows, run.translations)
    record_property("retry_outputs_with_forbidden_letters", leaked)


@pytest.fixture(scope="module")
def leak_of_the_converter(tiny_run: TinyRun, tiny_ct2: Path) -> ConverterLeak:
    return converter_leak(
        tiny_ct2,
        tiny_run.spm_model,
        [word for word in tiny_run.lines("cyrl2mong.train.src") if word],
        TO_SCRIPT_TAG,
    )


def test_the_d5_converter_writes_the_script_it_wants_unless_the_list_forbids_it(
    leak_of_the_converter: ConverterLeak, record_property: Any
) -> None:
    leak = leak_of_the_converter
    record_property("converter_words", leak.words)
    record_property("converter_words_with_script_unmasked", leak.with_script_unmasked)
    record_property("converter_words_with_script_masked", leak.with_script_masked)
    record_property("converter_listed_entries", leak.listed_entries)

    assert leak.words >= MIN_CONVERTER_WORDS_WITH_SCRIPT
    assert leak.with_script_unmasked >= MIN_CONVERTER_WORDS_WITH_SCRIPT
    assert leak.with_script_masked == 0
    assert leak.listed_entries > 0


# =================================================================================
# G5: the three surfaces.


@pytest.fixture(scope="module")
def sentences(d11_workspace: Workspace) -> list[str]:
    return list(d11_workspace.inputs.dev["eng_Latn-khk_Cyrl"].src[:SURFACE_SENTENCES])


@pytest.fixture(scope="module")
def engine_texts(d11_exports: Exports, sentences: list[str]) -> list[Translation]:
    engine = open_engine(d11_exports.float32.directory)
    try:
        return engine.translate(
            [Row(text, "eng_Latn", "khk_Cyrl") for text in sentences], DecodeOptions()
        )
    finally:
        engine.close()


def test_the_translator_returns_what_the_engine_returns(
    d11_exports: Exports, sentences: list[str], engine_texts: list[Translation]
) -> None:
    with rinchen.Translator(d11_exports.float32.directory, device="cpu") as translator:
        answers = translator.translate(sentences, tgt="khk_Cyrl", src="eng_Latn")
        single = translator.translate(sentences[0], tgt="khk_Cyrl", src="eng_Latn")
        info = translator.info

    assert len(answers) == SURFACE_SENTENCES
    assert [a.text for a in answers] == [e.text for e in engine_texts]
    assert answers == engine_texts
    assert single == engine_texts[0]
    assert all(e.status == "ok" for e in engine_texts)
    assert info.compute_type == "float32"
    assert info.device == "cpu"


def test_rinchen_translate_writes_the_engines_lines_one_per_input_line(
    d11_exports: Exports,
    sentences: list[str],
    engine_texts: list[Translation],
    record_property: Any,
) -> None:
    done = run_translate(
        d11_exports.float32.directory,
        sentences,
        "--tgt",
        "khk_Cyrl",
        "--src",
        "eng_Latn",
    )
    record_property("translate_50_lines_seconds", round(done.seconds, 1))

    assert done.returncode == 0, done.stderr
    assert done.lines == [e.text for e in engine_texts]
    assert len(done.lines) == len(sentences)
    assert done.stdout.endswith(b"\n")
    assert (
        "translated 50 lines: ok 50, empty 0, too_long 0, truncated 0, no_output 0"
        in (done.stderr)
    )


def test_rinchen_translate_gives_one_output_line_per_input_line_whatever_the_status(
    d11_exports: Exports, sentences: list[str]
) -> None:
    lines = [sentences[0], "", sentences[1], "   ", sentences[2]]
    done = run_translate(
        d11_exports.float32.directory, lines, "--tgt", "khk_Cyrl", "--src", "eng_Latn"
    )

    assert done.returncode == 0, done.stderr
    assert len(done.lines) == len(lines)
    assert done.lines[1] == done.lines[3] == ""
    assert all(done.lines[index] for index in (0, 2, 4))
    assert "ok 3, empty 2" in done.stderr
    assert "empty: first lines 2, 4" in done.stderr


def test_rinchen_translate_refuses_a_target_the_model_does_not_write_and_prints_nothing(
    d11_exports: Exports, sentences: list[str]
) -> None:
    done = run_translate(
        d11_exports.float32.directory, sentences[:3], "--tgt", "zzz_Latn"
    )

    assert done.returncode == 2
    assert done.stdout == b""


def test_rinchen_translate_refuses_a_target_before_it_empties_the_output_file(
    d11_exports: Exports, sentences: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"
    out.write_bytes(b"PRECIOUS\n")

    done = run_translate(
        d11_exports.float32.directory,
        sentences[:3],
        "--tgt",
        "khk_Cyrll",
        "--output",
        out,
    )
    empty = run_translate(d11_exports.float32.directory, [], "--tgt", "khk_Cyrll")

    assert done.returncode == 2
    assert out.read_bytes() == b"PRECIOUS\n"
    assert empty.returncode == 2


def test_rinchen_translate_into_a_closed_pipe_exits_1_and_says_so_once(
    d11_exports: Exports, sentences: list[str], tmp_path: Path
) -> None:
    lines = tmp_path / "lines.txt"
    lines.write_text("".join(f"{sentence}\n" for sentence in sentences * 40), "utf-8")

    done = run_translate_into_a_closed_pipe(
        d11_exports.float32.directory,
        lines,
        "--tgt",
        "khk_Cyrl",
        "--src",
        "eng_Latn",
        "--batch-size",
        "8",
    )

    assert done.stdout.endswith(b"\n") and len(done.stdout) > 1
    assert done.returncode == 1, done.stderr
    assert "error: cannot write the output to standard output" in done.stderr
    assert "Exception ignored" not in done.stderr
    assert "Traceback" not in done.stderr


def test_rinchen_translate_through_the_target_tag_export_gives_text_without_the_tag(
    d11_exports: Exports, sentences: list[str]
) -> None:
    done = run_translate(
        d11_exports.target_float32.directory,
        sentences[:10],
        "--tgt",
        "khk_Cyrl",
        "--src",
        "eng_Latn",
    )

    assert done.returncode == 0, done.stderr
    assert len(done.lines) == 10
    assert all(line and "<2" not in line for line in done.lines)
    assert "ok 10" in done.stderr


def test_the_engine_over_the_target_tag_export_writes_what_ctranslate2_writes_without_the_tag(
    d11_workspace: Workspace, d11_exports: Exports, record_property: Any
) -> None:
    rows, references = dev_rows(d11_workspace)
    engine = open_engine(d11_exports.target_float32.directory)
    try:
        # No script mask: the comparison is of the forced tag alone (the default `retry` decodes
        # again the outputs of this 1,000-step model that hold a Latin word in Cyrillic text).
        translations = engine.translate(rows, replace(GREEDY_128, script_mask="off"))
    finally:
        engine.close()
    raw = raw_greedy_texts(d11_exports.target_float32.directory, rows, placement="tgt")
    score = corpus_scores([t.text for t in translations], references).chrf
    # A 1,000-step teacher translates the toy worse than the 2,000-step one (chrF 98.2): the
    # number is a measurement, not a gate; the gate is that the engine adds and takes off
    # the forced tag without changing a word.
    record_property("target_tag_engine_chrf", round(score, 3))

    assert [t.text for t in translations] == raw
    assert all(t.status == "ok" for t in translations)
    assert all("<2" not in t.text for t in translations)


@pytest.fixture(scope="module")
def converter_runs(
    d11_exports: Exports, sentences: list[str], tiny_run: TinyRun, tiny_ct2: Path
) -> dict[str, Any]:
    arguments = (
        "--tgt",
        "khk_Cyrl",
        "--src",
        "eng_Latn",
        "--script-converter-ct2",
        tiny_ct2,
        "--script-converter-spm",
        tiny_run.spm_model,
    )
    return {
        script: run_translate(
            d11_exports.float32.directory, sentences, *arguments, "--script", script
        )
        for script in ("Mong", "Cyrl")
    }


def test_the_traditional_script_is_made_by_the_d5_converter_and_the_cyrillic_one_is_not(
    converter_runs: dict[str, Any], record_property: Any
) -> None:
    mong, cyrl = converter_runs["Mong"], converter_runs["Cyrl"]
    with_script = sum(has_mongolian_letter(line) for line in mong.lines)
    record_property("mong_lines_with_the_script", with_script)
    record_property("mong_seconds", round(mong.seconds, 1))

    assert mong.returncode == 0, mong.stderr
    assert len(mong.lines) == SURFACE_SENTENCES
    # The converter knows 448 words of a fixture: a sentence of the toy converts partly. The
    # gate asserts a count, not a spelling.
    assert with_script >= 1
    conversions = _counter(mong.stderr, "script_conversions")
    record_property("mong_script_conversions", conversions)
    assert conversions >= 1

    assert cyrl.returncode == 0, cyrl.stderr
    assert len(cyrl.lines) == SURFACE_SENTENCES
    assert not any(has_mongolian_letter(line) for line in cyrl.lines)
    assert _counter(cyrl.stderr, "script_conversions") == 0


def _counter(stderr: str, name: str) -> int:
    for line in stderr.splitlines():
        if line.startswith("counters: "):
            for part in line.removeprefix("counters: ").split(", "):
                counter, _, value = part.rpartition(" ")
                if counter == name:
                    return int(value)
    return 0


# =================================================================================
# G6: documents.


@dataclass(frozen=True)
class Documents:
    """The toy test documents translated by the engine over the document model."""

    cases: list[Case]
    results: list[DocumentResult]
    chrf: float
    counters: dict[str, int]


def _translate_cases(
    engine: Engine, cases: list[Case], options: DecodeOptions
) -> list[DocumentResult]:
    """One call per direction (a call has one source and one target), in the order of
    `cases`."""
    results: list[DocumentResult | None] = [None] * len(cases)
    for direction in sorted({(case.src_lang, case.tgt_lang) for case in cases}):
        indices = [
            index
            for index, case in enumerate(cases)
            if (case.src_lang, case.tgt_lang) == direction
        ]
        answers = engine.translate_documents(
            [cases[index].text for index in indices], *direction, options
        )
        for index, answer in zip(indices, answers, strict=True):
            results[index] = answer
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


@pytest.fixture(scope="module")
def document_engine(d11_document_model: DocumentModel) -> Iterator[Engine]:
    engine = open_engine(d11_document_model.export.directory)
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture(scope="module")
def documents(d11_document_model: DocumentModel, document_engine: Engine) -> Documents:
    cases = cases_of(d11_document_model.documents.test)
    options = DecodeOptions(
        beam_size=1, max_decoding_length=128, max_source_pieces=WINDOW_PIECES
    )
    before = document_engine.counters()
    results = _translate_cases(document_engine, cases, options)
    counters = delta(before, document_engine.counters())
    texts = [t.text for result in results for t in result.translations]
    references = [text for case in cases for text in case.references]
    return Documents(cases, results, corpus_scores(texts, references).chrf, counters)


def test_the_document_export_derives_its_level_from_the_saved_config(
    d11_document_model: DocumentModel, record_property: Any
) -> None:
    export = d11_document_model.export
    record_property("document_model_seconds", round(d11_document_model.seconds, 1))
    record_property("docify_seconds", round(d11_document_model.docify.seconds, 1))
    record_property(
        "continuation_seconds", round(d11_document_model.continuation.seconds, 1)
    )

    assert export.record["document_level"] is True
    assert export.manifest.status["document_level_check"] == "derived"
    assert "document level: derived from the saved config (true)" in (
        export.invocation.stdout
    )
    assert export.record["quantization"] == "float32"
    assert export.record["tag_placement"] == "src"


def test_every_document_has_one_translation_per_sentence_and_most_windows_line_up(
    documents: Documents, record_property: Any
) -> None:
    sentences = windows = aligned = singles = 0
    statuses: Counter[str] = Counter()
    for case, result in zip(documents.cases, documents.results, strict=True):
        assert len(result.translations) == len(case.sentences)
        assert result.stats.sentences == len(case.sentences)
        statuses.update(t.status for t in result.translations)
        sentences += result.stats.sentences
        windows += result.stats.windows
        aligned += result.stats.aligned
        singles += result.stats.singles
    decoded_alone = documents.counters["sources"]
    record_property("documents", len(documents.cases))
    record_property("sentences", sentences)
    record_property("windows", windows)
    record_property("aligned_windows", aligned)
    record_property("single_sentences", singles)
    record_property("sentences_decoded_alone", decoded_alone)
    record_property("sentence_statuses", dict(statuses))
    record_property("document_chrf", round(documents.chrf, 3))

    assert windows > 0
    assert aligned / windows >= MIN_ALIGNMENT
    # The model was continued on windows of 2 to 4 sentences only, so a sentence decoded alone
    # (a single the windows left over, or a sentence of a window that did not line up) comes
    # back with a `<sep>` after it; the engine cuts it there, so no sentence is missing.
    assert statuses == {"ok": sentences}


@pytest.mark.parametrize("size", WINDOW_SIZES)
def test_windows_of_each_size_line_up(
    d11_document_model: DocumentModel,
    document_engine: Engine,
    size: int,
    record_property: Any,
) -> None:
    windows = windows_of_size(cases_of(d11_document_model.documents.test), size)
    options = DecodeOptions(
        beam_size=1, max_decoding_length=128, max_source_pieces=WINDOW_PIECES
    )
    results = _translate_cases(document_engine, windows, options)
    aligned = sum(result.stats.aligned for result in results)
    record_property(f"windows_of_{size}", len(windows))
    record_property(f"aligned_windows_of_{size}", aligned)

    # A document of `size` sentences is one window of that size for the chunker.
    assert all(result.stats.windows == 1 for result in results)
    assert aligned / len(windows) >= MIN_ALIGNMENT


def test_a_window_cut_before_its_end_token_falls_back_to_its_sentences(
    d11_document_model: DocumentModel, document_engine: Engine, record_property: Any
) -> None:
    cases = cases_of(d11_document_model.documents.test)
    options = DecodeOptions(
        beam_size=1, max_decoding_length=SHORT_CAP, max_source_pieces=WINDOW_PIECES
    )
    before = document_engine.counters()
    results = _translate_cases(document_engine, cases, options)
    moved = delta(before, document_engine.counters())
    fallbacks = sum(len(result.stats.fallbacks) for result in results)
    windows = sum(result.stats.windows for result in results)
    record_property("capped_windows", windows)
    record_property("capped_window_fallbacks", fallbacks)

    assert fallbacks >= 1
    assert moved["window_fallbacks"] == fallbacks
    assert moved["windows"] == windows
    # Every document still has one translation per sentence.
    for case, result in zip(cases, results, strict=True):
        assert len(result.translations) == len(case.sentences)


def test_a_window_cut_inside_its_last_segment_still_falls_back(
    d11_document_model: DocumentModel, document_engine: Engine, record_property: Any
) -> None:
    """Counting the separators of a window cannot tell a cut that took the last piece or two of
    the last segment: it has one segment per sentence, and text. Only the missing end token
    does. Each window of two sentences is decoded with a cap two pieces below the length of its
    own full translation, so that the model is cut a piece or two before its end."""
    windows = windows_of_size(cases_of(d11_document_model.documents.test), 2)
    lengths = window_lengths(d11_document_model.export.directory, windows)
    by_length: dict[int, list[Case]] = {}
    for window, length in zip(windows, lengths, strict=True):
        by_length.setdefault(length, []).append(window)

    checked = 0
    for length, group in sorted(by_length.items()):
        options = DecodeOptions(
            beam_size=1, max_decoding_length=length - 2, max_source_pieces=WINDOW_PIECES
        )
        results = _translate_cases(document_engine, group, options)
        for result in results:
            assert result.stats.fallbacks == (0,), length
            assert result.stats.aligned == 0
        checked += len(group)
    record_property("windows_cut_before_the_end", checked)
    record_property("distinct_window_lengths", len(by_length))

    assert checked == len(windows)
    assert min(by_length) > 3


def test_rinchen_translate_translates_documents_with_the_document_model(
    d11_document_model: DocumentModel, record_property: Any
) -> None:
    cases = [
        case
        for case in cases_of(d11_document_model.documents.test)
        if (case.src_lang, case.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    ][:3]
    lines: list[str] = []
    for case in cases:
        lines += [*case.sentences, ""]
    done = run_translate(
        d11_document_model.export.directory,
        lines,
        "--tgt",
        "khk_Cyrl",
        "--src",
        "eng_Latn",
        "--doc",
        "--max-source-pieces",
        str(WINDOW_PIECES),
    )

    assert done.returncode == 0, done.stderr
    # One line per document, a blank line between two.
    assert len(done.lines) == 5
    assert done.lines[1] == done.lines[3] == ""
    assert all(done.lines[index] for index in (0, 2, 4))
    assert "translated 3 documents: ok 3" in done.stderr


def test_a_sentence_only_export_refuses_documents(d11_exports: Exports) -> None:
    engine = open_engine(d11_exports.float32.directory)
    try:
        with pytest.raises(ConfigError, match="document"):
            engine.translate_documents(
                ["Hello there. This is a star."], "eng_Latn", "khk_Cyrl"
            )
        assert engine.counters()["windows"] == 0
    finally:
        engine.close()
