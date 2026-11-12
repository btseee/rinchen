"""`rinchen.tiny.commands`: the readers of what the CLI stages wrote (D15, spec P3, brief G4.3), and the
recorded outputs under `tests/fixtures/tiny_run/` they are tested over.

The recordings are copies of the files a real run of the pipeline wrote (a spike on the CPU, no
network), with every absolute path of that machine replaced by a placeholder: a test of this module
reads nothing outside the repository. The values the readers return equal the recorded ones; a file
that is not what the command writes is refused, and the refusal names the file and the field and
quotes no value. The one reader of printed text is the one of `convert-script eval`, whose five lines
are refused by shape.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError
from rinchen.tiny import commands

RECORDINGS: Final = REPO_ROOT / "tests" / "fixtures" / "tiny_run"
SENTINEL: Final = "SENTINEL-VALUE-93"
LIMIT: Final = 100 * 1024
BYTE_ORDER_MARK: Final = chr(0xFEFF)


def recorded(name: str) -> Path:
    return RECORDINGS / name


def recorded_json(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(recorded(name).read_text(encoding="utf-8"))
    return document


def written(tmp_path: Path, document: object, name: str = "document.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ---- the recordings --------------------------------------------------------------------------------


def test_every_recording_is_small_and_names_no_path_of_the_machine_that_made_it() -> (
    None
):
    files = sorted(RECORDINGS.iterdir())

    assert files
    for path in files:
        data = path.read_bytes()
        assert len(data) < LIMIT, path.name
        text = data.decode("utf-8")
        assert "/tmp/" not in text, path.name
        assert "/home/" not in text, path.name
        assert "\r" not in text, path.name
        assert text.endswith("\n"), path.name


def test_every_recording_is_valid_in_its_own_format() -> None:
    for path in sorted(RECORDINGS.glob("*.json")):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict), path.name
    for path in sorted(RECORDINGS.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert isinstance(json.loads(line), dict), path.name


@pytest.mark.parametrize(
    "name",
    [
        "parity.json",
        "counts.json",
        "conv_eval.log",
        "synth.yaml",
        "spm.yaml",
        "matrix.yaml",
        "robustness_report.json",
        "result_memorised_eng_khk_int8.json",
        "result_memorised_eng_khk_float32.json",
        "result_heldout_eng_khk_int8.json",
        "hypotheses_heldout_eng_int8.jsonl",
        "hypotheses_memorised_eng_int8.jsonl",
    ],
)
def test_the_recordings_the_pipeline_tests_use_are_there(name: str) -> None:
    assert recorded(name).is_file()


def test_the_hypotheses_are_lines_of_a_direction_an_id_a_status_and_a_text() -> None:
    for name in (
        "hypotheses_heldout_eng_int8.jsonl",
        "hypotheses_memorised_eng_int8.jsonl",
    ):
        rows = [
            json.loads(line)
            for line in recorded(name).read_text(encoding="utf-8").splitlines()
        ]
        assert {tuple(sorted(row)) for row in rows} == {
            ("direction", "id", "status", "text")
        }
        assert {row["status"] for row in rows} == {"ok"}
        assert {row["direction"] for row in rows} == {
            "eng_Latn-khk_Cyrl",
            "khk_Cyrl-eng_Latn",
        }


# ---- read_json -------------------------------------------------------------------------------------


def test_read_json_returns_the_object_of_the_file() -> None:
    assert commands.read_json(recorded("parity.json")) == recorded_json("parity.json")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "3",
        '"x"',
        "null",
        "{",
        '{"a": 1,}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1, "b": 2}}',
        '{"a": 1} trailing',
        BYTE_ORDER_MARK + "{}",
    ],
)
def test_read_json_refuses_what_is_not_one_object(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DataError, match=r"bad\.json"):
        commands.read_json(path)


def test_read_json_refuses_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"a": "\xff"}')

    with pytest.raises(DataError, match="UTF-8"):
        commands.read_json(path)


def test_read_json_refuses_a_missing_file_and_a_directory(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"missing\.json"):
        commands.read_json(tmp_path / "missing.json")
    with pytest.raises(DataError):
        commands.read_json(tmp_path)


def test_a_refusal_names_the_file_and_not_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "x.json"
    path.parent.mkdir()
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DataError) as error:
        commands.read_json(path)

    assert "x.json" in str(error.value)
    assert str(tmp_path) not in str(error.value)


# ---- read_parity -----------------------------------------------------------------------------------


def test_the_parity_of_the_recorded_run_is_read_whole() -> None:
    parity = commands.read_parity(recorded("parity.json"))

    assert parity == recorded_json("parity.json")
    assert parity["passes"] is True
    assert parity["exact"] == 70
    assert parity["sentences"] == 70
    assert parity["exact_share"] == 1.0
    assert parity["mean_abs_delta"] == 1.6221680949143363e-06
    assert parity["quantization"] == "float32"
    assert parity["placement"] == "src"


REQUIRED_PARITY_KEYS: Final = (
    "passes",
    "gated",
    "exact",
    "sentences",
    "exact_share",
    "mean_abs_delta",
    "max_abs_delta",
    "min_exact_share",
    "max_mean_delta",
    "quantization",
    "placement",
)


@pytest.mark.parametrize("key", REQUIRED_PARITY_KEYS)
def test_a_parity_file_without_a_key_the_report_needs_is_refused(
    tmp_path: Path, key: str
) -> None:
    document = recorded_json("parity.json")
    del document[key]

    with pytest.raises(DataError, match=key):
        commands.read_parity(written(tmp_path, document))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("passes", "true"),
        ("passes", 1),
        ("passes", None),
        ("gated", 0),
        ("exact", True),
        ("exact", 70.5),
        ("exact", -1),
        ("exact", SENTINEL),
        ("sentences", 0),
        ("sentences", True),
        ("exact_share", 1.5),
        ("exact_share", -0.1),
        ("exact_share", True),
        ("exact_share", SENTINEL),
        ("mean_abs_delta", -1e-6),
        ("mean_abs_delta", None),
        ("mean_abs_delta", [1]),
        ("max_abs_delta", SENTINEL),
        ("min_exact_share", 2),
        ("max_mean_delta", -1),
        ("quantization", 8),
        ("quantization", ""),
        ("placement", None),
    ],
)
def test_a_parity_file_with_a_value_of_the_wrong_kind_is_refused_without_quoting_it(
    tmp_path: Path, key: str, value: object
) -> None:
    document = recorded_json("parity.json")
    document[key] = value

    with pytest.raises(DataError, match=key) as error:
        commands.read_parity(written(tmp_path, document))

    assert SENTINEL not in str(error.value)


def test_a_parity_file_that_says_more_than_it_must_is_still_read(
    tmp_path: Path,
) -> None:
    document = recorded_json("parity.json")
    document["something_new"] = {"a": [1, 2]}

    assert commands.read_parity(written(tmp_path, document)) == document


def test_the_exact_count_cannot_exceed_the_sentences(tmp_path: Path) -> None:
    document = recorded_json("parity.json")
    document["exact"] = document["sentences"] + 1

    with pytest.raises(DataError, match="exact"):
        commands.read_parity(written(tmp_path, document))


# ---- read_converter_eval ---------------------------------------------------------------------------

GOOD_LOG: Final = (
    "cer 0.0285\nword_accuracy 0.9160\nn 500\nrule_fallbacks 0\ninvalid_outputs 0\n"
)
GOOD_LINES: Final = GOOD_LOG.splitlines()
ARABIC_FIVE: Final = chr(0x665)
FULL_WIDTH_ONE: Final = chr(0xFF11)


def test_the_five_lines_of_the_recorded_log_are_the_five_numbers() -> None:
    text = recorded("conv_eval.log").read_text(encoding="utf-8")

    assert commands.read_converter_eval(text) == {
        "cer": 0.0285,
        "word_accuracy": 0.9160,
        "n": 500.0,
        "rule_fallbacks": 0.0,
        "invalid_outputs": 0.0,
    }


def test_the_last_line_needs_no_line_feed() -> None:
    assert commands.read_converter_eval(GOOD_LOG.rstrip("\n")) == (
        commands.read_converter_eval(GOOD_LOG)
    )


def test_the_numbers_are_read_as_the_command_prints_them() -> None:
    rng = random.Random(11)
    for _ in range(200):
        cer = rng.random() * rng.choice([1, 10])
        accuracy = rng.random()
        counts = [rng.randrange(0, 100_000) for _ in range(3)]
        text = (
            f"cer {cer:.4f}\nword_accuracy {accuracy:.4f}\nn {counts[0]}\n"
            f"rule_fallbacks {counts[1]}\ninvalid_outputs {counts[2]}\n"
        )

        assert commands.read_converter_eval(text) == {
            "cer": float(f"{cer:.4f}"),
            "word_accuracy": float(f"{accuracy:.4f}"),
            "n": float(counts[0]),
            "rule_fallbacks": float(counts[1]),
            "invalid_outputs": float(counts[2]),
        }


def replace_line(index: int, line: str) -> str:
    lines = list(GOOD_LINES)
    lines[index] = line
    return "\n".join(lines) + "\n"


REFUSED_LOGS: Final = {
    "empty": "",
    "only a line feed": "\n",
    "four lines": "\n".join(GOOD_LINES[:4]) + "\n",
    "one line": GOOD_LINES[0] + "\n",
    "six lines": GOOD_LOG + "extra 1\n",
    "a blank line at the end": GOOD_LOG + "\n",
    "a blank line first": "\n" + GOOD_LOG,
    "a fifth line that is not a number": replace_line(4, "invalid_outputs many"),
    "a fifth line without a value": replace_line(4, "invalid_outputs"),
    "cer twice": replace_line(1, "cer 0.9160"),
    "the last key twice": replace_line(3, "invalid_outputs 0"),
    "two lines swapped": "\n".join([GOOD_LINES[1], GOOD_LINES[0], *GOOD_LINES[2:]])
    + "\n",
    "a key missing and another added": replace_line(3, "something_else 0"),
    "windows line ends": GOOD_LOG.replace("\n", "\r\n"),
    "a space before": " " + GOOD_LOG,
    "a space after a number": replace_line(0, "cer 0.0285 "),
    "a tab": replace_line(0, "cer\t0.0285"),
    "two spaces": replace_line(0, "cer  0.0285"),
    "a colon": replace_line(0, "cer: 0.0285"),
    "an upper-case key": replace_line(0, "CER 0.0285"),
    "three decimals": replace_line(0, "cer 0.028"),
    "no leading digit": replace_line(0, "cer .0285"),
    "negative": replace_line(0, "cer -0.0285"),
    "an exponent": replace_line(0, "cer 2e-2"),
    "not a number": replace_line(0, "cer nan"),
    "infinite": replace_line(0, "cer inf"),
    "a plus sign": replace_line(2, "n +500"),
    "a count with a decimal point": replace_line(2, "n 500.0"),
    "a count with an exponent": replace_line(2, "n 5e2"),
    "a negative count": replace_line(3, "rule_fallbacks -1"),
    "a count with a comma": replace_line(2, "n 1,000"),
    "arabic-indic digits": replace_line(2, f"n {ARABIC_FIVE}00"),
    "full-width digits": replace_line(2, f"n {FULL_WIDTH_ONE}0"),
    "the spike's separator": GOOD_LOG + "\n--- stderr ---\n",
    "a line of the other stream first": "warning: slow\n" + GOOD_LOG,
    "a log line after": GOOD_LOG + '{"event": "done"}\n',
    "a progress line inside": "\n".join(
        [*GOOD_LINES[:2], "loading...", *GOOD_LINES[2:]]
    )
    + "\n",
}


@pytest.mark.parametrize("name", list(REFUSED_LOGS))
def test_a_log_that_is_not_the_five_lines_is_refused(name: str) -> None:
    with pytest.raises(DataError):
        commands.read_converter_eval(REFUSED_LOGS[name])


def test_a_refused_log_is_not_quoted() -> None:
    text = replace_line(4, f"invalid_outputs {SENTINEL}")

    with pytest.raises(DataError) as error:
        commands.read_converter_eval(text)

    assert SENTINEL not in str(error.value)


def test_a_log_with_a_very_long_extra_line_is_refused() -> None:
    with pytest.raises(DataError):
        commands.read_converter_eval(GOOD_LOG + "x" * 100_000)


# ---- read_result_metrics ---------------------------------------------------------------------------

RESULT_FILES: Final = (
    "result_memorised_eng_khk_int8.json",
    "result_memorised_eng_khk_float32.json",
    "result_heldout_eng_khk_int8.json",
)


@pytest.mark.parametrize("name", RESULT_FILES)
def test_the_metrics_of_a_recorded_result_are_the_recorded_ones(name: str) -> None:
    metrics = commands.read_result_metrics(recorded(name))

    assert metrics == recorded_json(name)["metrics"]
    assert list(metrics) == [
        "bleu",
        "chrf",
        "chrf_pp",
        "comet22",
        "metricx24",
        "metricx24_qe",
    ]
    assert metrics["comet22"] is None


def test_the_recorded_figures_are_those_of_the_spike() -> None:
    assert commands.read_result_metrics(recorded(RESULT_FILES[0]))["chrf"] == (
        99.9593961645435
    )
    assert commands.read_result_metrics(recorded(RESULT_FILES[1]))["chrf"] == (
        99.9593961645435
    )
    assert commands.read_result_metrics(recorded(RESULT_FILES[2]))["chrf"] == (
        71.99500932252622
    )


def test_a_neural_score_that_was_given_is_a_number(tmp_path: Path) -> None:
    document = recorded_json(RESULT_FILES[2])
    document["metrics"]["metricx24_qe"] = 3.5

    metrics = commands.read_result_metrics(written(tmp_path, document))

    assert metrics["metricx24_qe"] == 3.5
    assert metrics["comet22"] is None


def altered(**changes: object) -> dict[str, Any]:
    document = copy.deepcopy(recorded_json(RESULT_FILES[0]))
    document["metrics"].update(changes)
    return document


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"metrics": []},
        {"metrics": None},
        {"metrics": {}},
        altered(chrf=None),
        altered(bleu=None),
        altered(chrf_pp=None),
        altered(chrf=100.5),
        altered(chrf=-0.1),
        altered(bleu=101),
        altered(chrf=True),
        altered(chrf=SENTINEL),
        altered(chrf=float("nan")),
        altered(comet22=SENTINEL),
        altered(metricx24=True),
        altered(metricx24_qe=[1]),
        altered(unknown_metric=1.0),
    ],
    ids=lambda document: str(sorted(document))[:40],
)
def test_a_result_whose_metrics_are_not_the_six_is_refused(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DataError, match=r"result\.json") as error:
        commands.read_result_metrics(path)

    assert SENTINEL not in str(error.value)


def test_a_missing_metric_is_named(tmp_path: Path) -> None:
    document = altered()
    del document["metrics"]["chrf"]

    with pytest.raises(DataError, match="chrf"):
        commands.read_result_metrics(written(tmp_path, document))


# ---- read_synth_counts -----------------------------------------------------------------------------


def test_the_counts_of_the_recorded_synthetic_run_are_read_whole() -> None:
    counts = commands.read_synth_counts(recorded("counts.json"))

    assert counts == recorded_json("counts.json")
    assert counts["counts"]["selected"] == 64
    assert counts["counts"]["sources"] == 64
    assert counts["qe_status"] == "skipped: --no-qe"
    assert counts["job"] == "bt_eng_khk"
    assert counts["shard_stats"]["generated"] == 2


@pytest.mark.parametrize(
    "key", ["counts", "qe_status", "job", "generator", "shard_stats"]
)
def test_counts_without_a_key_are_refused(tmp_path: Path, key: str) -> None:
    document = recorded_json("counts.json")
    del document[key]

    with pytest.raises(DataError, match=key):
        commands.read_synth_counts(written(tmp_path, document))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["counts"].pop("selected"),
        lambda d: d["counts"].update(selected=-1),
        lambda d: d["counts"].update(selected=True),
        lambda d: d["counts"].update(selected=1.5),
        lambda d: d["counts"].update(selected=SENTINEL),
        lambda d: d["counts"].update(sources=0),
        lambda d: d["counts"].update(selected=65),
        lambda d: d.update(counts=[]),
        lambda d: d.update(qe_status=3),
        lambda d: d.update(job=None),
        lambda d: d["shard_stats"].update(generated=-1),
        lambda d: d["shard_stats"].update(planned=SENTINEL),
        lambda d: d.update(shard_stats=3),
    ],
    ids=[
        "count missing",
        "negative count",
        "boolean count",
        "fractional count",
        "text count",
        "no sources",
        "more selected than sources",
        "counts not a mapping",
        "status not text",
        "job not text",
        "negative shard count",
        "text shard count",
        "shard stats not a mapping",
    ],
)
def test_counts_of_the_wrong_shape_are_refused_without_quoting_a_value(
    tmp_path: Path, change: Any
) -> None:
    document = recorded_json("counts.json")
    change(document)

    with pytest.raises(DataError) as error:
        commands.read_synth_counts(written(tmp_path, document))

    assert SENTINEL not in str(error.value)
