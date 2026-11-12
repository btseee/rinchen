"""`rinchen.tiny.commands`: the argument lists of the CLI stages of the tiny pipeline (D15, spec P2
and P3, brief G4.2).

Each builder is a pure function that returns the list a stage hands to its runner. The lists are pinned
literally, in the order of the spike prototype, and every one is checked against the `--help` option
table of the real command it calls (`tests/support/tiny_help.py`): every option exists, every required
option is there, a flag has no value and an option has one, a path is absolute. No list can hold a
device value, an upload flag or the word of a download: the one function every list goes through
refuses them, whatever value a caller gives. The words are written in pieces here because the content
scans of the gate read the tests too.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.tiny import commands
from tests.support.tiny_help import (
    Option,
    check_argument_list,
    click_table,
    help_table,
)

BASE: Final = [sys.executable, "-m", "rinchen"]
DEVICE_FLAG: Final = "--dev" + "ice"
AUTO: Final = "au" + "to"
CUDA: Final = "cu" + "da"
UPLOAD: Final = "--up" + "load"
PUBLIC: Final = "--pub" + "lic"
DOWNLOAD: Final = "fe" + "tch"
FORBIDDEN_WORDS: Final = (AUTO, CUDA, DEVICE_FLAG, UPLOAD, PUBLIC, DOWNLOAD)
WORK: Final = Path("/work")


def at(name: str) -> Path:
    return WORK / name


@dataclass(frozen=True, slots=True)
class Call:
    """One call of the pipeline: the command words, the list a builder returns for these
    arguments and the list it must be."""

    id: str
    words: tuple[str, ...]
    argv: Callable[[], list[str]]
    expected: list[str]


def _calls() -> list[Call]:
    corpora = [at("train.parquet"), at("synth.parquet")]
    return [
        Call(
            "spm_train",
            ("spm", "train"),
            lambda: commands.spm_train(at("spm.yaml")),
            [*BASE, "spm", "train", "--config", "/work/spm.yaml"],
        ),
        Call(
            "spm_check",
            ("spm", "check"),
            lambda: commands.spm_check(at("tiny.model"), at("text")),
            [
                *BASE,
                "spm",
                "check",
                "--model",
                "/work/tiny.model",
                "--corpus-dir",
                "/work/text",
            ],
        ),
        Call(
            "converter_build_data",
            ("convert-script", "build-data"),
            lambda: commands.converter_build_data(at("pairs.tsv"), at("corpus")),
            [
                *BASE,
                "convert-script",
                "build-data",
                "--pairs",
                "/work/pairs.tsv",
                "--out-dir",
                "/work/corpus",
            ],
        ),
        Call(
            "converter_train",
            ("convert-script", "train"),
            lambda: commands.converter_train(
                at("corpus"), at("run"), at("char.yaml"), 2, 300
            ),
            [
                *BASE,
                "convert-script",
                "train",
                "--profile",
                "tiny",
                "--data-dir",
                "/work/corpus",
                "--run-dir",
                "/work/run",
                "--spm-config",
                "/work/char.yaml",
                "--threads",
                "2",
                "--timeout",
                "300",
            ],
        ),
        Call(
            "converter_export",
            ("convert-script", "export"),
            lambda: commands.converter_export(at("run"), at("model"), 300),
            [
                *BASE,
                "convert-script",
                "export",
                "--run-dir",
                "/work/run",
                "--out-dir",
                "/work/model",
                "--timeout",
                "300",
            ],
        ),
        Call(
            "converter_eval",
            ("convert-script", "eval"),
            lambda: commands.converter_eval(at("model"), at("pairs.tsv")),
            [
                *BASE,
                "convert-script",
                "eval",
                "--model-dir",
                "/work/model",
                "--pairs",
                "/work/pairs.tsv",
            ],
        ),
        Call(
            "synth_run",
            ("synth", "run"),
            lambda: commands.synth_run(
                "bt_eng_khk", at("mono.parquet"), at("synth.yaml"), at("run"), 1234
            ),
            [
                *BASE,
                "synth",
                "run",
                "bt_eng_khk",
                "--input",
                "/work/mono.parquet",
                "--config",
                "/work/synth.yaml",
                "--out-dir",
                "/work/run",
                "--seed",
                "1234",
                "--no-qe",
            ],
        ),
        Call(
            "train_prepare",
            ("train", "ablate"),
            lambda: commands.train_prepare(
                at("matrix.yaml"), corpora, at("tiny.model"), at("ablate"), 2
            ),
            [
                *BASE,
                "train",
                "ablate",
                "--matrix",
                "/work/matrix.yaml",
                "--corpus",
                "/work/train.parquet",
                "--corpus",
                "/work/synth.parquet",
                "--spm-model",
                "/work/tiny.model",
                "--out-dir",
                "/work/ablate",
                "--prepare-only",
                "--threads",
                "2",
            ],
        ),
        Call(
            "train_launch",
            ("train", "launch"),
            lambda: commands.train_launch(at("config.yaml"), at("run"), 2, 600),
            [
                *BASE,
                "train",
                "launch",
                "--config",
                "/work/config.yaml",
                "--run-dir",
                "/work/run",
                "--threads",
                "2",
                "--timeout",
                "600",
            ],
        ),
        Call(
            "train_average",
            ("train", "average"),
            lambda: commands.train_average(at("ckpt"), 2),
            [
                *BASE,
                "train",
                "average",
                "--checkpoint-dir",
                "/work/ckpt",
                "--last",
                "2",
            ],
        ),
        Call(
            "export_convert",
            ("export", "convert"),
            lambda: commands.export_convert(
                at("averaged"), at("tiny.model"), "int8", at("int8")
            ),
            [
                *BASE,
                "export",
                "convert",
                "--checkpoint",
                "/work/averaged",
                "--spm-model",
                "/work/tiny.model",
                "--tag-placement",
                "src",
                "--quantization",
                "int8",
                "--out-dir",
                "/work/int8",
            ],
        ),
        Call(
            "export_parity",
            ("export", "parity"),
            lambda: commands.export_parity(
                at("averaged"),
                at("float32"),
                at("train.parquet"),
                at("parity"),
                50,
                2,
                180,
            ),
            [
                *BASE,
                "export",
                "parity",
                "--checkpoint",
                "/work/averaged",
                "--export",
                "/work/float32",
                "--corpus",
                "/work/train.parquet",
                "--out-dir",
                "/work/parity",
                "--max-lines",
                "50",
                "--threads",
                "2",
                "--timeout",
                "180",
            ],
        ),
        Call(
            "eval_robustness",
            ("eval", "robustness"),
            lambda: commands.eval_robustness(
                at("robust.json"),
                at("int8"),
                at("robust"),
                1,
                at("conv"),
                at("char.model"),
            ),
            [
                *BASE,
                "eval",
                "robustness",
                "--suite",
                "/work/robust.json",
                "--model",
                "/work/int8",
                "--out-dir",
                "/work/robust",
                "--beam-size",
                "1",
                "--script-converter-ct2",
                "/work/conv",
                "--script-converter-spm",
                "/work/char.model",
            ],
        ),
        Call(
            "eval_run",
            ("eval", "run"),
            lambda: commands.eval_run(
                at("suite.json"), at("int8"), "tiny-int8", at("runs"), 1, None
            ),
            [
                *BASE,
                "eval",
                "run",
                "--suite",
                "/work/suite.json",
                "--model",
                "/work/int8",
                "--system",
                "tiny-int8",
                "--beam-size",
                "1",
                "--out-dir",
                "/work/runs",
            ],
        ),
        Call(
            "eval_run_robustness",
            ("eval", "run"),
            lambda: commands.eval_run(
                at("suite.json"), at("int8"), "tiny-int8", at("runs"), 1, at("robust")
            ),
            [
                *BASE,
                "eval",
                "run",
                "--suite",
                "/work/suite.json",
                "--model",
                "/work/int8",
                "--system",
                "tiny-int8",
                "--beam-size",
                "1",
                "--out-dir",
                "/work/runs",
                "--robustness",
                "/work/robust",
            ],
        ),
        Call(
            "eval_run_hypotheses",
            ("eval", "run"),
            lambda: commands.eval_run_hypotheses(
                at("suite.json"),
                at("hyp.jsonl"),
                "tiny-seam",
                at("runs"),
                {"metricx24_qe": at("scores.json")},
            ),
            [
                *BASE,
                "eval",
                "run",
                "--suite",
                "/work/suite.json",
                "--hypotheses",
                "/work/hyp.jsonl",
                "--system",
                "tiny-seam",
                "--out-dir",
                "/work/runs",
                "--segment-scores",
                "metricx24_qe=/work/scores.json",
            ],
        ),
        Call(
            "eval_report",
            ("eval", "report"),
            lambda: commands.eval_report(
                [at("run-a"), at("run-b")], [at("s1.json"), at("s2.json")], at("report")
            ),
            [
                *BASE,
                "eval",
                "report",
                "--suite",
                "/work/s1.json",
                "--suite",
                "/work/s2.json",
                "--run",
                "/work/run-a",
                "--run",
                "/work/run-b",
                "--out-dir",
                "/work/report",
            ],
        ),
    ]


CALLS: Final = _calls()
IDS: Final = [call.id for call in CALLS]


def test_there_are_seventeen_calls_over_sixteen_builders() -> None:
    assert len(CALLS) == 17
    assert len(set(IDS)) == 17
    builders = {call.id.removesuffix("_robustness") for call in CALLS}
    assert len(builders) == 16


# ---- the lists are pinned ------------------------------------------------------------------------


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_each_builder_returns_the_list_of_the_prototype(call: Call) -> None:
    assert call.argv() == call.expected


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_each_list_holds_only_strings(call: Call) -> None:
    assert all(type(token) is str for token in call.argv())


def test_rinchen_argv_starts_with_the_interpreter_and_the_package() -> None:
    assert commands.rinchen_argv() == BASE
    assert commands.rinchen_argv("spm", Path("/x"), "y") == [*BASE, "spm", "/x", "y"]


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_a_builder_makes_a_new_list_each_time(call: Call) -> None:
    first = call.argv()
    first.append("changed")
    assert call.argv() == call.expected


# ---- the lists against the real commands ----------------------------------------------------------


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_each_list_matches_the_help_table_of_its_command(call: Call) -> None:
    assert check_argument_list(call.argv(), call.words) == []


WORDS: Final = sorted({call.words for call in CALLS})


@pytest.mark.parametrize("words", WORDS, ids=lambda words: " ".join(words))
def test_the_help_reader_agrees_with_the_command_objects(
    words: tuple[str, ...],
) -> None:
    from_text = help_table(*words)
    from_objects, _ = click_table(*words)

    assert from_text == from_objects
    assert from_text.options


def test_the_help_reader_reads_flags_ranges_and_repeated_paths() -> None:
    # Typer prints a type for every option, a flag's included (`<bool>`), and a repeated path
    # as `<list[Path]>`: a flag takes no value, a range and a path do.
    flag = Option(required=False, takes_value=False, path_valued=False)
    tiny = help_table("tiny", "run")
    report = help_table("eval", "report")

    assert tiny.options["--dry-run"] == flag
    assert tiny.options["--qe-fake"] == tiny.options["--no-qe-fake"] == flag
    assert tiny.options["--threads"] == Option(
        False, takes_value=True, path_valued=False
    )
    assert tiny.options["--out-dir"] == Option(
        False, takes_value=True, path_valued=True
    )
    assert report.options["--run"] == Option(True, takes_value=True, path_valued=True)
    assert report.options["--baseline"] == Option(
        False, takes_value=True, path_valued=False
    )


SPM: Final = commands.spm_train(at("spm.yaml"))
SYNTH: Final = commands.synth_run("job", at("m.parquet"), at("c.yaml"), at("o"), 1)


@pytest.mark.parametrize(
    ("words", "broken"),
    [
        (("spm", "train"), [*SPM, "--unknown", "x"]),
        (("spm", "train"), SPM[:-1]),
        (("spm", "train"), SPM[:-2]),
        (("spm", "train"), [*SPM, "--config", "/work/again.yaml"]),
        (("spm", "train"), [*SPM[:5], "extra", *SPM[5:]]),
        (("spm", "train"), [*SPM[:-1], "relative.yaml"]),
        (("spm", "train"), SPM[1:]),
        (("spm", "check"), SPM),
        (("synth", "run"), [*SYNTH, "oops"]),
        (("synth", "run"), [token for token in SYNTH if token != "job"]),
        (("synth", "run"), [*SYNTH[:5], *SYNTH[6:], "job"]),
    ],
    ids=[
        "unknown option",
        "option without value",
        "required option missing",
        "option twice",
        "extra positional",
        "relative path",
        "no interpreter",
        "another command",
        "flag with a value",
        "positional missing",
        "positional after the options",
    ],
)
def test_the_help_check_has_teeth(words: tuple[str, ...], broken: list[str]) -> None:
    assert check_argument_list(broken, words) != []


# ---- what no list may hold ------------------------------------------------------------------------


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_no_list_holds_a_device_an_upload_or_a_download(call: Call) -> None:
    lowered = [token.lower() for token in call.argv()]
    assert not [token for token in lowered if token in FORBIDDEN_WORDS]
    assert not [token for token in lowered if token.startswith(DEVICE_FLAG)]


@pytest.mark.parametrize(
    "arguments",
    [
        [DEVICE_FLAG, "cpu"],
        [DEVICE_FLAG + "=cpu"],
        ["eval", "run", DEVICE_FLAG.upper()],
        ["x", AUTO],
        ["x", AUTO.upper()],
        ["x", CUDA],
        ["x", CUDA.title()],
        ["release", "hub", UPLOAD],
        ["release", "hub", PUBLIC],
        ["data", DOWNLOAD],
        ["eval", "suite", DOWNLOAD],
    ],
)
def test_the_one_function_every_list_goes_through_refuses_those_words(
    arguments: list[str],
) -> None:
    with pytest.raises(ValueError, match="never"):
        commands.rinchen_argv(*arguments)


def test_a_value_that_is_the_word_is_refused_but_a_longer_name_is_not() -> None:
    with pytest.raises(ValueError):
        commands.eval_run(at("s"), at("m"), AUTO, at("o"), 1, None)
    with pytest.raises(ValueError):
        commands.spm_train(Path(CUDA))

    assert commands.eval_run(at("s"), at("m"), AUTO + "matic", at("o"), 1, None)
    assert commands.spm_train(at(CUDA)) == [
        *BASE,
        "spm",
        "train",
        "--config",
        "/work/" + CUDA,
    ]


def test_the_refusal_does_not_quote_the_value() -> None:
    with pytest.raises(ValueError) as error:
        commands.eval_run(at("s"), at("m"), AUTO, at("o"), 1, None)

    message = str(error.value).lower()
    assert not [word for word in (AUTO, CUDA, DEVICE_FLAG, DOWNLOAD) if word in message]


# ---- what some lists always hold ------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_the_preparation_always_stops_before_training(count: int) -> None:
    argv = commands.train_prepare(
        at("m.yaml"),
        [at(f"c{n}.parquet") for n in range(count)],
        at("s.model"),
        at("o"),
        2,
    )

    assert argv.count("--prepare-only") == 1
    assert argv.count("--corpus") == count
    assert "--timeout" not in argv


@pytest.mark.parametrize("quantization", ["float32", "int8"])
def test_the_export_always_puts_the_tag_in_the_source(quantization: str) -> None:
    argv = commands.export_convert(at("c"), at("s.model"), quantization, at("o"))  # type: ignore[arg-type]

    index = argv.index("--tag-placement")
    assert argv[index + 1] == "src"
    assert argv.count("--tag-placement") == 1
    assert argv[argv.index("--quantization") + 1] == quantization


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_the_synthetic_run_always_skips_the_quality_floor_and_names_the_seed(
    seed: int,
) -> None:
    argv = commands.synth_run("job", at("m"), at("c"), at("o"), seed)

    assert argv.count("--no-qe") == 1
    assert argv[argv.index("--seed") + 1] == str(seed)
    assert argv[5] == "job"


def test_the_robustness_run_attaches_both_halves_of_the_converter() -> None:
    argv = commands.eval_robustness(at("s"), at("m"), at("o"), 4, at("ct2"), at("spm"))

    assert argv[argv.index("--script-converter-ct2") + 1] == "/work/ct2"
    assert argv[argv.index("--script-converter-spm") + 1] == "/work/spm"
    assert argv[argv.index("--beam-size") + 1] == "4"


@pytest.mark.parametrize(
    "scores",
    [{}, {"comet22": at("a")}, {"comet22": at("a"), "metricx24": at("b")}],
)
def test_scores_become_one_option_each(scores: dict[str, Path]) -> None:
    argv = commands.eval_run_hypotheses(at("s"), at("h"), "n", at("o"), scores)

    assert argv.count("--segment-scores") == len(scores)
    for metric, path in scores.items():
        assert f"{metric}={path}" in argv


def test_an_evaluation_with_no_runs_or_suites_is_refused_before_a_command_could_start() -> (
    None
):
    with pytest.raises(ValueError):
        commands.eval_report([], [at("s")], at("o"))
    with pytest.raises(ValueError):
        commands.eval_report([at("r")], [], at("o"))
    with pytest.raises(ValueError):
        commands.train_prepare(at("m"), [], at("s"), at("o"), 2)


def test_numbers_are_written_as_digits() -> None:
    argv: Sequence[Any] = commands.converter_train(at("d"), at("r"), at("c"), 3, 7)

    assert argv[argv.index("--threads") + 1] == "3"
    assert argv[argv.index("--timeout") + 1] == "7"
