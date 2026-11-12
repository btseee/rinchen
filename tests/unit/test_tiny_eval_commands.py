"""The two commands of the evaluation stage that take `--resamples` (D15, amendment 5 of Task 5).

`rinchen eval run` bootstraps its intervals with the draws of the metrics configuration unless it is told
how many to make, and the profile of the pipeline names a count of its own (`evaluation.resamples`).
`commands.eval_run` and `commands.eval_run_hypotheses` take it as a keyword-only argument that is written
into the argument list only when it is given, so that every list of the earlier tasks is unchanged. The
lists are checked against the options of the real command, read from its `--help`.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Final

import pytest

from rinchen.tiny import commands
from tests.support.tiny_help import check_argument_list

OUT: Final = Path("/work/runs")
SUITE: Final = Path("/work/suites/tiny-heldout-eng.json")
MODEL: Final = Path("/work/export/int8")
ROBUSTNESS: Final = Path("/work/robustness")
HYPOTHESES: Final = Path("/work/runs/tiny-heldout-eng/tiny-int8/hypotheses.jsonl")
SCORES: Final = Path("/work/scores/metricx24_qe.jsonl")


def run(**options: object) -> list[str]:
    return commands.eval_run(SUITE, MODEL, "tiny-int8", OUT, 1, None, **options)  # type: ignore[arg-type]


def hypotheses_run(**options: object) -> list[str]:
    return commands.eval_run_hypotheses(
        SUITE,
        HYPOTHESES,
        "tiny-int8-neural",
        OUT,
        {"metricx24_qe": SCORES},
        **options,  # type: ignore[arg-type]
    )


def test_a_model_run_without_a_count_is_the_list_of_the_earlier_task() -> None:
    assert run() == [
        *commands.rinchen_argv("eval", "run")[:3],
        "eval",
        "run",
        "--suite",
        str(SUITE),
        "--model",
        str(MODEL),
        "--system",
        "tiny-int8",
        "--beam-size",
        "1",
        "--out-dir",
        str(OUT),
    ]
    assert "--resamples" not in run()
    assert "--resamples" not in run(resamples=None)


def test_a_model_run_with_a_count_ends_with_it() -> None:
    assert run(resamples=1000)[-2:] == ["--resamples", "1000"]
    assert run(resamples=1000)[:-2] == run()


def test_the_count_follows_the_robustness_block_when_there_is_one() -> None:
    with_block = commands.eval_run(
        SUITE, MODEL, "tiny-int8", OUT, 1, ROBUSTNESS, resamples=250
    )

    assert with_block[-4:] == ["--robustness", str(ROBUSTNESS), "--resamples", "250"]


def test_a_hypotheses_run_without_a_count_is_the_list_of_the_earlier_task() -> None:
    assert hypotheses_run()[3:] == [
        "eval",
        "run",
        "--suite",
        str(SUITE),
        "--hypotheses",
        str(HYPOTHESES),
        "--system",
        "tiny-int8-neural",
        "--out-dir",
        str(OUT),
        "--segment-scores",
        f"metricx24_qe={SCORES}",
    ]
    assert "--resamples" not in hypotheses_run(resamples=None)


def test_a_hypotheses_run_with_a_count_ends_with_it() -> None:
    assert hypotheses_run(resamples=1000)[-2:] == ["--resamples", "1000"]
    assert hypotheses_run(resamples=1000)[:-2] == hypotheses_run()


def test_the_count_is_a_keyword_only_argument_with_no_default_value_but_none() -> None:
    for function in (commands.eval_run, commands.eval_run_hypotheses):
        parameter = inspect.signature(function).parameters["resamples"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


@pytest.mark.parametrize("count", [None, 100, 1000])
def test_the_lists_hold_only_options_of_the_real_command(count: int | None) -> None:
    words = ("eval", "run")

    assert check_argument_list(run(resamples=count), words) == []
    assert check_argument_list(hypotheses_run(resamples=count), words) == []
    assert (
        check_argument_list(
            commands.eval_run(
                SUITE, MODEL, "tiny-int8", OUT, 1, ROBUSTNESS, resamples=count
            ),
            words,
        )
        == []
    )


def test_the_count_is_written_as_digits_and_its_option_once() -> None:
    argv = run(resamples=1000)

    assert argv.count("--resamples") == 1
    assert argv[argv.index("--resamples") + 1] == "1000"
    assert all(isinstance(part, str) for part in argv)


def test_the_report_has_no_count() -> None:
    assert "resamples" not in inspect.signature(commands.eval_report).parameters
    assert (
        check_argument_list(
            commands.eval_report([HYPOTHESES.parent], [SUITE], OUT / "report"),
            ("eval", "report"),
        )
        == []
    )
