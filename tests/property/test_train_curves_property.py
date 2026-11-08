"""Property tests for `parse_train_log` over generated Eole logs.

A generated log has the lines of Eole 0.6.4's training log (reports, validations, saves)
with noise lines between them: warnings, blank lines, progress-bar redraws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.train.curves import TrainReport, parse_train_log

_NOISE = [
    "",
    "   ",
    "/venv/eole/constants.py:8: RuntimeWarning: Could not import flash_attn",
    "  from eole.ops import _CPP_OPS_AVAILABLE",
    "[2026-11-08 10:07:38,963 INFO] Random seed set to: 1234",
    "[2026-11-08 10:07:42,053 INFO] Weighted corpora loaded so far:",
    "Loading:  10%|#  | 1/10\rLoading:  50%|#####  | 5/10",
    "Traceback (most recent call last):",
    "[2026-11-08 10:07:46,530 INFO] Train perplexity: 175.065",
    "[2026-11-08 10:07:46,538 INFO] Saving config and vocab to /tmp/run/ckpt",
]

_figure = st.decimals(
    min_value="0.01",
    max_value="999.99",
    places=2,
    allow_nan=False,
    allow_infinity=False,
)


@dataclass(frozen=True)
class _Generated:
    """Generated reports and the log text that holds them."""

    reports: list[TrainReport]
    validations: list[float]
    saved: list[int]
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _report_line(report: TrainReport, shown: tuple[str, str, str, str]) -> str:
    acc, ppl, xent, lr = shown
    return (
        f"[2026-11-08 10:07:41,809 INFO] Step {report.step:2d}/{report.total:5d}; "
        f"acc: {acc}; ppl: {ppl:>6}; xent: {xent}; aux: 0.000; mtp: 0.000; "
        f"attn_ent: 1.965; lr: {lr}; sents:    1512; bsz: 1131/1231/151; "
        f"3995/4347 tok/s;      3 sec;"
    )


@st.composite
def logs(draw: st.DrawFn) -> _Generated:
    step_gaps = draw(st.lists(st.integers(1, 50), min_size=1, max_size=12))
    total = 100_000
    lines: list[str] = []
    reports: list[TrainReport] = []
    validations: list[float] = []
    saved: list[int] = []
    step = 0
    for gap in step_gaps:
        step += gap
        acc, ppl, xent = (draw(_figure) for _ in range(3))
        lr = draw(st.integers(1, 999))
        shown = (str(acc), str(ppl), str(xent), f"{lr / 100:.2f}e-03")
        report = TrainReport(
            step,
            total,
            float(shown[0]),
            float(shown[1]),
            float(shown[2]),
            float(shown[3]),
        )
        reports.append(report)
        lines.append(_report_line(report, shown))
        if draw(st.booleans()):
            perplexity, accuracy = draw(_figure), draw(_figure)
            validations.append(float(perplexity))
            lines.append(f"[x INFO] Validation perplexity: {perplexity}")
            lines.append(f"[x INFO] Validation accuracy: {accuracy}")
            saved.append(step)
            lines.append(
                f"[x INFO] Saving optimizer and weights to step_{step}, and symlink "
                "to /tmp/run/ckpt"
            )
        lines.extend(draw(st.lists(st.sampled_from(_NOISE), max_size=3)))
    return _Generated(reports, validations, saved, lines)


@given(logs())
def test_the_reports_validations_and_saves_of_a_generated_log_are_read_back(
    generated: _Generated,
) -> None:
    curve = parse_train_log(generated.text)

    assert list(curve.reports) == generated.reports
    assert list(curve.validations) == generated.validations
    assert list(curve.saved_steps) == generated.saved
    assert curve.runs == 1
    assert curve.non_finite_step is None
    assert curve.unparsed_reports == 0
    assert curve.early_stopped is False


@given(logs(), st.data())
def test_a_log_cut_anywhere_gives_a_prefix_of_what_the_whole_log_gives(
    generated: _Generated, data: st.DataObject
) -> None:
    text = generated.text
    cut = data.draw(st.integers(0, len(text)))

    whole = parse_train_log(text)
    part = parse_train_log(text[:cut])

    assert part.reports == whole.reports[: len(part.reports)]
    assert part.validations == whole.validations[: len(part.validations)]
    assert part.accuracies == whole.accuracies[: len(part.accuracies)]
    assert part.saved_steps == whole.saved_steps[: len(part.saved_steps)]
    assert part.unparsed_reports == 0
    assert part.non_finite_step is None


@given(logs(), logs())
def test_appending_a_run_that_starts_again_gives_that_run(
    first: _Generated, second: _Generated
) -> None:
    # A generated run ends below step 10,000 and the next starts at a step of at most
    # 50, so the steps go back where the second begins.
    bridge = "Step 10000/100000; acc: 1.0; ppl: 2.0; xent: 3.0; lr: 1.00e-03;\n"

    curve = parse_train_log(first.text + bridge + second.text)

    assert list(curve.reports) == second.reports
    assert list(curve.validations) == second.validations
    assert list(curve.saved_steps) == second.saved
    assert curve.runs == 2
    assert curve.restarted is True


@given(st.text())
def test_parsing_never_raises_on_any_text(text: str) -> None:
    curve = parse_train_log(text)

    assert curve.runs >= 0
    assert len(curve.reports) >= curve.runs


@given(
    st.lists(
        st.one_of(
            st.sampled_from(_NOISE),
            st.sampled_from(
                [
                    "Step 10/  200; acc: 0.0; ppl: 229.78; xent: 5.44; lr: 6.25e-04;",
                    "Step 5/ 200; acc: nan; ppl: inf; xent: 1; lr: 6.25e-04;",
                    "Validation perplexity: nan",
                    "Validation perplexity: 1e400",
                    "Validation accuracy: -",
                    "Saving optimizer and weights to step_7, and symlink",
                    "Training finished after not improving. Early Stop!",
                ]
            ),
            st.text(max_size=40),
        ),
        max_size=30,
    )
)
def test_parsing_a_shuffle_of_log_lines_and_junk_never_raises(
    lines: list[str],
) -> None:
    curve = parse_train_log("\n".join(lines))

    assert all(isinstance(r.step, int) for r in curve.reports)


@given(logs(), st.data())
def test_a_non_finite_figure_in_a_report_is_found_at_that_report(
    generated: _Generated, data: st.DataObject
) -> None:
    index = data.draw(st.integers(0, len(generated.reports) - 1))
    word = data.draw(st.sampled_from(["nan", "inf", "-inf", "NaN"]))
    field = data.draw(st.sampled_from(["acc", "ppl", "xent", "lr"]))
    target = generated.reports[index]
    lines = list(generated.lines)
    prefix = f"[2026-11-08 10:07:41,809 INFO] Step {target.step:2d}/"
    position = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    lines[position] = _with_figure(lines[position], field, word)

    curve = parse_train_log("\n".join(lines) + "\n")

    assert curve.non_finite_step == target.step
    assert curve.non_finite_field == field
    assert not math.isfinite(getattr(curve.reports[index], field))
    assert len(curve.reports) == len(generated.reports)


def _with_figure(line: str, field: str, word: str) -> str:
    head, _, rest = line.partition(f"{field}: ")
    _, _, tail = rest.partition(";")
    return f"{head}{field}: {word};{tail}"


@settings(max_examples=50)
@given(st.lists(st.integers(1, 40), min_size=2, max_size=10))
def test_the_run_count_is_one_more_than_the_number_of_times_the_steps_go_back(
    gaps: list[int],
) -> None:
    steps = [gaps[0]]
    for gap in gaps[1:]:
        # alternate between going forward and starting again
        steps.append(steps[-1] + gap if gap % 2 else max(1, steps[-1] - gap))
    lines = [
        f"Step {step}/ 999; acc: 1.0; ppl: 2.0; xent: 3.0; lr: 1.00e-03;"
        for step in steps
    ]
    goes_back = sum(1 for before, after in pairwise(steps) if after <= before)

    curve = parse_train_log("\n".join(lines) + "\n")

    assert curve.runs == goes_back + 1
