"""The training curve of an Eole run, read from its log, and the gate on it.

Eole 0.6.4 logs, every `report_every` steps, a line such as

    [2026-11-08 10:07:41,809 INFO] Step 10/  200; acc: 0.0; ppl: 229.78; xent: 5.44;
    aux: 0.000; mtp: 0.000; attn_ent: 1.965; lr: 6.25e-04; sents:    1512; ...

(one line in the log, wrapped here; the figures are those of the steps since the
previous report, and `ppl` and `xent` include the label smoothing). At each validation
it logs `Validation perplexity: 83.6869` and `Validation accuracy: 12.6863`, with no
step on them, then `Saving optimizer and weights to step_50, and symlink to
<model_path>`. Early stopping logs `Early Stop!` and, after it, the save of the final
step. A loss that has become NaN prints `nan` and training goes on to the end with exit
code 0 (spike, D8), so the log is the only place that says so:
`TrainCurve.non_finite_step` is what a harness checks.

Nothing here runs Eole or opens a file: the caller reads the log.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, NamedTuple

from rinchen.errors import DataError

# A report line. The value of a field runs to its `;`, so a line cut inside a figure
# does not match.
_REPORT: Final = re.compile(
    r"Step\s+(?P<step>[0-9]+)/\s*(?P<total>[0-9]+);"
    r"\s*acc:\s*(?P<acc>[^;\s]+);"
    r"\s*ppl:\s*(?P<ppl>[^;\s]+);"
    r"\s*xent:\s*(?P<xent>[^;\s]+);"
    r"(?P<middle>.*?)\blr:\s*(?P<lr>[^;\s]+);"
)
_REPORT_START: Final = re.compile(r"Step\s+[0-9]+/\s*[0-9]+;")
_FIELD: Final = re.compile(r"(?P<name>\w+):\s*(?P<value>[^;]*);")
_VALIDATION_PERPLEXITY: Final = re.compile(r"Validation perplexity:\s*(\S+)")
_VALIDATION_ACCURACY: Final = re.compile(r"Validation accuracy:\s*(\S+)")
_SAVING: Final = re.compile(r"Saving optimizer and weights to step_([0-9]+)\b")
_EARLY_STOP: Final = "Early Stop!"
# A carriage return ends a line too: progress bars redraw with it.
_LINE_BREAK: Final = re.compile(r"[\r\n]")


class TrainReport(NamedTuple):
    """One `Step N/T` line: the step `step` of `total`, and the accuracy (in percent),
    perplexity, cross-entropy and learning rate of the steps since the previous report.
    A value can be NaN or infinite: the log said so."""

    step: int
    total: int
    acc: float
    ppl: float
    xent: float
    lr: float


@dataclass(frozen=True, slots=True)
class TrainCurve:
    """What a log says about a run; `TrainCurve()` is what an empty log says.

    A log can hold several runs appended (the steps go back to a smaller number than the
    last report's): everything below but `non_finite_*` and `unparsed_reports` is that
    of the LAST run, and `runs` counts them.

    reports: the `Step N/T` lines, in order.
    validations, accuracies: the `Validation perplexity` and `Validation accuracy`
        values, in order (a validation prints no step).
    saved_steps: N of every `Saving optimizer and weights to step_N`, in order.
    early_stopped: the log says `Early Stop!` and no report follows it.
    early_stop_step: the step saved right after `Early Stop!`, which is where the run
        stopped; None when it is not early stopped or the log ends before that save.
    non_finite_step: None when every figure in the whole log is finite, else the step of
        the first report with a NaN or infinite figure (any field, `lr` included), or of
        the last report before a validation figure that is, 0 if there is none.
    non_finite_field: the name of that figure (`xent`, `lr`, `validation perplexity`).
    runs: the number of runs with at least one report; `restarted` says there are
        several.
    unparsed_reports: lines that start like a report and could not be read as one,
        in the whole log; a log cut in the middle of its last line is not counted.
    """

    reports: tuple[TrainReport, ...] = ()
    validations: tuple[float, ...] = ()
    accuracies: tuple[float, ...] = ()
    saved_steps: tuple[int, ...] = ()
    early_stopped: bool = False
    early_stop_step: int | None = None
    non_finite_step: int | None = None
    non_finite_field: str | None = None
    runs: int = 0
    unparsed_reports: int = 0

    @property
    def restarted(self) -> bool:
        return self.runs > 1


def parse_train_log(text: str) -> TrainCurve:
    """The curve of the last run in the log `text`.

    Lines end at `\\n` or `\\r`. Python warnings, tracebacks, progress bars and anything
    else that is not one of the lines above are ignored. The last line, when `text` does
    not end with a line break, may have been cut in the middle of a figure, so it is read
    as a report line only, which a cut cannot make wrong (it must reach the `;` after
    `lr`). A new run starts at a report whose step is not above the previous report's.
    A validation that comes before the first report of a run (`valid_steps` below
    `report_every`) is taken for the previous run's.
    """
    *lines, last = _LINE_BREAK.split(text)
    reading = _Reading()
    for line in lines:
        reading.line(line)
    reading.report_line(last, may_be_cut=True)
    return reading.curve()


def loss_gate(curve: TrainCurve, min_drop: float = 0.3) -> str | None:
    """None when `curve` shows a run that learned, else why it did not.

    A run learned when no figure in its log is NaN or infinite, the `xent` of its last
    report is at most `1 - min_drop` times that of its first (a fall of at least
    `min_drop`: with 0.3, 0.7 times the first is accepted and 0.7000001 times is not),
    and its last validation perplexity is below its first. The curve is of the last run
    only (`TrainCurve`), so a resumed run is judged from where its last attempt began.

    Raises:
        DataError: the log has fewer than two reports or fewer than two validations to
            compare.
        ValueError: `min_drop` is not in [0, 1).
    """
    if not 0 <= min_drop < 1:
        raise ValueError(f"min_drop must be in [0, 1), got {min_drop!r}")
    if curve.non_finite_step is not None:
        return f"{curve.non_finite_field} is not finite at step {curve.non_finite_step}"
    if len(curve.reports) < 2:
        raise DataError(
            f"the log has {len(curve.reports)} training report(s); the loss gate "
            "compares the first and the last of at least two"
        )
    if len(curve.validations) < 2:
        raise DataError(
            f"the log has {len(curve.validations)} validation(s); the loss gate "
            "compares the first and the last of at least two"
        )
    first, last = curve.reports[0], curve.reports[-1]
    figures = (first.xent, last.xent, curve.validations[0], curve.validations[-1])
    if not all(math.isfinite(figure) for figure in figures):
        return "a figure the gate compares is not finite"
    if last.xent > (1 - min_drop) * first.xent:
        return (
            f"xent went from {first.xent} at step {first.step} to {last.xent} at step "
            f"{last.step}, a fall of less than {min_drop:.0%}"
        )
    if not curve.validations[-1] < curve.validations[0]:
        return (
            f"validation perplexity went from {curve.validations[0]} to "
            f"{curve.validations[-1]}, it did not fall"
        )
    return None


class _Reading:
    """The state of one pass over a log."""

    def __init__(self) -> None:
        self._reports: list[TrainReport] = []
        self._validations: list[float] = []
        self._accuracies: list[float] = []
        self._saved: list[int] = []
        self._early_stopped = False
        self._early_stop_step: int | None = None
        self._awaiting_stop_save = False
        self._non_finite: tuple[int, str] | None = None
        self._runs = 0
        self._unparsed = 0

    def line(self, line: str) -> None:
        if self.report_line(line):
            return
        for pattern, label, store in (
            (_VALIDATION_PERPLEXITY, "validation perplexity", self._validations),
            (_VALIDATION_ACCURACY, "validation accuracy", self._accuracies),
        ):
            match = pattern.search(line)
            if match is not None:
                self._validation_figure(match.group(1), label, store)
                return
        saved = _SAVING.search(line)
        if saved is not None:
            step = int(saved.group(1))
            self._saved.append(step)
            if self._awaiting_stop_save:
                self._early_stop_step = step
                self._awaiting_stop_save = False
        elif _EARLY_STOP in line:
            self._early_stopped = True
            self._early_stop_step = None
            self._awaiting_stop_save = True

    def report_line(self, line: str, *, may_be_cut: bool = False) -> bool:
        """Read `line` as a report; whether it starts like one. A line that does and
        cannot be read is counted, unless it may be a cut one."""
        if _REPORT_START.search(line) is None:
            return False
        match = _REPORT.search(line)
        report = None if match is None else _report_of(match)
        if match is None or report is None:
            if not may_be_cut:
                self._unparsed += 1
            return True
        self._enter_the_run_of(report.step)
        self._reports.append(report)
        self._early_stopped = False
        self._early_stop_step = None
        self._awaiting_stop_save = False
        self._note_non_finite_fields(report.step, match)
        return True

    def curve(self) -> TrainCurve:
        return TrainCurve(
            reports=tuple(self._reports),
            validations=tuple(self._validations),
            accuracies=tuple(self._accuracies),
            saved_steps=tuple(self._saved),
            early_stopped=self._early_stopped,
            early_stop_step=self._early_stop_step,
            non_finite_step=None if self._non_finite is None else self._non_finite[0],
            non_finite_field=None if self._non_finite is None else self._non_finite[1],
            runs=self._runs,
            unparsed_reports=self._unparsed,
        )

    def _enter_the_run_of(self, step: int) -> None:
        # The first report opens the first run; a step that is not above the previous
        # report's opens the next one and forgets the run before.
        if self._reports and step <= self._reports[-1].step:
            self._runs += 1
            self._reports.clear()
            self._validations.clear()
            self._accuracies.clear()
            self._saved.clear()
        elif not self._reports:
            self._runs += 1

    def _note_non_finite_fields(self, step: int, match: re.Match[str]) -> None:
        named = [
            ("acc", match["acc"]),
            ("ppl", match["ppl"]),
            ("xent", match["xent"]),
            ("lr", match["lr"]),
        ]
        named += [
            (field["name"], field["value"].strip())
            for field in _FIELD.finditer(match["middle"])
        ]
        for name, value in named:
            figure = _figure_of(value)
            if figure is not None and not math.isfinite(figure):
                self._remember_non_finite(step, name)
                return

    def _validation_figure(self, value: str, label: str, store: list[float]) -> None:
        figure = _figure_of(value)
        if figure is None:
            return
        store.append(figure)
        if not math.isfinite(figure):
            last = self._reports[-1].step if self._reports else 0
            self._remember_non_finite(last, label)

    def _remember_non_finite(self, step: int, name: str) -> None:
        if self._non_finite is None:
            self._non_finite = (step, name)


def _figure_of(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _report_of(match: re.Match[str]) -> TrainReport | None:
    try:
        acc, ppl, xent, lr = (
            float(match[name]) for name in ("acc", "ppl", "xent", "lr")
        )
    except ValueError:
        return None
    return TrainReport(int(match["step"]), int(match["total"]), acc, ppl, xent, lr)
