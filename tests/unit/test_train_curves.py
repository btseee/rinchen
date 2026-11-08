"""Tests for `rinchen.train.curves`: the training curve read from an Eole log, and the
loss gate.

The sample lines are copied from the logs of the D8 spike (Eole 0.6.4 training the tiny
multilingual model on the CPU), apart from those marked as made up.
"""

from __future__ import annotations

import math

import pytest

from rinchen.errors import DataError
from rinchen.train.curves import TrainCurve, TrainReport, loss_gate, parse_train_log

# --- lines of the spike's logs ---------------------------------------------------

STEP_10 = (
    "[2026-11-08 10:07:41,809 INFO] Step 10/  200; acc: 0.0; ppl: 229.78; xent: 5.44; "
    "aux: 0.000; mtp: 0.000; attn_ent: 1.965; lr: 6.25e-04; sents:    1512; "
    "bsz: 1131/1231/151; 3995/4347 tok/s;      3 sec;"
)
STEP_50 = (
    "[2026-11-08 10:07:42,844 INFO] Step 50/  200; acc: 13.0; ppl: 103.71; xent: 4.64; "
    "aux: 0.000; mtp: 0.000; attn_ent: 2.065; lr: 3.13e-03; sents:    1483; "
    "bsz: 1241/1142/148; 42905/39476 tok/s;      4 sec;"
)
STEP_100 = (
    "[2026-11-08 10:07:47,817 INFO] Step 100/  200; acc: 52.1; ppl: 16.53; xent: 2.81; "
    "aux: 0.000; mtp: 0.000; attn_ent: 2.017; lr: 6.25e-03; sents:    1489; "
    "bsz: 1175/1201/149; 29842/30495 tok/s;      9 sec;"
)
STEP_140 = (  # a padded figure: `ppl:  9.66`
    "[2026-11-08 10:07:52,849 INFO] Step 140/  200; acc: 57.4; ppl:  9.66; xent: 2.27; "
    "aux: 0.000; mtp: 0.000; attn_ent: 2.037; lr: 5.28e-03; sents:    1442; "
    "bsz: 1159/1193/144; 50632/52126 tok/s;     14 sec;"
)
STEP_200 = (
    "[2026-11-08 10:07:57,693 INFO] Step 200/  200; acc: 58.3; ppl:  8.53; xent: 2.14; "
    "aux: 0.000; mtp: 0.000; attn_ent: 1.480; lr: 4.42e-03; sents:    1447; "
    "bsz: 1187/1182/145; 54316/54046 tok/s;     19 sec;"
)
# NaN run (learning_rate 1e10): padded `nan` in ppl, `nan` in xent and attn_ent.
NAN_STEP_10 = (
    "[2026-11-08 10:11:19,120 INFO] Step 10/   60; acc: 0.0; ppl:   nan; xent: nan; "
    "aux: 0.000; mtp: 0.000; attn_ent: nan; lr: 1.25e+07; sents:    1512; "
    "bsz: 1131/1231/151; 4727/5143 tok/s;      2 sec;"
)
VALIDATION_BLOCK_50 = [
    "[2026-11-08 10:07:46,530 INFO] Train perplexity: 175.065",
    "[2026-11-08 10:07:46,530 INFO] Train accuracy: 10.1401",
    "[2026-11-08 10:07:46,530 INFO] Validation perplexity: 83.6869",
    "[2026-11-08 10:07:46,530 INFO] Validation accuracy: 12.6863",
    (
        "[2026-11-08 10:07:46,531 INFO] Saving optimizer and weights to step_50, and "
        "symlink to /tmp/run/ckpt"
    ),
    (
        "[2026-11-08 10:07:46,537 INFO] Saving transforms artifacts, if any, to "
        "/tmp/run/ckpt/step_50"
    ),
    "[2026-11-08 10:07:46,538 INFO] Saving config and vocab to /tmp/run/ckpt",
]
VALIDATION_BLOCK_200 = [
    "[2026-11-08 10:08:00,993 INFO] Validation perplexity: 8.81733",
    "[2026-11-08 10:08:00,993 INFO] Validation accuracy: 58.5791",
    (
        "[2026-11-08 10:08:00,993 INFO] Saving optimizer and weights to step_200, and "
        "symlink to /tmp/run/ckpt"
    ),
]
# The early stop of the spike's lr-8 run (patience 2, validation every 20 steps).
EARLY_STOP_BLOCK = [
    "[2026-11-08 10:25:31,746 INFO] Validation perplexity: 14.092",
    "[2026-11-08 10:25:31,747 INFO] Decreasing patience: 0/2",
    "[2026-11-08 10:25:31,747 INFO] Training finished after not improving. Early Stop!",
    "[2026-11-08 10:25:31,747 INFO] Best model found at step 80",
    "[2026-11-08 10:25:31,747 INFO] earlystopper has_stopped!",
    (
        "[2026-11-08 10:25:32,296 INFO] Saving optimizer and weights to step_120, and "
        "symlink to /tmp/run/ckpt"
    ),
]
WARNINGS = [
    (
        "/home/u/.venv/lib/python3.14/site-packages/eole/constants.py:8: "
        "RuntimeWarning: Could not import flash_attn: No module named 'flash_attn'."
    ),
    "  from eole.ops import _CPP_OPS_AVAILABLE, gelu_and_mul",
    "[2026-11-08 10:07:38,962 INFO] Logging initialized (log_file: )",
    "[2026-11-08 10:07:38,963 INFO] Random seed set to: 1234",
]


def text_of(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def report(step: int, xent: float, **changes: float) -> str:
    """A made-up report line in the format of the spike's."""
    fields = {"acc": 10.0, "ppl": 2.0**xent, "lr": 1.0e-3, **changes}
    return (
        f"[2026-11-08 10:00:00,000 INFO] Step {step:2d}/{1000:5d}; "
        f"acc: {fields['acc']}; ppl: {fields['ppl']:5.2f}; xent: {xent}; "
        f"aux: 0.000; mtp: 0.000; attn_ent: 1.000; lr: {fields['lr']:.2e}; "
        f"sents:    1500; bsz: 1100/1100/150; 5000/5000 tok/s;      3 sec;"
    )


# ---- the empty curve ------------------------------------------------------------


def test_an_empty_log_gives_the_empty_curve() -> None:
    assert parse_train_log("") == TrainCurve()


def test_the_empty_curve_has_nothing() -> None:
    curve = TrainCurve()

    assert curve.reports == ()
    assert curve.validations == ()
    assert curve.accuracies == ()
    assert curve.saved_steps == ()
    assert curve.early_stopped is False
    assert curve.early_stop_step is None
    assert curve.non_finite_step is None
    assert curve.non_finite_field is None
    assert curve.runs == 0
    assert curve.restarted is False
    assert curve.unparsed_reports == 0


@pytest.mark.parametrize("text", ["\n", "\n\n\n", "\r\r", "   ", "\r\n"])
def test_a_log_of_nothing_but_blank_lines_gives_the_empty_curve(text: str) -> None:
    assert parse_train_log(text) == TrainCurve()


def test_a_log_with_only_warnings_and_startup_lines_gives_the_empty_curve() -> None:
    assert parse_train_log(text_of(*WARNINGS)) == TrainCurve()


# ---- the reports ----------------------------------------------------------------


def test_a_real_report_line_is_read_field_by_field() -> None:
    curve = parse_train_log(text_of(STEP_10))

    assert curve.reports == (TrainReport(10, 200, 0.0, 229.78, 5.44, 6.25e-4),)
    assert curve.runs == 1
    assert curve.restarted is False


def test_a_report_is_a_tuple_of_step_total_acc_ppl_xent_lr() -> None:
    (only,) = parse_train_log(text_of(STEP_50)).reports

    assert only == (50, 200, 13.0, 103.71, 4.64, 3.13e-3)
    assert (only.step, only.total, only.acc) == (50, 200, 13.0)
    assert (only.ppl, only.xent, only.lr) == (103.71, 4.64, 3.13e-3)


def test_figures_padded_with_spaces_are_read() -> None:
    (only,) = parse_train_log(text_of(STEP_140)).reports

    assert only.ppl == 9.66
    assert only.step == 140


def test_the_step_and_the_total_are_read_whatever_their_padding() -> None:
    padded = "Step  5/ 1000; acc: 1.0; ppl: 2.0; xent: 0.5; lr: 1.0e-03;"
    tight = "Step 12345/123456; acc: 1.0; ppl: 2.0; xent: 0.5; lr: 1.0e-03;"

    steps = [r.step for r in parse_train_log(text_of(padded, tight)).reports]
    totals = [r.total for r in parse_train_log(text_of(padded, tight)).reports]

    assert steps == [5, 12345]
    assert totals == [1000, 123456]


def test_the_reports_of_a_whole_log_come_in_order_with_the_noise_around_them() -> None:
    log = text_of(
        *WARNINGS,
        STEP_10,
        "[2026-11-08 10:07:42,053 INFO] Weighted corpora loaded so far:",
        "\t\t\t * en-mn.auth: 3",
        STEP_50,
        *VALIDATION_BLOCK_50,
        STEP_100,
        STEP_140,
        STEP_200,
        *VALIDATION_BLOCK_200,
    )

    curve = parse_train_log(log)

    assert [r.step for r in curve.reports] == [10, 50, 100, 140, 200]
    assert [r.xent for r in curve.reports] == [5.44, 4.64, 2.81, 2.27, 2.14]
    assert curve.validations == (83.6869, 8.81733)
    assert curve.accuracies == (12.6863, 58.5791)
    assert curve.saved_steps == (50, 200)
    assert curve.early_stopped is False
    assert curve.non_finite_step is None
    assert curve.runs == 1


def test_a_report_line_does_not_need_the_log_prefix() -> None:
    bare = STEP_10.split("] ", 1)[1]

    assert parse_train_log(text_of(bare)).reports[0].step == 10


def test_a_report_without_a_trailing_newline_at_the_end_of_the_log_is_read() -> None:
    assert parse_train_log(STEP_10).reports[0].step == 10


def test_windows_line_endings_are_read() -> None:
    curve = parse_train_log(STEP_10 + "\r\n" + STEP_50 + "\r\n")

    assert [r.step for r in curve.reports] == [10, 50]


def test_progress_bar_redraws_with_carriage_returns_are_ignored() -> None:
    noise = "Loading:  10%|#         | 1/10\rLoading:  50%|#####     | 5/10\r"
    log = noise + STEP_10 + "\n" + noise + "\n" + STEP_50 + "\n"

    assert [r.step for r in parse_train_log(log).reports] == [10, 50]


def test_a_carriage_return_ends_a_line_like_a_line_feed() -> None:
    # No line feed anywhere, and a validation figure last: it is whole because a
    # carriage return ends it.
    log = (
        "[x INFO] Validation perplexity: 83.6869\r"
        "[x INFO] Validation accuracy: 12.6863\r"
    )

    curve = parse_train_log(log)

    assert curve.validations == (83.6869,)
    assert curve.accuracies == (12.6863,)


def test_a_report_after_a_progress_bar_on_the_same_physical_line_is_read() -> None:
    log = "Loading:  10%|#         | 1/10\r" + STEP_10 + "\n"

    assert parse_train_log(log).reports[0].step == 10


def test_python_warnings_and_tracebacks_between_the_lines_are_ignored() -> None:
    log = text_of(
        STEP_10,
        "/venv/torch/jit/_script.py:1485: FutureWarning: `torch.jit.script` is not "
        "supported in Python 3.14+",
        "  warnings.warn(",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "RuntimeError: Timed out waiting 20000ms for send operation to complete",
        STEP_50,
    )

    assert [r.step for r in parse_train_log(log).reports] == [10, 50]


def test_a_line_that_only_mentions_a_step_is_not_a_report() -> None:
    log = text_of(
        "Saving optimizer and weights to step_50, and symlink to /tmp/x",
        "Step 5 of the plan is done",
        "Best model found at step 80",
    )

    assert parse_train_log(log).reports == ()
    assert parse_train_log(log).unparsed_reports == 0


# ---- a report that cannot be read -----------------------------------------------


def test_a_report_with_a_figure_that_is_not_a_number_is_counted_not_dropped_silently() -> (
    None
):
    broken = STEP_50.replace("xent: 4.64", "xent: 4,64")

    curve = parse_train_log(text_of(STEP_10, broken, STEP_100))

    assert [r.step for r in curve.reports] == [10, 100]
    assert curve.unparsed_reports == 1


def test_a_report_with_a_field_missing_is_counted() -> None:
    no_lr = "Step 10/ 200; acc: 1.0; ppl: 2.0; xent: 0.5; aux: 0.0; sents: 5;"

    curve = parse_train_log(text_of(no_lr))

    assert curve.reports == ()
    assert curve.unparsed_reports == 1


# ---- a log cut in the middle of a line ------------------------------------------


@pytest.mark.parametrize("cut", range(0, len(STEP_50), 7))
def test_a_log_cut_inside_its_last_report_never_reads_a_wrong_figure(cut: int) -> None:
    log = text_of(STEP_10) + STEP_50[:cut]

    curve = parse_train_log(log)

    assert curve.reports[0] == TrainReport(10, 200, 0.0, 229.78, 5.44, 6.25e-4)
    if len(curve.reports) == 2:
        assert curve.reports[1] == TrainReport(50, 200, 13.0, 103.71, 4.64, 3.13e-3)
    assert curve.unparsed_reports == 0


def test_a_report_cut_right_after_its_lr_is_read_because_every_figure_is_whole() -> (
    None
):
    head = STEP_50.split(" sents:")[0]

    curve = parse_train_log(text_of(STEP_10) + head)

    assert [r.step for r in curve.reports] == [10, 50]


def test_a_report_cut_inside_the_lr_figure_is_not_read() -> None:
    cut = STEP_50[: STEP_50.index("lr: 3.13") + len("lr: 3.1")]

    assert [r.step for r in parse_train_log(text_of(STEP_10) + cut).reports] == [10]


def test_a_validation_figure_cut_in_the_last_line_is_not_read() -> None:
    log = text_of(STEP_10, "[x INFO] Validation perplexity: 83.6869") + (
        "[x INFO] Validation accuracy: 12.68"
    )

    curve = parse_train_log(log)

    assert curve.validations == (83.6869,)
    assert curve.accuracies == ()


def test_a_validation_figure_in_a_complete_last_line_is_read() -> None:
    log = text_of(STEP_10, "[x INFO] Validation perplexity: 83.6869")

    assert parse_train_log(log).validations == (83.6869,)


def test_a_save_line_cut_in_the_last_line_is_not_read() -> None:
    log = text_of(STEP_10) + "[x INFO] Saving optimizer and weights to step_5"

    assert parse_train_log(log).saved_steps == ()


def test_a_log_cut_after_early_stop_before_the_final_save_has_no_stop_step() -> None:
    log = text_of(
        STEP_10, EARLY_STOP_BLOCK[0], EARLY_STOP_BLOCK[1], EARLY_STOP_BLOCK[2]
    )

    curve = parse_train_log(log)

    assert curve.early_stopped is True
    assert curve.early_stop_step is None


# ---- validations and saves ------------------------------------------------------


def test_validation_perplexities_and_accuracies_are_read_in_order() -> None:
    log = text_of(
        "[x INFO] Validation perplexity: 83.6869",
        "[x INFO] Validation accuracy: 12.6863",
        "[x INFO] Validation perplexity: 16.6115",
        "[x INFO] Validation accuracy: 49.3498",
    )

    curve = parse_train_log(log)

    assert curve.validations == (83.6869, 16.6115)
    assert curve.accuracies == (12.6863, 49.3498)


def test_the_train_perplexity_line_is_not_a_validation() -> None:
    log = text_of(
        "[x INFO] Train perplexity: 175.065", "[x INFO] Train accuracy: 10.1401"
    )

    curve = parse_train_log(log)

    assert curve.validations == ()
    assert curve.accuracies == ()


def test_every_save_line_gives_its_step() -> None:
    log = text_of(*VALIDATION_BLOCK_50, *VALIDATION_BLOCK_200)

    assert parse_train_log(log).saved_steps == (50, 200)


def test_the_other_saving_lines_are_not_a_saved_step() -> None:
    log = text_of(*VALIDATION_BLOCK_50[-2:])

    assert parse_train_log(log).saved_steps == ()


# ---- early stopping -------------------------------------------------------------


def test_early_stopping_gives_the_step_of_the_save_that_follows_it() -> None:
    log = text_of(STEP_100, *EARLY_STOP_BLOCK)

    curve = parse_train_log(log)

    assert curve.early_stopped is True
    assert curve.early_stop_step == 120
    assert curve.saved_steps == (120,)
    assert curve.validations == (14.092,)


def test_the_stop_step_is_that_of_the_first_save_after_the_stop_not_an_earlier_one() -> (
    None
):
    log = text_of(*VALIDATION_BLOCK_50, STEP_100, *EARLY_STOP_BLOCK)

    curve = parse_train_log(log)

    assert curve.saved_steps == (50, 120)
    assert curve.early_stop_step == 120


def test_a_run_that_did_not_stop_early_says_so() -> None:
    curve = parse_train_log(text_of(STEP_10, *VALIDATION_BLOCK_50))

    assert curve.early_stopped is False
    assert curve.early_stop_step is None


def test_an_early_stop_followed_by_more_training_is_not_the_end_of_the_run() -> None:
    # The run stopped, the target was raised and a later launch resumed it.
    log = text_of(STEP_10, *EARLY_STOP_BLOCK, STEP_140)

    curve = parse_train_log(log)

    assert curve.early_stopped is False
    assert curve.early_stop_step is None
    assert curve.saved_steps == (120,)


def test_two_early_stops_in_one_log_give_the_last_one() -> None:
    other = [line.replace("step_120", "step_300") for line in EARLY_STOP_BLOCK]
    log = text_of(STEP_10, *EARLY_STOP_BLOCK, STEP_140, STEP_200, *other)

    assert parse_train_log(log).early_stop_step == 300


# ---- NaN and infinity -----------------------------------------------------------


def test_nan_in_the_loss_is_reported_with_its_step_and_field() -> None:
    log = text_of(STEP_10, NAN_STEP_10.replace("Step 10/   60", "Step 20/   60"))

    curve = parse_train_log(log)

    assert curve.non_finite_step == 20
    assert curve.non_finite_field == "ppl"
    assert [math.isnan(r.xent) for r in curve.reports] == [False, True]
    assert curve.reports[1].lr == 1.25e7


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [
        ("acc", "acc: 13.0", "acc: nan"),
        ("ppl", "ppl: 103.71", "ppl: inf"),
        ("xent", "xent: 4.64", "xent: -inf"),
        ("xent", "xent: 4.64", "xent: NaN"),
        ("lr", "lr: 3.13e-03", "lr: nan"),
        ("lr", "lr: 3.13e-03", "lr: inf"),
        ("aux", "aux: 0.000", "aux: nan"),
        ("attn_ent", "attn_ent: 2.065", "attn_ent: nan"),
    ],
)
def test_a_non_finite_figure_in_any_field_of_a_report_is_reported(
    field: str, old: str, new: str
) -> None:
    log = text_of(STEP_10, STEP_50.replace(old, new), STEP_100)

    curve = parse_train_log(log)

    assert curve.non_finite_step == 50
    assert curve.non_finite_field == field
    assert [r.step for r in curve.reports] == [10, 50, 100]


def test_the_first_non_finite_figure_wins() -> None:
    log = text_of(
        STEP_10,
        STEP_50.replace("xent: 4.64", "xent: nan"),
        STEP_100.replace("lr: 6.25e-03", "lr: inf"),
    )

    curve = parse_train_log(log)

    assert (curve.non_finite_step, curve.non_finite_field) == (50, "xent")


@pytest.mark.parametrize("word", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_a_non_finite_validation_perplexity_is_reported_with_the_last_report_before_it(
    word: str,
) -> None:
    log = text_of(STEP_10, STEP_50, f"[x INFO] Validation perplexity: {word}")

    curve = parse_train_log(log)

    assert curve.non_finite_step == 50
    assert curve.non_finite_field == "validation perplexity"
    assert len(curve.validations) == 1
    assert not math.isfinite(curve.validations[0])


def test_a_non_finite_validation_accuracy_is_reported() -> None:
    log = text_of(STEP_10, "[x INFO] Validation accuracy: nan")

    curve = parse_train_log(log)

    assert (curve.non_finite_step, curve.non_finite_field) == (
        10,
        "validation accuracy",
    )


def test_a_non_finite_validation_before_any_report_is_at_step_zero() -> None:
    curve = parse_train_log(text_of("[x INFO] Validation perplexity: nan"))

    assert (curve.non_finite_step, curve.non_finite_field) == (
        0,
        "validation perplexity",
    )


def test_the_real_nan_log_of_the_spike_is_non_finite_from_its_first_report() -> None:
    log = text_of(
        NAN_STEP_10,
        "[2026-11-08 10:11:22,502 INFO] Train perplexity: nan",
        "[2026-11-08 10:11:22,502 INFO] Validation perplexity: nan",
        "[2026-11-08 10:11:22,502 INFO] Validation accuracy: 0",
    )

    curve = parse_train_log(log)

    assert curve.non_finite_step == 10
    assert curve.validations[0] != curve.validations[0]  # nan
    assert curve.accuracies == (0.0,)


def test_finite_figures_report_no_problem() -> None:
    curve = parse_train_log(text_of(STEP_10, STEP_50, STEP_100, *VALIDATION_BLOCK_50))

    assert curve.non_finite_step is None
    assert curve.non_finite_field is None


def test_sents_and_bsz_and_timing_fields_are_not_mistaken_for_non_finite_figures() -> (
    None
):
    # `bsz: 1131/1231/151` is no number and must not matter.
    assert parse_train_log(text_of(STEP_10)).non_finite_step is None


def test_a_non_finite_figure_in_a_cut_last_report_is_not_read() -> None:
    cut = STEP_50[: STEP_50.index("xent:") + len("xent: n")]

    assert parse_train_log(text_of(STEP_10) + cut).non_finite_step is None


def test_a_non_finite_figure_of_an_earlier_run_is_still_reported() -> None:
    # A NaN checkpoint is the one a later attempt resumes from.
    log = text_of(STEP_10, STEP_50.replace("xent: 4.64", "xent: nan"), STEP_10, STEP_50)

    curve = parse_train_log(log)

    assert curve.runs == 2
    assert curve.non_finite_step == 50


# ---- several runs in one log ----------------------------------------------------


def test_a_log_with_two_runs_appended_gives_the_last_run() -> None:
    first = [report(10, 5.0), report(20, 4.0), report(30, 3.0)]
    second = [report(10, 5.1), report(20, 4.1)]

    curve = parse_train_log(text_of(*first, *second))

    assert [(r.step, r.xent) for r in curve.reports] == [(10, 5.1), (20, 4.1)]
    assert curve.runs == 2
    assert curve.restarted is True


def test_a_resume_from_an_earlier_checkpoint_starts_a_new_run() -> None:
    # The first attempt reached step 80, the second resumed from step_50.
    log = text_of(
        *(report(step, 5.0 - step / 100) for step in (10, 20, 30, 40, 50, 60, 70, 80)),
        *(report(step, 4.0 - step / 100) for step in (60, 70, 80, 90)),
    )

    curve = parse_train_log(log)

    assert [r.step for r in curve.reports] == [60, 70, 80, 90]
    assert curve.runs == 2


def test_a_step_equal_to_the_previous_one_starts_a_new_run() -> None:
    curve = parse_train_log(text_of(report(10, 5.0), report(10, 4.0)))

    assert [r.xent for r in curve.reports] == [4.0]
    assert curve.runs == 2


def test_a_resume_that_continues_forward_is_one_run() -> None:
    # The first attempt reached step 100 and its checkpoint; the second resumed there.
    log = text_of(report(90, 3.0), report(100, 2.9), report(110, 2.8), report(120, 2.7))

    curve = parse_train_log(log)

    assert [r.step for r in curve.reports] == [90, 100, 110, 120]
    assert curve.runs == 1


def test_raising_the_target_between_attempts_is_one_run() -> None:
    first = "Step 100/  100; acc: 52.1; ppl: 16.53; xent: 2.81; lr: 6.25e-03;"
    second = "Step 110/  200; acc: 49.6; ppl: 14.79; xent: 2.69; lr: 5.96e-03;"

    curve = parse_train_log(text_of(first, second))

    assert [(r.step, r.total) for r in curve.reports] == [(100, 100), (110, 200)]
    assert curve.runs == 1


def test_validations_and_saves_of_the_earlier_run_are_not_in_the_last_runs_curve() -> (
    None
):
    log = text_of(
        STEP_10,
        *VALIDATION_BLOCK_50,
        *EARLY_STOP_BLOCK,
        STEP_10,
        STEP_50,
    )

    curve = parse_train_log(log)

    assert [r.step for r in curve.reports] == [10, 50]
    assert curve.validations == ()
    assert curve.accuracies == ()
    assert curve.saved_steps == ()
    assert curve.early_stopped is False
    assert curve.runs == 2


def test_three_runs_give_the_last_and_count_them() -> None:
    log = text_of(report(10, 5.0), report(10, 4.0), report(10, 3.0), report(20, 2.0))

    curve = parse_train_log(log)

    assert [r.xent for r in curve.reports] == [3.0, 2.0]
    assert curve.runs == 3


# ---- loss_gate ------------------------------------------------------------------


def curve_of(
    first: float,
    last: float,
    validations: tuple[float, ...] = (80.0, 10.0),
) -> TrainCurve:
    return TrainCurve(
        reports=(
            TrainReport(10, 200, 1.0, 2.0, first, 1e-3),
            TrainReport(200, 200, 1.0, 2.0, last, 1e-3),
        ),
        validations=validations,
        runs=1,
    )


def test_the_spikes_real_curve_passes_the_gate() -> None:
    log = text_of(
        STEP_10,
        STEP_50,
        *VALIDATION_BLOCK_50,
        STEP_100,
        STEP_200,
        *VALIDATION_BLOCK_200,
    )

    assert loss_gate(parse_train_log(log)) is None


def test_a_fall_of_exactly_30_percent_passes() -> None:
    assert loss_gate(curve_of(10.0, 7.0)) is None


def test_a_fall_a_hair_short_of_30_percent_does_not() -> None:
    problem = loss_gate(curve_of(10.0, 7.0000001))

    assert problem is not None
    assert "xent went from 10.0 at step 10 to 7.0000001 at step 200" in problem
    assert "less than 30%" in problem


def test_a_fall_of_more_than_30_percent_passes() -> None:
    assert loss_gate(curve_of(10.0, 2.0)) is None


def test_a_rise_does_not_pass() -> None:
    assert loss_gate(curve_of(5.0, 6.0)) is not None


def test_no_fall_does_not_pass() -> None:
    assert loss_gate(curve_of(5.0, 5.0)) is not None


def test_the_required_fall_is_a_parameter() -> None:
    assert loss_gate(curve_of(10.0, 6.0), min_drop=0.5) is not None
    assert loss_gate(curve_of(10.0, 5.0), min_drop=0.5) is None
    assert loss_gate(curve_of(10.0, 9.9), min_drop=0.0) is None
    assert loss_gate(curve_of(10.0, 10.1), min_drop=0.0) is not None


@pytest.mark.parametrize("min_drop", [-0.1, 1.0, 1.5, float("nan")])
def test_a_required_fall_outside_zero_to_one_is_a_value_error(min_drop: float) -> None:
    with pytest.raises(ValueError, match="min_drop"):
        loss_gate(curve_of(10.0, 1.0), min_drop=min_drop)


def test_the_gate_compares_the_first_and_the_last_report_not_the_extremes() -> None:
    curve = TrainCurve(
        reports=(
            TrainReport(10, 200, 1.0, 2.0, 10.0, 1e-3),
            TrainReport(50, 200, 1.0, 2.0, 1.0, 1e-3),
            TrainReport(200, 200, 1.0, 2.0, 8.0, 1e-3),
        ),
        validations=(80.0, 10.0),
        runs=1,
    )

    assert loss_gate(curve) is not None


def test_the_last_validation_must_be_below_the_first() -> None:
    problem = loss_gate(curve_of(10.0, 2.0, validations=(10.0, 10.0)))

    assert problem is not None
    assert "validation perplexity went from 10.0 to 10.0" in problem
    assert loss_gate(curve_of(10.0, 2.0, validations=(10.0, 12.0))) is not None
    assert loss_gate(curve_of(10.0, 2.0, validations=(10.0, 40.0, 9.9))) is None


def test_a_non_finite_figure_anywhere_fails_the_gate_with_its_step() -> None:
    log = text_of(STEP_10, STEP_50.replace("xent: 4.64", "xent: nan"), STEP_200)

    problem = loss_gate(parse_train_log(log))

    assert problem == "xent is not finite at step 50"


def test_a_non_finite_validation_fails_the_gate() -> None:
    log = text_of(
        STEP_10,
        "[x INFO] Validation perplexity: nan",
        "[x INFO] Validation perplexity: 9.0",
        STEP_200,
    )

    problem = loss_gate(parse_train_log(log))

    assert problem == "validation perplexity is not finite at step 10"


@pytest.mark.parametrize(
    "curve",
    [
        curve_of(float("nan"), 1.0),
        curve_of(10.0, float("nan")),
        curve_of(float("inf"), 1.0),
        curve_of(10.0, 1.0, validations=(float("nan"), 1.0)),
        curve_of(10.0, 1.0, validations=(80.0, float("nan"))),
        curve_of(10.0, 1.0, validations=(80.0, float("-inf"))),
    ],
)
def test_a_curve_built_with_non_finite_figures_is_refused_too(
    curve: TrainCurve,
) -> None:
    assert loss_gate(curve) == "a figure the gate compares is not finite"


def test_nan_is_never_a_fall() -> None:
    # `nan <= x` is false, `nan > x` is false: only the explicit check refuses it.
    assert loss_gate(curve_of(10.0, float("nan"))) is not None


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_reports_is_a_data_error(count: int) -> None:
    curve = TrainCurve(
        reports=(TrainReport(10, 200, 1.0, 2.0, 5.0, 1e-3),)[:count],
        validations=(80.0, 10.0),
        runs=count,
    )

    with pytest.raises(DataError, match=f"{count} training report"):
        loss_gate(curve)


def test_a_single_report_in_a_log_is_a_data_error() -> None:
    with pytest.raises(DataError, match="at least two"):
        loss_gate(parse_train_log(text_of(STEP_10)))


@pytest.mark.parametrize("validations", [(), (80.0,)])
def test_fewer_than_two_validations_is_a_data_error(
    validations: tuple[float, ...],
) -> None:
    with pytest.raises(DataError, match=f"{len(validations)} validation"):
        loss_gate(curve_of(10.0, 2.0, validations=validations))


def test_the_empty_curve_is_a_data_error_not_a_pass() -> None:
    with pytest.raises(DataError):
        loss_gate(TrainCurve())


def test_a_log_of_two_appended_runs_is_judged_on_the_last_run_alone() -> None:
    good = [report(10, 5.0), report(200, 1.0)]
    bad = [report(10, 5.0), report(200, 4.9)]
    validations = [
        "[x INFO] Validation perplexity: 9.0",
        "[x INFO] Validation perplexity: 2.0",
    ]

    log = text_of(*good, *validations, *bad, *validations)

    assert loss_gate(parse_train_log(log)) is not None
