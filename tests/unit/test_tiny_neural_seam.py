"""The neural seam of the `evaluate` stage (D15, spec P2 and P4, brief G5.4).

D13 built no neural scorer: scores of COMET and MetricX enter `rinchen eval run --hypotheses` through a
scores file whose header names the suite and the hypotheses file of the run by their SHA-256. With
`--qe-fake` the stage scores the hypotheses of the int8 held-out English run with the planted stand-in
(`ConceptQE`, a lookup table summed from fixed vectors that ranks nothing) into such a file, written by
D13's own writer, runs `eval run --hypotheses` over it and checks that the metric is in the result of
every direction and rests on every row scored. Without the flag the part is skipped for its closed reason
(tested in `test_tiny_eval_stages.py`).

The recording runner reads the scores file it is given with D13's own reader against the suite file and
the hypotheses file it is also given, so a file with a wrong digest, a missing segment or a segment that
was not scored is refused there as it would be by the real command. What is checked here is recomputed
from the files the stage wrote, by code that is not the stage's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.eval import suites as suite_reader
from rinchen.eval.outputs import OutputKey
from rinchen.eval.segment_scores import SCORES_SCHEMA, read_scores
from rinchen.tiny import commands, eval_stages
from rinchen.tiny.fakes import ConceptEncoder, ConceptQE
from rinchen.tiny.states import (
    FAIL_INPUT,
    SKIP_EARLIER_FAILED,
    fail_check,
    fail_exit,
    is_closed_reason,
)
from tests.support import tiny_run, tiny_run_eval
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_run_eval import hypotheses_of, run_evaluation
from tests.support.tiny_stage_inputs import shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of the support modules, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
model_outputs = tiny_run_eval.model_outputs
evaluated_inputs = tiny_run_eval.evaluated_inputs
eval_out = tiny_run_eval.eval_out

run = run_evaluation
SUITE: Final = "tiny-heldout-eng"
SYSTEM: Final = "tiny-int8"
NEURAL: Final = "tiny-int8-qe"
SEAM_KEY: Final = f"eval run {SUITE} {NEURAL}"
METRIC: Final = "metricx24_qe"
DIRECTIONS: Final = ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn")


def directory(out: Path) -> Path:
    return out.resolve() / "11_evaluate"


def hypotheses_path(out: Path) -> Path:
    return directory(out) / "runs" / SUITE / SYSTEM / "hypotheses.jsonl"


def scores_path(out: Path) -> Path:
    return directory(out) / "scores" / f"{METRIC}.jsonl"


def suite_path(out: Path) -> Path:
    return directory(out) / "suites" / f"{SUITE}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_lines(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lines[0], lines[1:]


def recomputed(out: Path, *, against: str = "source") -> list[float]:
    """The scores of the hypotheses of the held-out English run, worked out from the suite file and
    the hypotheses file with the planted stand-in, each hypothesis against its source or its
    reference."""
    suite, _ = suite_reader.read_suite(suite_path(out))
    rows = hypotheses_of(hypotheses_path(out))
    side = 0 if against == "source" else 1
    texts = []
    for row in rows:
        view = suite.view(row["direction"])
        position = view.ids.index(row["id"])
        texts.append((view.sources[position], view.references[position])[side])
    fixtures = shipped_fixtures()
    scorer = ConceptQE(ConceptEncoder(fixtures.nouns, fixtures.entities))
    return scorer.score(texts, [row["text"] for row in rows])


# ---- the scores file ----------------------------------------------------------------------------------


def test_the_header_names_the_suite_and_the_hypotheses_by_their_digests(
    eval_out: Path,
) -> None:
    run(eval_out)

    header, _ = score_lines(scores_path(eval_out))
    assert header == {
        "schema": SCORES_SCHEMA,
        "metric": METRIC,
        "suite_sha256": sha256(suite_path(eval_out)),
        "hypotheses_sha256": sha256(hypotheses_path(eval_out)),
    }


def test_the_header_digests_are_those_of_the_files_the_first_run_wrote_and_not_of_any_other(
    eval_out: Path,
) -> None:
    run(eval_out)

    header, _ = score_lines(scores_path(eval_out))
    other_suite = directory(eval_out) / "suites" / "tiny-heldout-rus.json"
    other_hypotheses = (
        directory(eval_out) / "runs" / "tiny-heldout-rus" / SYSTEM / "hypotheses.jsonl"
    )
    assert header["suite_sha256"] != sha256(other_suite)
    assert header["hypotheses_sha256"] != sha256(other_hypotheses)


def test_the_real_reader_of_the_scores_accepts_the_file(eval_out: Path) -> None:
    run(eval_out)

    rows = hypotheses_of(hypotheses_path(eval_out))
    expected = [OutputKey(row["direction"], row["id"]) for row in rows]

    found = read_scores(
        scores_path(eval_out).read_bytes(),
        metric=METRIC,
        suite_sha256=sha256(suite_path(eval_out)),
        hypotheses_sha256=sha256(hypotheses_path(eval_out)),
        expected=expected,
    )

    assert list(found) == expected
    assert len(found) == 60


def test_there_is_one_score_for_each_hypothesis_in_the_order_of_the_file(
    eval_out: Path,
) -> None:
    run(eval_out)

    _, lines = score_lines(scores_path(eval_out))
    rows = hypotheses_of(hypotheses_path(eval_out))
    assert [(line["direction"], line["id"]) for line in lines] == [
        (row["direction"], row["id"]) for row in rows
    ]
    assert sorted({line["direction"] for line in lines}) == sorted(DIRECTIONS)


def test_each_score_is_the_planted_scorers_for_the_hypothesis_against_its_source(
    eval_out: Path,
) -> None:
    run(eval_out)

    _, lines = score_lines(scores_path(eval_out))
    assert [line["score"] for line in lines] == recomputed(eval_out)


def test_the_scores_are_of_the_source_and_not_of_the_reference(eval_out: Path) -> None:
    # In the fixtures a source and its reference say the same thing to the planted scorer, so the
    # Khalkha side of the held-out English rows is rotated: now a reference says another row's thing.
    fixtures = shipped_fixtures()
    pair = "eng_Latn-khk_Cyrl"
    test_rows = [row for row in fixtures.pairs[pair] if row.split == "test"]
    shifted = {
        row.id: dataclasses.replace(row, tgt=test_rows[(n + 1) % len(test_rows)].tgt)
        for n, row in enumerate(test_rows)
    }
    rows = tuple(shifted.get(row.id, row) for row in fixtures.pairs[pair])
    altered = dataclasses.replace(fixtures, pairs={**fixtures.pairs, pair: rows})

    run(eval_out, fixtures=altered)

    from_source = recomputed(eval_out, against="source")
    from_reference = recomputed(eval_out, against="reference")
    assert from_source != from_reference
    _, lines = score_lines(scores_path(eval_out))
    assert [line["score"] for line in lines] == from_source


def test_the_scorer_is_built_from_the_nouns_and_the_entities_of_the_fixtures(
    eval_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, Any]] = []
    real = eval_stages.ConceptEncoder

    def recording(nouns: Any, entities: Any) -> Any:
        seen.append((nouns, entities))
        return real(nouns, entities)

    monkeypatch.setattr(eval_stages, "ConceptEncoder", recording)

    run(eval_out)

    fixtures = shipped_fixtures()
    assert len(seen) == 1
    assert seen[0][0] is fixtures.nouns
    assert seen[0][1] is fixtures.entities


# ---- the command --------------------------------------------------------------------------------------


def test_the_run_takes_the_hypotheses_and_the_scores_and_nothing_a_model_would(
    eval_out: Path,
) -> None:
    _, runner = run(eval_out)

    call = runner.calls[-2]
    assert call.key == SEAM_KEY
    assert list(call.argv) == commands.eval_run_hypotheses(
        suite_path(eval_out),
        hypotheses_path(eval_out),
        NEURAL,
        directory(eval_out) / "runs",
        {METRIC: scores_path(eval_out)},
        resamples=1000,
    )
    for name in ("--model", "--beam-size", "--robustness"):
        assert name not in call.argv


def test_the_seam_does_not_change_the_hypotheses_it_scores(eval_out: Path) -> None:
    _, runner = run(eval_out)
    first = hypotheses_path(eval_out).read_bytes()
    seam_run = directory(eval_out) / "runs" / SUITE / NEURAL / "hypotheses.jsonl"

    assert seam_run.read_bytes() == first
    assert runner.calls[-2].argv[
        runner.calls[-2].argv.index("--hypotheses") + 1
    ] == str(hypotheses_path(eval_out))


def test_the_seam_run_is_a_system_of_its_own_beside_the_int8_one(
    eval_out: Path,
) -> None:
    run(eval_out)

    runs = directory(eval_out) / "runs" / SUITE
    assert sorted(path.name for path in runs.iterdir()) == [SYSTEM, NEURAL]


# ---- the facts ------------------------------------------------------------------------------------------------


def test_the_facts_give_the_metric_of_each_direction_and_the_rows_it_rests_on(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out)

    _, lines = score_lines(scores_path(eval_out))
    by_direction = {
        direction: [line["score"] for line in lines if line["direction"] == direction]
        for direction in DIRECTIONS
    }
    seam = result.facts["neural_seam"]
    assert sorted(seam) == ["metric", "rows", "scores", "system"]
    assert (seam["metric"], seam["system"]) == (METRIC, NEURAL)
    assert seam["rows"] == {direction: 30 for direction in DIRECTIONS}
    assert seam["scores"] == {
        direction: sum(values) / len(values)
        for direction, values in by_direction.items()
    }
    assert result.record.detail["neural_seam"] == seam


def test_the_part_is_ok_and_the_record_lists_it(eval_out: Path) -> None:
    result, _ = run(eval_out)

    assert result.record.state == "ok"
    part = next(item for item in result.record.parts if item.id == "neural_seam")
    assert (part.state, part.reason) == ("ok", None)


# ---- what goes wrong ------------------------------------------------------------------------------------------


def parts_of(result: Any) -> dict[str, tuple[str, str | None]]:
    return {part.id: (part.state, part.reason) for part in result.record.parts}


def test_a_result_without_the_metric_fails_the_part_with_its_check(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out, ignore_scores=True)

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("neural_scores")
    assert parts_of(result) == {
        "memorised": ("ok", None),
        "heldout": ("ok", None),
        "neural_seam": ("fail", fail_check("neural_scores")),
        "report": ("skipped", SKIP_EARLIER_FAILED),
    }
    assert result.record.detail == {}
    assert "memorised" in result.facts and "heldout" in result.facts
    assert "neural_seam" not in result.facts


def alter_a_direction(change: Any) -> Any:
    def hook(call: Any) -> None:
        path = (
            Path(call.one("--out-dir")) / SUITE / NEURAL / DIRECTIONS[1] / "result.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        change(document)
        path.write_text(json.dumps(document), encoding="utf-8")

    return hook


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["metrics"].update({METRIC: None}),
        lambda d: d.update(n_segments=29, statuses={"ok": 29}),
        lambda d: d.update(n_segments=31, statuses={"ok": 31}),
    ],
    ids=["no-score-in-a-direction", "one-row-fewer", "one-row-more"],
)
def test_a_direction_whose_metric_is_missing_or_rests_on_other_rows_fails_the_part(
    eval_out: Path, change: Any
) -> None:
    result, _ = run(eval_out, after={SEAM_KEY: alter_a_direction(change)})

    assert result.record.reason == fail_check("neural_scores")
    assert parts_of(result)["neural_seam"] == ("fail", fail_check("neural_scores"))


def test_a_run_that_the_command_refuses_fails_the_part_with_its_exit_code(
    eval_out: Path,
) -> None:
    result, runner = run(eval_out, fail={SEAM_KEY: 3})

    assert result.record.reason == fail_exit(3)
    assert runner.keys[-1] == SEAM_KEY
    assert parts_of(result)["neural_seam"] == ("fail", fail_exit(3))
    assert parts_of(result)["report"] == ("skipped", SKIP_EARLIER_FAILED)


def test_a_result_that_the_command_did_not_write_is_a_failed_input(
    eval_out: Path,
) -> None:
    def remove(call: Any) -> None:
        (
            Path(call.one("--out-dir")) / SUITE / NEURAL / DIRECTIONS[0] / "result.json"
        ).unlink()

    result, _ = run(eval_out, after={SEAM_KEY: remove})

    assert result.record.reason == FAIL_INPUT
    assert parts_of(result)["neural_seam"] == ("fail", FAIL_INPUT)


def test_a_hypothesis_that_is_not_a_row_of_the_suite_is_a_failed_input_before_any_scoring(
    eval_out: Path,
) -> None:
    def add_a_row(call: Any) -> None:
        path = Path(call.one("--out-dir")) / SUITE / SYSTEM / "hypotheses.jsonl"
        extra = {
            "direction": DIRECTIONS[0],
            "id": "unknown-9",
            "status": "ok",
            "text": "x",
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(extra) + "\n")

    result, runner = run(eval_out, after={f"eval run {SUITE} {SYSTEM}": add_a_row})

    assert result.record.reason == FAIL_INPUT
    assert SEAM_KEY not in runner.keys
    assert not scores_path(eval_out).exists()


def test_hypotheses_that_are_not_readable_are_a_failed_input(eval_out: Path) -> None:
    def spoil(call: Any) -> None:
        path = Path(call.one("--out-dir")) / SUITE / SYSTEM / "hypotheses.jsonl"
        path.write_text("not json\n", encoding="utf-8")

    result, runner = run(eval_out, after={f"eval run {SUITE} {SYSTEM}": spoil})

    assert result.record.reason == FAIL_INPUT
    assert SEAM_KEY not in runner.keys


def test_every_reason_of_the_seam_is_of_the_closed_list(eval_out: Path) -> None:
    result, _ = run(eval_out, ignore_scores=True)

    assert result.record.reason is not None
    assert is_closed_reason(result.record.reason)
    assert fail_check("neural_scores") == eval_stages.NEURAL_CHECK


def test_without_the_flag_no_score_is_made_and_no_scorer_is_built(
    eval_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object) -> None:
        raise AssertionError("the planted scorer was built without the flag")

    monkeypatch.setattr(eval_stages, "ConceptEncoder", refuse)

    result, runner = run(eval_out, qe_fake=False)

    assert result.record.state == "ok"
    assert SEAM_KEY not in runner.keys
    assert not (directory(eval_out) / "scores").exists()


def test_the_scores_file_is_written_whole_and_nothing_partial_is_left(
    eval_out: Path,
) -> None:
    run(eval_out)

    assert sorted(path.name for path in scores_path(eval_out).parent.iterdir()) == [
        f"{METRIC}.jsonl"
    ]
