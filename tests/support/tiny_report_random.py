"""Random observations for the tiny report's property and differential tests (D15 Task 6).

`random_report(seed)` starts from the golden run and, driven by one seeded generator, cuts it short, fails
a stage with or without facts and with a reason of the closed list, changes the figures (chrF, parity,
robustness, the held-out table, the agreement of the exports), drops facts, skips the neural part, picks
the profile name and the stand-in flag and draws an elapsed time (an overrun, an impossible number). Every
record it makes is one a stage could hand back, and every document it writes must parse back.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import Report
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_NEURAL,
    PartRecord,
    StageRecord,
    fail_check,
    fail_environment,
    fail_exit,
    fail_timeout,
)
from tests.support.tiny_report_build import build
from tests.support.tiny_report_inputs import MEMORISED_DIRECTIONS, golden_results

CHECKS = (
    "planted_noise",
    "stage_skipped",
    "command_not_started",
    "service_probe",
    "neural_scores",
)
PROFILES = ("tiny", "full", "control", "a1", "t_" + "x" * 13)


def random_reason(rng: random.Random) -> str:
    choices = [
        lambda: fail_exit(rng.randint(1, 255)),
        lambda: fail_timeout(rng.randint(1, 5000)),
        lambda: FAIL_BUDGET,
        lambda: FAIL_INPUT,
        lambda: fail_check(rng.choice(CHECKS)),
        lambda: fail_environment(rng.randint(0, 9), rng.randint(0, 9)),
    ]
    return rng.choice(choices)()


# Figures a hair either side of the bounds and the floors a table writes them beside.
NEAR_SHARES = (0.95, 0.9500000000000001, 0.94999, 0.99, 0.99999, 0.9999999999999999)
NEAR_SCORES = (50.0, 50.00001, 49.99999, 49.99999999999999)
NEAR_DELTAS = (0.001, 0.0010000001, 0.00099999)
NEAR_DROPS = (1.0, 1.0000001, 0.9999999)


def _share(rng: random.Random) -> float:
    return rng.choice([0.0, 1.0, rng.random(), rng.random(), rng.choice(NEAR_SHARES)])


def _score(rng: random.Random) -> float:
    return rng.choice(
        [
            0.0,
            100.0,
            rng.uniform(0, 100),
            rng.uniform(40, 100),
            rng.choice(NEAR_SCORES),
        ]
    )


def _change_figures(rng: random.Random, facts: dict[str, dict[str, Any]]) -> None:
    evaluation = facts.get("evaluate")
    if evaluation:
        for direction in MEMORISED_DIRECTIONS:
            for system in ("int8", "float32"):
                score = _score(rng)
                evaluation["memorised_chrf"][direction][system] = score
                evaluation["memorised"][direction][system]["metrics"]["chrf"] = score
        for direction in MEMORISED_DIRECTIONS:
            if rng.random() < 0.15:
                held = evaluation["memorised_chrf"][direction]
                near = min(100.0, held["int8"] + rng.choice(NEAR_DROPS))
                held["float32"] = near
                evaluation["memorised"][direction]["float32"]["metrics"]["chrf"] = near
        for by_system in evaluation["heldout"].values():
            by_system["int8"]["metrics"]["bleu"] = _score(rng)
        evaluation["agreement"]["share"] = _share(rng)
        evaluation["int8_chrf_drop"] = {
            "least": rng.uniform(-5, 0),
            "largest": rng.choice([rng.uniform(0, 5), *NEAR_DROPS]),
        }
    parity = facts.get("parity")
    if parity:
        parity["exact_share"] = _share(rng)
        parity["mean_abs_delta"] = rng.choice(
            [0.0, 1.6e-06, rng.random() * 0.01, *NEAR_DELTAS]
        )
    robustness = facts.get("robustness")
    if robustness:
        for key in (
            "numbers_acc",
            "placeholder_acc",
            "script_purity",
            "entity_acc",
            "term_acc",
        ):
            robustness["values"][key] = rng.choice([_share(rng), None])
        robustness["values"]["mong_source_retention"] = rng.choice(
            [rng.uniform(0, 2), None]
        )


def _drop_facts(rng: random.Random, facts: dict[str, dict[str, Any]]) -> None:
    for stage in list(facts):
        if rng.random() < 0.08:
            facts[stage] = {}
        elif rng.random() < 0.05:
            keys = list(facts[stage])
            if keys:
                del facts[stage][rng.choice(keys)]


def random_results(rng: random.Random) -> list[StageResult]:
    results = golden_results()
    facts = {r.record.id: copy.deepcopy(dict(r.facts)) for r in results}
    if rng.random() < 0.6:
        _change_figures(rng, facts)
    if rng.random() < 0.3:
        _drop_facts(rng, facts)
    if rng.random() < 0.25:
        facts["evaluate"]["neural_seam"] = None
    rebuilt = []
    for result in results:
        record = result.record
        parts = tuple(record.parts)
        if facts["evaluate"].get("neural_seam") is None and record.id == "evaluate":
            parts = tuple(
                PartRecord(p.id, "skipped", SKIP_NO_NEURAL)
                if p.id == "neural_seam"
                else p
                for p in parts
            )
        rebuilt.append(
            StageResult(
                StageRecord(
                    record.id, record.state, record.reason, record.detail, parts
                ),
                facts[record.id],
            )
        )
    results = rebuilt
    if rng.random() < 0.45:
        index = rng.randrange(len(STAGE_IDS))
        stage_id = STAGE_IDS[index]
        parts = _failed_parts(rng, stage_id)
        keep = copy.deepcopy(dict(results[index].facts)) if rng.random() < 0.5 else {}
        record = StageRecord(stage_id, "fail", random_reason(rng), {}, parts)
        results[index] = StageResult(record, keep)
        if rng.random() < 0.7:
            for later in range(index + 1, len(results)):
                results[later] = StageResult(
                    StageRecord(STAGE_IDS[later], "skipped", SKIP_EARLIER_FAILED), {}
                )
    elif rng.random() < 0.3:
        results = results[: rng.randint(0, len(results))]
    rng.shuffle(results)
    return results


def _failed_parts(rng: random.Random, stage_id: str) -> tuple[PartRecord, ...]:
    wanted = PART_IDS.get(stage_id, ())
    if not wanted or rng.random() < 0.4:
        return ()
    done = rng.randint(0, len(wanted))
    parts = [PartRecord(p, "ok") for p in wanted[:done]]
    if done < len(wanted):
        parts.append(PartRecord(wanted[done], "fail", random_reason(rng)))
    return tuple(parts)


def random_elapsed(rng: random.Random) -> float:
    return rng.choice(
        [
            rng.uniform(0, 1200),
            rng.uniform(0, 5000),
            1200.0,
            0.0,
            -1.0,
            math.nan,
            math.inf,
            rng.uniform(100, 400),
        ]
    )


def random_report(seed: int) -> Report:
    rng = random.Random(seed)
    results = random_results(rng)
    return build(
        results,
        elapsed=random_elapsed(rng),
        qe_fake=rng.random() < 0.5,
        profile=rng.choice(PROFILES),
    )
