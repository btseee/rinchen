"""Property tests of `rinchen eval run` (D13, Task 7), over the golden suite extended with a Russian
sentence and a third direction: whatever the selection of directions and whatever directions the
outputs file holds, in whatever order its lines are, every declared direction has exactly one state
that is a function of the two, the files written hold the directions that ran and nothing else, and
`hypotheses.jsonl` is the canonical rewrite of what was used (equal to the file itself when the file
is canonical). Under the `no_network` guard like every D13 test module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.eval.outputs import Output, OutputKey, outputs_to_bytes
from tests.support.eval_cli import (
    THIRD,
    invoke,
    make_workspace,
    run_dir_of,
    three_language_suite,
)
from tests.support.eval_fakes import golden_outputs
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=40,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
DIRECTIONS = ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl", THIRD)


def outputs_of(covered: list[str]) -> dict[OutputKey, Output]:
    """Outputs for the covered directions: the golden `alpha` texts for the two golden ones and
    the source's own text (a stand-in) for the third."""
    alpha = golden_outputs("alpha")
    found: dict[OutputKey, Output] = {}
    for direction in DIRECTIONS:
        if direction not in covered:
            continue
        for index in range(10):
            key = OutputKey(direction, f"g{index + 1:02d}")
            found[key] = alpha.get(key, Output(f"русский {key.id}", "ok"))
    return found


@SETTINGS
@given(
    covered=st.lists(st.sampled_from(DIRECTIONS), unique=True, max_size=3),
    selection=st.one_of(
        st.none(),
        st.lists(st.sampled_from(DIRECTIONS), unique=True, min_size=1, max_size=3),
    ),
    seed=st.randoms(use_true_random=False),
)
def test_every_direction_has_one_state_that_is_a_function_of_the_file_and_the_selection(
    covered: list[str], selection: list[str] | None, seed: object
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = make_workspace(Path(directory))
        suite = three_language_suite(Path(directory) / "three.json")
        outputs = outputs_of(covered)
        lines = outputs_to_bytes(outputs).decode().splitlines()
        seed.shuffle(lines)  # type: ignore[attr-defined]
        file = Path(directory) / "in" / "sys.jsonl"
        file.write_text("".join(line + "\n" for line in lines), "utf-8")
        extra = [x for d in selection or [] for x in ("--direction", d)]

        result = invoke(workspace.hypotheses_args(*extra, hypotheses=file, suite=suite))

        assert result.exit_code == 0, result.output
        run_dir = run_dir_of(workspace.out_dir, "sys")
        document = json.loads((run_dir / "run.json").read_text("utf-8"))
        states = {
            entry["direction"]: entry["state"] for entry in document["directions"]
        }
        assert list(states) == list(DIRECTIONS)
        expected = {
            d: "not_selected"
            if selection is not None and d not in selection
            else ("run" if d in covered else "not_in_file")
            for d in DIRECTIONS
        }
        assert states == expected
        ran = [d for d in DIRECTIONS if expected[d] == "run"]
        written = sorted(
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("result.json")
        )
        assert written == sorted(f"{d}/result.json" for d in ran)
        canonical = outputs_to_bytes(
            {k: v for k, v in outputs.items() if k.direction in ran}
        )
        assert (run_dir / "hypotheses.jsonl").read_bytes() == canonical
        stdout_states = [
            line.split(": ", 1)[1] for line in result.stdout.splitlines()[3:6]
        ]
        for direction, line in zip(DIRECTIONS, stdout_states, strict=True):
            if expected[direction] == "run":
                assert line.startswith("chrF++ ")
            else:
                assert line == f"n/a ({expected[direction]})"
