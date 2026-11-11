"""Property test of a pytest rerun over the REAL pytest (D14, spec M8): a scratch project of 0 to 6
passing, failing, erroring and skipping tests, with a random floor and skip allowance, gives the
state, the reason and the counters that a small pure-Python model of the rule predicts from the
generated numbers.

The model is the rule of M8 and knows nothing of pytest or of JUnit: failures and errors first,
then the floor of passed tests, then the skip allowance; exit codes follow from the counters (1
for a failure or an error, 5 when nothing was collected, else 0) and decide nothing the counters
have not already decided. Each of the 50 generated cases (and the six explicit ones, one for each
outcome) writes one test file under `tmp_path`, runs the real runner and the real pytest over it
with the GPU hidden, and never selects the repository's own tests. Under the `no_network` guard
like every D14 test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from rinchen.maintain.commands import SubprocessRunner
from rinchen.maintain.reruns import TestCounts, run_pytest_rerun
from rinchen.maintain.states import fail_floor, fail_skipped, fail_tests
from tests.support.maintain_fakes import pytest_rerun
from tests.support.maintain_project import make_project, parent_environment
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = [pytest.mark.usefixtures("no_network"), pytest.mark.subprocess]

SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
GENERATED = "tests/test_generated.py"
HEADER = (
    "import pytest\n\n\n"
    "@pytest.fixture\n"
    "def broken():\n"
    "    raise RuntimeError('the fixture is broken')\n"
)


def generated_source(passing: int, failing: int, erroring: int, skipping: int) -> str:
    """A test module with this many tests of each kind."""
    parts = [HEADER]
    parts += [f"\n\ndef test_passes_{n}():\n    assert True\n" for n in range(passing)]
    parts += [f"\n\ndef test_fails_{n}():\n    assert False\n" for n in range(failing)]
    parts += [f"\n\ndef test_errors_{n}(broken):\n    pass\n" for n in range(erroring)]
    parts += [
        f"\n\ndef test_skips_{n}():\n    pytest.skip('on purpose')\n"
        for n in range(skipping)
    ]
    return "".join(parts)


def predicted(
    passing: int,
    failing: int,
    erroring: int,
    skipping: int,
    floor: int,
    allowance: int,
) -> tuple[str, str | None]:
    """The state and reason the rule of M8 gives for these numbers."""
    if failing + erroring > 0:
        return "fail", fail_tests(failing, erroring)
    if passing < floor:
        return "fail", fail_floor(passing, floor)
    if skipping > allowance:
        return "fail", fail_skipped(skipping, allowance)
    return "ok", None


# Failures and errors are left out more often than not, so that the floor, the skip allowance and
# the pass are reached as often as the failures that decide first.
FAILURES = st.sampled_from([0, 0, 0, 1, 2, 3])


# Six explicit cases (each outcome once, whatever the generator draws) and 50 generated ones.
@SETTINGS
@given(
    passing=st.integers(0, 6),
    failing=FAILURES,
    erroring=FAILURES,
    skipping=st.integers(0, 6),
    floor=st.integers(1, 8),
    allowance=st.integers(0, 6),
)
@example(passing=3, failing=0, erroring=0, skipping=1, floor=3, allowance=1)  # ok
@example(passing=3, failing=1, erroring=0, skipping=0, floor=1, allowance=0)  # failure
@example(passing=2, failing=0, erroring=1, skipping=0, floor=1, allowance=0)  # error
@example(passing=2, failing=0, erroring=0, skipping=0, floor=3, allowance=0)  # floor
@example(passing=3, failing=0, erroring=0, skipping=2, floor=3, allowance=1)  # skips
@example(passing=0, failing=0, erroring=0, skipping=0, floor=1, allowance=0)  # exit 5
def test_a_rerun_over_a_generated_project_equals_the_model(
    tmp_path_factory: pytest.TempPathFactory,
    passing: int,
    failing: int,
    erroring: int,
    skipping: int,
    floor: int,
    allowance: int,
) -> None:
    base = tmp_path_factory.mktemp("rerun", numbered=True)
    source = generated_source(passing, failing, erroring, skipping)
    project = make_project(base, tests={GENERATED: source})

    result = run_pytest_rerun(
        pytest_rerun(
            GENERATED,
            min_passed=floor,
            max_skipped=allowance,
            timeout=120,
        ),
        runner=SubprocessRunner(),
        python=sys.executable,
        project_dir=project,
        out_dir=Path(base) / "out",
        parent_env=parent_environment(),
    )

    expected = predicted(passing, failing, erroring, skipping, floor, allowance)
    assert (result.state, result.reason) == expected
    assert result.tests == TestCounts(passing, failing, erroring, skipping)
    assert result.figures is None
