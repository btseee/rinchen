"""A small project on disk for the D14 tests: a `pyproject.toml`, a `uv.lock` and a `tests/`
directory with a few tiny test files whose outcomes are known (passing, failing, erroring,
skipping, sleeping, marked `network`, one that asserts the hidden environment, an empty
directory), and the helpers that look at the processes the sleeping test leaves.

The project is a stand-in for the repository: the reruns of `rinchen.maintain.reruns` run
`python -m pytest` in it, and a real run selects the files named here and never the repository's
own `tests/`. For every pytest target of a configuration the project also holds an empty test
file (a target that ends in `.py`) or an empty directory, so that the shipped table's targets
exist in it and a fake runner can be scripted against them; a real pytest run over such a target
collects nothing.

The sleeping test starts a grandchild, writes both pids next to its own file, and sleeps for
two minutes: a rerun that is killed on its timeout must take both away, and `read_pids` hands
the two to the test that checks it. The helpers read `/proc`, so the tests that use them are
Linux-only.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from rinchen.config.maintain import MaintainConfig

SAMPLE_LOCK: Final = (
    Path(__file__).resolve().parents[1] / "fixtures" / "maintain" / "uv.lock.sample"
)
PYPROJECT: Final = '[project]\nname = "scratch-project"\nversion = "0.0.0"\n'

EMPTY_TARGET: Final = "tests/empty"
PASSES_A: Final = "tests/test_passes_a.py"
PASSES_B: Final = "tests/test_passes_b.py"
FAILS: Final = "tests/test_fails.py"
ERRORS: Final = "tests/test_errors.py"
SKIPS: Final = "tests/test_skips.py"
SLEEPS: Final = "tests/test_sleeps.py"
NETWORK_ONLY: Final = "tests/test_network_only.py"
HIDDEN_ENVIRONMENT_TEST: Final = "tests/test_hidden_environment.py"
PIDS_FILE: Final = str(Path(SLEEPS).with_suffix(".pids"))

STANDARD_TESTS: Final[Mapping[str, str]] = {
    PASSES_A: "def test_passes():\n    assert 1 + 1 == 2\n",
    PASSES_B: "def test_also_passes():\n    assert 'a' + 'b' == 'ab'\n",
    FAILS: "def test_fails():\n    assert 1 + 1 == 3\n",
    ERRORS: (
        "import pytest\n\n\n"
        "@pytest.fixture\n"
        "def broken():\n"
        "    raise RuntimeError('the fixture is broken')\n\n\n"
        "def test_errors(broken):\n"
        "    pass\n"
    ),
    SKIPS: (
        "import pytest\n\n\ndef test_skips():\n    pytest.skip('skipped on purpose')\n"
    ),
    NETWORK_ONLY: (
        "import pytest\n\n\n"
        "@pytest.mark.network\n"
        "def test_would_need_the_network():\n"
        "    assert False\n"
    ),
    HIDDEN_ENVIRONMENT_TEST: (
        "import os\n\n\n"
        "def test_the_device_and_the_hub_are_hidden():\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "    assert os.environ['HF_DATASETS_OFFLINE'] == '1'\n"
        "    assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
    ),
    SLEEPS: (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n\n"
        "def test_sleeps():\n"
        "    grandchild = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(120)']\n"
        "    )\n"
        "    pids = Path(__file__).with_suffix('.pids')\n"
        "    pids.write_text(f'{os.getpid()} {grandchild.pid}')\n"
        "    time.sleep(120)\n"
    ),
}


def make_project(
    tmp_path: Path,
    *,
    config: MaintainConfig | None = None,
    lock: bytes | None = None,
    tests: Mapping[str, str] | None = None,
) -> Path:
    """A project directory (`<tmp_path>/project`) and its path.

    `lock` is the text of `uv.lock` (the hand-written sample by default); `tests` maps paths
    under the project (`tests/test_x.py`) to source text (`STANDARD_TESTS` by default, `{}` for
    none); `config` adds an empty file or directory for each of its pytest targets that does
    not exist yet.
    """
    project = tmp_path / "project"
    (project / EMPTY_TARGET).mkdir(parents=True)
    (project / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (project / "uv.lock").write_bytes(
        SAMPLE_LOCK.read_bytes() if lock is None else lock
    )
    for relative, source in (STANDARD_TESTS if tests is None else tests).items():
        written = project / relative
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text(source, encoding="utf-8")
    if config is not None:
        for rerun in config.reruns:
            for target in rerun.targets:
                _ensure_target(project / target)
    return project


def _ensure_target(path: Path) -> None:
    if path.exists():
        return
    if path.suffix == ".py":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    else:
        path.mkdir(parents=True)


def parent_environment() -> dict[str, str]:
    """The environment of the test process without what steers a pytest started from it: the
    `PYTEST_*` variables, so that a rerun in a scratch project is not given the options of the
    run that tests it."""
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTEST_", "COV_CORE_"))
    }


# ---- the processes the sleeping test leaves ------------------------------------------------------


def read_pids(pids_file: Path) -> list[int]:
    """The pids a process wrote to `pids_file` (the sleeping test's is `project / PIDS_FILE`), or
    a failure that says it did not get to write them."""
    if not pids_file.exists():
        raise AssertionError("the sleeping test did not get to write its pids in time")
    return [int(pid) for pid in pids_file.read_text().split()]


def is_running(pid: int) -> bool:
    """Whether a process with this pid exists and is not a zombie (a zombie has exited; only its
    parent has not reaped it)."""
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except ProcessLookupError, FileNotFoundError:
        return False
    return state != "Z"


def wait_until_gone(pid: int, seconds: float = 5.0) -> bool:
    """Whether the process is gone within `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not is_running(pid):
            return True
        time.sleep(0.05)
    return not is_running(pid)


def kill_quietly(pids: list[int]) -> None:
    """SIGKILL each pid that still exists: the cleanup of a test whose assertion may have failed
    with the processes alive. Only pids the test itself started are ever passed."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
