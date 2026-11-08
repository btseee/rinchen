"""Tests for `rinchen.train.launch_command`: the `command.sh` an operator runs by hand.

The text is pure, so the golden strings are exact. Bash itself checks them twice: `bash
-n` must accept every variant, and one test runs the script against a stand-in `eole`
that records its arguments and environment. The `eole` path is injected; no test starts
Eole.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rinchen.train.launch_command import render_command_sh

RUN = Path("/runs/t1")
HERE = Path("/work")
EOLE = "/venv/bin/eole"


def render(**overrides: Any) -> str:
    arguments: dict[str, Any] = {
        "run_dir": RUN,
        "working_directory": HERE,
        "eole": EOLE,
        "config": RUN / "launch" / "train-1.yaml",
        "src_vocab": RUN / "vocab.src",
        "threads": 2,
        "cuda_visible_devices": "",
        "attempt": 1,
        "resumes_from": None,
    }
    arguments.update(overrides)
    return render_command_sh(**arguments)


CPU_FRESH = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# Unlike `rinchen train launch` it writes no launch.json and does not look for NaN.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
cd /work
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml
"""

EIGHT_GPUS = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# Unlike `rinchen train launch` it writes no launch.json and does not look for NaN.
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /work
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml
"""

RESUME = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 3 of the run in "/runs/t1".
# It resumes from step 150 (training.train_from of the config it names).
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# Unlike `rinchen train launch` it writes no launch.json and does not look for NaN.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
cd /work
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-3.yaml -n_sample -1
/venv/bin/eole train -config /runs/t1/launch/train-3.yaml
"""

GPU_NOT_REQUESTED = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# Unlike `rinchen train launch` it writes no launch.json and does not look for NaN.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd /work
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml
"""


def test_a_cpu_run_from_scratch_is_rendered_exactly() -> None:
    assert render() == CPU_FRESH


def test_an_eight_gpu_run_exports_the_requested_devices_and_its_thread_count() -> None:
    assert render(threads=4, cuda_visible_devices="0,1,2,3,4,5,6,7") == EIGHT_GPUS


def test_a_resume_names_its_step_and_its_own_attempt_config() -> None:
    text = render(attempt=3, resumes_from=150, config=RUN / "launch" / "train-3.yaml")

    assert text == RESUME


def test_devices_nobody_asked_for_are_left_to_the_environment() -> None:
    assert render(cuda_visible_devices=None) == GPU_NOT_REQUESTED


def test_the_text_is_the_same_every_time() -> None:
    assert render() == render()


def test_the_text_ends_with_one_line_feed_and_has_no_carriage_return() -> None:
    text = render()

    assert text.endswith("train-1.yaml\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text


def test_the_guarded_vocabulary_line_comes_before_the_training_line() -> None:
    lines = render().splitlines()

    vocab = next(i for i, line in enumerate(lines) if "build_vocab" in line)
    train = next(i for i, line in enumerate(lines) if " train -config " in line)
    assert vocab < train
    assert lines[vocab].startswith("[ -f /runs/t1/vocab.src ] || ")


def test_the_vocabulary_is_built_on_the_config_of_the_same_attempt() -> None:
    text = render(attempt=7, config=RUN / "launch" / "train-7.yaml")

    assert text.count("/runs/t1/launch/train-7.yaml") == 2
    assert "train-1.yaml" not in text


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_dir": Path("runs/t1")},
        {"working_directory": Path("work")},
        {"eole": "eole"},
        {"config": Path("launch/train-1.yaml")},
        {"src_vocab": Path("vocab.src")},
    ],
)
def test_a_relative_path_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="absolute"):
        render(**overrides)


@pytest.mark.parametrize("threads", [0, -1])
def test_fewer_than_one_thread_is_refused(threads: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        render(threads=threads)


def test_a_path_with_spaces_and_quotes_is_quoted_for_bash() -> None:
    run = Path("/runs/it's here")

    text = render(
        run_dir=run, config=run / "launch" / "train-1.yaml", src_vocab=run / "v s"
    )

    assert "'/runs/it'\"'\"'s here/v s'" in text
    assert "-config '/runs/it'\"'\"'s here/launch/train-1.yaml'" in text


def test_a_line_break_in_the_run_directory_stays_inside_the_comment_that_names_it() -> (
    None
):
    run = Path("/runs/a\nrm -rf x")

    text = render(
        run_dir=run, config=run / "launch" / "train-1.yaml", src_vocab=run / "v"
    )

    assert text.splitlines()[1] == (
        '# Written by `rinchen train launch` for attempt 1 of the run in "/runs/a\\nrm -rf x".'
    )


# ---- bash reads it --------------------------------------------------------------

BASH = shutil.which("bash")


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "command.sh"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"threads": 4, "cuda_visible_devices": "0,1,2,3,4,5,6,7"},
        {"attempt": 3, "resumes_from": 150},
        {"cuda_visible_devices": None},
        {"run_dir": Path("/runs/it's here"), "src_vocab": Path("/runs/it's here/v")},
    ],
)
def test_bash_accepts_every_variant_as_a_script(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    assert BASH is not None, "bash is needed to check the script"
    script = write(tmp_path, render(**overrides))

    done = subprocess.run(
        [BASH, "-n", str(script)], capture_output=True, text=True, check=False
    )

    assert (done.returncode, done.stderr) == (0, "")


STAND_IN = """\
#!/usr/bin/env bash
{
  printf 'ARGS'
  printf ' <%s>' "$@"
  printf '\\n'
  printf 'ENV CUDA=%s OMP=%s MKL=%s\\n' "${CUDA_VISIBLE_DEVICES-unset}" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"
} >> "$STAND_IN_LOG"
exit "${STAND_IN_EXIT:-0}"
"""


def run_script(
    tmp_path: Path,
    *,
    vocab_exists: bool,
    exit_code: int = 0,
    run_name: str = "the run",
    **overrides: Any,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the rendered script with a stand-in `eole` (in a directory with a space in
    its name) and return the process and the text the stand-in recorded."""
    assert BASH is not None
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    eole = bin_dir / "eole"
    eole.write_text(STAND_IN, encoding="utf-8", newline="\n")
    eole.chmod(0o755)
    run_dir = tmp_path / run_name
    (run_dir / "launch").mkdir(parents=True)
    vocab = run_dir / "vocab.src"
    if vocab_exists:
        vocab.write_text("x", encoding="utf-8")
    log = tmp_path / "stand-in.log"
    script = write(
        tmp_path,
        render(
            run_dir=run_dir,
            working_directory=tmp_path,
            eole=str(eole),
            config=run_dir / "launch" / "train-1.yaml",
            src_vocab=vocab,
            **overrides,
        ),
    )
    env = {
        "PATH": os.environ["PATH"],
        "STAND_IN_LOG": str(log),
        "STAND_IN_EXIT": str(exit_code),
        "CUDA_VISIBLE_DEVICES": "should be replaced",
    }
    done = subprocess.run(
        [BASH, str(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd="/",
    )
    return done, log.read_text(encoding="utf-8") if log.exists() else ""


def test_the_script_builds_the_vocabulary_then_trains_when_the_vocabulary_is_missing(
    tmp_path: Path,
) -> None:
    done, text = run_script(tmp_path, vocab_exists=False)
    lines = text.splitlines()

    config = f"<{tmp_path}/the run/launch/train-1.yaml>"
    assert done.returncode == 0, done.stderr
    assert lines == [
        f"ARGS <build_vocab> <-config> {config} <-n_sample> <-1>",
        "ENV CUDA= OMP=2 MKL=2",
        f"ARGS <train> <-config> {config}",
        "ENV CUDA= OMP=2 MKL=2",
    ]


def test_the_script_only_trains_when_the_vocabulary_is_there(tmp_path: Path) -> None:
    done, text = run_script(tmp_path, vocab_exists=True)
    lines = text.splitlines()

    assert done.returncode == 0, done.stderr
    assert lines == [
        f"ARGS <train> <-config> <{tmp_path}/the run/launch/train-1.yaml>",
        "ENV CUDA= OMP=2 MKL=2",
    ]


def test_the_script_exports_the_requested_devices_and_threads(tmp_path: Path) -> None:
    _, text = run_script(
        tmp_path,
        vocab_exists=True,
        threads=4,
        cuda_visible_devices="0,1,2,3,4,5,6,7",
    )
    lines = text.splitlines()

    assert lines[-1] == "ENV CUDA=0,1,2,3,4,5,6,7 OMP=4 MKL=4"


def test_the_script_leaves_the_devices_alone_when_none_were_requested(
    tmp_path: Path,
) -> None:
    _, text = run_script(tmp_path, vocab_exists=True, cuda_visible_devices=None)
    lines = text.splitlines()

    assert lines[-1] == "ENV CUDA=should be replaced OMP=2 MKL=2"


def test_the_script_stops_at_a_failing_command_and_passes_its_status_on(
    tmp_path: Path,
) -> None:
    done, text = run_script(tmp_path, vocab_exists=False, exit_code=3)
    lines = text.splitlines()

    assert done.returncode == 3
    # The vocabulary command failed, so training never started.
    assert [line for line in lines if line.startswith("ARGS")] == [
        (
            f"ARGS <build_vocab> <-config> <{tmp_path}/the run/launch/train-1.yaml> "
            "<-n_sample> <-1>"
        )
    ]


def test_the_script_runs_from_the_working_directory_it_was_written_for(
    tmp_path: Path,
) -> None:
    assert BASH is not None
    here = tmp_path / "somewhere else"
    here.mkdir()
    marker = tmp_path / "cwd.txt"
    fake = tmp_path / "eole"
    fake.write_text(f'#!/usr/bin/env bash\npwd > "{marker}"\n', encoding="utf-8")
    fake.chmod(0o755)
    script = write(
        tmp_path,
        render(
            run_dir=tmp_path,
            working_directory=here,
            eole=str(fake),
            config=tmp_path / "train-1.yaml",
            src_vocab=tmp_path / "vocab.src",
        ),
    )

    subprocess.run([BASH, str(script)], check=True, capture_output=True, cwd="/")

    assert marker.read_text(encoding="utf-8").strip() == str(here)


@pytest.mark.parametrize(
    "run_name",
    [
        "it's a run",
        'say "hi"',
        "line\nbreak",
        "tab\there",
        "back\\slash",
        "a;b&c|d",
        "*",
    ],
)
def test_the_script_hands_eole_a_run_directory_with_any_character_in_one_piece(
    tmp_path: Path, run_name: str
) -> None:
    done, text = run_script(tmp_path, vocab_exists=True, run_name=run_name)

    assert done.returncode == 0, done.stderr
    config = tmp_path / run_name / "launch" / "train-1.yaml"
    assert text.startswith(f"ARGS <train> <-config> <{config}>\nENV ")
