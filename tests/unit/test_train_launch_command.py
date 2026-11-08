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
        "attempt_log": RUN / "launch" / "attempt-1.log",
        "build_vocab_log": RUN / "launch" / "build_vocab.log",
        "threads": 2,
        "cuda_visible_devices": "",
        "attempt": 1,
        "resumes_from": None,
    }
    arguments.update(overrides)
    return render_command_sh(**arguments)


def render_in(run: Path, **overrides: Any) -> str:
    """`render` for the run directory `run`, with every path of the script below it."""
    paths = {
        "run_dir": run,
        "config": run / "launch" / "train-1.yaml",
        "src_vocab": run / "vocab.src",
        "attempt_log": run / "launch" / "attempt-1.log",
        "build_vocab_log": run / "launch" / "build_vocab.log",
    }
    return render(**{**paths, **overrides})


CPU_FRESH = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# launch.json beside it records the run, so `rinchen train launch` can resume a run started
# with this script. The output of `eole train` is appended to "launch/attempt-1.log", which
# `rinchen train launch` reads for a NaN or an early stop when it takes the run over.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
cd /work
mkdir -p /runs/t1/launch
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1 2>&1 | tee -a /runs/t1/launch/build_vocab.log
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml 2>&1 | tee -a /runs/t1/launch/attempt-1.log
"""

EIGHT_GPUS = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# launch.json beside it records the run, so `rinchen train launch` can resume a run started
# with this script. The output of `eole train` is appended to "launch/attempt-1.log", which
# `rinchen train launch` reads for a NaN or an early stop when it takes the run over.
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /work
mkdir -p /runs/t1/launch
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1 2>&1 | tee -a /runs/t1/launch/build_vocab.log
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml 2>&1 | tee -a /runs/t1/launch/attempt-1.log
"""

RESUME = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 3 of the run in "/runs/t1".
# It resumes from step 150 (training.train_from of the config it names).
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# launch.json beside it records the run, so `rinchen train launch` can resume a run started
# with this script. The output of `eole train` is appended to "launch/attempt-3.log", which
# `rinchen train launch` reads for a NaN or an early stop when it takes the run over.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
cd /work
mkdir -p /runs/t1/launch
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-3.yaml -n_sample -1 2>&1 | tee -a /runs/t1/launch/build_vocab.log
/venv/bin/eole train -config /runs/t1/launch/train-3.yaml 2>&1 | tee -a /runs/t1/launch/attempt-3.log
"""

GPU_NOT_REQUESTED = """\
#!/usr/bin/env bash
# Written by `rinchen train launch` for attempt 1 of the run in "/runs/t1".
# It starts the run from scratch: no checkpoint of it was complete when this was written.
# What it resumes from is fixed at that moment: write it again with
# `rinchen train launch --write-only` after the run changed.
# launch.json beside it records the run, so `rinchen train launch` can resume a run started
# with this script. The output of `eole train` is appended to "launch/attempt-1.log", which
# `rinchen train launch` reads for a NaN or an early stop when it takes the run over.
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd /work
mkdir -p /runs/t1/launch
[ -f /runs/t1/vocab.src ] || /venv/bin/eole build_vocab -config /runs/t1/launch/train-1.yaml -n_sample -1 2>&1 | tee -a /runs/t1/launch/build_vocab.log
/venv/bin/eole train -config /runs/t1/launch/train-1.yaml 2>&1 | tee -a /runs/t1/launch/attempt-1.log
"""


def test_a_cpu_run_from_scratch_is_rendered_exactly() -> None:
    assert render() == CPU_FRESH


def test_an_eight_gpu_run_exports_the_requested_devices_and_its_thread_count() -> None:
    assert render(threads=4, cuda_visible_devices="0,1,2,3,4,5,6,7") == EIGHT_GPUS


def test_a_resume_names_its_step_and_its_own_attempt_config_and_log() -> None:
    text = render(
        attempt=3,
        resumes_from=150,
        config=RUN / "launch" / "train-3.yaml",
        attempt_log=RUN / "launch" / "attempt-3.log",
    )

    assert text == RESUME


def test_devices_nobody_asked_for_are_left_to_the_environment() -> None:
    assert render(cuda_visible_devices=None) == GPU_NOT_REQUESTED


def test_the_text_is_the_same_every_time() -> None:
    assert render() == render()


def test_the_text_ends_with_one_line_feed_and_has_no_carriage_return() -> None:
    text = render()

    assert text.endswith("attempt-1.log\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text


def test_the_guarded_vocabulary_line_comes_before_the_training_line() -> None:
    lines = render().splitlines()

    vocab = next(i for i, line in enumerate(lines) if "build_vocab -config" in line)
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
        {"attempt_log": Path("launch/attempt-1.log")},
        {"build_vocab_log": Path("launch/build_vocab.log")},
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

    text = render_in(run, src_vocab=run / "v s")

    assert "'/runs/it'\"'\"'s here/v s'" in text
    assert "-config '/runs/it'\"'\"'s here/launch/train-1.yaml'" in text


def test_the_log_arguments_and_the_launch_directory_are_quoted_for_bash() -> None:
    run = Path("/runs/it's here")

    lines = render_in(run).splitlines()

    assert lines[-3:] == [
        "mkdir -p '/runs/it'\"'\"'s here/launch'",
        (
            "[ -f '/runs/it'\"'\"'s here/vocab.src' ] || /venv/bin/eole build_vocab "
            "-config '/runs/it'\"'\"'s here/launch/train-1.yaml' -n_sample -1 2>&1 | "
            "tee -a '/runs/it'\"'\"'s here/launch/build_vocab.log'"
        ),
        (
            "/venv/bin/eole train -config '/runs/it'\"'\"'s here/launch/train-1.yaml' "
            "2>&1 | tee -a '/runs/it'\"'\"'s here/launch/attempt-1.log'"
        ),
    ]


def test_a_non_ascii_run_directory_with_a_space_is_quoted_in_every_path() -> None:
    run = Path("/runs/\u0431\u0430\u0433\u0448 \u0442\u0435\u0441\u0442")

    lines = render_in(run).splitlines()

    quoted = "'/runs/\u0431\u0430\u0433\u0448 \u0442\u0435\u0441\u0442"
    assert lines[-3] == f"mkdir -p {quoted}/launch'"
    assert lines[-1] == (
        f"/venv/bin/eole train -config {quoted}/launch/train-1.yaml' "
        f"2>&1 | tee -a {quoted}/launch/attempt-1.log'"
    )


def test_a_line_break_in_the_run_directory_stays_inside_the_comment_that_names_it() -> (
    None
):
    run = Path("/runs/a\nrm -rf x")

    text = render_in(run, src_vocab=run / "v")

    assert text.splitlines()[1] == (
        '# Written by `rinchen train launch` for attempt 1 of the run in "/runs/a\\nrm -rf x".'
    )


def test_the_header_names_the_log_that_launch_reads() -> None:
    text = render(attempt=4, attempt_log=RUN / "launch" / "attempt-4.log")

    assert 'is appended to "launch/attempt-4.log", which' in text
    assert "`rinchen train launch` reads for a NaN or an early stop" in text
    assert "does not look for NaN" not in text


def test_a_line_break_in_the_log_name_stays_inside_the_header_comment() -> None:
    text = render(attempt_log=RUN / "launch" / "attempt-1.log\nrm -rf x")

    header = text.split("set -euo pipefail")[0].splitlines()
    assert all(line.startswith("#") for line in header)
    assert 'launch/attempt-1.log\\nrm -rf x"' in "\n".join(header)


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
        {"run": Path("/runs/it's here")},
        {"run": Path("/runs/багш тест")},
    ],
)
def test_bash_accepts_every_variant_as_a_script(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    assert BASH is not None, "bash is needed to check the script"
    run = overrides.get("run")
    options = {name: value for name, value in overrides.items() if name != "run"}
    script = write(
        tmp_path, render(**options) if run is None else render_in(run, **options)
    )

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
echo "$1 says hello on stdout"
echo "$1 says hello on stderr" >&2
exit "${STAND_IN_EXIT:-0}"
"""


def run_script(
    tmp_path: Path,
    *,
    vocab_exists: bool,
    exit_code: int = 0,
    run_name: str = "the run",
    launch_directory_exists: bool = True,
    times: int = 1,
    **overrides: Any,
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the rendered script `times` times with a stand-in `eole` (in a directory with
    a space in its name) and return the last process and the text the stand-in
    recorded. The run directory is `tmp_path / run_name`, with its `launch` directory
    unless `launch_directory_exists` is false."""
    assert BASH is not None
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    eole = bin_dir / "eole"
    eole.write_text(STAND_IN, encoding="utf-8", newline="\n")
    eole.chmod(0o755)
    run_dir = tmp_path / run_name
    (run_dir / "launch").mkdir(parents=True)
    if not launch_directory_exists:
        (run_dir / "launch").rmdir()
    vocab = run_dir / "vocab.src"
    if vocab_exists:
        vocab.write_text("x", encoding="utf-8")
    log = tmp_path / "stand-in.log"
    script = write(
        tmp_path,
        render_in(
            run_dir,
            working_directory=tmp_path,
            eole=str(eole),
            **overrides,
        ),
    )
    env = {
        "PATH": os.environ["PATH"],
        "STAND_IN_LOG": str(log),
        "STAND_IN_EXIT": str(exit_code),
        "CUDA_VISIBLE_DEVICES": "should be replaced",
    }
    for _ in range(times):
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
            attempt_log=tmp_path / "attempt-1.log",
            build_vocab_log=tmp_path / "build_vocab.log",
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


# ---- the output goes to the logs that `launch` reads -------------------------------


def logs_of(tmp_path: Path, run_name: str = "the run") -> tuple[str, str]:
    """The text of `build_vocab.log` and of `attempt-1.log` of the run, empty when a log
    does not exist."""
    launch_dir = tmp_path / run_name / "launch"

    def text(name: str) -> str:
        path = launch_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return text("build_vocab.log"), text("attempt-1.log")


def test_the_output_of_eole_train_is_appended_to_the_attempt_log_and_still_shown(
    tmp_path: Path,
) -> None:
    done, _ = run_script(tmp_path, vocab_exists=True)

    assert logs_of(tmp_path) == (
        "",
        "train says hello on stdout\ntrain says hello on stderr\n",
    )
    assert "train says hello on stdout\n" in done.stdout


def test_the_output_of_the_vocabulary_command_goes_to_its_own_log(
    tmp_path: Path,
) -> None:
    run_script(tmp_path, vocab_exists=False)

    assert logs_of(tmp_path) == (
        "build_vocab says hello on stdout\nbuild_vocab says hello on stderr\n",
        "train says hello on stdout\ntrain says hello on stderr\n",
    )


def test_a_second_run_of_the_script_appends_to_the_attempt_log(tmp_path: Path) -> None:
    run_script(tmp_path, vocab_exists=True, times=2)

    assert logs_of(tmp_path)[1] == (
        "train says hello on stdout\ntrain says hello on stderr\n" * 2
    )


def test_the_script_makes_the_launch_directory_when_it_is_missing(
    tmp_path: Path,
) -> None:
    done, _ = run_script(tmp_path, vocab_exists=True, launch_directory_exists=False)

    assert done.returncode == 0, done.stderr
    assert logs_of(tmp_path)[1].startswith("train says hello")


def test_a_failing_training_leaves_its_status_after_the_log_is_written(
    tmp_path: Path,
) -> None:
    done, _ = run_script(tmp_path, vocab_exists=True, exit_code=5)

    # The status is eole's, not the status of `tee` at the end of the pipe.
    assert done.returncode == 5
    assert logs_of(tmp_path)[1] == (
        "train says hello on stdout\ntrain says hello on stderr\n"
    )


def test_a_failing_vocabulary_command_stops_the_script_with_its_status(
    tmp_path: Path,
) -> None:
    done, _ = run_script(tmp_path, vocab_exists=False, exit_code=3)

    assert done.returncode == 3
    assert logs_of(tmp_path)[1] == ""  # training never started
    assert logs_of(tmp_path)[0].startswith("build_vocab says hello")
