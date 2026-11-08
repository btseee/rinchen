"""Tests for `rinchen.train.eole_run`: the Eole subprocesses every Rinchen model shares.

No test starts Eole. `run_logged` is replaced, in the module that calls it, by a
recorder that writes what the real tool would (the checkpoint, the prediction output
file), so each test pins the commands, the environment and the failure handling. The
runner's own process handling is tested with real children in
`test_utils_logged_process.py`; the converter's wrappers of these functions in
`test_script_conv_train.py`; the real Eole run is the integration tests.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train import eole_run
from rinchen.train.eole_run import (
    CHECKPOINT_FILE,
    eole_command,
    eole_commands,
    predict_files,
    predict_tokens,
    run_training,
    split_lines,
    training_environment,
    write_lines,
    written_since,
)

EOLE_NAMES = ("eole", "eole.exe")
NNBSP = chr(0x202F)
LINE_SEPARATOR = chr(0x2028)
NEXT_LINE = chr(0x0085)


@dataclass
class Launch:
    """One `run_logged` call as the fake saw it, with the files it read at launch."""

    command: list[str]
    env: dict[str, str]
    log_path: Path
    what: str
    timeout: float | None
    config: dict[str, Any]
    source_lines: list[str]  # the `src` file of a `predict`, else empty
    prefix_lines: list[str]  # the `tgt` file of a forced `predict`, else empty

    @property
    def subcommand(self) -> str:
        return self.command[1]


class FakeRunLogged:
    """Stands in for `run_logged`: records every launch, writes `output` lines to the
    log it is given, raises `failures[subcommand]` when one is set and otherwise
    leaves what the finished tool would. A finished `train` leaves a checkpoint file
    `checkpoint_age` seconds old; a finished `predict` its output file with
    `hypotheses` (one token line each) or, when that is None, the source lines.
    `writes_results` false leaves nothing."""

    def __init__(self) -> None:
        self.launches: list[Launch] = []
        self.failures: dict[str, UpstreamError] = {}
        self.output: Sequence[str] = ("a log line",)
        self.writes_results = True
        self.checkpoint_age = 0.0
        self.hypotheses: list[str] | None = None

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str],
        log_path: Path,
        what: str,
        timeout: float | None = None,
    ) -> None:
        config = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        launch = Launch(
            command=list(command),
            env=dict(env),
            log_path=log_path,
            what=what,
            timeout=timeout,
            config=config,
            source_lines=self._lines_of(config, "src"),
            prefix_lines=self._lines_of(config, "tgt"),
        )
        self.launches.append(launch)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "".join(f"{line}\n" for line in self.output), encoding="utf-8"
        )
        if launch.subcommand in self.failures:
            raise self.failures[launch.subcommand]
        if self.writes_results:
            self._leave_results(launch)

    @staticmethod
    def _lines_of(config: dict[str, Any], key: str) -> list[str]:
        if key not in config:
            return []
        return (
            Path(config[key]).read_text(encoding="utf-8").removesuffix("\n").split("\n")
        )

    def _leave_results(self, launch: Launch) -> None:
        if launch.subcommand == "train":
            model_path = Path(launch.config["training"]["model_path"])
            model_path.mkdir(parents=True, exist_ok=True)
            weights = model_path / CHECKPOINT_FILE
            weights.write_bytes(b"weights")
            written = time.time() - self.checkpoint_age
            os.utime(weights, (written, written))
        elif launch.subcommand == "predict":
            hypotheses = self.hypotheses
            if hypotheses is None:
                hypotheses = launch.source_lines
            Path(launch.config["output"]).write_text(
                "".join(f"{hypothesis}\n" for hypothesis in hypotheses),
                encoding="utf-8",
            )

    @property
    def subcommands(self) -> list[str]:
        return [launch.subcommand for launch in self.launches]


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRunLogged:
    fake = FakeRunLogged()
    monkeypatch.setattr(eole_run, "run_logged", fake)
    return fake


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


def _accept_anything(cfg: object) -> None:
    return None


def _write_config(
    tmp_path: Path, run_dir: Path, *, gpu_ranks: object = None
) -> tuple[Path, dict[str, Any]]:
    """A config of the shape `run_training` reads: the checks of its content are the
    validator's, so only `training` matters here."""
    training: dict[str, Any] = {"model_path": (run_dir / "ckpt").as_posix()}
    if gpu_ranks is not None:
        training["gpu_ranks"] = gpu_ranks
    cfg = {"seed": 1, "training": training}
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, cfg


@pytest.fixture
def config(tmp_path: Path, run_dir: Path) -> Path:
    return _write_config(tmp_path, run_dir)[0]


# ---- run_training: the validator ------------------------------------------------


def test_run_training_hands_the_parsed_config_to_validate(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    seen: list[object] = []

    run_training(config, run_dir, validate=seen.append)

    assert seen == [
        {"seed": 1, "training": {"model_path": (run_dir / "ckpt").as_posix()}}
    ]


def test_run_training_stops_before_any_command_when_validate_raises(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    def refuse(cfg: object) -> None:
        raise ConfigError("invalid Eole config: seed: nope")

    with pytest.raises(ConfigError, match="seed: nope"):
        run_training(config, run_dir, validate=refuse)

    assert fake_run.launches == []
    assert not run_dir.exists()


def test_run_training_lets_the_validator_judge_a_config_that_is_not_a_mapping(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path = tmp_path / "list.yaml"
    cfg_path.write_text("- just\n- a list\n", encoding="utf-8")
    seen: list[object] = []

    with pytest.raises(ConfigError, match=r"training\.model_path"):
        run_training(cfg_path, run_dir, validate=seen.append)

    assert seen == [["just", "a list"]]
    assert fake_run.launches == []


# ---- run_training: commands and environment -------------------------------------


def test_run_training_runs_build_vocab_then_train_on_the_config(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    run_training(config, run_dir, validate=_accept_anything)

    assert [launch.command[1:] for launch in fake_run.launches] == [
        ["build_vocab", "-config", str(config), "-n_sample", "-1"],
        ["train", "-config", str(config)],
    ]
    assert {Path(launch.command[0]).parent for launch in fake_run.launches} == {
        Path(sys.executable).parent
    }
    assert [launch.what for launch in fake_run.launches] == [
        "eole build_vocab",
        "eole train",
    ]


def test_run_training_runs_the_commands_eole_commands_names(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    build_vocab, train = eole_commands(config)

    run_training(config, run_dir, validate=_accept_anything)

    assert [list(build_vocab), list(train)] == [
        launch.command for launch in fake_run.launches
    ]


def test_run_training_pins_the_threads_and_defaults_to_two(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    run_training(config, run_dir, validate=_accept_anything)
    run_training(config, run_dir, validate=_accept_anything, threads=5)

    assert [
        (launch.env["OMP_NUM_THREADS"], launch.env["MKL_NUM_THREADS"])
        for launch in fake_run.launches
    ] == [("2", "2"), ("2", "2"), ("5", "5"), ("5", "5")]


@pytest.mark.parametrize(
    ("gpu_ranks", "visible"),
    [(None, ""), ([], ""), ([0], "1")],
    ids=["no gpu_ranks key", "empty gpu_ranks", "a gpu rank"],
)
def test_run_training_hides_the_gpu_unless_the_config_trains_on_one(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: FakeRunLogged,
    tmp_path: Path,
    run_dir: Path,
    gpu_ranks: object,
    visible: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cfg_path, _ = _write_config(tmp_path, run_dir, gpu_ranks=gpu_ranks)

    run_training(cfg_path, run_dir, validate=_accept_anything)

    assert [launch.env["CUDA_VISIBLE_DEVICES"] for launch in fake_run.launches] == [
        visible,
        visible,
    ]


@pytest.mark.parametrize("timeout", [None, 7.5])
def test_run_training_bounds_each_command_by_the_timeout_only_when_asked(
    fake_run: FakeRunLogged, config: Path, run_dir: Path, timeout: float | None
) -> None:
    run_training(config, run_dir, validate=_accept_anything, timeout=timeout)

    assert [launch.timeout for launch in fake_run.launches] == [timeout, timeout]


@pytest.mark.parametrize("threads", [0, -1])
def test_run_training_rejects_fewer_than_one_thread_before_it_launches_anything(
    fake_run: FakeRunLogged, config: Path, run_dir: Path, threads: int
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_training(config, run_dir, validate=_accept_anything, threads=threads)

    assert fake_run.launches == []
    assert not run_dir.exists()


def test_run_training_writes_one_log_per_command_in_the_run_directory(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    run_training(config, run_dir, validate=_accept_anything)

    assert [launch.log_path for launch in fake_run.launches] == [
        run_dir / "build_vocab.log",
        run_dir / "train.log",
    ]


def test_run_training_returns_the_model_path_the_config_names(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    assert run_training(config, run_dir, validate=_accept_anything) == run_dir / "ckpt"
    assert (run_dir / "ckpt" / CHECKPOINT_FILE).is_file()


# ---- run_training: the model path -----------------------------------------------


@pytest.mark.parametrize("what", ["Eole config", "converter config"])
@pytest.mark.parametrize(
    "model_path", [None, 5, ""], ids=["missing", "not a string", "empty"]
)
def test_run_training_rejects_a_config_without_a_usable_model_path(
    fake_run: FakeRunLogged,
    tmp_path: Path,
    run_dir: Path,
    model_path: object,
    what: str,
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    if model_path is None:
        del cfg["training"]["model_path"]
    else:
        cfg["training"]["model_path"] = model_path
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match=rf"^invalid {what}: training\.model_path: must be a non-empty path$",
    ):
        run_training(cfg_path, run_dir, validate=_accept_anything, what=what)

    assert fake_run.launches == []
    assert not run_dir.exists()


@pytest.mark.parametrize("what", ["Eole config", "converter config"])
@pytest.mark.parametrize(
    "outside",
    ["elsewhere/ckpt", "run", "run/../elsewhere", "run_two/ckpt"],
    ids=["sibling", "the run directory itself", "escaping with ..", "prefix sibling"],
)
def test_run_training_rejects_a_model_path_outside_the_run_directory(
    fake_run: FakeRunLogged,
    tmp_path: Path,
    run_dir: Path,
    outside: str,
    what: str,
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["training"]["model_path"] = (tmp_path / outside).as_posix()
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match=rf"^invalid {what}: training\.model_path: .* must be inside the run "
        rf"directory {re.escape(str(run_dir))}$",
    ):
        run_training(cfg_path, run_dir, validate=_accept_anything, what=what)

    assert fake_run.launches == []
    assert not run_dir.exists()


def test_run_training_names_the_config_an_eole_config_by_default(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    del cfg["training"]["model_path"]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ConfigError, match="^invalid Eole config: training"):
        run_training(cfg_path, run_dir, validate=_accept_anything)


def test_run_training_accepts_a_model_path_other_than_the_default_inside_the_run_directory(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    model_path = run_dir / "models" / "tiny"
    cfg["training"]["model_path"] = model_path.as_posix()
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert run_training(cfg_path, run_dir, validate=_accept_anything) == model_path
    assert (model_path / CHECKPOINT_FILE).is_file()


def test_run_training_accepts_relative_paths(
    monkeypatch: pytest.MonkeyPatch, fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = _write_config(tmp_path, Path("run"))

    result = run_training(cfg_path, Path("run"), validate=_accept_anything)

    assert result == Path("run") / "ckpt"


# ---- run_training: the config file and the dollar sign ---------------------------


def test_run_training_rejects_a_missing_config_file(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    missing = tmp_path / "absent.yaml"

    with pytest.raises(
        ConfigError, match=f"^Eole config file not found: {re.escape(str(missing))}$"
    ):
        run_training(missing, run_dir, validate=_accept_anything)

    assert fake_run.launches == []


def test_run_training_rejects_a_config_that_does_not_parse(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("key: [unclosed", encoding="utf-8")

    with pytest.raises(
        ConfigError, match=f"^invalid Eole config {re.escape(str(cfg_path))}: cannot"
    ):
        run_training(cfg_path, run_dir, validate=_accept_anything)

    assert fake_run.launches == []


def test_run_training_rejects_a_dollar_sign_in_the_config_path(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, _ = _write_config(tmp_path, run_dir)
    dollar_path = cfg_path.with_name("$HOME.yaml")
    cfg_path.rename(dollar_path)

    with pytest.raises(ConfigError, match=r"the config path holds '\$'"):
        run_training(dollar_path, run_dir, validate=_accept_anything)

    assert fake_run.launches == []


def test_run_training_rejects_a_dollar_sign_in_the_run_directory(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    dollar_run_dir = tmp_path / "run$1"
    cfg_path, _ = _write_config(tmp_path, dollar_run_dir)

    with pytest.raises(ConfigError, match=r"the run directory holds '\$'"):
        run_training(cfg_path, dollar_run_dir, validate=_accept_anything)

    assert fake_run.launches == []
    assert not dollar_run_dir.exists()


def test_run_training_rejects_a_dollar_sign_anywhere_in_the_config_text(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    # Eole expands `$VAR` in the whole file, so a value with one is rewritten too.
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["note"] = "corpus/$CORPUS/valid.src"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"the text of the config .* holds '\$'"):
        run_training(cfg_path, run_dir, validate=_accept_anything)

    assert fake_run.launches == []


def test_run_training_checks_for_a_dollar_sign_before_it_validates(
    fake_run: FakeRunLogged, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["note"] = "$HOME"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    seen: list[object] = []

    with pytest.raises(ConfigError, match=r"holds '\$'"):
        run_training(cfg_path, run_dir, validate=seen.append)

    assert seen == []


# ---- run_training: the checkpoint is new and failures ----------------------------


def test_run_training_fails_when_training_writes_no_checkpoint(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    fake_run.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under") as raised:
        run_training(config, run_dir, validate=_accept_anything)

    assert f"expected a fresh {CHECKPOINT_FILE}" in str(raised.value)
    assert str(run_dir / "train.log") in str(raised.value)


def test_run_training_fails_when_only_an_older_runs_checkpoint_is_there(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    stale = run_dir / "ckpt" / CHECKPOINT_FILE
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old weights")
    written = time.time() - 3600
    os.utime(stale, (written, written))
    fake_run.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(config, run_dir, validate=_accept_anything)


def test_run_training_accepts_a_checkpoint_stamped_just_inside_the_two_second_slack(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    fake_run.checkpoint_age = 1.8

    assert run_training(config, run_dir, validate=_accept_anything) == run_dir / "ckpt"


def test_run_training_rejects_a_checkpoint_stamped_just_outside_the_two_second_slack(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    fake_run.checkpoint_age = 2.3

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(config, run_dir, validate=_accept_anything)


def test_run_training_stops_after_a_failed_build_vocab(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    fake_run.failures = {"build_vocab": UpstreamError("eole build_vocab failed: boom")}

    with pytest.raises(UpstreamError, match="build_vocab failed: boom"):
        run_training(config, run_dir, validate=_accept_anything)

    assert fake_run.subcommands == ["build_vocab"]


def test_run_training_reports_a_failed_train_as_it_was_raised(
    fake_run: FakeRunLogged, config: Path, run_dir: Path
) -> None:
    fake_run.failures = {"train": UpstreamError("eole train failed: boom")}

    with pytest.raises(UpstreamError, match="train failed: boom"):
        run_training(config, run_dir, validate=_accept_anything)

    assert fake_run.subcommands == ["build_vocab", "train"]


def test_run_training_reports_a_missing_eole_script_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: FakeRunLogged,
    tmp_path: Path,
    config: Path,
    run_dir: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        run_training(config, run_dir, validate=_accept_anything)

    assert fake_run.launches == []


# ---- the small helpers ----------------------------------------------------------


def test_the_checkpoint_name_is_the_file_eole_links_to_its_newest_checkpoint() -> None:
    assert CHECKPOINT_FILE == "model.00.safetensors"


def test_training_environment_sets_both_thread_variables_and_hides_the_gpu_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    hidden = training_environment(3, hide_gpu=True)
    shown = training_environment(3, hide_gpu=False)

    assert (hidden["OMP_NUM_THREADS"], hidden["MKL_NUM_THREADS"]) == ("3", "3")
    assert hidden["CUDA_VISIBLE_DEVICES"] == ""
    assert shown["CUDA_VISIBLE_DEVICES"] == "0"


def test_the_eole_command_is_the_console_script_next_to_the_interpreter() -> None:
    command = Path(eole_command())

    assert command.parent == Path(sys.executable).parent
    assert command.name in EOLE_NAMES


def test_the_eole_command_names_the_directory_it_looked_in_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    with pytest.raises(UpstreamError, match=f"not installed in {tmp_path}"):
        eole_command()


def test_written_since_compares_the_modification_time_with_the_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights"
    path.write_bytes(b"w")
    os.utime(path, (100.0, 100.0))

    assert written_since(path, 100.0)
    assert written_since(path, 99.0)
    assert not written_since(path, 100.5)


def test_written_since_is_false_for_a_missing_file_and_a_link_that_points_nowhere(
    tmp_path: Path,
) -> None:
    (tmp_path / "dangling").symlink_to("nowhere")

    assert not written_since(tmp_path / "absent", 0.0)
    assert not written_since(tmp_path / "dangling", 0.0)


@pytest.mark.parametrize(
    ("text", "lines"),
    [
        ("", []),
        ("\n", [""]),
        ("a\nb", ["a", "b"]),
        ("a\n\nb\n", ["a", "", "b"]),
        ("a\r\nb\r\n", ["a\r", "b\r"]),
        (f"a{LINE_SEPARATOR}b{NEXT_LINE}c\n", [f"a{LINE_SEPARATOR}b{NEXT_LINE}c"]),
    ],
    ids=repr,
)
def test_split_lines_breaks_at_lf_only_and_write_lines_ends_each_line_with_one(
    tmp_path: Path, text: str, lines: list[str]
) -> None:
    assert split_lines(text) == lines

    path = tmp_path / "lines.txt"
    write_lines(path, lines)

    assert path.read_bytes() == "".join(f"{line}\n" for line in lines).encode("utf-8")


# ---- predict_tokens: the config -------------------------------------------------

SOURCE = ["<2xx> a b", "<2xx> c a", "<2xx> b c"]


def _predict(
    tmp_path: Path, source: Sequence[str] = SOURCE, **overrides: Any
) -> list[str]:
    arguments: dict[str, Any] = {"out_dir": tmp_path / "out"}
    arguments.update(overrides)
    return predict_tokens(tmp_path / "ckpt", source, **arguments)


def test_predict_tokens_dumps_the_greedy_cpu_config_with_no_transforms(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    _predict(tmp_path)

    files = predict_files(tmp_path / "out", "eole")
    (launch,) = fake_run.launches
    assert launch.config == {
        "model_path": (tmp_path / "ckpt").as_posix(),
        "src": files.token_source.as_posix(),
        "output": files.token_output.as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
    }


def test_predict_tokens_names_the_target_file_and_forces_it_as_a_prefix_when_asked(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    prefixes = ["<2yy>", "<2zz>", "<2yy>"]

    _predict(tmp_path, forced_prefix_tokens=prefixes)

    files = predict_files(tmp_path / "out", "eole")
    (launch,) = fake_run.launches
    assert launch.config["tgt"] == files.token_target.as_posix()
    assert launch.config["tgt_file_prefix"] is True
    assert launch.prefix_lines == prefixes
    assert launch.config["transforms"] == []
    assert launch.source_lines == SOURCE


def test_predict_tokens_adds_neither_tgt_key_without_forced_prefixes(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    _predict(tmp_path)

    (launch,) = fake_run.launches
    assert "tgt" not in launch.config
    assert "tgt_file_prefix" not in launch.config
    assert not predict_files(tmp_path / "out", "eole").token_target.exists()


def test_predict_tokens_passes_the_batch_size_on(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    _predict(tmp_path, batch_size=7)

    assert fake_run.launches[0].config["batch_size"] == 7


def test_predict_tokens_writes_the_source_token_lines_one_per_line(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    source = ["<2xx> a b", "", f"<2xx> a{NNBSP}b {LINE_SEPARATOR} c\r"]

    _predict(tmp_path, source)

    token_source = predict_files(tmp_path / "out", "eole").token_source
    assert token_source.read_bytes() == "".join(f"{line}\n" for line in source).encode()


def test_predict_tokens_runs_eole_predict_on_the_dumped_config(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    _predict(tmp_path)

    files = predict_files(tmp_path / "out", "eole")
    (launch,) = fake_run.launches
    assert Path(launch.command[0]).name in EOLE_NAMES
    assert launch.command[1:] == ["predict", "-config", str(files.config)]
    assert launch.what == "eole predict"
    assert launch.log_path == files.log


def test_predict_tokens_pins_the_threads_and_hides_the_gpu(
    monkeypatch: pytest.MonkeyPatch, fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    _predict(tmp_path)
    _predict(tmp_path, threads=4)

    assert [
        (
            launch.env["OMP_NUM_THREADS"],
            launch.env["MKL_NUM_THREADS"],
            launch.env["CUDA_VISIBLE_DEVICES"],
        )
        for launch in fake_run.launches
    ] == [("2", "2", ""), ("4", "4", "")]


@pytest.mark.parametrize("timeout", [None, 3.5])
def test_predict_tokens_bounds_the_command_by_the_timeout_only_when_asked(
    fake_run: FakeRunLogged, tmp_path: Path, timeout: float | None
) -> None:
    _predict(tmp_path, timeout=timeout)

    assert fake_run.launches[0].timeout == timeout


def test_predict_tokens_names_its_files_after_the_stem_under_the_output_directory(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    out_dir = tmp_path / "a" / "b"

    _predict(tmp_path, out_dir=out_dir, stem="dev.mn", forced_prefix_tokens=["t"] * 3)

    assert predict_files(out_dir, "dev.mn") == (
        out_dir / "dev.mn.log",
        out_dir / "dev.mn.predict.yaml",
        out_dir / "dev.mn.tokens.src",
        out_dir / "dev.mn.tokens.tgt",
        out_dir / "dev.mn.tokens.out",
    )
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "dev.mn.log",
        "dev.mn.predict.yaml",
        "dev.mn.tokens.out",
        "dev.mn.tokens.src",
        "dev.mn.tokens.tgt",
    ]


# ---- predict_tokens: the result -------------------------------------------------


def test_predict_tokens_returns_one_output_line_per_source_line_in_order(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.hypotheses = ["x y", "z", "w"]

    assert _predict(tmp_path) == ["x y", "z", "w"]


def test_predict_tokens_returns_the_output_lines_as_eole_wrote_them(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    # Pieces are separated by U+0020 only: the suffix gap, a carriage return and the
    # Unicode line separators belong to their line, and an empty hypothesis is a line.
    fake_run.hypotheses = [f"a{NNBSP}b", "", f"c\r {LINE_SEPARATOR} d {NEXT_LINE}"]

    assert _predict(tmp_path) == [
        f"a{NNBSP}b",
        "",
        f"c\r {LINE_SEPARATOR} d {NEXT_LINE}",
    ]


def test_predict_tokens_keeps_the_forced_prefix_the_output_starts_with(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.hypotheses = ["<2yy> p", "<2zz> q", "<2yy> r"]

    result = _predict(tmp_path, forced_prefix_tokens=["<2yy>", "<2zz>", "<2yy>"])

    assert result == ["<2yy> p", "<2zz> q", "<2yy> r"]


def test_predict_tokens_reports_an_output_with_another_line_count_than_the_source(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.hypotheses = ["x", "y"]

    with pytest.raises(
        UpstreamError, match="wrote 2 lines for 3 source lines"
    ) as raised:
        _predict(tmp_path)

    assert str(predict_files(tmp_path / "out", "eole").token_output) in str(
        raised.value
    )


def test_predict_tokens_reports_an_output_with_more_lines_than_the_source(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.hypotheses = ["x", "y", "z", "w"]

    with pytest.raises(UpstreamError, match="wrote 4 lines for 3 source lines"):
        _predict(tmp_path)


def test_predict_tokens_fails_when_eole_exits_zero_without_an_output_file(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.writes_results = False

    with pytest.raises(
        UpstreamError, match="exited with code 0 but wrote no output file"
    ):
        _predict(tmp_path)


def test_predict_tokens_does_not_mistake_an_output_of_an_earlier_run_for_a_new_one(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    token_output = predict_files(tmp_path / "out", "eole").token_output
    token_output.parent.mkdir()
    token_output.write_text("x\ny\nz\n", encoding="utf-8")
    fake_run.writes_results = False

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path)


def test_predict_tokens_lets_a_failed_run_propagate(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    fake_run.failures = {"predict": UpstreamError("eole predict failed: exit code 1")}

    with pytest.raises(UpstreamError, match="eole predict failed: exit code 1"):
        _predict(tmp_path)


def test_predict_tokens_reports_a_missing_eole_script_without_writing_anything(
    monkeypatch: pytest.MonkeyPatch, fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        _predict(tmp_path)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


# ---- predict_tokens: refused before anything is written --------------------------


def test_predict_tokens_rejects_a_dollar_sign_in_the_file_stem(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"^the file stem holds '\$'"):
        _predict(tmp_path, stem="pred$1")

    assert fake_run.launches == []
    assert list(tmp_path.iterdir()) == []


def test_predict_tokens_rejects_a_dollar_sign_in_the_output_directory(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"^the output directory holds '\$'"):
        _predict(tmp_path, out_dir=tmp_path / "out$dir")

    assert fake_run.launches == []
    assert list(tmp_path.iterdir()) == []


def test_predict_tokens_rejects_a_dollar_sign_in_the_checkpoint_directory(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"^the checkpoint directory holds '\$'"):
        predict_tokens(tmp_path / "$ckpt", SOURCE, out_dir=tmp_path / "out")

    assert fake_run.launches == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("threads", [0, -1])
def test_predict_tokens_rejects_fewer_than_one_thread_before_it_writes_anything(
    fake_run: FakeRunLogged, tmp_path: Path, threads: int
) -> None:
    with pytest.raises(ValueError, match="threads must be at least 1"):
        _predict(tmp_path, threads=threads)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("prefix_count", [0, 2, 4])
def test_predict_tokens_rejects_prefixes_of_a_different_length_than_the_source(
    fake_run: FakeRunLogged, tmp_path: Path, prefix_count: int
) -> None:
    with pytest.raises(
        ValueError, match=rf"{prefix_count} forced prefixes for 3 source"
    ):
        _predict(tmp_path, forced_prefix_tokens=["<2yy>"] * prefix_count)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("forced", [None, []])
def test_predict_tokens_rejects_an_empty_source_list_before_it_writes_anything(
    fake_run: FakeRunLogged, tmp_path: Path, forced: list[str] | None
) -> None:
    # Eole would read no examples and exit zero, and the line count then agrees.
    with pytest.raises(ValueError, match="no source lines"):
        _predict(tmp_path, source=[], forced_prefix_tokens=forced)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("empty_at", [0, 1, 2])
def test_predict_tokens_rejects_an_empty_forced_prefix_line_before_it_writes_anything(
    fake_run: FakeRunLogged, tmp_path: Path, empty_at: int
) -> None:
    # Eole silently drops an example whose forced prefix is empty, so the output
    # would be a line short and the line count check would blame the wrong thing.
    forced = ["<2yy>"] * 3
    forced[empty_at] = ""

    with pytest.raises(
        ValueError, match=rf"forced prefix for source line {empty_at + 1} is empty"
    ):
        _predict(tmp_path, forced_prefix_tokens=forced)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


def test_predict_tokens_names_the_first_empty_forced_prefix_line(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="source line 2 is empty"):
        _predict(tmp_path, forced_prefix_tokens=["<2yy>", "", ""])


def test_predict_tokens_accepts_an_empty_source_line_with_a_forced_prefix_of_its_own(
    fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    # Only the source list and the forced prefixes are checked for emptiness: what
    # Eole does with an empty source line is the caller's to know, the line count
    # check reports it if Eole drops the line.
    assert _predict(
        tmp_path, source=["", "<2xx> a"], forced_prefix_tokens=["<2yy>", "<2yy>"]
    ) == ["", "<2xx> a"]


# ---- the order of the checks ----------------------------------------------------
# A missing `eole` is reported before a bad thread count, as D5 always did: the
# environment cannot run anything, whatever the thread count.


def _hide_eole(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "no_eole"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))


def test_run_training_reports_a_missing_eole_before_a_bad_thread_count(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: FakeRunLogged,
    tmp_path: Path,
    config: Path,
    run_dir: Path,
) -> None:
    _hide_eole(monkeypatch, tmp_path)

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        run_training(config, run_dir, validate=_accept_anything, threads=0)

    assert fake_run.launches == []
    assert not run_dir.exists()


def test_predict_tokens_reports_a_missing_eole_before_a_bad_thread_count(
    monkeypatch: pytest.MonkeyPatch, fake_run: FakeRunLogged, tmp_path: Path
) -> None:
    _hide_eole(monkeypatch, tmp_path)

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        _predict(tmp_path, threads=0)

    assert fake_run.launches == []
    assert not (tmp_path / "out").exists()


# ---- predict_tokens: stale files ------------------------------------------------


@pytest.fixture
def stale(tmp_path: Path) -> Path:
    """A file an earlier run left, which the caller's run replaces."""
    path = tmp_path / "earlier.result"
    path.write_text("earlier run\n", encoding="utf-8")
    return path


def test_predict_tokens_removes_a_stale_file_before_it_launches_eole(
    fake_run: FakeRunLogged, tmp_path: Path, stale: Path
) -> None:
    fake_run.failures = {"predict": UpstreamError("eole predict failed: exit code 1")}

    with pytest.raises(UpstreamError, match="exit code 1"):
        _predict(tmp_path, stale_files=[stale])

    assert not stale.exists()


def test_predict_tokens_removes_every_stale_file_and_accepts_one_that_is_missing(
    fake_run: FakeRunLogged, tmp_path: Path, stale: Path
) -> None:
    other = tmp_path / "other.result"
    other.write_text("earlier run\n", encoding="utf-8")

    _predict(tmp_path, stale_files=[stale, tmp_path / "never_written", other])

    assert not stale.exists()
    assert not other.exists()


def test_predict_tokens_leaves_a_file_nobody_named_alone(
    fake_run: FakeRunLogged, tmp_path: Path, stale: Path
) -> None:
    _predict(tmp_path)

    assert stale.read_text(encoding="utf-8") == "earlier run\n"


@pytest.mark.parametrize(
    ("problem", "message", "arguments"),
    [
        ("threads", "threads must be at least 1", {"threads": 0}),
        ("source", "no source lines", {"source": []}),
        (
            "prefixes",
            "forced prefixes for 3 source",
            {"forced_prefix_tokens": ["<2yy>"]},
        ),
        (
            "empty-prefix",
            "forced prefix for source line 1 is empty",
            {"forced_prefix_tokens": ["", "<2yy>", "<2yy>"]},
        ),
        ("dollar", "the file stem holds", {"stem": "pred$1"}),
    ],
)
def test_predict_tokens_keeps_a_stale_file_when_a_check_refuses_the_call(
    fake_run: FakeRunLogged,
    tmp_path: Path,
    stale: Path,
    problem: str,
    message: str,
    arguments: dict[str, Any],
) -> None:
    with pytest.raises((ValueError, ConfigError), match=message):
        _predict(tmp_path, stale_files=[stale], **arguments)

    assert stale.read_text(encoding="utf-8") == "earlier run\n"


def test_predict_tokens_keeps_a_stale_file_when_eole_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: FakeRunLogged,
    tmp_path: Path,
    stale: Path,
) -> None:
    _hide_eole(monkeypatch, tmp_path)

    with pytest.raises(UpstreamError, match="not installed"):
        _predict(tmp_path, stale_files=[stale])

    assert stale.read_text(encoding="utf-8") == "earlier run\n"


def test_predict_tokens_keeps_a_stale_file_when_the_output_directory_cannot_be_made(
    fake_run: FakeRunLogged, tmp_path: Path, stale: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _predict(tmp_path, out_dir=blocker, stale_files=[stale])

    assert stale.read_text(encoding="utf-8") == "earlier run\n"
