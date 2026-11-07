"""Tests for `rinchen.script_conv.train`: the Eole subprocesses of the converter.

No test starts Eole. `subprocess.run` is replaced by a recorder that writes what the
real tool would (log lines, the checkpoint directory, the prediction file), so each
test pins the commands, the environment and the failure handling. The real run is
the integration test in `tests/integration/test_script_conv_tiny.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.script_conv.config import dump_config, render_converter_config
from rinchen.script_conv.train import predict, run_training, training_environment

TO_SCRIPT_TAG = "<2mn_Mong>"
EOLE_NAMES = ("eole", "eole.exe")


class FakeEole:
    """Stands in for `subprocess.run`: records every call, writes `output` lines to
    the log handle it is given and exits with `exit_codes[subcommand]` (default 0).
    A finished `train` leaves the checkpoint directory, a finished `predict` its
    output file, unless told not to."""

    def __init__(
        self,
        *,
        exit_codes: dict[str, int] | None = None,
        output: Sequence[str] = ("a log line",),
        make_results: bool = True,
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.exit_codes = exit_codes or {}
        self.output = output
        self.make_results = make_results

    def __call__(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(argv), kwargs))
        subcommand = argv[1]
        for line in self.output:
            kwargs["stdout"].write(f"{line}\n".encode())
        code = self.exit_codes.get(subcommand, 0)
        if code == 0 and self.make_results:
            self._leave_results(subcommand, Path(argv[3]))
        return subprocess.CompletedProcess(list(argv), code)

    @staticmethod
    def _leave_results(subcommand: str, config_path: Path) -> None:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if subcommand == "train":
            Path(cfg["training"]["model_path"]).mkdir(parents=True)
        elif subcommand == "predict":
            Path(cfg["output"]).write_text("predicted\n", encoding="utf-8")

    @property
    def subcommands(self) -> list[str]:
        return [argv[1] for argv, _ in self.calls]


@pytest.fixture
def fake_eole(monkeypatch: pytest.MonkeyPatch) -> FakeEole:
    fake = FakeEole()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeEole) -> FakeEole:
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


def _write_config(
    tmp_path: Path, run_dir: Path, *, profile: str = "tiny"
) -> tuple[Path, dict[str, Any]]:
    cfg = render_converter_config(
        profile,  # type: ignore[arg-type]
        data_dir=tmp_path / "corpus",
        spm_model=tmp_path / "converter_char.model",
        run_dir=run_dir,
    )
    cfg_path = tmp_path / "converter.yaml"
    dump_config(cfg, cfg_path)
    return cfg_path, cfg


@pytest.fixture
def tiny_config(tmp_path: Path, run_dir: Path) -> Path:
    return _write_config(tmp_path, run_dir)[0]


# ---- training_environment -------------------------------------------------------


def test_training_environment_sets_both_thread_variables_as_strings_and_keeps_path() -> (
    None
):
    env = training_environment(3, hide_gpu=False)

    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["PATH"] == os.environ["PATH"]


def test_training_environment_hides_the_gpu_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert training_environment(2, hide_gpu=True)["CUDA_VISIBLE_DEVICES"] == ""
    assert training_environment(2, hide_gpu=False)["CUDA_VISIBLE_DEVICES"] == "0"


def test_training_environment_leaves_an_unset_gpu_variable_unset_when_not_hiding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert "CUDA_VISIBLE_DEVICES" not in training_environment(2, hide_gpu=False)


def test_training_environment_is_a_copy_of_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    before = dict(os.environ)

    training_environment(2, hide_gpu=True)["OMP_NUM_THREADS"] = "x"

    assert dict(os.environ) == before


# ---- run_training: commands and environment -------------------------------------


def test_run_training_runs_build_vocab_then_train_with_the_config(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(monkeypatch, FakeEole())

    run_training(tiny_config, run_dir)

    assert [Path(argv[0]).name for argv, _ in fake.calls] == ["eole", "eole"]
    assert [argv[1:] for argv, _ in fake.calls] == [
        ["build_vocab", "-config", str(tiny_config), "-n_sample", "-1"],
        ["train", "-config", str(tiny_config)],
    ]


def test_run_training_calls_the_eole_script_next_to_the_interpreter(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(monkeypatch, FakeEole())

    run_training(tiny_config, run_dir)

    interpreter_dir = Path(sys.executable).parent
    assert {Path(argv[0]).parent for argv, _ in fake.calls} == {interpreter_dir}


def test_run_training_sets_the_thread_variables_and_hides_the_gpu_of_a_cpu_config(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fake = _install(monkeypatch, FakeEole())

    run_training(tiny_config, run_dir, threads=3)

    for _, kwargs in fake.calls:
        assert kwargs["env"]["OMP_NUM_THREADS"] == "3"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "3"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_run_training_leaves_the_gpu_visible_for_a_config_that_trains_on_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_dir: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cfg_path, _ = _write_config(tmp_path, run_dir, profile="full")
    fake = _install(monkeypatch, FakeEole())

    run_training(cfg_path, run_dir)

    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in fake.calls] == [
        "1",
        "1",
    ]


def test_run_training_defaults_to_two_threads(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(monkeypatch, FakeEole())

    run_training(tiny_config, run_dir)

    assert fake.calls[0][1]["env"]["OMP_NUM_THREADS"] == "2"


# ---- run_training: results and logs ---------------------------------------------


def test_run_training_returns_the_checkpoint_directory_under_the_run_directory(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    assert run_training(tiny_config, run_dir) == run_dir / "ckpt"


def test_run_training_creates_the_run_directory_and_writes_one_log_per_command(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.output = ("first line", "second line")

    run_training(tiny_config, run_dir)

    for log_name in ("build_vocab.log", "train.log"):
        assert (run_dir / log_name).read_text(encoding="utf-8") == (
            "first line\nsecond line\n"
        )


def test_run_training_combines_stdout_and_stderr_in_the_log(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    kwargs = fake_eole.calls[0][1]
    assert kwargs["stderr"] == subprocess.STDOUT


def test_run_training_fails_when_the_checkpoint_directory_is_missing_after_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    _install(monkeypatch, FakeEole(make_results=False))

    with pytest.raises(UpstreamError, match="no checkpoint directory"):
        run_training(tiny_config, run_dir)


# ---- run_training: failures -----------------------------------------------------


def test_run_training_validates_the_config_before_it_launches_anything(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["model"]["embeddings"]["position_encoding_type"] = "Rotary"
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match="position_encoding_type"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []
    assert not run_dir.exists()


def test_run_training_rejects_a_missing_config_file(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    with pytest.raises(ConfigError, match="not found"):
        run_training(tmp_path / "absent.yaml", run_dir)

    assert fake_eole.calls == []


@pytest.mark.parametrize("text", ["key: [unclosed", "- just\n- a list\n"])
def test_run_training_rejects_a_config_that_is_not_a_mapping_of_settings(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path, text: str
) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_rejects_a_dollar_sign_in_the_config_path(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, _ = _write_config(tmp_path, run_dir)
    dollar_path = cfg_path.with_name("$HOME.yaml")
    cfg_path.rename(dollar_path)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(dollar_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_rejects_a_dollar_sign_in_the_run_directory(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    dollar_run_dir = tmp_path / "run$1"
    cfg_path, _ = _write_config(tmp_path, dollar_run_dir)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(cfg_path, dollar_run_dir)

    assert fake_eole.calls == []
    assert not dollar_run_dir.exists()


def test_run_training_rejects_a_dollar_sign_anywhere_in_the_config_text(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    # Eole expands `$VAR` in the whole file, so a data path with one is rewritten too.
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["data"]["valid"]["path_src"] = "corpus/$CORPUS/valid.src"
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_reports_a_failed_build_vocab_with_the_log_tail_and_stops(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(
        monkeypatch,
        FakeEole(
            exit_codes={"build_vocab": 2},
            output=[f"line {number}" for number in range(1, 31)],
        ),
    )

    with pytest.raises(UpstreamError) as raised:
        run_training(tiny_config, run_dir)

    message = str(raised.value)
    assert "exit code 2" in message
    assert message.endswith("\n".join(f"line {number}" for number in range(11, 31)))
    assert "line 10\n" not in message
    assert fake.subcommands == ["build_vocab"]


def test_run_training_reports_a_failed_train_with_the_log_tail(
    monkeypatch: pytest.MonkeyPatch, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(
        monkeypatch,
        FakeEole(exit_codes={"train": 1}, output=["Traceback", "boom"]),
    )

    with pytest.raises(UpstreamError) as raised:
        run_training(tiny_config, run_dir)

    message = str(raised.value)
    assert "train" in message
    assert str(run_dir / "train.log") in message
    assert message.endswith("Traceback\nboom")
    assert fake.subcommands == ["build_vocab", "train"]


def test_run_training_reports_a_missing_eole_script_without_running_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_config: Path, run_dir: Path
) -> None:
    fake = _install(monkeypatch, FakeEole())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    with pytest.raises(UpstreamError, match="eole"):
        run_training(tiny_config, run_dir)

    assert fake.calls == []


# ---- predict --------------------------------------------------------------------


def _predict(tmp_path: Path, **overrides: Any) -> Path:
    arguments: dict[str, Any] = {
        "tag": TO_SCRIPT_TAG,
        "spm_model": tmp_path / "converter_char.model",
        "out_file": tmp_path / "out" / "pred.tgt",
    }
    arguments.update(overrides)
    return predict(tmp_path / "run" / "ckpt", tmp_path / "words.src", **arguments)


def test_predict_writes_the_predict_config_next_to_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path)

    cfg = yaml.safe_load(
        Path(f"{out_file}.predict.yaml").read_text(encoding="utf-8"),
    )
    assert cfg == {
        "model_path": (tmp_path / "run" / "ckpt").as_posix(),
        "src": (tmp_path / "words.src").as_posix(),
        "output": out_file.as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": ["sentencepiece", "prefix"],
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": (tmp_path / "converter_char.model").as_posix(),
                "tgt_subword_model": (tmp_path / "converter_char.model").as_posix(),
            },
            "prefix": {"src_prefix": TO_SCRIPT_TAG, "tgt_prefix": ""},
        },
    }


def test_predict_puts_the_requested_tag_in_the_prefix_transform(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path, tag="<2mn_Cyrl>")

    cfg = yaml.safe_load(Path(f"{out_file}.predict.yaml").read_text(encoding="utf-8"))
    assert cfg["transforms_configs"]["prefix"]["src_prefix"] == "<2mn_Cyrl>"


def test_predict_runs_eole_predict_on_the_cpu_and_returns_the_output_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fake = _install(monkeypatch, FakeEole())

    out_file = _predict(tmp_path, threads=4)

    assert out_file == tmp_path / "out" / "pred.tgt"
    ((argv, kwargs),) = fake.calls
    assert Path(argv[0]).name in EOLE_NAMES
    assert argv[1:] == ["predict", "-config", f"{out_file}.predict.yaml"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "4"


def test_predict_writes_the_combined_log_next_to_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    fake_eole.output = ("decoding",)

    out_file = _predict(tmp_path)

    assert Path(f"{out_file}.log").read_text(encoding="utf-8") == "decoding\n"


def test_predict_reports_a_failure_with_the_log_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, FakeEole(exit_codes={"predict": 1}, output=["bad model"]))

    with pytest.raises(UpstreamError) as raised:
        _predict(tmp_path)

    assert "exit code 1" in str(raised.value)
    assert str(raised.value).endswith("bad model")


def test_predict_fails_when_eole_exits_zero_without_an_output_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, FakeEole(make_results=False))

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path)


def test_predict_does_not_mistake_an_earlier_output_for_a_new_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_file = tmp_path / "out" / "pred.tgt"
    out_file.parent.mkdir()
    out_file.write_text("left from an earlier run\n", encoding="utf-8")
    _install(monkeypatch, FakeEole(make_results=False))

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path)


def test_predict_creates_the_directory_of_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path, out_file=tmp_path / "a" / "b" / "pred.tgt")

    assert out_file.parent.is_dir()


@pytest.mark.parametrize("field", ["spm_model", "out_file"])
def test_predict_rejects_a_dollar_sign_in_a_path(
    fake_eole: FakeEole, tmp_path: Path, field: str
) -> None:
    with pytest.raises(ConfigError, match=r"\$"):
        _predict(tmp_path, **{field: tmp_path / "out" / "$x"})

    assert fake_eole.calls == []


def test_predict_rejects_a_dollar_sign_in_the_checkpoint_or_source_path(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"\$"):
        predict(
            tmp_path / "run$" / "ckpt",
            tmp_path / "words.src",
            tag=TO_SCRIPT_TAG,
            spm_model=tmp_path / "converter_char.model",
            out_file=tmp_path / "pred.tgt",
        )

    assert fake_eole.calls == []
