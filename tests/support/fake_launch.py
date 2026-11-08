"""A stand-in for `eole build_vocab` and `eole train` for the tests of the launcher.

`FakeEole` is put in place of `run_logged` in `rinchen.train.launch`. It reads the config
the launcher hands it (`-config`), and does what the real command leaves behind: a log
in the format Eole 0.6.4 writes (`curves.parse_train_log` reads it), the step directories
a run saves (complete, as `tests.support.step_dirs` fabricates them, or cut), and the
vocabulary file. A `Script` says how one `eole train` goes: how far it gets, whether it
early-stops, saves a step it cuts short, logs a NaN, or fails the way `run_logged` does
(raising `UpstreamError`, `KeyboardInterrupt`, `SystemExit`). Without a script a train
call is a healthy run to `training.train_steps`.

Every call is recorded with the environment, the parsed config and the state of
`launch.json` at that moment, so a test can say what the launcher did before Eole started.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tests.support import step_dirs
from tests.support.eole_train_config import make_config

EOLE = "/fake/venv/bin/eole"
BUILT_VOCABULARY = "a\t2\nb\t1\n"
REPORT_EVERY = 10
SAVE_EVERY = 50
_STEP_DIRECTORY = re.compile(r"step_([0-9]+)")


def launch_config(run_dir: Path, **training: Any) -> dict[str, Any]:
    """The D7 test config laid out for a launch into `run_dir`: absolute vocabulary and
    checkpoint paths inside it, 200 steps, a report every 10, a validation and a save
    every 50 and the last 4 saves kept. `training` overrides keys of `training`."""
    cfg = make_config()
    cfg["src_vocab"] = str(run_dir / "vocab.src")
    cfg["report_every"] = REPORT_EVERY
    cfg["training"].update(
        model_path=str(run_dir / "ckpt"),
        train_steps=200,
        valid_steps=SAVE_EVERY,
        save_checkpoint_steps=SAVE_EVERY,
        keep_checkpoint=4,
    )
    cfg["training"].update(training)
    return cfg


def saved_model_section(model: dict[str, Any]) -> dict[str, Any]:
    """The `model` section Eole saves into `config.json` for the `model` section `model`
    of a config (as `launch_config` writes it): the real layout with the sizes the config
    sets."""
    return step_dirs.eole_model_section(
        hidden_size=model["hidden_size"],
        heads=model["heads"],
        transformer_ff=model["transformer_ff"],
        encoder_layers=model["encoder"]["layers"],
        decoder_layers=model["decoder"]["layers"],
    )


def write_config(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@dataclass
class Script:
    """How one `eole train` goes.

    to_step: the step it reaches; the config's `train_steps` when None.
    early_stop_at: it early-stops there (log `Early Stop!`, then the save of that step).
    nan_from: the first report step whose figures are NaN; `nan_field` names the figure.
    skip_final_save: it ends (as after Ctrl-C) without saving the step `to_step`.
    cut: steps it saves and then cuts short, so they are not complete.
    error: raised after the files are written (what `run_logged` raises, or an interrupt).
    """

    to_step: int | None = None
    early_stop_at: int | None = None
    nan_from: int | None = None
    nan_field: str = "xent"
    skip_final_save: bool = False
    cut: tuple[int, ...] = ()
    error: BaseException | None = None


@dataclass
class Call:
    """One command the launcher ran: as `run_logged` was given it, plus what was on disk."""

    command: tuple[str, ...]
    env: dict[str, str]
    log_path: Path
    what: str
    timeout: float | None
    config: dict[str, Any]
    launch_json: bytes | None  # run_dir/launch.json when the command started
    inherit_fds: tuple[int, ...]  # the descriptors the launcher handed the command
    # (st_dev, st_ino) of the file behind each of them when the command started: the
    # numbers alone do not say what they were
    inherited_files: tuple[tuple[int, int], ...]
    vocabulary: bytes | None  # the config's `src_vocab` file when the command started

    @property
    def subcommand(self) -> str:
        return self.command[1]


@dataclass
class FakeEole:
    """Stands in for `run_logged`; see the module docstring."""

    run_dir: Path
    calls: list[Call] = field(default_factory=list)
    scripts: list[Script] = field(default_factory=list)
    vocabulary_error: BaseException | None = None
    writes_vocabulary: bool = True  # false: `build_vocab` exits 0 and leaves no file

    def queue(self, *scripts: Script) -> None:
        """The scripts of the next `eole train` calls, in order."""
        self.scripts.extend(scripts)

    @property
    def trains(self) -> list[Call]:
        return [call for call in self.calls if call.subcommand == "train"]

    @property
    def vocabularies(self) -> list[Call]:
        return [call for call in self.calls if call.subcommand == "build_vocab"]

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str],
        log_path: Path,
        what: str,
        timeout: float | None = None,
        inherit_fds: Sequence[int] = (),
    ) -> None:
        launch_json = self.run_dir / "launch.json"
        config = yaml.safe_load(Path(command[3]).read_text(encoding="utf-8"))
        vocabulary_file = Path(config["src_vocab"])
        call = Call(
            command=tuple(command),
            env=dict(env),
            log_path=log_path,
            what=what,
            timeout=timeout,
            config=config,
            launch_json=launch_json.read_bytes() if launch_json.exists() else None,
            inherit_fds=tuple(inherit_fds),
            inherited_files=tuple(
                (status.st_dev, status.st_ino)
                for status in (os.fstat(number) for number in inherit_fds)
            ),
            vocabulary=vocabulary_file.read_bytes()
            if vocabulary_file.is_file()
            else None,
        )
        self.calls.append(call)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if call.subcommand == "build_vocab":
            self._build_vocabulary(call)
        else:
            self._train(call)

    def _build_vocabulary(self, call: Call) -> None:
        call.log_path.write_text("vocabulary counted\n", encoding="utf-8")
        if self.vocabulary_error is not None:
            raise self.vocabulary_error
        if not self.writes_vocabulary:
            return
        vocabulary = Path(call.config["src_vocab"])
        vocabulary.parent.mkdir(parents=True, exist_ok=True)
        # The format of `eole build_vocab`: one token and its count per line.
        vocabulary.write_text(BUILT_VOCABULARY, encoding="utf-8")

    def _train(self, call: Call) -> None:
        script = self.scripts.pop(0) if self.scripts else Script()
        model_path = Path(call.config["training"]["model_path"])
        outcome = simulate_training(call.config, script)
        call.log_path.write_text("\n".join(outcome.log_lines) + "\n", encoding="utf-8")
        for step in outcome.saves:
            existing = model_path / f"step_{step}"
            if existing.exists():
                shutil.rmtree(existing)
            made = step_dirs.make_step_dir(
                model_path, step, model=saved_model_section(call.config["model"])
            )
            if step in script.cut:
                step_dirs.truncate(made, step_dirs.WEIGHTS_FILE, 10)
        if script.error is not None:
            raise script.error


@dataclass(frozen=True)
class Training:
    """What an `eole train` on a config leaves, as `simulate_training` says: the lines of
    its log and the steps it saves."""

    log_lines: list[str]
    saves: list[int]


def simulate_training(cfg: dict[str, Any], script: Script) -> Training:
    """The log and the saved steps of an `eole train` of `script` on the config `cfg`
    (resuming from its `training.train_from` when it has one)."""
    training = cfg["training"]
    start = _step_of(training.get("train_from"))
    end = script.early_stop_at or script.to_step or training["train_steps"]
    saves = _saves(start, end, script, cfg)
    model_path = Path(training["model_path"])
    return Training(_log_lines(start, end, saves, script, cfg, model_path), saves)


def _step_of(train_from: object) -> int:
    if not isinstance(train_from, str):
        return 0
    match = _STEP_DIRECTORY.fullmatch(Path(train_from).name)
    assert match is not None, train_from
    return int(match.group(1))


def _saves(start: int, end: int, script: Script, cfg: dict[str, Any]) -> list[int]:
    """The steps a run from `start` to `end` leaves: every multiple of the save interval,
    and `end` itself, which Eole saves after the loop (a run interrupted at `end` saves
    nothing there)."""
    every = cfg["training"]["save_checkpoint_steps"]
    steps = [step for step in range(start + 1, end + 1) if step % every == 0]
    if end > start and end not in steps:
        steps.append(end)
    if script.skip_final_save and steps[-1:] == [end]:
        steps.pop()
    return steps


def _log_lines(
    start: int,
    end: int,
    saves: list[int],
    script: Script,
    cfg: dict[str, Any],
    model_path: Path,
) -> list[str]:
    total = cfg["training"]["train_steps"]
    valid_every = cfg["training"]["valid_steps"]
    lines = ["[2026-11-08 10:00:00,000 INFO] Training on CPU - this may be very slow"]
    validations = 0
    for step in range(start + 1, end + 1):
        if step % REPORT_EVERY == 0:
            lines.append(_report(step, total, script))
        if step % valid_every == 0:
            validations += 1
            lines.append(
                f"[2026-11-08 10:00:01,000 INFO] Validation perplexity: {_ppl(step, script)}"
            )
            lines.append(
                f"[2026-11-08 10:00:01,000 INFO] Validation accuracy: {12.5 + step / 10}"
            )
            if script.early_stop_at is not None and step == script.early_stop_at:
                lines.append(
                    "[2026-11-08 10:00:02,000 INFO] Training finished after not "
                    "improving. Early Stop!"
                )
        if step in saves:
            lines.append(
                f"[2026-11-08 10:00:03,000 INFO] Saving optimizer and weights to "
                f"step_{step}, and symlink to {model_path}"
            )
    if script.skip_final_save:
        lines.append("[2026-11-08 10:00:04,000 INFO] Training interrupted by user")
    return lines


def _ppl(step: int, script: Script) -> str:
    if script.nan_from is not None and step >= script.nan_from:
        return "nan"
    return f"{max(2.0, 300.0 / (1 + step / 10)):.4f}"


def _report(step: int, total: int, script: Script) -> str:
    xent = f"{max(2.0, 5.5 - step / 60):.2f}"
    lr = "6.25e-04"
    if script.nan_from is not None and step >= script.nan_from:
        if script.nan_field == "lr":
            lr = "inf"
        else:
            xent = "nan"
    return (
        f"[2026-11-08 10:00:00,500 INFO] Step {step:2d}/{total:5d}; acc: 20.1; "
        f"ppl: {300.0 / (1 + step / 10):.2f}; xent: {xent}; aux: 0.000; mtp: 0.000; "
        f"attn_ent: 1.965; lr: {lr}; sents:    1512; bsz: 1131/1231/151; "
        f"3995/4347 tok/s;      3 sec;"
    )
