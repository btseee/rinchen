"""Inputs and fakes for the tests of `rinchen.train.ablation`: the toy corpus as
Parquet, matrices over the tiny base config, plans, scripted dev scores, a fake clock,
and a fake `train` and `evaluate` that stand in for Eole.

No fake starts a process. `FakeTrain` writes a checkpoint file (its bytes name the
variant and the call, so a retrained variant has other bytes), `FakeEvaluate` returns
scripted scores, and both record what they were handed and advance a `FakeClock` by
scripted seconds, so the durations an orchestrator reports are exact.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rinchen.config.ablation import AblationMatrix, TagPlacement
from rinchen.config.loader import REPO_ROOT
from rinchen.train.ablation import (
    AblationPlan,
    AblationResult,
    VariantProgress,
    plan_ablation,
    run_ablation,
)
from rinchen.train.eole_cfg import read_config
from rinchen.train.eole_run import CHECKPOINT_FILE
from rinchen.train.evaluate import DevResult, DirectionScores
from rinchen.train.mixture import DevSet
from tests.support.corpus_rows import corpus_row
from tests.support.tiny_parallel import TOY_DIRECTIONS, build_toy, write_corpus_parquet

BASE_CONFIG = REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"
# The four toy directions, in the order the dev results list them.
DIRECTIONS = tuple(sorted(TOY_DIRECTIONS))
DEV_LINES = 100
CHRF_SIGNATURE = "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1"
BLEU_SIGNATURE = "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1"

ID_SRC = "t0.5_bt0.5_src_s1234"
ID_TGT = "t0.5_bt0.5_tgt_s1234"


def colliding_rows(pairs: int = 100) -> list[dict[str, str]]:
    """`pairs` Mongolian sentences, each the source of one row into English and of
    one into Russian: 2 * `pairs` training rows whose source also occurs with
    another target language."""
    rows = []
    for index in range(pairs):
        source = f"Давхцсан өгүүлбэр {index}."
        rows.append(
            corpus_row(
                source, f"Overlap {index}.", src_lang="khk_Cyrl", tgt_lang="eng_Latn"
            )
        )
        rows.append(
            corpus_row(
                source,
                f"Пересечение {index}.",
                src_lang="khk_Cyrl",
                tgt_lang="rus_Cyrl",
            )
        )
    return rows


def write_toy_corpus(
    directory: Path,
    *,
    changed_bt_row: bool = False,
    extra_rows: Sequence[Mapping[str, Any]] = (),
    name: str = "toy.parquet",
) -> Path:
    """The toy corpus (training, back-translated, dev and test rows, then `extra_rows`)
    as Parquet in `directory`. With `changed_bt_row` the first back-translated row has
    one more character in its target, which gives it another id."""
    toy = build_toy()
    bt = list(toy.bt)
    if changed_bt_row:
        bt[0] = {**bt[0], "tgt": bt[0]["tgt"] + "!"}
    rows = [*toy.train, *bt, *toy.dev, *toy.test, *extra_rows]
    return write_corpus_parquet(rows, directory / name)


def make_matrix(**fields: Any) -> AblationMatrix:
    """A matrix over the tiny base config. By default it is two variants, one per tag
    placement, at temperature 0.5 and bt ratio 0.5 with seed 1234; `fields` replace
    whole fields (`axes`, `seeds`, `overrides`, `weight_total`, ...)."""
    defaults: dict[str, Any] = {
        "name": "toy",
        "base": str(BASE_CONFIG),
        "axes": {
            "temperature": [0.5],
            "bt_ratio": [0.5],
            "tag_placement": ["src", "tgt"],
        },
        "seeds": [1234],
    }
    return AblationMatrix.model_validate(defaults | fields)


def one_variant_axes(
    placement: TagPlacement = "src", *, bt_ratio: float = 0.5
) -> dict[str, Any]:
    return {"temperature": [0.5], "bt_ratio": [bt_ratio], "tag_placement": [placement]}


def toy_plan(
    root: Path, spm_model: Path, *, corpus: Path | None = None, **fields: Any
) -> AblationPlan:
    """The plan of `make_matrix(**fields)` over the toy corpus (written to
    `root / "in"` unless `corpus` is given) with its output directory `root / "out"`."""
    corpus = corpus if corpus is not None else write_toy_corpus(root / "in")
    return plan_ablation(make_matrix(**fields), [corpus], spm_model, root / "out")


def dev_result(
    chrf: float | Mapping[str, float],
    bleu: float | Mapping[str, float] | None = None,
    *,
    tag_leaks: int = 0,
    lines: int = DEV_LINES,
) -> DevResult:
    """Scripted dev scores. A number is every direction's score; a mapping names
    the directions it scores. BLEU defaults to half the chrF. The mean and the minimum
    are those of the chrF values."""
    chrfs = dict.fromkeys(DIRECTIONS, chrf) if isinstance(chrf, float | int) else chrf
    if bleu is None:
        bleus = {direction: value / 2 for direction, value in chrfs.items()}
    elif isinstance(bleu, float | int):
        bleus = dict.fromkeys(chrfs, bleu)
    else:
        bleus = dict(bleu)
    return DevResult(
        directions={
            direction: DirectionScores(
                lines=lines, chrf=float(chrfs[direction]), bleu=float(bleus[direction])
            )
            for direction in sorted(chrfs)
        },
        chrf_mean=statistics.fmean(chrfs.values()),
        chrf_min=min(chrfs.values()),
        tag_leaks=tag_leaks,
        chrf_signature=CHRF_SIGNATURE,
        bleu_signature=BLEU_SIGNATURE,
    )


class FakeClock:
    """A clock that moves only when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True)
class TrainCall:
    variant_id: str
    cfg_path: Path
    run_dir: Path
    validate: Callable[[Any], None]
    threads: int
    timeout: float | None
    config_text: str
    checkpoint_dir: Path | None


class FakeTrain:
    """Stands in for `run_training`: records the call, applies `validate` to the
    config as the real one does, writes `run/ckpt/model.00.safetensors` and returns
    its directory. `fail` maps a variant id to the exception it raises; `seconds` maps
    a variant id to the time it advances the clock (1.0 otherwise). The variant is
    named by the directory of the config."""

    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        seconds: Mapping[str, float] | None = None,
        fail: Mapping[str, BaseException] | None = None,
        write_checkpoint: bool = True,
    ) -> None:
        self.clock = clock
        self.seconds = dict(seconds or {})
        self.fail = dict(fail or {})
        self.write_checkpoint = write_checkpoint
        self.calls: list[TrainCall] = []

    @property
    def variant_ids(self) -> list[str]:
        return [call.variant_id for call in self.calls]

    def __call__(
        self,
        cfg_path: Path,
        run_dir: Path,
        *,
        validate: Callable[[Any], None],
        threads: int,
        timeout: float | None,
    ) -> Path:
        variant_id = cfg_path.parent.name
        validate(read_config(cfg_path, "Eole config"))
        ckpt_dir = run_dir / "ckpt"
        self.calls.append(
            TrainCall(
                variant_id=variant_id,
                cfg_path=cfg_path,
                run_dir=run_dir,
                validate=validate,
                threads=threads,
                timeout=timeout,
                config_text=cfg_path.read_text(encoding="utf-8"),
                checkpoint_dir=ckpt_dir,
            )
        )
        if variant_id in self.fail:
            raise self.fail[variant_id]
        if self.clock is not None:
            self.clock.advance(self.seconds.get(variant_id, 1.0))
        if self.write_checkpoint:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / CHECKPOINT_FILE).write_bytes(
                f"checkpoint of {variant_id}, training call {len(self.calls)}".encode()
            )
        return ckpt_dir


@dataclass(frozen=True)
class EvaluateCall:
    variant_id: str
    ckpt_dir: Path
    dev: Mapping[str, DevSet]
    spm_model: Path
    placement: TagPlacement
    out_dir: Path
    threads: int
    timeout: float | None
    result_existed: bool


class FakeEvaluate:
    """Stands in for `evaluate_variant`: records the call (and whether the variant's
    `result.json` existed at that moment) and returns the scripted `DevResult` of the
    variant (`dev_result(80.0)` for one not scripted). `fail` maps a variant id to the
    exception it raises; `seconds` the time it advances the clock (0.5 otherwise).
    The variant is named by the parent of `out_dir`."""

    def __init__(
        self,
        scores: Mapping[str, DevResult] | None = None,
        *,
        clock: FakeClock | None = None,
        seconds: Mapping[str, float] | None = None,
        fail: Mapping[str, BaseException] | None = None,
    ) -> None:
        self.scores = dict(scores or {})
        self.clock = clock
        self.seconds = dict(seconds or {})
        self.fail = dict(fail or {})
        self.calls: list[EvaluateCall] = []

    @property
    def variant_ids(self) -> list[str]:
        return [call.variant_id for call in self.calls]

    def __call__(
        self,
        ckpt_dir: Path,
        dev: Mapping[str, DevSet],
        *,
        spm_model: Path,
        placement: TagPlacement,
        out_dir: Path,
        threads: int,
        timeout: float | None,
    ) -> DevResult:
        variant_id = out_dir.parent.name
        self.calls.append(
            EvaluateCall(
                variant_id=variant_id,
                ckpt_dir=ckpt_dir,
                dev=dev,
                spm_model=spm_model,
                placement=placement,
                out_dir=out_dir,
                threads=threads,
                timeout=timeout,
                result_existed=(out_dir.parent / "result.json").exists(),
            )
        )
        if variant_id in self.fail:
            raise self.fail[variant_id]
        if self.clock is not None:
            self.clock.advance(self.seconds.get(variant_id, 0.5))
        return self.scores.get(variant_id, dev_result(80.0))


@dataclass
class ProgressLog:
    """A progress callback that keeps the events."""

    events: list[VariantProgress] = field(default_factory=list)

    def __call__(self, event: VariantProgress) -> None:
        self.events.append(event)

    @property
    def outcomes(self) -> list[tuple[str, str]]:
        return [(event.variant_id, event.outcome) for event in self.events]


def flip_a_byte(path: Path) -> None:
    """Change the first byte of `path`, keeping its size."""
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))


class AblationRun:
    """One `run_ablation` call with fresh fakes (or the ones given), and its outcome.
    `clock` goes to `run_ablation` only when given."""

    def __init__(
        self,
        plan: AblationPlan,
        *,
        train: FakeTrain | None = None,
        evaluate: FakeEvaluate | None = None,
        clock: FakeClock | None = None,
        progress: Callable[[VariantProgress], None] | None = None,
        threads: int = 2,
        timeout: float | None = 30.0,
        force: bool = False,
    ) -> None:
        self.plan = plan
        self.train = train if train is not None else FakeTrain(clock=clock)
        self.evaluate = evaluate if evaluate is not None else FakeEvaluate(clock=clock)
        self.result: AblationResult = run_ablation(
            plan,
            threads=threads,
            timeout=timeout,
            force=force,
            train=self.train,
            evaluate=self.evaluate,
            progress=progress,
            **({} if clock is None else {"clock": clock}),
        )

    def variant_dir(self, variant_id: str) -> Path:
        return self.plan.out_dir / "variants" / variant_id

    def result_path(self, variant_id: str) -> Path:
        return self.variant_dir(variant_id) / "result.json"

    def config(self, variant_id: str) -> dict[str, Any]:
        loaded = read_config(self.variant_dir(variant_id) / "config.yaml", "config")
        assert isinstance(loaded, dict)
        return loaded
