"""Synthetic-data config (`configs/data/synth.yaml`): the generators, the jobs
that run them and the selection thresholds. Every number is an unvalidated
prior until a real generator has run; a missing section or field keeps its
default, but a `generators` or `jobs` table you supply replaces the default
one instead of merging into it. The default jobs name the default generators,
so supplying `generators` alone leaves `jobs` empty."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)

from rinchen.config.filters import QeConfig
from rinchen.config.languages import load_languages
from rinchen.config.loader import REPO_ROOT, anchor_at_repo_root, reject_blank_path
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

SYNTH_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "synth.yaml"

_MONGOLIAN = "khk_Cyrl"

# Strict so a quoted number or a YAML boolean is a typo to report, not a
# value to coerce.
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
_NonNegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
_Probability = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
_Temperature = Annotated[float, Field(strict=True, ge=0, le=2, allow_inf_nan=False)]
_CandidateCount = Annotated[int, Field(strict=True, ge=1, le=64)]


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


# A name or URL the operator fills in.
_NonBlankString = Annotated[str, AfterValidator(_reject_blank)]
# Relative paths mean the same thing from any working directory.
_RepoPath = Annotated[
    Path, BeforeValidator(reject_blank_path), AfterValidator(anchor_at_repo_root)
]


class _GeneratorBase(BaseModel):
    """What every generator shares: how many candidates to make per source
    text and how to sample them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: _CandidateCount = 4
    temperature: _Temperature = 0.7
    top_p: _Probability = 0.95
    max_tokens: _PositiveInt = 256


class _ServerGeneratorBase(_GeneratorBase):
    """A generator behind an OpenAI-compatible chat endpoint."""

    base_url: _NonBlankString
    # A placeholder: the operator names the model the server really serves.
    model: _NonBlankString = "Hy-MT2-7B"
    concurrency: _PositiveInt = 32
    timeout_s: _PositiveNumber = 120.0
    retries: _NonNegativeInt = 4
    backoff_s: _NonNegativeNumber = 1.0


class VllmConfig(_ServerGeneratorBase):
    kind: Literal["vllm"] = "vllm"
    base_url: _NonBlankString = "http://127.0.0.1:8000/v1"


class LlamaCppConfig(_ServerGeneratorBase):
    kind: Literal["llamacpp"] = "llamacpp"
    base_url: _NonBlankString = "http://127.0.0.1:8080/v1"
    concurrency: _PositiveInt = 8


class Ct2Config(_GeneratorBase):
    """A CTranslate2 translation model. It uses `n` as the number of
    hypotheses and `beam_size` for the search, and ignores `temperature`,
    `top_p` and `max_tokens`, which exist only so every generator has the
    same fields. `n` may not exceed `beam_size`: CTranslate2 raises "The
    number of hypotheses cannot be greater than beam_size * patience" on the
    first batch, which would end a long run late."""

    kind: Literal["ct2"] = "ct2"
    model_dir: _RepoPath
    spm_model: _RepoPath
    family: Literal["nllb", "madlad"]
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    beam_size: _PositiveInt = 4
    batch_size: _PositiveInt = 16

    @model_validator(mode="after")
    def _hypotheses_fit_the_beam(self) -> Ct2Config:
        if self.n > self.beam_size:
            raise ValueError(
                f"n ({self.n}) cannot be greater than beam_size "
                f"({self.beam_size}): CTranslate2 returns at most beam_size "
                "hypotheses"
            )
        return self


def _generator_kind(value: Any) -> Any:
    """The `kind` of a generator given as a mapping or as a model; whatever it
    is, the discriminator rejects a value that is not a known kind."""
    if isinstance(value, dict):
        return value.get("kind")
    return getattr(value, "kind", None)


# A callable discriminator, so a missing or unknown `kind` names the valid ones.
GeneratorConfig = Annotated[
    Annotated[VllmConfig, Tag("vllm")]
    | Annotated[LlamaCppConfig, Tag("llamacpp")]
    | Annotated[Ct2Config, Tag("ct2")],
    Discriminator(
        _generator_kind,
        custom_error_type="generator_kind",
        custom_error_message="kind must be one of vllm, llamacpp, ct2",
    ),
]


def _default_generators() -> dict[str, GeneratorConfig]:
    # The CT2 paths are placeholders for the operator; nothing creates them.
    return {
        "vllm_default": VllmConfig(),
        "llamacpp_default": LlamaCppConfig(),
        "ct2_nllb_default": Ct2Config(
            model_dir=Path("models/ct2/nllb-200-3.3B"),
            spm_model=Path("models/ct2/nllb-200-3.3B/sentencepiece.bpe.model"),
            family="nllb",
        ),
    }


def _supported_job_languages() -> frozenset[str]:
    return frozenset(
        entry.code
        for entry in load_languages()
        if Script.MONGOLIAN not in entry.scripts
    )


class JobConfig(BaseModel):
    """One synthetic-data run over a monolingual corpus. For `bt` the
    monolingual text is the target of the pairs and the generator translates
    it into `other_lang` to make the source; for `ft` the monolingual text is
    the source and the generator translates it into `other_lang` to make the
    target. Both languages must be registry languages a generator can write:
    the registry's Mongolian-script entries are left out, because the D5 script
    converter owns that script. The generator name is checked by
    `SynthConfig`, which owns the generators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["bt", "ft"]
    mono_lang: str
    other_lang: str
    generator: Annotated[str, Field(min_length=1)]
    shard_size: _PositiveInt = 512

    @field_validator("mono_lang", "other_lang")
    @classmethod
    def _is_a_supported_language(cls, value: str) -> str:
        supported = _supported_job_languages()
        if value not in supported:
            raise ValueError(
                f"{value!r} is not a supported job language; "
                f"supported: {sorted(supported)}"
            )
        return value

    @model_validator(mode="after")
    def _pairs_mongolian_with_another_language(self) -> JobConfig:
        if self.mono_lang == self.other_lang:
            raise ValueError(
                f"mono_lang and other_lang must differ, both are {self.mono_lang!r}"
            )
        if _MONGOLIAN not in (self.mono_lang, self.other_lang):
            raise ValueError(
                f"one of mono_lang and other_lang must be {_MONGOLIAN}, "
                f"got {self.mono_lang!r} and {self.other_lang!r}"
            )
        return self


def _default_jobs() -> dict[str, JobConfig]:
    return {
        "bt_eng_khk": JobConfig(
            kind="bt",
            mono_lang=_MONGOLIAN,
            other_lang="eng_Latn",
            generator="vllm_default",
        ),
        "ft_khk_eng": JobConfig(
            kind="ft",
            mono_lang=_MONGOLIAN,
            other_lang="eng_Latn",
            generator="vllm_default",
        ),
    }


class SelectionConfig(BaseModel):
    """How candidates are kept and chosen. `qe` is the D4 filter's QeConfig,
    so both stages mean the same thing by "QE"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Share of the synthetic side's letters that must belong to the registry
    # scripts of its language.
    min_script_share: _Probability = 0.9
    qe: QeConfig = QeConfig()
    mbr_utility: Literal["chrf"] = "chrf"


class SynthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generators: dict[str, GeneratorConfig] = Field(default_factory=_default_generators)
    jobs: dict[str, JobConfig] = Field(default_factory=_default_jobs)
    selection: SelectionConfig = SelectionConfig()

    @model_validator(mode="before")
    @classmethod
    def _supplied_generators_replace_the_default_jobs(cls, data: Any) -> Any:
        # The default jobs name the default generators; without this they
        # would dangle as soon as `generators` is replaced.
        if isinstance(data, dict) and "generators" in data and "jobs" not in data:
            return {**data, "jobs": {}}
        return data

    @model_validator(mode="after")
    def _jobs_name_configured_generators(self) -> SynthConfig:
        for name, job in self.jobs.items():
            if job.generator not in self.generators:
                raise ValueError(
                    f"job {name!r} names unknown generator {job.generator!r}; "
                    f"configured generators: {sorted(self.generators)}"
                )
        return self


def load_synth_config(path: Path = SYNTH_CONFIG_PATH) -> SynthConfig:
    """Load and validate a `configs/data/synth.yaml`-shaped file. Raises
    ConfigError on any violation (missing, unreadable or unparseable file,
    non-mapping top level, unknown field, out-of-range or wrongly typed value,
    a job naming an unknown generator or unsupported languages)."""
    if not path.is_file():
        raise ConfigError(f"synth config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid synth config {path}: cannot read or parse: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid synth config {path}: top level must be a mapping of "
            f"section names to values, got {type(raw).__name__}"
        )
    try:
        return SynthConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid synth config {path}: {exc}") from exc
