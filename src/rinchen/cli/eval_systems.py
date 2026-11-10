"""The systems `rinchen eval run` builds (D13, spec T5): a Rinchen export through the engine, a
baseline through the generator configuration of D6 and a file of hypotheses. What each can run, the
one call and the states of the directions are `rinchen.eval.systems`'; this module is what that
layer may not know: where a model is opened (the two seams the tests replace, `open_engine` and
`build_translator`), how a generator's configuration is read and bent to an evaluation, and the
guard that stops a baseline whose vocabulary has none of its family's language tokens.

A generator's synthetic-data defaults are SAMPLING settings (`n` 4, `temperature` 0.7): an
evaluation translates once, greedily. The effective settings are `n` 1 for every generator and,
for the servers, `temperature` 0.0 and `top_p` 1.0; for a CTranslate2 model the beam stays as
configured. They are what the system's record holds (never a path or an address). `device` `auto`
is refused because CTranslate2 would pick the GPU when one is visible; `cuda` is the operator's
decision and was never run here (unmeasured).

The vocabulary guard exists because D6's `CT2Translator` is silent about a family whose language
tokens are not in the model's vocabulary: CTranslate2 maps an unknown token to `<unk>` and the
model answers noise, the same noise for any source (measured on the toy export, which has
`<2eng_Latn>` and no NLLB or MADLAD token). So a CTranslate2 baseline is run only when its
`shared_vocabulary.json` holds the token of every language of its family's table; the toy
export's vocabulary is the negative control. The baseline adapter was only ever exercised over
fakes: no NLLB, MADLAD or Hy-MT2 model was loaded for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import yaml
from pydantic import ValidationError

from rinchen.cli.eval_robustness_plan import RunOptions
from rinchen.cli.synth import build_translator
from rinchen.cli.train_common import hash_input
from rinchen.config.synth import Ct2Config, GeneratorConfig, SynthConfig
from rinchen.errors import ConfigError
from rinchen.eval.outputs import outputs_from_bytes
from rinchen.eval.robust_hypotheses import Translates
from rinchen.eval.suite_rules import split_direction
from rinchen.eval.suites import Suite
from rinchen.eval.systems import (
    CandidateSystem,
    EngineSystem,
    FileSystem,
    Scalar,
    Verdict,
)
from rinchen.export.record import CT2_FILES
from rinchen.export.verified import string_list, verified_content
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import LoadOptions, ModelInfo
from rinchen.manifest.model import ManifestFile
from rinchen.synth.ct2_baselines import MADLAD_CODES, NLLB_CODES
from rinchen.synth.prompts import LANGUAGE_NAMES, supported_directions

__all__ = [
    "GeneratorSystem",
    "RunEngine",
    "build_translator",
    "engine_system",
    "file_system",
    "generator_system",
    "open_engine",
    "vocabulary_guard",
]

Family = Literal["nllb", "madlad"]
_VOCABULARY_FILE: Final = CT2_FILES[1]
_BASELINE_FILE: Final = "baseline model file"
_FAMILY_TOKENS: Final[dict[str, dict[str, str]]] = {
    "nllb": dict(NLLB_CODES),
    "madlad": dict(MADLAD_CODES),
}
UNSUPPORTED_PAIR_REASON: Final = (
    "the baseline's prompts cover only Khalkha to and from a partner language"
)


class RunEngine(Translates, Protocol):
    """What the command asks of an engine: `rinchen.infer.engine.Engine` has all of it."""

    @property
    def info(self) -> ModelInfo: ...

    def close(self) -> None: ...


def open_engine(model: Path, load: LoadOptions) -> RunEngine:
    """The engine over the export `model`: the one place the command touches a Rinchen model
    (the tests replace it). The caller closes it.

    Raises:
        ConfigError, DataError: as `Engine.open`.
    """
    return Engine.open(model, load)


# ---- the systems of a model and of a file --------------------------------------------------------


def engine_system(engine: RunEngine, options: RunOptions, name: str) -> EngineSystem:
    """The Rinchen system over `engine` under `name`: it runs the directions the model was
    trained for, with `options`, and records them (`converter` is not one: this command attaches
    none)."""
    record = {
        key: value
        for key, value in options.record(converter=False).items()
        if key != "converter"
    }
    return EngineSystem(engine, engine.info, options.decode(), name, record)


def file_system(data: bytes, digest: str, suite: Suite, name: str) -> FileSystem:
    """The system that answers from the outputs file `data` (whose SHA-256 is `digest`) under
    `name`.

    Raises:
        DataError: `data` is not the outputs of `suite` (`outputs_from_bytes`).
    """
    return FileSystem(outputs_from_bytes(data, suite), name, digest)


# ---- the baselines -------------------------------------------------------------------------------------


class GeneratorSystem(CandidateSystem):
    """A baseline of D6's configuration. `inputs` are the files of its model that were hashed
    before the translator loaded them (none for a server)."""

    inputs: tuple[ManifestFile, ...] = ()


class _ServerSystem(GeneratorSystem):
    """A chat model behind an HTTP endpoint. D6's prompts exist only for Khalkha to and from a
    partner language, so any other pair would raise in the middle of the run: it is a state of
    the direction instead."""

    def verdict(self, direction: str) -> Verdict:
        if split_direction(direction) in supported_directions():
            return Verdict("run")
        return Verdict("unsupported_language", UNSUPPORTED_PAIR_REASON)


def generator_system(
    config_bytes: bytes, name: str, *, system_name: str | None = None
) -> GeneratorSystem:
    """The baseline `name` of the generators configuration `config_bytes` (the bytes that were
    hashed), under `system_name` (default `name`), with the evaluation's settings.

    A CTranslate2 model's files are hashed (`inputs`), its vocabulary is guarded and only then is
    the translator built; a server is not contacted until it translates.

    Raises:
        ConfigError: the configuration is not usable, `name` is not in it, the device is `auto`,
            or a file or token of the model is missing (see `vocabulary_guard`).
        DataError, UpstreamError: as `build_translator`.
    """
    generators = _generators_of(config_bytes).generators
    configured = generators.get(name)
    if configured is None:
        raise ConfigError(
            f"no generator named {name} in the generators config (it has "
            f"{', '.join(sorted(generators))})"
        )
    effective = _evaluation_config(name, configured)
    inputs: tuple[ManifestFile, ...] = ()
    if isinstance(effective, Ct2Config):
        inputs = _hash_model_files(effective)
        languages = tuple(_codes(effective))
        vocabulary_guard(
            effective.model_dir,
            effective.family,
            languages,
            hashed=_entry_named(inputs, _VOCABULARY_FILE),
        )
        system: GeneratorSystem = GeneratorSystem(
            build_translator(effective),
            system_name or name,
            languages,
            _options_of(name, effective),
        )
    else:
        system = _ServerSystem(
            build_translator(effective),
            system_name or name,
            tuple(LANGUAGE_NAMES),
            _options_of(name, effective),
        )
    system.inputs = inputs
    return system


def _generators_of(data: bytes) -> SynthConfig:
    """The generators configuration in the YAML `data`.

    Raises:
        ConfigError: `data` is not UTF-8, not YAML, not a mapping or breaks the schema; the
            messages name fields and never quote a value.
    """
    try:
        raw: Any = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError("invalid generators config: not UTF-8 text") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = (
            ""
            if mark is None
            else f" at line {mark.line + 1}, column {mark.column + 1}"
        )
        raise ConfigError(
            f"invalid generators config: cannot parse the YAML{where}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            "invalid generators config: the top level must be a mapping of section "
            f"names to values, not {type(raw).__name__}"
        )
    try:
        return SynthConfig.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
            for item in error.errors(include_input=False)
        )
        raise ConfigError(f"invalid generators config: {problems}") from error


def _evaluation_config(name: str, configured: GeneratorConfig) -> GeneratorConfig:
    """`configured` with the settings of an evaluation: one candidate, and for a server greedy
    decoding. The configuration is frozen, so this is a copy.

    Raises:
        ConfigError: a CTranslate2 model on device `auto`.
    """
    if isinstance(configured, Ct2Config):
        if configured.device == "auto":
            raise ConfigError(
                f"the generator {name} has device 'auto': CTranslate2 would choose the GPU "
                "when one is visible, so say cpu or cuda in the configuration"
            )
        return configured.model_copy(update={"n": 1})
    return configured.model_copy(update={"n": 1, "temperature": 0.0, "top_p": 1.0})


def _options_of(name: str, effective: GeneratorConfig) -> dict[str, Scalar]:
    """The effective settings that go into the record: no path and no address."""
    if isinstance(effective, Ct2Config):
        return {
            "generator": name,
            "kind": "ct2",
            "family": effective.family,
            "device": effective.device,
            "beam_size": effective.beam_size,
            "batch_size": effective.batch_size,
            "n": effective.n,
            "max_tokens": effective.max_tokens,
        }
    return {
        "generator": name,
        "kind": effective.kind,
        "model": effective.model,
        "n": effective.n,
        "temperature": effective.temperature,
        "top_p": effective.top_p,
        "max_tokens": effective.max_tokens,
    }


def _codes(config: Ct2Config) -> dict[str, str]:
    """The family's table: registry code to the token the family's model reads."""
    return _FAMILY_TOKENS[config.family]


def _entry_named(entries: Sequence[ManifestFile], name: str) -> ManifestFile:
    (entry,) = (entry for entry in entries if Path(entry.path).name == name)
    return entry


def _hash_model_files(config: Ct2Config) -> tuple[ManifestFile, ...]:
    """The converter's three files and the SentencePiece model, each hashed before anything
    reads it."""
    paths = [config.model_dir / name for name in CT2_FILES]
    return tuple(
        hash_input(path, what=_BASELINE_FILE, missing=ConfigError)
        for path in (*paths, config.spm_model)
    )


# ---- the vocabulary guard (F14) ------------------------------------------------------------------------


def vocabulary_guard(
    model_dir: Path,
    family: Family,
    languages: Sequence[str],
    *,
    hashed: ManifestFile | None = None,
) -> None:
    """Refuse a model directory whose vocabulary lacks the token of a language of `languages`
    in the table of `family` (NLLB: the language code; MADLAD: the target tag `<2xx>`). The
    vocabulary (`shared_vocabulary.json`) is hashed first and parsed from the bytes that were
    hashed; `hashed` is the entry of a hash the caller already took (and recorded), which the
    bytes read must still match.

    Raises:
        ConfigError: the file is missing, is not a JSON list of strings, changed after it was
            hashed, or has no token for a language (the first one in `languages`; the message
            names the family and the language and quotes no entry of the vocabulary).
    """
    entry = hashed or hash_input(
        model_dir / _VOCABULARY_FILE, what=_BASELINE_FILE, missing=ConfigError
    )
    tokens = string_list(verified_content(Path(entry.path), entry.sha256))
    if tokens is None:
        raise ConfigError(
            f"{entry.path} must be a JSON list of strings: it is the vocabulary of the "
            f"{family} model"
        )
    known = frozenset(tokens)
    table = _FAMILY_TOKENS[family]
    for language in languages:
        token = table.get(language)
        if token is None:
            raise ConfigError(
                f"the {family} table has no token for the language {language}"
            )
        if token not in known:
            # MADLAD's token is a tag (`<2mn>`) that differs from the code; NLLB's is the code.
            shown = "" if token == language else f" ({token})"
            raise ConfigError(
                f"the {family} model's vocabulary {entry.path} has no token for the "
                f"language {language}{shown}: the model would read it as unknown and "
                "answer with noise"
            )
