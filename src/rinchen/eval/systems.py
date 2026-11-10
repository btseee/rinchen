"""The systems of D13 (spec T5): what turns the requests of a suite into outputs, and which
directions each of them can run.

A system is anything that answers `translate(requests)` with one `Output` per request, in order,
and `verdict(direction)` with whether it can run a direction and, if not, why. There are three
kinds (`SystemKind`), each an adapter over a protocol, so nothing here imports an engine, a
generator or an HTTP client:

- `EngineSystem` (kind `rinchen`) wraps anything with `translate(rows, options)`, the engine of
  `rinchen.infer`. It runs only the directions the model was trained for: the engine answers an
  untrained pair with status `ok` in the wrong language, so a direction outside
  `info.directions` is `not_trained`, never translated and never scored. ONE call per
  `translate`.
- `CandidateSystem` (kind `baseline`) wraps D6's translator protocol, `translate(texts, src,
  tgt)` giving a list of candidates per text. It runs the directions whose two languages it
  knows (`unsupported_language` otherwise), one call per direction in order; the first candidate
  is the output with status `ok`, an empty list is `no_output` with an empty text.
- `FileSystem` (kind `hypotheses`) holds outputs read from a file (`outputs_from_bytes`): the
  boundary for every other system. It runs the directions the file holds (`not_in_file`
  otherwise).

A `SystemRecord` says which system it was (kind, name, the digests of the model and of the
manifest that vouched for it, the effective options, the digest of the file a file system was
read from); it goes into every result. Nothing here knows how a model or a generator is opened:
that is the command's, which also owns closing the engine. The baseline adapter was only ever
exercised over fakes: no NLLB, MADLAD or Hy-MT2 model was loaded for it (UNMEASURED).
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol, get_args

from rinchen.errors import DataError, UpstreamError
from rinchen.eval.outputs import (
    Output,
    OutputKey,
    direction_place,
    output_place,
)
from rinchen.eval.robust_hypotheses import Translates
from rinchen.eval.suite_rules import direction_name, split_direction
from rinchen.infer.engine_types import DecodeOptions, Row

__all__ = [
    "CandidateSystem",
    "Candidates",
    "DirectionState",
    "EngineSystem",
    "FileSystem",
    "ModelFacts",
    "Scalar",
    "SuiteRequest",
    "SuiteSystem",
    "SystemKind",
    "SystemRecord",
    "Verdict",
]

SystemKind = Literal["rinchen", "baseline", "hypotheses"]
DirectionState = Literal[
    "run", "not_trained", "unsupported_language", "not_in_file", "not_selected"
]
Scalar = str | int | float | bool

NOT_TRAINED_REASON: Final = "not one of the model's trained directions"
UNSUPPORTED_LANGUAGE_REASON: Final = (
    "a language of the direction is not one the baseline knows"
)
NOT_IN_FILE_REASON: Final = "the file holds no output for the direction"
_KINDS: Final = get_args(SystemKind)
_STATES: Final = get_args(DirectionState)
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SuiteRequest:
    """One row of a direction that a system is asked to translate: the `direction`, the row
    `id`, the two languages and the source `text`.

    Raises:
        ValueError: `direction` is not `<src_lang>-<tgt_lang>`.
    """

    direction: str
    id: str
    src_lang: str
    tgt_lang: str
    text: str

    def __post_init__(self) -> None:
        if self.direction != direction_name(self.src_lang, self.tgt_lang):
            raise ValueError(
                "direction must be the source and target language joined by '-'"
            )


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a system can run a direction: its `state` (`run`, or why not) and, for a state
    other than `run`, a short fixed sentence saying why.

    Raises:
        ValueError: `state` is not one of the five.
    """

    state: DirectionState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise ValueError(f"state must be one of {', '.join(_STATES)}")


@dataclass(frozen=True, slots=True)
class SystemRecord:
    """Which system produced the outputs: its `kind`, its `name`, the SHA-256 of the model
    (`model_sha256`) and of the manifest that vouched for it (`manifest_sha256`) when it has
    them, the effective `options` (a mapping of strings to strings, numbers and booleans) and,
    for a file system, the digest of the file it was read from (`outputs_sha256`).

    Raises:
        TypeError: `name` is not a string, or `options` is not a mapping of strings to
            strings, numbers and booleans.
        ValueError: `kind` is not one of the three, a digest is not 64 lowercase hexadecimal
            characters (or None), or an option is a number that is not finite.
    """

    kind: SystemKind
    name: str
    model_sha256: str | None = None
    manifest_sha256: str | None = None
    options: Mapping[str, Scalar] = field(default_factory=dict)
    outputs_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {', '.join(_KINDS)}")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        for digest_field in ("model_sha256", "manifest_sha256", "outputs_sha256"):
            digest = getattr(self, digest_field)
            if digest is not None and not (
                isinstance(digest, str) and _DIGEST.fullmatch(digest)
            ):
                raise ValueError(
                    f"{digest_field} must be 64 lowercase hexadecimal characters or None"
                )
        object.__setattr__(self, "options", _checked_options(self.options))

    def to_dict(self) -> dict[str, object]:
        """The record as a JSON-ready dictionary with the six keys; the options in sorted
        order."""
        return {
            "kind": self.kind,
            "name": self.name,
            "model_sha256": self.model_sha256,
            "manifest_sha256": self.manifest_sha256,
            "options": dict(self.options),
            "outputs_sha256": self.outputs_sha256,
        }


def _checked_options(options: Mapping[str, Scalar]) -> Mapping[str, Scalar]:
    """A read-only copy of `options` in sorted order."""
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    for key, value in options.items():
        if not isinstance(key, str):
            raise TypeError("options must be named by strings")
        if not isinstance(value, str | int | float):  # a bool is an int
            raise TypeError("options must hold strings, numbers and booleans only")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("an option that is a number must be finite")
    return MappingProxyType(dict(sorted(options.items())))


class SuiteSystem(Protocol):
    """What the runner asks of a system."""

    @property
    def record(self) -> SystemRecord: ...

    def verdict(self, direction: str) -> Verdict: ...

    def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]: ...


class ModelFacts(Protocol):
    """What an engine system reads of a loaded model (`rinchen.infer.engine_types.ModelInfo`
    has it): the directions it was trained for and the digests of the model and of the
    manifest that vouched for it."""

    @property
    def directions(self) -> Sequence[str]: ...

    @property
    def model_sha256(self) -> str: ...

    @property
    def manifest_sha256(self) -> str: ...


class Candidates(Protocol):
    """D6's translator protocol (`rinchen.synth.translator.Translator`): one list of candidates
    per text, best first, possibly empty."""

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]: ...


class EngineSystem:
    """A Rinchen model, through anything with `translate(rows, options)`.

    `info` says which directions the model was trained for and its digests, `options` is what
    the engine decodes with, `name` names the system and `options_record` is the effective
    settings that go into the record."""

    def __init__(
        self,
        engine: Translates,
        info: ModelFacts,
        options: DecodeOptions,
        name: str,
        options_record: Mapping[str, Scalar],
    ) -> None:
        self._engine = engine
        self._directions = tuple(info.directions)
        self._options = options
        self.record: Final = SystemRecord(
            kind="rinchen",
            name=name,
            model_sha256=info.model_sha256,
            manifest_sha256=info.manifest_sha256,
            options=options_record,
        )

    def verdict(self, direction: str) -> Verdict:
        if direction in self._directions:
            return Verdict("run")
        return Verdict("not_trained", NOT_TRAINED_REASON)

    def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]:
        """One output per request, in order, from ONE call of the engine.

        Raises:
            UpstreamError: the engine answered a different number of translations.
        """
        if not requests:
            return []
        rows = [Row(r.text, r.src_lang, r.tgt_lang) for r in requests]
        answers = self._engine.translate(rows, self._options)
        if len(answers) != len(requests):
            raise UpstreamError(
                f"the engine answered {len(answers)} translations for "
                f"{len(requests)} requests"
            )
        return [Output(answer.text, answer.status) for answer in answers]


class CandidateSystem:
    """A baseline, through D6's translator protocol.

    `languages` are the registry codes the baseline's family has a code for; `options_record`
    is the effective settings that go into the record."""

    def __init__(
        self,
        translator: Candidates,
        name: str,
        languages: Collection[str],
        options_record: Mapping[str, Scalar],
    ) -> None:
        self._translator = translator
        self._languages = frozenset(languages)
        self.record: Final = SystemRecord(
            kind="baseline", name=name, options=options_record
        )

    def verdict(self, direction: str) -> Verdict:
        src, tgt = split_direction(direction)
        if src in self._languages and tgt in self._languages:
            return Verdict("run")
        return Verdict("unsupported_language", UNSUPPORTED_LANGUAGE_REASON)

    def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]:
        """One output per request, in the order of the requests, from one call per direction
        (the directions in the order they first appear).

        Raises:
            UpstreamError: the translator answered a different number of candidate lists for
                the texts of a direction.
        """
        groups: dict[str, list[int]] = {}
        for index, request in enumerate(requests):
            groups.setdefault(request.direction, []).append(index)
        answered: dict[int, Output] = {}
        for direction, indexes in groups.items():
            first = requests[indexes[0]]
            texts = [requests[index].text for index in indexes]
            candidate_lists = self._translator.translate(
                texts, first.src_lang, first.tgt_lang
            )
            if len(candidate_lists) != len(texts):
                raise UpstreamError(
                    f"the translator answered {len(candidate_lists)} candidate lists for "
                    f"{len(texts)} texts of {direction_place(direction)}"
                )
            for index, candidates in zip(indexes, candidate_lists, strict=True):
                answered[index] = (
                    Output(candidates[0], "ok")
                    if candidates
                    else Output("", "no_output")
                )
        return [answered[index] for index in range(len(requests))]


class FileSystem:
    """Outputs read from a file, as `outputs_from_bytes` returns them (every direction in it
    complete); `outputs_sha256` is the digest of the bytes they were read from."""

    def __init__(
        self, outputs: Mapping[OutputKey, Output], name: str, outputs_sha256: str
    ) -> None:
        self._outputs = outputs
        self._directions = frozenset(key.direction for key in outputs)
        self.record: Final = SystemRecord(
            kind="hypotheses", name=name, outputs_sha256=outputs_sha256
        )

    def verdict(self, direction: str) -> Verdict:
        if direction in self._directions:
            return Verdict("run")
        return Verdict("not_in_file", NOT_IN_FILE_REASON)

    def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]:
        """The output of every request, in order.

        Raises:
            DataError: the file holds no output for a request.
        """
        found: list[Output] = []
        for request in requests:
            output = self._outputs.get(OutputKey(request.direction, request.id))
            if output is None:
                raise DataError(
                    f"the file holds no output for {output_place(request.direction, request.id)}"
                )
            found.append(output)
        return found
