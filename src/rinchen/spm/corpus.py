"""Find the per-language text files a SentencePiece model trains on, and draw a
temperature-balanced sample from them."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping
from pathlib import Path

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, DataError
from rinchen.utils.sampling import temperature_weights

_CORPUS_SUFFIX = ".txt"


def discover_corpus(corpus_dir: Path) -> dict[str, Path]:
    """Map each language code to its `<code>.txt` file directly in `corpus_dir`,
    sorted by code. Other extensions and subdirectories are ignored.

    Raises:
        ConfigError: `corpus_dir` is not a directory, or a `.txt` file is not
            named after a registry language code.
        DataError: there is no `.txt` file, a file has no non-blank line, or a
            file cannot be read or decoded as UTF-8.
    """
    files = _list_corpus_files(corpus_dir)
    _require_registry_codes(files)
    corpus = {path.stem: path for path in files}
    _require_text(corpus)
    return corpus


def read_lines(path: Path) -> list[str]:
    """Read a corpus file as UTF-8, one entry per non-blank line.

    Lines end at ``"\\n"`` only, with any trailing ``"\\r"`` dropped, and a
    leading BOM is removed. Nothing else is stripped: a NNBSP at a line's edge
    is meaningful in mn_Mong.

    Raises:
        DataError: the file cannot be read or is not valid UTF-8; the message
            names the file and the cause is chained.
    """
    return list(_iter_text_lines(path))


def sample_balanced(
    lines_by_lang: Mapping[str, list[str]],
    *,
    input_sentence_size: int,
    temperature: float,
    seed: int,
) -> list[str]:
    """Draw about ``input_sentence_size`` lines, tilted toward small languages.

    When the corpus already fits the budget, every line is returned. Otherwise
    each language gets ``floor(weight * input_sentence_size)`` lines, with
    weights from `temperature_weights` over the language sizes and the quota
    capped at the language's own size; a capped language's unused quota is not
    redistributed. Each language is sampled without replacement and keeps its
    original line order. Languages come out in mapping order.

    Raises:
        ValueError: ``input_sentence_size`` is not positive, or (when sampling
            is needed) `temperature_weights` rejects the temperature.
    """
    if input_sentence_size <= 0:
        raise ValueError(f"input_sentence_size must be > 0, got {input_sentence_size}")
    total = sum(len(lines) for lines in lines_by_lang.values())
    if total <= input_sentence_size:
        return [line for lines in lines_by_lang.values() for line in lines]

    weights = temperature_weights(
        [len(lines) for lines in lines_by_lang.values()], temperature
    )
    sample: list[str] = []
    for (language, lines), weight in zip(lines_by_lang.items(), weights, strict=True):
        quota = min(len(lines), math.floor(weight * input_sentence_size))
        sample.extend(_sample_language(language, lines, quota, seed))
    return sample


def _list_corpus_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        raise ConfigError(
            f"corpus directory is missing or not a directory: {corpus_dir}"
        )
    files = sorted(
        (path for path in corpus_dir.glob(f"*{_CORPUS_SUFFIX}") if path.is_file()),
        key=lambda path: path.stem,
    )
    if not files:
        raise DataError(f"no {_CORPUS_SUFFIX} corpus files in {corpus_dir}")
    return files


def _require_registry_codes(files: list[Path]) -> None:
    known = known_language_codes()
    for path in files:
        if path.stem not in known:
            raise ConfigError(
                f"corpus file {path.name} is not named after a registry language "
                f"code ({', '.join(sorted(known))})"
            )


def _require_text(corpus: Mapping[str, Path]) -> None:
    for language, path in corpus.items():
        if next(_iter_text_lines(path), None) is None:
            raise DataError(
                f"corpus for language {language} has no non-blank line: {path}"
            )


def _iter_text_lines(path: Path) -> Iterator[str]:
    # newline="\n" ends lines at "\n" only; the default would also end them at a
    # lone "\r". The "\r" of a CRLF ending is dropped below. "utf-8-sig" drops a
    # leading BOM, which would otherwise put U+FEFF into the first line and so
    # into the SentencePiece vocabulary.
    try:
        with path.open(encoding="utf-8-sig", newline="\n") as handle:
            for raw_line in handle:
                line = raw_line.removesuffix("\n").rstrip("\r")
                if line.strip():
                    yield line
    except (UnicodeDecodeError, OSError) as exc:
        raise DataError(f"cannot read corpus file {path}: {exc}") from exc


def _sample_language(
    language: str, lines: list[str], quota: int, seed: int
) -> list[str]:
    # A stream per language: string seeding is SHA-512 based, so it ignores
    # PYTHONHASHSEED, and adding or reordering a language shifts no other sample.
    rng = random.Random(f"{seed}-{language}")
    chosen = sorted(rng.sample(range(len(lines)), quota))
    return [lines[index] for index in chosen]
