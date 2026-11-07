"""Tests for `rinchen.script_conv.char_model`: the converter's own character-level
SentencePiece model, its check and its config.

The model trains on the Cyrillic and script columns of the tiny reference list. The
traditional-script characters under test are `chr()` calls and the words that hold
them come from the fixture, so the exact code points stay reviewable. Every file
lands under pytest's temporary directories.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import sentencepiece
import yaml
from pydantic import ValidationError

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.script_conv.alphabet import FVS, MVS, NNBSP
from rinchen.script_conv.char_model import (
    CONVERTER_TAGS,
    CharModelConfig,
    check_char_model,
    load_char_model_config,
    train_char_model,
)
from tests.support.script_reference import FIXTURE_DIR

TINY_WORDS = FIXTURE_DIR / "tiny_words.tsv"
CONFIG_PATH = REPO_ROOT / "configs" / "spm" / "converter_char.yaml"

TAG_TO_SCRIPT = "<2mn_Mong>"
TAG_TO_CYRILLIC = "<2mn_Cyrl>"
# Far above the fixture's character count: with `hard_vocab_limit=False` the trainer
# settles on the characters it saw, so the size only has to be large enough.
VOCAB_SIZE = 200
# Below the five pieces every model holds (unknown, two sentence markers, two tags),
# so the trainer refuses.
TOO_SMALL_VOCAB_SIZE = 4
# Above that but far below the fixture's characters: the trainer accepts it.
TRUNCATING_VOCAB_SIZE = 40
# Pieces every model holds besides the characters and the tags: unknown, the two
# sentence markers and the dummy prefix.
SPECIAL_PIECES = 4
DUMMY_PREFIX = chr(0x2581)
FVS_FIRST = chr(0x180B)


def _read_pairs() -> list[tuple[str, str]]:
    with TINY_WORDS.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)  # the header
        return [(cyrillic, script) for cyrillic, script in reader]


PAIRS = _read_pairs()
CYRILLIC_WORDS = [cyrillic for cyrillic, _ in PAIRS]
SCRIPT_WORDS = [script for _, script in PAIRS]


def _fixture_word_with(character: str) -> str | None:
    return next((word for word in SCRIPT_WORDS if character in word), None)


def _word_with(character: str) -> str:
    """A script word that holds `character`. The fixture has no word with the third
    or fourth free variation selector, so for those the word of the first selector
    stands in, with the selector swapped."""
    word = _fixture_word_with(character)
    if word is not None:
        return word
    assert character in FVS, f"no fixture word holds U+{ord(character):04X}"
    base = _fixture_word_with(FVS_FIRST)
    assert base is not None
    return base.replace(FVS_FIRST, character)


# The characters whose survival matters most: they are invisible or look like a gap.
SEPARATORS = {
    "NNBSP": NNBSP,
    "MVS": MVS,
    **{f"FVS-U+{ord(selector):04X}": selector for selector in sorted(FVS)},
}
SEPARATOR_CASES = [pytest.param(char, id=name) for name, char in SEPARATORS.items()]

STAND_IN_WORDS = [
    word
    for word in (_word_with(selector) for selector in sorted(FVS))
    if word not in SCRIPT_WORDS
]
TRAINING_TEXTS = [*CYRILLIC_WORDS, *SCRIPT_WORDS, *STAND_IN_WORDS]


def _processor(model: Path) -> sentencepiece.SentencePieceProcessor:
    return sentencepiece.SentencePieceProcessor(model_file=str(model))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_variant(
    texts: list[str],
    prefix: Path,
    *,
    symbols: list[str],
    symbol_option: str = "user_defined_symbols",
) -> Path:
    """Train through the real trainer with the converter's options, except for the
    reserved symbols, which the caller sets, and the option that carries them."""
    corpus = prefix.parent / f"{prefix.name}.txt"
    corpus.write_text("".join(f"{text}\n" for text in texts), encoding="utf-8")
    options: dict[str, object] = {
        "input": str(corpus),
        "model_prefix": str(prefix),
        "vocab_size": VOCAB_SIZE,
        "model_type": "char",
        "character_coverage": 1.0,
        "byte_fallback": False,
        "normalization_rule_name": "identity",
        "add_dummy_prefix": True,
        "hard_vocab_limit": False,
        "minloglevel": 2,
    }
    if symbols:
        options[symbol_option] = symbols
    sentencepiece.SentencePieceTrainer.train(**options)
    return Path(f"{prefix}.model")


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    prefix = tmp_path_factory.mktemp("char_model") / "converter_char"
    return train_char_model(TRAINING_TEXTS, prefix, vocab_size=VOCAB_SIZE)


# --- train_char_model: what the trainer is asked to do -------------------------------


def test_converter_tags_are_the_two_target_script_tags() -> None:
    assert CONVERTER_TAGS == (TAG_TO_SCRIPT, TAG_TO_CYRILLIC)


def test_trainer_receives_exactly_the_policy_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def record_options(**options: object) -> None:
        captured.update(options)

    monkeypatch.setattr(
        sentencepiece.SentencePieceTrainer, "train", staticmethod(record_options)
    )
    prefix = tmp_path / "converter_char"

    train_char_model(["ab"], prefix, vocab_size=VOCAB_SIZE)

    # A fixed name beside the prefix, so the model file does not depend on a random
    # input path.
    assert captured.pop("input") == str(tmp_path / "converter_char.input.txt")
    # Spelled out rather than imported, so changing a policy option fails here. An
    # option not listed (a seed, shuffling) is left at the trainer's default.
    assert captured == {
        "model_prefix": str(prefix),
        "vocab_size": VOCAB_SIZE,
        "model_type": "char",
        "character_coverage": 1.0,
        "byte_fallback": False,
        "normalization_rule_name": "identity",
        "add_dummy_prefix": True,
        "user_defined_symbols": [TAG_TO_SCRIPT, TAG_TO_CYRILLIC],
        "hard_vocab_limit": False,
        "minloglevel": 2,
    }
    assert captured["byte_fallback"] is False
    assert captured["add_dummy_prefix"] is True
    assert captured["hard_vocab_limit"] is False


def test_trainer_reads_the_texts_one_per_line_and_skips_blank_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, bytes] = {}

    def read_input(**options: object) -> None:
        seen["input"] = Path(str(options["input"])).read_bytes()

    monkeypatch.setattr(
        sentencepiece.SentencePieceTrainer, "train", staticmethod(read_input)
    )
    texts = (text for text in ["ab", "", "  ", "cd"])  # one pass only

    train_char_model(texts, tmp_path / "converter_char", vocab_size=VOCAB_SIZE)

    assert seen["input"] == b"ab\ncd\n"


def test_a_text_of_only_narrow_no_break_spaces_is_not_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only U+0020 separates words, so U+202F alone is a (strange) word to keep.
    seen: dict[str, bytes] = {}

    def read_input(**options: object) -> None:
        seen["input"] = Path(str(options["input"])).read_bytes()

    monkeypatch.setattr(
        sentencepiece.SentencePieceTrainer, "train", staticmethod(read_input)
    )

    train_char_model([NNBSP], tmp_path / "converter_char", vocab_size=VOCAB_SIZE)

    assert seen["input"] == (NNBSP + "\n").encode("utf-8")


@pytest.mark.parametrize(
    "texts",
    [
        pytest.param([], id="no-texts"),
        pytest.param(["", "   ", " "], id="only-blank-texts"),
    ],
)
def test_texts_without_a_non_blank_line_raise_data_error_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, texts: list[str]
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        sentencepiece.SentencePieceTrainer, "train", staticmethod(calls.append)
    )

    with pytest.raises(DataError, match="no non-blank text"):
        train_char_model(texts, tmp_path / "converter_char", vocab_size=VOCAB_SIZE)

    assert calls == []
    assert list(tmp_path.iterdir()) == []


# --- train_char_model: what a real run leaves behind ---------------------------------


def test_real_run_returns_the_model_path_and_leaves_only_model_and_vocab(
    trained_model: Path,
) -> None:
    assert trained_model.name == "converter_char.model"
    assert sorted(path.name for path in trained_model.parent.iterdir()) == [
        "converter_char.model",
        "converter_char.vocab",
    ]


def test_missing_prefix_directory_is_created(tmp_path: Path) -> None:
    prefix = tmp_path / "nested" / "dir" / "converter_char"

    model = train_char_model(["ab", "ba"], prefix, vocab_size=VOCAB_SIZE)

    assert model == Path(f"{prefix}.model")
    assert model.is_file()


def test_training_three_times_into_one_prefix_gives_identical_files(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "converter_char"
    model_hashes = []
    vocab_hashes = []

    for _ in range(3):
        model = train_char_model(TRAINING_TEXTS, prefix, vocab_size=VOCAB_SIZE)
        model_hashes.append(_sha256(model))
        vocab_hashes.append(_sha256(Path(f"{prefix}.vocab")))

    assert len(set(model_hashes)) == 1
    assert len(set(vocab_hashes)) == 1


def test_vocab_size_is_only_a_cap_on_the_pieces(tmp_path: Path) -> None:
    model = train_char_model(TRAINING_TEXTS, tmp_path / "capped", vocab_size=512)

    characters = {char for text in TRAINING_TEXTS for char in text}
    pieces = _processor(model).get_piece_size()
    assert pieces == len(characters) + len(CONVERTER_TAGS) + SPECIAL_PIECES
    assert pieces < 512


def test_a_vocab_too_small_for_the_special_pieces_raises_upstream_error_with_cause(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "converter_char"

    with pytest.raises(UpstreamError, match=re.escape(str(prefix))) as excinfo:
        train_char_model(TRAINING_TEXTS, prefix, vocab_size=TOO_SMALL_VOCAB_SIZE)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) in str(excinfo.value)
    assert not Path(f"{prefix}.input.txt").exists()


def test_the_prefix_directory_is_created_even_when_training_fails(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "nested" / "converter_char"

    with pytest.raises(UpstreamError):
        train_char_model(TRAINING_TEXTS, prefix, vocab_size=TOO_SMALL_VOCAB_SIZE)

    assert prefix.parent.is_dir()
    assert list(prefix.parent.iterdir()) == []


def test_a_text_iterator_that_raises_midway_leaves_no_input_file(
    tmp_path: Path,
) -> None:
    def failing_texts() -> Iterator[str]:
        yield "ab"
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed") as excinfo:
        train_char_model(
            failing_texts(), tmp_path / "converter_char", vocab_size=VOCAB_SIZE
        )

    # The source's own error, not SentencePiece's.
    assert not isinstance(excinfo.value, UpstreamError)
    assert list(tmp_path.iterdir()) == []


def test_a_vocab_below_the_character_count_trains_a_model_the_check_rejects(
    tmp_path: Path,
) -> None:
    # `hard_vocab_limit=False` lets the trainer drop the rarest characters, so the
    # check is what catches a vocab size that is too small for the texts.
    model = train_char_model(
        TRAINING_TEXTS, tmp_path / "truncated", vocab_size=TRUNCATING_VOCAB_SIZE
    )

    with pytest.raises(DataError, match="unknown to the model"):
        check_char_model(model, TRAINING_TEXTS)


def test_the_input_file_is_removed_when_the_trainer_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**options: object) -> None:
        assert Path(str(options["input"])).is_file()
        raise RuntimeError("boom")

    monkeypatch.setattr(sentencepiece.SentencePieceTrainer, "train", staticmethod(fail))

    with pytest.raises(UpstreamError, match="boom"):
        train_char_model(["ab"], tmp_path / "converter_char", vocab_size=VOCAB_SIZE)

    assert list(tmp_path.iterdir()) == []


# --- the trained model ---------------------------------------------------------------


@pytest.mark.parametrize("tag", [TAG_TO_SCRIPT, TAG_TO_CYRILLIC])
def test_each_tag_is_one_known_piece(trained_model: Path, tag: str) -> None:
    processor = _processor(trained_model)

    tag_id = processor.piece_to_id(tag)

    assert tag_id != processor.unk_id()
    assert processor.encode(tag) in (
        [tag_id],
        [processor.piece_to_id(DUMMY_PREFIX), tag_id],
    )


def test_every_training_text_round_trips(trained_model: Path) -> None:
    processor = _processor(trained_model)

    changed = [
        text
        for text in TRAINING_TEXTS
        if processor.decode(processor.encode(text)) != text
    ]

    assert changed == []


def test_no_training_text_has_an_unknown_piece(trained_model: Path) -> None:
    processor = _processor(trained_model)

    with_unknown = [
        text for text in TRAINING_TEXTS if processor.unk_id() in processor.encode(text)
    ]

    assert with_unknown == []


@pytest.mark.parametrize("character", SEPARATOR_CASES)
def test_gap_and_selector_characters_survive_inside_a_word(
    trained_model: Path, character: str
) -> None:
    processor = _processor(trained_model)
    word = _word_with(character)

    pieces = processor.encode(word, out_type=str)

    assert character in pieces
    assert processor.decode(processor.encode(word)) == word


# --- check_char_model ----------------------------------------------------------------


def test_check_accepts_the_trained_model_on_its_training_texts(
    trained_model: Path,
) -> None:
    check_char_model(trained_model, TRAINING_TEXTS)


@pytest.mark.parametrize(
    "extra_texts",
    [
        # The tag's characters are unknown, so the pieces merge into one unknown
        # piece whose surface text is the tag: it looks atomic in string form.
        pytest.param([], id="tag-characters-unknown"),
        pytest.param([TAG_TO_SCRIPT, TAG_TO_CYRILLIC], id="tag-split-to-characters"),
    ],
)
def test_check_names_a_tag_the_model_does_not_hold_as_one_piece(
    tmp_path: Path, extra_texts: list[str]
) -> None:
    model = _train_variant(
        [*TRAINING_TEXTS, *extra_texts], tmp_path / "plain", symbols=[]
    )

    with pytest.raises(DataError, match=re.escape(TAG_TO_SCRIPT)) as excinfo:
        check_char_model(model, TRAINING_TEXTS)

    assert str(model) in str(excinfo.value)


@pytest.mark.parametrize(
    "extra_texts",
    [
        pytest.param([], id="tag-characters-unknown"),
        pytest.param([TAG_TO_SCRIPT, TAG_TO_CYRILLIC], id="tag-characters-known"),
    ],
)
def test_check_names_a_tag_that_is_a_piece_but_never_matched_in_text(
    tmp_path: Path, extra_texts: list[str]
) -> None:
    # A control symbol is a piece of the model, but text never matches it.
    model = _train_variant(
        [*TRAINING_TEXTS, *extra_texts],
        tmp_path / "control",
        symbols=list(CONVERTER_TAGS),
        symbol_option="control_symbols",
    )
    processor = _processor(model)
    assert processor.piece_to_id(TAG_TO_SCRIPT) != processor.unk_id()

    with pytest.raises(DataError, match="not matched as a symbol") as excinfo:
        check_char_model(model, TRAINING_TEXTS)

    message = str(excinfo.value)
    assert TAG_TO_SCRIPT in message
    assert "encodes to ids" in message
    assert "split into" not in message


def test_check_names_the_second_tag_when_only_the_first_is_defined(
    tmp_path: Path,
) -> None:
    model = _train_variant(TRAINING_TEXTS, tmp_path / "one", symbols=[TAG_TO_SCRIPT])

    with pytest.raises(DataError, match=re.escape(TAG_TO_CYRILLIC)):
        check_char_model(model, TRAINING_TEXTS)


def test_check_looks_at_the_tags_even_when_there_are_no_texts(tmp_path: Path) -> None:
    model = _train_variant(TRAINING_TEXTS, tmp_path / "plain", symbols=[])

    with pytest.raises(DataError, match=re.escape(TAG_TO_SCRIPT)):
        check_char_model(model, [])


def test_check_names_the_unknown_character_and_the_text_holding_it(
    tmp_path: Path,
) -> None:
    model = train_char_model(
        CYRILLIC_WORDS, tmp_path / "cyrillic", vocab_size=VOCAB_SIZE
    )
    script_word = SCRIPT_WORDS[0]

    with pytest.raises(DataError) as excinfo:
        check_char_model(model, [CYRILLIC_WORDS[0], script_word])

    assert f"U+{ord(script_word[0]):04X}" in str(excinfo.value)
    assert ascii(script_word) in str(excinfo.value)


@pytest.mark.parametrize("character", SEPARATOR_CASES)
def test_check_names_a_gap_or_selector_character_the_model_never_saw(
    tmp_path: Path, character: str
) -> None:
    seen = [text for text in TRAINING_TEXTS if character not in text]
    model = train_char_model(seen, tmp_path / "without", vocab_size=VOCAB_SIZE)

    with pytest.raises(DataError, match=f"U\\+{ord(character):04X}"):
        check_char_model(model, [_word_with(character)])


def test_check_names_a_text_that_does_not_round_trip(trained_model: Path) -> None:
    # Every character is known, but the doubled space collapses on decode.
    text = f"{CYRILLIC_WORDS[0]}  {CYRILLIC_WORDS[1]}"

    with pytest.raises(DataError, match="round-trip") as excinfo:
        check_char_model(trained_model, [CYRILLIC_WORDS[2], text])

    assert ascii(text) in str(excinfo.value)
    assert ascii(CYRILLIC_WORDS[2]) not in str(excinfo.value)


def test_check_reads_texts_from_a_one_pass_iterator(trained_model: Path) -> None:
    check_char_model(trained_model, iter(TRAINING_TEXTS))


@pytest.mark.parametrize(
    ("kind", "expected_cause"),
    [
        pytest.param("missing", OSError, id="missing-file"),
        pytest.param("not-a-model", RuntimeError, id="not-a-model"),
        pytest.param("directory", None, id="directory"),
    ],
)
def test_check_reports_a_model_that_cannot_be_loaded(
    tmp_path: Path, kind: str, expected_cause: type[Exception] | None
) -> None:
    model = tmp_path / "bad.model"
    if kind == "not-a-model":
        model.write_text("not a model", encoding="utf-8")
    elif kind == "directory":
        model.mkdir()

    with pytest.raises(DataError, match="cannot load") as excinfo:
        check_char_model(model, ["ab"])

    assert str(model) in str(excinfo.value)
    if expected_cause is None:
        assert excinfo.value.__cause__ is None
    else:
        assert isinstance(excinfo.value.__cause__, expected_cause)


# --- config --------------------------------------------------------------------------

_VALID_FIELDS: dict[str, str | None] = {
    "vocab_size": "512",
    "output_prefix": "out/converter_char",
}


def _config_yaml(**fields: str | None) -> str:
    """A valid config document; a field set to a string replaces that value's raw
    YAML text, a field set to None is dropped, and an unknown name is appended."""
    merged = {**_VALID_FIELDS, **fields}
    return "".join(
        f"{key}: {value}\n" for key, value in merged.items() if value is not None
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "converter_char.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_config_loads() -> None:
    config = load_char_model_config(CONFIG_PATH)

    assert config.vocab_size == 512
    assert config.output_prefix == REPO_ROOT / "data" / "script_conv" / "converter_char"
    assert config.output_prefix.is_absolute()


def test_shipped_config_has_exactly_the_two_keys() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert set(raw) == {"vocab_size", "output_prefix"}


def test_relative_prefix_resolves_against_repo_root_and_absolute_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute = tmp_path / "abs" / "converter_char"

    relative_config = load_char_model_config(_write(tmp_path, _config_yaml()))
    absolute_config = load_char_model_config(
        _write(tmp_path, _config_yaml(output_prefix=str(absolute)))
    )

    assert relative_config.output_prefix == REPO_ROOT / "out" / "converter_char"
    assert absolute_config.output_prefix == absolute


def test_valid_config_round_trips_every_field(tmp_path: Path) -> None:
    config = load_char_model_config(_write(tmp_path, _config_yaml()))

    assert config == CharModelConfig(
        vocab_size=512, output_prefix=REPO_ROOT / "out" / "converter_char"
    )


def test_loaded_config_is_immutable(tmp_path: Path) -> None:
    config = load_char_model_config(_write(tmp_path, _config_yaml()))

    with pytest.raises(ValidationError):
        config.vocab_size = 1024  # type: ignore[misc]


@pytest.mark.parametrize(
    ("yaml_text", "message_fragment"),
    [
        pytest.param(
            _config_yaml(vocab_size=None), "vocab_size.*Field required", id="no-vocab"
        ),
        pytest.param(
            _config_yaml(output_prefix=None),
            "output_prefix.*Field required",
            id="no-prefix",
        ),
        pytest.param(
            _config_yaml(seed="1234"),
            "seed.*Extra inputs are not permitted",
            id="seed-is-not-a-key",
        ),
        pytest.param(
            _config_yaml(vocab_size="0"), "vocab_size.*greater than 0", id="zero-vocab"
        ),
        pytest.param(
            _config_yaml(vocab_size="-5"),
            "vocab_size.*greater than 0",
            id="negative-vocab",
        ),
        pytest.param(
            _config_yaml(vocab_size="512.5"),
            "vocab_size.*valid integer",
            id="fractional-vocab",
        ),
        pytest.param(
            _config_yaml(vocab_size="'512'"),
            "vocab_size.*Input should be a valid integer",
            id="quoted-vocab",
        ),
        pytest.param(
            _config_yaml(vocab_size="true"),
            "vocab_size.*Input should be a valid integer",
            id="boolean-vocab",
        ),
        pytest.param(
            _config_yaml(output_prefix="[a, b]"),
            "output_prefix.*Input is not a valid path",
            id="prefix-list",
        ),
        pytest.param(
            _config_yaml(output_prefix='""'),
            "Value error, output_prefix must not be empty",
            id="empty-prefix",
        ),
        pytest.param(
            _config_yaml(output_prefix='"   "'),
            "Value error, output_prefix must not be empty",
            id="blank-prefix",
        ),
        pytest.param(
            "- vocab_size: 512\n", "top level must be a mapping", id="top-level-list"
        ),
        pytest.param("", "top level must be a mapping", id="empty-file"),
    ],
)
def test_malformed_config_raises_config_error(
    tmp_path: Path, yaml_text: str, message_fragment: str
) -> None:
    path = _write(tmp_path, yaml_text)

    # (?s): pydantic puts the field name and its message on separate lines.
    with pytest.raises(ConfigError, match=f"(?s){message_fragment}") as excinfo:
        load_char_model_config(path)

    assert path.name in str(excinfo.value)


def test_field_violation_chains_the_validation_error(tmp_path: Path) -> None:
    path = _write(tmp_path, _config_yaml(vocab_size="0"))

    with pytest.raises(ConfigError) as excinfo:
        load_char_model_config(path)

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_yaml_syntax_error_raises_config_error_with_cause(tmp_path: Path) -> None:
    path = _write(tmp_path, "vocab_size: [512\n")

    with pytest.raises(ConfigError, match="converter_char.yaml") as excinfo:
        load_char_model_config(path)

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_non_utf8_config_raises_config_error_with_cause(tmp_path: Path) -> None:
    path = tmp_path / "converter_char.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="converter_char.yaml") as excinfo:
        load_char_model_config(path)

    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_unreadable_config_raises_config_error_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, _config_yaml())

    def deny(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(ConfigError, match="converter_char.yaml") as excinfo:
        load_char_model_config(path)

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_char_model_config(tmp_path / "nonexistent.yaml")
