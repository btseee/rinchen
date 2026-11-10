"""No message of an error of `rinchen.infer` or `rinchen.serve` interpolates a text of a request
(D11, fix wave).

A text of a request must never come back in an error body, a log line or an exception message
(it may be a document, or a secret). The way to keep that true as the code changes is to look at
every place where a `DataError`, `ConfigError`, `UpstreamError` or `RequestTextError` is built
from something that is not a literal: this module finds each by reading the source (an AST scan)
and compares the list with the one below, in which every entry was reviewed and says why it
holds no text of a request. A new interpolation fails the test until someone has looked at it
and added it here with its reason; an entry that no longer matches code fails too, so the list
cannot rot.

What a reviewed entry may hold: an index or a count, the name of a type, an entry of the model's
vocabulary or of the language registry, a path or an option of the operator, the number of an
option the service validated as an integer, a language code cut by `shown`. Never a text, a
sentence or a document of a request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import rinchen
from tests.support.error_messages import Finding, interpolations

ROOT = Path(rinchen.__file__).parent
PACKAGES: Final = ("infer", "serve")

# (file, function, expression, why it holds no text of a request)
REVIEWED: Final[tuple[tuple[str, str, str, str], ...]] = (
    # ---- the chunker ----
    (
        "infer/doc_chunker.py",
        "_refuse_separators",
        "position",
        "the index of a sentence, an integer; the sentence itself is not quoted",
    ),
    (
        "infer/doc_chunker.py",
        "_refuse_separators",
        "SEGMENT_SEPARATOR",
        "a constant",
    ),
    # ---- the engine, on a request ----
    (
        "infer/engine.py",
        "Engine._route",
        "shown(src)",
        "a language code, cut short and escaped by `shown`; the schema allows 32 characters",
    ),
    (
        "infer/engine.py",
        "Engine._route",
        "selected",
        "the target of the request, which equals a source already found in the language registry",
    ),
    (
        "infer/engine.py",
        "Engine._route",
        "refusal(selected, self._record)",
        "`refusal` names a target (cut by `shown`) and the targets of the record",
    ),
    (
        "infer/engine.py",
        "Engine._translate_batch",
        "len(sources)",
        "a count",
    ),
    (
        "infer/engine.py",
        "Engine._translate_batch",
        "type(error).__name__",
        "the name of a type; the library's own message stays in the chained cause",
    ),
    (
        "infer/engine.py",
        "Engine._translate_batch",
        "len(results)",
        "a count",
    ),
    (
        "infer/engine.py",
        "Engine._parse",
        "job.label",
        "`row N` or `document N`: the place of a text, an index made by the engine",
    ),
    (
        "infer/engine.py",
        "Engine._parse",
        "job.route.tag",
        "a direction tag of the record, an entry of the vocabulary",
    ),
    (
        "infer/engine.py",
        "Engine._convert",
        "type(error).__name__",
        "the name of a type; the converter's own message stays in the chained cause",
    ),
    # ---- the engine, at load: the operator's files and options, no request exists yet ----
    (
        "infer/engine.py",
        "_load_processor",
        "export.directory / SPM_FILE",
        "a path of the operator",
    ),
    (
        "infer/engine.py",
        "_require_usable_tags",
        "tag",
        "a direction tag of the record",
    ),
    (
        "infer/engine.py",
        "_require_usable_tags",
        "code",
        "a language code of the record",
    ),
    (
        "infer/engine.py",
        "_require_usable_tags",
        "export.directory / _VOCABULARY_FILE",
        "a path of the operator",
    ),
    (
        "infer/engine.py",
        "_require_usable_tags",
        "export.directory / SPM_FILE",
        "a path of the operator",
    ),
    (
        "infer/engine.py",
        "_load_ct2",
        "load.device",
        "an option of the operator",
    ),
    (
        "infer/engine.py",
        "_load_ct2",
        "load.compute_type",
        "an option of the operator",
    ),
    (
        "infer/engine.py",
        "_load_ct2",
        "error",
        "the library's message when it cannot load the model: no request exists yet",
    ),
    (
        "infer/engine.py",
        "_load_ct2",
        "export.directory",
        "a path of the operator",
    ),
    # ---- the vocabulary of what a request may name ----
    (
        "infer/engine_types.py",
        "reserved_symbol_error",
        "label",
        "`row N` or `document N`: the place of a text, an index made by the engine",
    ),
    (
        "infer/engine_types.py",
        "reserved_symbol_error",
        "symbol",
        "a reserved symbol, an entry of the model's vocabulary, never the text",
    ),
    (
        "infer/engine_types.py",
        "require_encodable",
        "label",
        "`row N` or `document N`: the place of a text, an index made by the engine",
    ),
    (
        "infer/engine_types.py",
        "require_encodable",
        "error.start",
        "the position of the lone surrogate, an integer",
    ),
    (
        "infer/engine_types.py",
        "selected_target",
        "shown(script)",
        "the script of the request, cut short and escaped by `shown`",
    ),
    (
        "infer/engine_types.py",
        "selected_target",
        "shown(tgt)",
        "the target of the request, cut short and escaped by `shown`",
    ),
    (
        "infer/engine_types.py",
        "require_document_model",
        "record.tag_placement",
        "`src` or `tgt`, from the record",
    ),
    (
        "infer/engine_types.py",
        "LoadOptions.__post_init__",
        "self.device",
        "an option of the operator, checked against a fixed list",
    ),
    (
        "infer/engine_types.py",
        "LoadOptions.__post_init__",
        "self.compute_type",
        "an option of the operator, checked against a fixed list",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "self.n",
        "an integer: the service validates `n_best` as one before it builds the options",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "self.beam_size",
        "an integer: the service validates `beam_size` as one",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "self.max_source_pieces",
        "an integer of the config",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "MAX_SOURCE_PIECES",
        "a constant",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "', '.join(SCRIPT_MODES)",
        "the three modes, constants",
    ),
    (
        "infer/engine_types.py",
        "DecodeOptions.__post_init__",
        "self.script_mask",
        "an option of the config, checked against the three modes",
    ),
    (
        "infer/engine_types.py",
        "_require_count",
        "name",
        "the name of an option",
    ),
    (
        "infer/engine_types.py",
        "_require_count",
        "minimum",
        "a constant",
    ),
    (
        "infer/engine_types.py",
        "_require_count",
        "value",
        "an option of the config or an integer the service validated: never a text",
    ),
    (
        "infer/script_mask.py",
        "allowed_scripts",
        "tgt",
        "a target the engine's route already found in the registry; a caller's code otherwise",
    ),
    # ---- the service ----
    (
        "serve/_translate.py",
        "decode_options",
        "body.n_best",
        "an integer: the request model is strict",
    ),
    (
        "serve/_translate.py",
        "decode_options",
        "beam",
        "an integer: the request model and the config are strict",
    ),
    (
        "serve/batcher.py",
        "MicroBatcher.__init__",
        "name",
        "the name of a setting",
    ),
    (
        "serve/batcher.py",
        "MicroBatcher.__init__",
        "value",
        "a setting of the config",
    ),
    (
        "serve/batcher.py",
        "MicroBatcher.__init__",
        "max_wait_seconds",
        "a setting of the config",
    ),
    (
        "serve/batcher.py",
        "MicroBatcher._on_flight_done",
        "len(entries)",
        "a count",
    ),
    (
        "serve/batcher.py",
        "MicroBatcher._on_flight_done",
        "len(results)",
        "a count",
    ),
    (
        "serve/schemas.py",
        "bounded_request_model",
        "name",
        "the name of a setting",
    ),
    (
        "serve/schemas.py",
        "bounded_request_model",
        "highest",
        "a constant",
    ),
    (
        "serve/schemas.py",
        "bounded_request_model",
        "value",
        "a setting of the config",
    ),
)


def scanned() -> list[Finding]:
    found: list[Finding] = []
    for package in PACKAGES:
        for path in sorted((ROOT / package).glob("*.py")):
            relative = f"{package}/{path.name}"
            found += interpolations(path.read_text(encoding="utf-8"), relative)
    return found


def test_every_interpolation_in_an_error_of_infer_or_serve_has_been_reviewed() -> None:
    found = {(f.path, f.function, f.expression) for f in scanned()}
    reviewed = {
        (path, function, expression) for path, function, expression, _ in REVIEWED
    }

    assert sorted(found - reviewed) == [], (
        "unreviewed: add it to REVIEWED with its reason"
    )


def test_every_reviewed_entry_still_matches_a_line_of_code() -> None:
    found = {(f.path, f.function, f.expression) for f in scanned()}
    reviewed = {
        (path, function, expression) for path, function, expression, _ in REVIEWED
    }

    assert sorted(reviewed - found) == [], "stale: remove it from REVIEWED"


def test_every_reviewed_entry_says_why_and_is_listed_once() -> None:
    keys = [(path, function, expression) for path, function, expression, _ in REVIEWED]

    assert len(keys) == len(set(keys))
    assert all(len(reason.split()) >= 2 for *_, reason in REVIEWED)


def test_no_reviewed_entry_quotes_a_sentence_or_a_text() -> None:
    # The kind of expression that put a sentence into a message once (`sentence[:60]`).
    suspicious = ("sentence", "text", "document", "row.text", "word", "line")
    quoting = [
        (path, function, expression)
        for path, function, expression, _ in REVIEWED
        if any(f"{name}[" in expression or expression == name for name in suspicious)
    ]

    assert quoting == []


# ---- the scan itself ----------------------------------------------------------------------


def found_in(source: str) -> list[tuple[str, str]]:
    return [(f.function, f.expression) for f in interpolations(source, "x.py")]


def test_the_scan_finds_each_way_of_putting_something_into_a_message() -> None:
    source = (
        "def one(x):\n"
        "    raise DataError(f'a {x} b')\n"
        "def two(x):\n"
        "    raise ConfigError('a {} b'.format(x.name))\n"
        "def three(x):\n"
        "    raise UpstreamError('a %s b' % x)\n"
        "def four(x):\n"
        "    raise RequestTextError('a ' + x)\n"
        "def five(x):\n"
        "    raise DataError(str(x))\n"
        "class K:\n"
        "    def six(self, y):\n"
        "        raise errors.DataError(f'{y!r:>3} and {self.z[:60]}')\n"
    )

    assert found_in(source) == [
        ("one", "x"),
        ("two", "x.name"),
        ("three", "x"),
        ("four", "x"),
        ("five", "str(x)"),
        ("K.six", "y"),
        ("K.six", "self.z[:60]"),
    ]


def test_the_scan_ignores_literals_and_other_calls() -> None:
    source = (
        "def one():\n"
        "    raise DataError('a literal ' 'in two parts')\n"
        "def two(x):\n"
        "    raise ValueError(f'not ours {x}')\n"
        "def three(x):\n"
        "    log(f'{x}')\n"
        "def four():\n"
        "    raise ConfigError('a ' + 'b')\n"
    )

    assert found_in(source) == []


def test_the_scan_reads_the_packages_it_is_told_to() -> None:
    files = {f.path for f in scanned()}

    assert {
        "infer/engine.py",
        "infer/engine_types.py",
        "infer/doc_chunker.py",
        "serve/batcher.py",
        "serve/schemas.py",
    } <= files
