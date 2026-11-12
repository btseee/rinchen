"""Tests for `rinchen.train.doc_corpus`: the corpus files of the windows, the Eole data
section that names them, the validation windows and the config of a continuation.

The pieces of the validation lines are those the toy SentencePiece model gives the toy
sentences (counted by hand in `tests.support.doc_rows`); the word-boundary piece U+2581
is `W` below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sentencepiece
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import make_row_id
from rinchen.errors import ConfigError, DataError
from rinchen.manifest.stage import sha256_of
from rinchen.train.corpus_files import CorpusFiles
from rinchen.train.doc_corpus import (
    build_doc_corpora,
    check_continuation_base,
    doc_data_section,
    doc_weights,
    planned_doc_corpora,
    render_continuation_config,
    write_doc_validation,
)
from rinchen.train.docify import DocWindows, docify
from rinchen.train.eole_cfg import read_config, validate_train_config
from rinchen.train.sampling import CorpusSize
from rinchen.train.validation_corpus import ValidPaths
from tests.support.corpus_rows import file_lines
from tests.support.doc_rows import EN_MN, JOIN, MN_EN, SENTENCES, document, rows_frame
from tests.support.tiny_parallel import train_toy_spm

W = chr(0x2581)
BASE = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
EN_MN_NAME = "eng_Latn-khk_Cyrl.doc"
MN_EN_NAME = "khk_Cyrl-eng_Latn.doc"


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("doc_corpus_spm"))


def windows_of(rows: list[dict[str, Any]], spm: Path, **options: Any) -> DocWindows:
    keywords: dict[str, Any] = {
        "min_size": 2,
        "max_size": 2,
        "max_pieces": 1000,
        "seed": 1,
    } | options
    return docify(rows_frame(rows), spm_model=spm, **keywords)


@pytest.fixture
def both_directions(toy_spm: Path) -> DocWindows:
    rows = document("d1", range(4)) + document("d1", range(4), languages=MN_EN)
    return windows_of(rows, toy_spm)


# ---- the corpus files --------------------------------------------------------------------


def test_one_corpus_per_direction_named_after_it_with_a_doc_group(
    both_directions: DocWindows, tmp_path: Path
) -> None:
    corpora = build_doc_corpora(both_directions, tmp_path / "data")

    assert [corpus.name for corpus in corpora] == [EN_MN_NAME, MN_EN_NAME]
    assert [corpus.direction for corpus in corpora] == [
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
    ]
    assert {corpus.group for corpus in corpora} == {"doc"}
    assert [(c.src_lang, c.tgt_lang, c.rows) for c in corpora] == [
        ("eng_Latn", "khk_Cyrl", 2),
        ("khk_Cyrl", "eng_Latn", 2),
    ]
    assert corpora[0].src_path == tmp_path / "data" / "doc" / f"{EN_MN_NAME}.src"
    assert corpora[0].tgt_path == tmp_path / "data" / "doc" / f"{EN_MN_NAME}.tgt"


def test_the_files_hold_one_window_per_line_in_the_order_of_the_ids_with_no_tags(
    both_directions: DocWindows, tmp_path: Path
) -> None:
    corpus = build_doc_corpora(both_directions, tmp_path)[0]

    pairs = [
        (
            JOIN.join(e for e, _ in SENTENCES[a : a + 2]),
            JOIN.join(m for _, m in SENTENCES[a : a + 2]),
        )
        for a in (0, 2)
    ]
    pairs.sort(key=lambda pair: make_row_id("eng_Latn", "khk_Cyrl", *pair))
    assert file_lines(corpus.src_path) == [source for source, _ in pairs]
    assert file_lines(corpus.tgt_path) == [target for _, target in pairs]
    text = corpus.src_path.read_text(encoding="utf-8")
    assert "<doc>" not in text and "<2" not in text
    assert "\r" not in text


def test_the_sidecar_records_the_rows_and_the_digests(
    both_directions: DocWindows, tmp_path: Path
) -> None:
    corpus = build_doc_corpora(both_directions, tmp_path)[0]

    sidecar = corpus.src_path.with_name(f"{EN_MN_NAME}.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "rows": 2,
        "src_sha256": sha256_of(corpus.src_path),
        "tgt_sha256": sha256_of(corpus.tgt_path),
    }
    assert (corpus.src_sha256, corpus.tgt_sha256) == (
        sha256_of(corpus.src_path),
        sha256_of(corpus.tgt_path),
    )


def test_an_unchanged_corpus_is_reused_and_a_changed_or_damaged_one_is_written_again(
    both_directions: DocWindows, toy_spm: Path, tmp_path: Path
) -> None:
    first = build_doc_corpora(both_directions, tmp_path)
    before = first[0].src_path.stat().st_mtime_ns

    second = build_doc_corpora(both_directions, tmp_path)

    assert [c.reused for c in first] == [False, False]
    assert [c.reused for c in second] == [True, True]
    assert second[0].src_path.stat().st_mtime_ns == before

    second[0].src_path.write_text("tampered\n", encoding="utf-8")
    third = build_doc_corpora(both_directions, tmp_path)
    assert [c.reused for c in third] == [False, True]
    assert sha256_of(third[0].src_path) == first[0].src_sha256

    other = windows_of(
        document("d9", range(6)) + document("d9", range(6), languages=MN_EN), toy_spm
    )
    fourth = build_doc_corpora(other, tmp_path)
    assert [c.reused for c in fourth] == [False, False]
    assert fourth[0].rows == 3


def test_the_planned_corpora_are_the_ones_the_build_writes_without_writing_them(
    both_directions: DocWindows, tmp_path: Path
) -> None:
    planned = planned_doc_corpora(both_directions)

    assert [(size.name, size.direction, size.rows) for size in planned] == [
        (EN_MN_NAME, "eng_Latn-khk_Cyrl", 2),
        (MN_EN_NAME, "khk_Cyrl-eng_Latn", 2),
    ]
    assert list(tmp_path.iterdir()) == []
    built = build_doc_corpora(both_directions, tmp_path)
    assert [(c.name, c.direction, c.rows) for c in built] == [
        (size.name, size.direction, size.rows) for size in planned
    ]
    assert doc_weights(planned, temperature=0.5) == doc_weights(built, temperature=0.5)


def test_planned_corpora_follow_the_window_counts_of_each_direction(
    toy_spm: Path,
) -> None:
    rows = document("d1", range(6)) + document("d1", range(4), languages=MN_EN)
    windows = windows_of(rows, toy_spm)

    planned = planned_doc_corpora(windows)

    assert [(size.name, size.rows) for size in planned] == [
        (EN_MN_NAME, 3),
        (MN_EN_NAME, 2),
    ]


def test_no_window_means_no_planned_corpus(toy_spm: Path) -> None:
    with pytest.raises(DataError, match="no document windows"):
        planned_doc_corpora(windows_of(document("d1", [0]), toy_spm))


def test_no_window_is_a_data_error(toy_spm: Path, tmp_path: Path) -> None:
    empty = windows_of(document("d1", [0]), toy_spm)

    with pytest.raises(DataError, match="no document windows"):
        build_doc_corpora(empty, tmp_path)

    assert not (tmp_path / "doc").exists()


# ---- the weights and the data section ----------------------------------------------------


def _corpus(name: str, direction: str, rows: int, tmp_path: Path) -> CorpusFiles:
    src_lang, tgt_lang = direction.split("-")
    return CorpusFiles(
        name=name,
        direction=direction,
        group="doc",
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_path=tmp_path / f"{name}.src",
        tgt_path=tmp_path / f"{name}.tgt",
        rows=rows,
        src_sha256="0" * 64,
        tgt_sha256="1" * 64,
        reused=False,
    )


def test_the_weights_follow_the_window_counts_through_the_temperature(
    tmp_path: Path,
) -> None:
    corpora = [
        _corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 300, tmp_path),
        _corpus(MN_EN_NAME, "khk_Cyrl-eng_Latn", 100, tmp_path),
    ]

    assert doc_weights(corpora, temperature=1.0) == {EN_MN_NAME: 75, MN_EN_NAME: 25}
    # 300 ** 0.5 : 100 ** 0.5 is 63.4 : 36.6, rounded to whole numbers of 100.
    assert doc_weights(corpora, temperature=0.5) == {EN_MN_NAME: 63, MN_EN_NAME: 37}
    assert doc_weights(corpora, temperature=1.0, total=8) == {
        EN_MN_NAME: 6,
        MN_EN_NAME: 2,
    }
    sizes = [CorpusSize(c.name, c.direction, c.rows) for c in corpora]
    assert doc_weights(sizes, temperature=0.5) == {EN_MN_NAME: 63, MN_EN_NAME: 37}


def test_weights_that_cannot_be_made_are_refused(tmp_path: Path) -> None:
    corpora = [_corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 5, tmp_path)]

    with pytest.raises(ValueError, match="temperature"):
        doc_weights(corpora, temperature=0.0)
    with pytest.raises(ValueError, match="at least one corpus"):
        doc_weights([], temperature=1.0)
    two = [*corpora, _corpus(MN_EN_NAME, "khk_Cyrl-eng_Latn", 5, tmp_path)]
    with pytest.raises(ConfigError, match="weight_total"):
        doc_weights(two, temperature=1.0, total=1)


def test_the_data_section_prefixes_the_tag_and_doc_and_runs_the_three_transforms(
    tmp_path: Path,
) -> None:
    corpora = [
        _corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 300, tmp_path),
        _corpus(MN_EN_NAME, "khk_Cyrl-eng_Latn", 100, tmp_path),
    ]
    valid = ValidPaths(tmp_path / "valid.src", tmp_path / "valid.tgt")

    section = doc_data_section(
        corpora, {EN_MN_NAME: 3, MN_EN_NAME: 1}, valid_paths=valid
    )

    transforms = ["sentencepiece", "prefix", "filtertoolong"]
    assert section == {
        EN_MN_NAME: {
            "path_src": (tmp_path / f"{EN_MN_NAME}.src").as_posix(),
            "path_tgt": (tmp_path / f"{EN_MN_NAME}.tgt").as_posix(),
            "transforms": transforms,
            "weight": 3,
            "src_prefix": "<2khk_Cyrl> <doc>",
            "tgt_prefix": "",
        },
        MN_EN_NAME: {
            "path_src": (tmp_path / f"{MN_EN_NAME}.src").as_posix(),
            "path_tgt": (tmp_path / f"{MN_EN_NAME}.tgt").as_posix(),
            "transforms": transforms,
            "weight": 1,
            "src_prefix": "<2eng_Latn> <doc>",
            "tgt_prefix": "",
        },
        "valid": {
            "path_src": (tmp_path / "valid.src").as_posix(),
            "path_tgt": (tmp_path / "valid.tgt").as_posix(),
        },
    }


def test_the_data_section_can_leave_filtertoolong_out(tmp_path: Path) -> None:
    corpora = [_corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 3, tmp_path)]
    valid = ValidPaths(tmp_path / "valid.src", tmp_path / "valid.tgt")

    section = doc_data_section(
        corpora, {EN_MN_NAME: 1}, valid_paths=valid, filtertoolong=False
    )

    assert section[EN_MN_NAME]["transforms"] == ["sentencepiece", "prefix"]


def test_a_corpus_without_a_weight_is_refused(tmp_path: Path) -> None:
    corpora = [_corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 3, tmp_path)]
    valid = ValidPaths(tmp_path / "valid.src", tmp_path / "valid.tgt")

    with pytest.raises(ValueError, match="no weight"):
        doc_data_section(corpora, {}, valid_paths=valid)


# ---- the validation windows ---------------------------------------------------------------


def _dev(toy_spm: Path, *, documents: int = 1) -> DocWindows:
    rows = [
        row
        for number in range(documents)
        for row in document(f"v{number}", range(4), split="dev")
    ]
    return docify(
        rows_frame(rows),
        spm_model=toy_spm,
        min_size=2,
        max_size=2,
        max_pieces=1000,
        seed=0,
        split="dev",
    )


def test_the_validation_lines_are_the_tag_doc_and_pieces_of_the_joined_windows(
    toy_spm: Path, tmp_path: Path
) -> None:
    dev = windows_of(document("v", [0, 1], split="dev"), toy_spm, split="dev")

    paths = write_doc_validation(dev, toy_spm, tmp_path / "out")

    assert (paths.src, paths.tgt) == (
        tmp_path / "out" / "valid.src",
        tmp_path / "out" / "valid.tgt",
    )
    english = f"{W} T h i s {W}is {W}the {W}book . {W} <sep> {W}The {W}water {W}is {W} h e r e ."
    mongolian = f"{W}Энэ {W}бол {W}ном . {W} <sep> {W}Ус {W}энд {W}байна ."
    assert file_lines(paths.src) == [f"<2khk_Cyrl> <doc> {english}"]
    assert file_lines(paths.tgt) == [mongolian]


def test_every_window_is_a_line_in_the_order_of_the_frame_with_its_own_tag(
    toy_spm: Path, tmp_path: Path
) -> None:
    rows = document("v", range(2), split="dev") + document(
        "v", range(2), split="dev", languages=MN_EN
    )
    dev = windows_of(rows, toy_spm, split="dev")

    paths = write_doc_validation(dev, toy_spm, tmp_path)

    sources = file_lines(paths.src)
    assert [line.split(" ")[0] for line in sources] == ["<2khk_Cyrl>", "<2eng_Latn>"]
    assert {line.split(" ")[1] for line in sources} == {"<doc>"}
    assert len(file_lines(paths.tgt)) == 2


def test_max_windows_keeps_the_windows_with_the_smallest_ids_of_each_direction(
    toy_spm: Path, tmp_path: Path
) -> None:
    # Three documents of two sentences each way, so three windows a direction.
    rows = [
        row
        for number in range(3)
        for languages in (EN_MN, MN_EN)
        for row in document(
            f"v{number}",
            [2 * number, 2 * number + 1],
            split="dev",
            languages=languages,
        )
    ]
    dev = windows_of(rows, toy_spm, split="dev")
    by_id = {
        (row["src_lang"], row["tgt_lang"], row["id"]): row["tgt"]
        for row in dev.frame.iter_rows(named=True)
    }
    expected = []
    for languages in (EN_MN, MN_EN):
        ids = sorted(id_ for src, tgt, id_ in by_id if (src, tgt) == languages)
        expected += [by_id[(*languages, id_)] for id_ in ids[:2]]
    processor = sentencepiece.SentencePieceProcessor(model_file=str(toy_spm))

    paths = write_doc_validation(dev, toy_spm, tmp_path, max_windows=2)

    kept = sorted(file_lines(paths.tgt))
    assert kept == sorted(
        " ".join(processor.encode(text, out_type=str)) for text in expected
    )
    assert len(file_lines(paths.src)) == 4


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_max_windows_must_be_a_positive_integer(
    toy_spm: Path, tmp_path: Path, bad: object
) -> None:
    dev = _dev(toy_spm)

    with pytest.raises(ValueError, match="max_windows"):
        write_doc_validation(dev, toy_spm, tmp_path, max_windows=bad)  # type: ignore[arg-type]

    assert not (tmp_path / "valid.src").exists()


def test_no_dev_window_is_a_data_error_and_writes_nothing(
    toy_spm: Path, tmp_path: Path
) -> None:
    empty = windows_of(document("v", [0], split="dev"), toy_spm, split="dev")

    with pytest.raises(DataError, match="no dev windows"):
        write_doc_validation(empty, toy_spm, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_a_model_without_doc_as_one_piece_is_a_config_error(
    toy_spm: Path, tmp_path: Path
) -> None:
    import sentencepiece

    corpus = tmp_path / "plain.txt"
    corpus.write_text(
        "".join(f"line {i} of some text\n" for i in range(200)), encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "plain"),
        vocab_size=40,
        minloglevel=2,
    )

    with pytest.raises(ConfigError, match="<doc>"):
        write_doc_validation(_dev(toy_spm), tmp_path / "plain.model", tmp_path / "out")


# ---- the config of the continuation ------------------------------------------------------


def _render(tmp_path: Path, spm: Path, **replace: Any) -> tuple[Path, dict[str, Any]]:
    corpora = [_corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 3, tmp_path / "data")]
    data = doc_data_section(
        corpora,
        {EN_MN_NAME: 1},
        valid_paths=ValidPaths(
            tmp_path / "v" / "valid.src", tmp_path / "v" / "valid.tgt"
        ),
    )
    keywords: dict[str, Any] = {
        "data": data,
        "spm_model": spm,
        "src_vocab": tmp_path / "run" / "vocab.src",
        "model_path": tmp_path / "run" / "ckpt",
        "out": tmp_path / "run" / "config.yaml",
    } | replace
    out = render_continuation_config(BASE, **keywords)
    return out, data


def test_the_continuation_config_is_the_base_with_the_per_run_values(
    toy_spm: Path, tmp_path: Path
) -> None:
    out, data = _render(tmp_path, toy_spm)

    cfg = read_config(out, "config")
    base = read_config(BASE, "base")
    assert out == tmp_path / "run" / "config.yaml"
    assert cfg["data"] == data
    assert cfg["src_vocab"] == (tmp_path / "run" / "vocab.src").as_posix()
    assert cfg["training"]["model_path"] == (tmp_path / "run" / "ckpt").as_posix()
    assert cfg["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": toy_spm.as_posix(),
        "tgt_subword_model": toy_spm.as_posix(),
    }
    assert (
        cfg["transforms_configs"]["filtertoolong"]
        == base["transforms_configs"]["filtertoolong"]
    )
    # The model and every option of the run stay the base's.
    assert cfg["model"] == base["model"]
    for key, value in base["training"].items():
        if key != "model_path":
            assert cfg["training"][key] == value
    validate_train_config(cfg)


def test_the_continuation_config_holds_neither_train_from_nor_reset_optim(
    toy_spm: Path, tmp_path: Path
) -> None:
    out, _ = _render(tmp_path, toy_spm)

    text = out.read_text(encoding="utf-8")
    assert "train_from" not in text
    assert "reset_optim" not in text


@pytest.mark.parametrize("key", ["train_from", "reset_optim"])
def test_a_base_that_holds_train_from_or_reset_optim_is_refused(
    toy_spm: Path, tmp_path: Path, key: str
) -> None:
    base = read_config(BASE, "base")
    base["training"][key] = "all" if key == "reset_optim" else "/elsewhere/step_1"
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    corpora = [_corpus(EN_MN_NAME, "eng_Latn-khk_Cyrl", 3, tmp_path)]
    data = doc_data_section(
        corpora, {EN_MN_NAME: 1}, valid_paths=ValidPaths(tmp_path / "a", tmp_path / "b")
    )

    with pytest.raises(ConfigError, match=key):
        render_continuation_config(
            path,
            data=data,
            spm_model=toy_spm,
            src_vocab=tmp_path / "vocab.src",
            model_path=tmp_path / "ckpt",
            out=tmp_path / "out.yaml",
        )

    assert not (tmp_path / "out.yaml").exists()


def test_a_base_with_corpora_of_its_own_is_refused(
    toy_spm: Path, tmp_path: Path
) -> None:
    base = read_config(BASE, "base")
    base["data"] = {
        "other.auth": {
            "path_src": "a.src",
            "path_tgt": "a.tgt",
            "transforms": [],
            "weight": 1,
            "src_prefix": "",
            "tgt_prefix": "",
        }
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"other\.auth"):
        render_continuation_config(
            path,
            data={},
            spm_model=toy_spm,
            src_vocab=tmp_path / "vocab.src",
            model_path=tmp_path / "ckpt",
            out=tmp_path / "out.yaml",
        )


@pytest.mark.parametrize("which", ["src_vocab", "model_path", "spm_model"])
def test_a_dollar_sign_in_a_path_is_refused(
    toy_spm: Path, tmp_path: Path, which: str
) -> None:
    with pytest.raises(ConfigError, match=r"\$"):
        _render(tmp_path, toy_spm, **{which: tmp_path / "a$b"})

    assert not (tmp_path / "run" / "config.yaml").exists()


def test_a_config_that_does_not_validate_writes_nothing(
    toy_spm: Path, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="data"):
        _render(tmp_path, toy_spm, data={"x": {"path_src": 3}})

    assert not (tmp_path / "run" / "config.yaml").exists()


def test_the_config_is_not_written_over_the_base(toy_spm: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base"):
        _render(tmp_path, toy_spm, out=BASE)


# ---- the base config of a continuation, checked on its own ---------------------------------


def test_the_base_config_is_returned_valid_and_as_it_is() -> None:
    cfg = check_continuation_base(BASE)

    assert cfg == read_config(BASE, "base")
    assert cfg["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": 64,
        "tgt_seq_length": 64,
    }


def test_the_base_check_refuses_what_the_render_refuses(tmp_path: Path) -> None:
    held = read_config(BASE, "base")
    held["training"]["reset_optim"] = "all"
    path = tmp_path / "held.yaml"
    path.write_text(yaml.safe_dump(held), encoding="utf-8")
    with pytest.raises(ConfigError, match="reset_optim"):
        check_continuation_base(path)

    corpora = read_config(BASE, "base")
    corpora["data"] = {
        "other.auth": {
            "path_src": "a.src",
            "path_tgt": "a.tgt",
            "transforms": [],
            "weight": 1,
            "src_prefix": "",
            "tgt_prefix": "",
        }
    }
    path = tmp_path / "corpora.yaml"
    path.write_text(yaml.safe_dump(corpora), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"other\.auth"):
        check_continuation_base(path)

    with pytest.raises(ConfigError, match="not found"):
        check_continuation_base(tmp_path / "absent.yaml")
