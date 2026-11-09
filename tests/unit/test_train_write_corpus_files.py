"""Tests for `rinchen.train.corpus_files.write_corpus_files` and `corpus_name`: the
corpus text files of any frame of rows, which D9's document windows are written with.
`test_train_corpus_files.py` has the D7 corpora built from the training rows.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from rinchen.train.corpus_files import corpus_name, write_corpus_files
from tests.support.corpus_rows import EN_MN, MN_EN


def test_a_corpus_name_is_the_direction_and_the_group() -> None:
    assert corpus_name(EN_MN, "auth") == "eng_Latn-khk_Cyrl.auth"
    assert corpus_name(MN_EN, "doc") == "khk_Cyrl-eng_Latn.doc"


def test_a_frame_is_written_as_the_corpus_of_a_direction_and_a_group(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame(
        {
            "id": ["b", "a"],
            "src": ["Two", "One"],
            "tgt": ["Хоёр", "Нэг"],
            "src_lang": ["eng_Latn", "eng_Latn"],
            "tgt_lang": ["khk_Cyrl", "khk_Cyrl"],
        }
    )

    first = write_corpus_files(frame, tmp_path, EN_MN, "doc")
    again = write_corpus_files(frame, tmp_path, EN_MN, "doc")

    assert first.name == "eng_Latn-khk_Cyrl.doc"
    assert (first.direction, first.group, first.rows) == (EN_MN, "doc", 2)
    assert (first.src_lang, first.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    assert first.src_path == tmp_path / "eng_Latn-khk_Cyrl.doc.src"
    assert first.src_path.read_bytes() == b"One\nTwo\n"
    assert first.tgt_path.read_bytes() == "Нэг\nХоёр\n".encode()
    assert (first.reused, again.reused) == (False, True)
    assert (tmp_path / "eng_Latn-khk_Cyrl.doc.json").is_file()
