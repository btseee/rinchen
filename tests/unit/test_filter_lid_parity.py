"""B7 parity: the `GlotLID` wrapper agrees with the raw fastText Python API on
the real GlotLID model for 200 fixture lines.

The golden file `glotlid_200.json` was captured once from `fasttext`'s
`predict(text, k=1)` on `model_v3.bin` of `cis-lmu/glotlid` (the fastText
command-line binary is not installed; spec ruling 9). Its label for a line of
traditional Mongolian script is `mn_Mong`, which the model does not know: the
wrapper answers those lines before the model runs. The model is downloaded to
the Hugging Face cache, outside the repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.languages import known_language_codes
from rinchen.config.loader import REPO_ROOT
from rinchen.filter.lid import GlotLID

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"
LINES_PATH = GOLDEN_DIR / "glotlid_200_lines.tsv"
GOLDEN_PATH = GOLDEN_DIR / "glotlid_200.json"
LINES_PER_LANGUAGE = 20
TRADITIONAL_FIRST = 0x1800
TRADITIONAL_LAST = 0x18AF
# The golden probabilities are stored rounded to six decimals.
PROBABILITY_TOLERANCE = 1e-5


def _fixture_lines() -> list[tuple[str, str]]:
    lines = []
    for row in LINES_PATH.read_text(encoding="utf-8").splitlines():
        label, text = row.split("\t", 1)
        lines.append((label, text))
    return lines


def _golden() -> dict[str, Any]:
    golden: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return golden


def _is_traditional(text: str) -> bool:
    letters = [char for char in text if not char.isspace()]
    in_block = sum(
        TRADITIONAL_FIRST <= ord(char) <= TRADITIONAL_LAST for char in letters
    )
    return bool(letters) and in_block * 2 >= len(letters)


# --- the fixture files --------------------------------------------------------------


def test_the_fixture_has_twenty_lines_for_every_registry_language() -> None:
    lines = _fixture_lines()

    assert len(lines) == len(known_language_codes()) * LINES_PER_LANGUAGE
    counts = {code: 0 for code in known_language_codes()}
    for label, text in lines:
        counts[label] += 1
        assert text.strip() == text
        assert text
    assert set(counts.values()) == {LINES_PER_LANGUAGE}
    assert len({text for _, text in lines}) == len(lines)


def test_only_the_mn_mong_lines_are_traditional_script() -> None:
    for label, text in _fixture_lines():
        assert _is_traditional(text) == (label == "mn_Mong"), (label, text)


def test_the_golden_file_has_one_entry_for_every_fixture_line() -> None:
    golden = _golden()

    assert golden["model"] == "cis-lmu/glotlid model_v3.bin"
    assert golden["reference"] == "fasttext python API predict, k=1"
    assert len(golden["lines"]) == len(_fixture_lines())
    for entry in golden["lines"]:
        assert set(entry) == {"label", "prob"}
        assert entry["label"]
        assert 0.0 <= entry["prob"] <= 1.0


def test_the_golden_label_of_a_traditional_script_line_is_mn_mong() -> None:
    for (_, text), entry in zip(_fixture_lines(), _golden()["lines"], strict=True):
        if _is_traditional(text):
            assert entry == {"label": "mn_Mong", "prob": 1.0}
        else:
            assert entry["label"] != "mn_Mong"


# --- the real model -----------------------------------------------------------------


@pytest.mark.network
def test_the_wrapper_top_1_label_equals_the_golden_label_for_every_line() -> None:
    from huggingface_hub import hf_hub_download

    model_file = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model_v3.bin")
    texts = [text for _, text in _fixture_lines()]
    golden = _golden()["lines"]

    predictions = GlotLID(Path(model_file)).predict(texts, k=1)

    assert len(predictions) == len(golden)
    mismatches = [
        (text, entry["label"], top[0][0] if top else None)
        for text, entry, top in zip(texts, golden, predictions, strict=True)
        if not top or top[0][0] != entry["label"]
    ]
    assert mismatches == []
    for entry, top in zip(golden, predictions, strict=True):
        assert top[0][1] == pytest.approx(entry["prob"], abs=PROBABILITY_TOLERANCE)
