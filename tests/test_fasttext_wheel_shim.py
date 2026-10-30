"""Verifies the fasttext-wheel shim (vendor/fasttext-wheel-shim/) actually
performs the one real-world task eole's `clean` transform depends on:
loading the lid.176 language-identification model and predicting with it.
Requires network access to download the real model file, exactly as
eole/transforms/clean.py does at runtime.
"""

import urllib.request

import fasttext
import pytest


@pytest.mark.network
def test_shim_loads_lid176_and_identifies_languages(tmp_path):
    model_path = tmp_path / "lid.176.ftz"
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        model_path,
    )

    model = fasttext.load_model(str(model_path))

    mongolian_labels, _ = model.predict(
        "Сайн байна уу, энэ бол монгол хэл дээрх өгүүлбэр юм"
    )
    assert mongolian_labels == ("__label__mn",)

    english_labels, _ = model.predict(
        "This is an English sentence for language identification."
    )
    assert english_labels == ("__label__en",)
