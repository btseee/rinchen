from probe_env import classify_wheel_tags


def test_classify_cp314_wheel():
    tags = classify_wheel_tags(["torch-2.14.1-cp314-cp314-win_amd64.whl"])
    assert tags == {"cp314"}


def test_classify_cp314t_wheel():
    tags = classify_wheel_tags(
        ["ctranslate2-4.8.4-cp314-cp314t-manylinux_2_28_x86_64.whl"]
    )
    assert tags == {"cp314t"}


def test_classify_abi3_wheel():
    tags = classify_wheel_tags(
        ["fasttext_new-0.1.0-cp39-abi3-manylinux_2_28_x86_64.whl"]
    )
    assert tags == {"abi3"}


def test_classify_pure_python_wheel():
    tags = classify_wheel_tags(["eole-0.6.4-py3-none-any.whl"])
    assert tags == {"py3-none"}


def test_classify_sdist_only():
    tags = classify_wheel_tags(["eole-0.6.4.tar.gz"])
    assert tags == {"sdist"}


def test_classify_no_files():
    tags = classify_wheel_tags([])
    assert tags == {"none"}


def test_classify_mixed_files_combine_tags():
    tags = classify_wheel_tags(
        [
            "pkg-1.0-cp314-cp314-win_amd64.whl",
            "pkg-1.0-cp314-cp314t-manylinux_2_28_x86_64.whl",
            "pkg-1.0.tar.gz",
        ]
    )
    assert tags == {"cp314", "cp314t", "sdist"}
