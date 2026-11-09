"""Tests for `rinchen.train.eole_cfg`, the pieces of an Eole config that every Rinchen
model shares (the forbidden keys, the `$` guard, reading the YAML), and for the places
that moved there or to `eole_run`: `script_conv` re-exports them, and nothing under
`rinchen.train` or `rinchen.utils` may import `script_conv`.

`dump_config` keeps its tests in `test_script_conv_config.py`, which reach it through
the re-export; what it does with a value YAML cannot dump is tested here.
"""

from __future__ import annotations

import ast
import enum
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.script_conv import config as converter_config
from rinchen.script_conv import train as converter_train
from rinchen.train import eole_cfg, eole_cfg_keys, eole_run

SOURCE_ROOT = REPO_ROOT / "src" / "rinchen"

# ---- the constants ------------------------------------------------------------------


def test_the_forbidden_keys_are_the_sixteen_field_names_d5_listed() -> None:
    assert isinstance(eole_cfg.FORBIDDEN_KEYS, frozenset)
    assert eole_cfg.FORBIDDEN_KEYS == {
        "rope_config",
        "rotary_interleave",
        "rotary_theta",
        "rotary_theta_local",
        "rotary_dim",
        "rotary_dim_global",
        "partial_rotary_factor",
        "interleave_local",
        "multidimensional_rope",
        "xdrope_section",
        "qk_norm_post_rope",
        "relative_positions_buckets",
        "sliding_window",
        "layer_types",
        "heads_kv",
        "global_heads_kv",
    }


def test_the_position_encoding_is_the_sinusoidal_interleaved_one() -> None:
    assert eole_cfg.POSITION_ENCODING == "SinusoidalInterleaved"


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in vars(eole_cfg_keys)
        if name.isupper() and not name.startswith("_")
    ],
)
def test_every_allow_list_is_still_a_name_of_eole_cfg(name: str) -> None:
    # They moved to `eole_cfg_keys` to keep `eole_cfg` short; callers and the drift
    # tests still read them as `eole_cfg.<NAME>`.
    assert getattr(eole_cfg, name) is getattr(eole_cfg_keys, name)


def test_the_allow_lists_are_those_the_module_of_keys_holds() -> None:
    assert {
        name
        for name in vars(eole_cfg_keys)
        if name.isupper() and not name.startswith("_")
    } == {
        "TOP_LEVEL_KEYS",
        "CORPUS_KEYS",
        "TRANSFORMS_CONFIGS_KEYS",
        "SENTENCEPIECE_KEYS",
        "FILTERTOOLONG_KEYS",
        "MODEL_KEYS",
        "ENCODER_KEYS",
        "DECODER_KEYS",
        "EMBEDDINGS_KEYS",
        "TRAINING_KEYS",
        "RESUME_TRAINING_KEYS",
        "CONTINUATION_TRAINING_KEYS",
        "CONTINUATION_RESET_OPTIM",
        "CONTINUATION_RESUME_RESET_OPTIM",
    }


# ---- reject_dollar ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["runs/$HOME/ckpt", "$", Path("runs") / "$x" / "ckpt"],
    ids=["a variable", "a lone dollar", "a path"],
)
def test_reject_dollar_names_what_holds_the_sign(value: str | Path) -> None:
    with pytest.raises(ConfigError) as raised:
        eole_cfg.reject_dollar("run directory", value)

    assert str(raised.value) == (
        "the run directory holds '$', which Eole would expand as a variable"
    )


@pytest.mark.parametrize("value", ["runs/ckpt", "", Path("runs") / "ckpt"])
def test_reject_dollar_accepts_a_value_without_the_sign(value: str | Path) -> None:
    assert eole_cfg.reject_dollar("run directory", value) is None


# ---- reject_glob_characters --------------------------------------------------------


@pytest.mark.parametrize("char", ["[", "]", "*", "?"])
def test_reject_glob_characters_names_the_path_and_the_character(char: str) -> None:
    path = Path("runs") / f"a{char}b" / "ckpt"

    with pytest.raises(ConfigError) as raised:
        eole_cfg.reject_glob_characters("checkpoint directory", path)

    assert str(raised.value) == (
        f"the checkpoint directory {path} holds {char!r}, which Eole reads as a glob "
        "pattern when it loads a checkpoint from it"
    )


def test_reject_glob_characters_says_the_callers_reason_instead_of_eoles() -> None:
    with pytest.raises(ConfigError) as raised:
        eole_cfg.reject_glob_characters(
            "output directory", "runs/a[b", reason="which this path rule refuses"
        )

    assert str(raised.value) == (
        "the output directory runs/a[b holds '[', which this path rule refuses"
    )


def test_reject_dollar_says_the_callers_reason_instead_of_eoles() -> None:
    with pytest.raises(ConfigError) as raised:
        eole_cfg.reject_dollar(
            "output directory", "runs/$HOME", reason="which a rule refuses"
        )

    assert str(raised.value) == "the output directory holds '$', which a rule refuses"


def test_reject_glob_characters_names_the_first_one_in_the_text() -> None:
    with pytest.raises(ConfigError, match=r"holds '\]'"):
        eole_cfg.reject_glob_characters("output directory", "runs/a]b[c*d?e/ckpt")


@pytest.mark.parametrize(
    "value",
    [
        "runs/ckpt",
        "",
        Path("runs") / "my run" / "ckpt",
        Path("runs") / "ünï" / "хураангуй",
        "runs/{1}/a!b-c.d,e",
    ],
    ids=["plain", "empty", "a space", "non-ASCII", "other punctuation"],
)
def test_reject_glob_characters_accepts_every_other_value(value: str | Path) -> None:
    assert eole_cfg.reject_glob_characters("run directory", value) is None


# ---- read_config -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "seed: 1\ntraining:\n  model_path: ckpt\n",
            {"seed": 1, "training": {"model_path": "ckpt"}},
        ),
        ("- just\n- a list\n", ["just", "a list"]),
        ("", None),
    ],
    ids=["a mapping", "a list", "an empty file"],
)
def test_read_config_returns_what_the_yaml_holds_and_leaves_its_shape_to_the_validator(
    tmp_path: Path, text: str, expected: Any
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    assert eole_cfg.read_config(path, "Eole config") == expected


def test_read_config_reports_a_missing_file_naming_the_kind_of_config(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.yaml"

    with pytest.raises(ConfigError) as raised:
        eole_cfg.read_config(missing, "ablation base config")

    assert str(raised.value) == f"ablation base config file not found: {missing}"


def test_read_config_reports_a_directory_as_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        eole_cfg.read_config(tmp_path, "Eole config")


@pytest.mark.parametrize(
    "content",
    [b"key: [unclosed", b"key: \xff\xfe\n"],
    ids=["not yaml", "not utf-8"],
)
def test_read_config_reports_a_file_it_cannot_parse_with_its_path(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_bytes(content)

    with pytest.raises(
        ConfigError,
        match=rf"^invalid Eole config {re.escape(str(path))}: cannot read or parse: ",
    ):
        eole_cfg.read_config(path, "Eole config")


def test_read_config_rejects_a_dollar_sign_anywhere_in_the_text(tmp_path: Path) -> None:
    # Eole expands `$VAR` in the whole text of the file it reads.
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  valid:\n    path_src: corpus/$CORPUS/valid.src\n")

    with pytest.raises(ConfigError) as raised:
        eole_cfg.read_config(path, "Eole config")

    assert str(raised.value) == (
        f"the text of the config {path} holds '$', which Eole would expand as a variable"
    )


def test_read_config_rejects_a_mapping_that_contains_itself(tmp_path: Path) -> None:
    # A YAML anchor can make a value hold itself; `merge` would recurse forever.
    path = tmp_path / "config.yaml"
    path.write_text("a: &x\n  self: *x\n", encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        eole_cfg.read_config(path, "base Eole config")

    assert str(raised.value) == (
        f"invalid base Eole config {path}: a.self: refers back to a mapping or list "
        "that contains it"
    )


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("&x [*x]\n", "[0]"),
        ("a: &x\n  - b: *x\n", "a[0].b"),
        ("a: &x\n  b:\n    c: [1, *x]\n", "a.b.c[1]"),
    ],
    ids=["a list in a list", "through a list", "deep"],
)
def test_read_config_names_where_a_recursive_anchor_is_used(
    tmp_path: Path, text: str, where: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        eole_cfg.read_config(path, "Eole config")

    assert str(raised.value) == (
        f"invalid Eole config {path}: {where}: refers back to a mapping or list that "
        "contains it"
    )


def test_read_config_keeps_an_anchor_that_is_only_shared_not_recursive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("a: &x [1, 2]\nb: *x\nc: [*x, *x]\n", encoding="utf-8")

    assert eole_cfg.read_config(path, "Eole config") == {
        "a": [1, 2],
        "b": [1, 2],
        "c": [[1, 2], [1, 2]],
    }


class _Mode(enum.StrEnum):
    FAST = "fast"


def test_dump_config_turns_a_value_yaml_cannot_dump_into_a_config_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config.yaml"

    with pytest.raises(ConfigError) as raised:
        eole_cfg.dump_config({"training": {"mode": _Mode.FAST}}, target)

    assert str(raised.value).startswith(f"cannot write the config {target}: ")
    assert isinstance(raised.value.__cause__, yaml.YAMLError)
    assert list(tmp_path.iterdir()) == []


def test_dump_config_keeps_the_old_file_when_a_value_cannot_be_dumped(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config.yaml"
    eole_cfg.dump_config({"seed": 1}, target)

    with pytest.raises(ConfigError):
        eole_cfg.dump_config({"seed": Path("x")}, target)

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {"seed": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["config.yaml"]


# ---- the moves -------------------------------------------------------------------


def test_script_conv_config_re_exports_the_names_that_moved_to_eole_cfg() -> None:
    assert converter_config.FORBIDDEN_KEYS is eole_cfg.FORBIDDEN_KEYS
    assert converter_config.dump_config is eole_cfg.dump_config


def test_script_conv_train_re_exports_the_names_that_moved_to_eole_run() -> None:
    assert converter_train.CHECKPOINT_FILE is eole_run.CHECKPOINT_FILE
    assert converter_train.eole_commands is eole_run.eole_commands
    assert converter_train.training_environment is eole_run.training_environment


def test_the_converters_run_training_names_its_config_a_converter_config(
    tmp_path: Path,
) -> None:
    # The converter's messages keep the wording they had before `eole_run` took over.
    run_dir = tmp_path / "run"
    cfg = converter_config.render_converter_config(
        "tiny",
        data_dir=tmp_path / "corpus",
        spm_model=tmp_path / "chars.model",
        run_dir=run_dir,
    )
    cfg["training"]["model_path"] = (tmp_path / "elsewhere").as_posix()
    cfg_path = tmp_path / "converter.yaml"
    converter_config.dump_config(cfg, cfg_path)

    with pytest.raises(
        ConfigError,
        match=r"^invalid converter config: training\.model_path: .* must be inside "
        r"the run directory ",
    ):
        converter_train.run_training(cfg_path, run_dir)

    assert not run_dir.exists()


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_nothing_under_train_or_utils_imports_script_conv() -> None:
    files = sorted(
        [*(SOURCE_ROOT / "train").glob("*.py"), *(SOURCE_ROOT / "utils").glob("*.py")]
    )
    importers = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in files
        if any(
            module == "rinchen.script_conv" or module.startswith("rinchen.script_conv.")
            for module in _imported_modules(path)
        )
    ]

    # The scan must have seen the modules this test exists for.
    names = {path.name for path in files}
    assert {"eole_cfg.py", "eole_run.py", "logged_process.py"} <= names
    assert importers == []
