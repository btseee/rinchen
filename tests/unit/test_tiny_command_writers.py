"""`rinchen.tiny.commands`: the files the CLI stages write for their commands (D15, spec P2, P7 and P17,
brief G4.4): the SentencePiece config, the character-model config, the ablation matrix and the
synthetic-data config.

Each file is loaded through the real loader of the command that reads it (a refusal fails the test) and,
where the spike recorded the file its prototype wrote, compared with the recording. A writer refuses a
path that is not absolute (the loaders anchor a relative one at the checkout, which would put a model
or a corpus there) and a server that is not on the loopback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from rinchen.config.ablation import load_ablation
from rinchen.config.spm import load_spm_config
from rinchen.config.synth import VllmConfig, load_synth_config
from rinchen.config.tiny import SpmSection, TrainSection
from rinchen.script_conv.char_model import load_char_model_config
from rinchen.tiny import commands
from rinchen.train.variants import expand_variants
from tests.support.tiny_stage_inputs import shipped_config

RECORDINGS: Final = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_run"
RECORDED_URL: Final = "http://127.0.0.1:41309/v1"


def recorded_yaml(name: str) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(
        (RECORDINGS / name).read_text(encoding="utf-8")
    )
    return document


def train_section(**changes: Any) -> TrainSection:
    return shipped_config().train.model_copy(update=changes)


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


# ---- the SentencePiece config ------------------------------------------------------------------------


def write_spm(
    tmp_path: Path, section: SpmSection | None = None, seed: int = 1234
) -> Path:
    path = tmp_path / "spm.yaml"
    commands.write_spm_config(
        path,
        section or shipped_config().spm,
        seed,
        tmp_path / "text",
        tmp_path / "tiny",
    )
    return path


def test_the_spm_config_loads_and_holds_the_profile_and_the_paths(
    tmp_path: Path,
) -> None:
    loaded = load_spm_config(write_spm(tmp_path))
    section = shipped_config().spm

    assert loaded.vocab_size == section.vocab_size == 640
    assert loaded.input_sentence_size == section.input_sentence_size
    assert loaded.character_coverage == section.character_coverage
    assert loaded.temperature == section.temperature
    assert loaded.seed == 1234
    assert loaded.corpus_dir == tmp_path / "text"
    assert loaded.output_prefix == tmp_path / "tiny"


def test_the_spm_config_is_the_one_the_prototype_wrote(tmp_path: Path) -> None:
    written = yaml.safe_load(write_spm(tmp_path).read_text(encoding="utf-8"))
    recorded = recorded_yaml("spm.yaml")

    assert recorded["corpus_dir"] == "SPM_TEXT_DIR"
    assert recorded["output_prefix"] == "SPM_OUTPUT_PREFIX"
    for key in (
        "vocab_size",
        "input_sentence_size",
        "character_coverage",
        "temperature",
        "seed",
    ):
        assert written[key] == recorded[key], key
    assert sorted(written) == sorted(recorded)


def test_the_spm_config_follows_its_arguments(tmp_path: Path) -> None:
    section = SpmSection(
        vocab_size=700, input_sentence_size=5, character_coverage=0.5, temperature=0.7
    )

    loaded = load_spm_config(write_spm(tmp_path, section, seed=9))

    assert (loaded.vocab_size, loaded.input_sentence_size) == (700, 5)
    assert (loaded.character_coverage, loaded.temperature, loaded.seed) == (0.5, 0.7, 9)


@pytest.mark.parametrize("which", ["corpus_dir", "output_prefix"])
def test_the_spm_config_refuses_a_relative_path(tmp_path: Path, which: str) -> None:
    paths = {"corpus_dir": tmp_path / "text", "output_prefix": tmp_path / "tiny"}
    paths[which] = Path("relative")

    with pytest.raises(ValueError, match="absolute"):
        commands.write_spm_config(
            tmp_path / "spm.yaml", shipped_config().spm, 1, **paths
        )

    assert names_in(tmp_path) == []


# ---- the character-model config ----------------------------------------------------------------------


def test_the_character_config_loads_with_the_profiles_vocabulary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "char.yaml"

    commands.write_char_config(path, shipped_config().converter.char_vocab_size)

    assert load_char_model_config(path).vocab_size == 200


@pytest.mark.parametrize("size", [0, -1])
def test_the_character_config_refuses_a_size_that_is_not_positive(
    tmp_path: Path, size: int
) -> None:
    with pytest.raises(ValueError):
        commands.write_char_config(tmp_path / "char.yaml", size)

    assert names_in(tmp_path) == []


# ---- the ablation matrix -----------------------------------------------------------------------------


def write_matrix(
    tmp_path: Path, config: TrainSection | None = None, seed: int = 1234
) -> Path:
    used = config or shipped_config().train
    path = tmp_path / "matrix.yaml"
    commands.write_matrix(path, used, used.base, seed)
    return path


def test_the_matrix_loads_through_the_real_loader_as_one_variant(
    tmp_path: Path,
) -> None:
    matrix = load_ablation(write_matrix(tmp_path))

    assert [variant.id for variant in expand_variants(matrix)] == ["t1_bt0.2_src_s1234"]
    assert matrix.base == shipped_config().train.base
    assert matrix.axes.tag_placement == ["src"]
    assert matrix.axes.temperature == [1.0]
    assert matrix.axes.bt_ratio == [0.2]
    assert matrix.seeds == [1234]
    assert matrix.dev_max_lines == 100


def test_the_matrix_overrides_only_the_steps(tmp_path: Path) -> None:
    matrix = load_ablation(write_matrix(tmp_path))

    assert matrix.overrides == {"training": {"train_steps": 1500, "valid_steps": 1500}}


def test_the_steps_of_the_profile_are_the_ones_the_matrix_states(
    tmp_path: Path,
) -> None:
    matrix = load_ablation(write_matrix(tmp_path, train_section(steps=400)))

    assert matrix.overrides == {"training": {"train_steps": 400, "valid_steps": 400}}


def test_the_matrix_follows_the_mixing_and_the_seed_of_the_profile(
    tmp_path: Path,
) -> None:
    config = train_section(temperature=0.5, bt_ratio=0.25, dev_max_lines=7)

    matrix = load_ablation(write_matrix(tmp_path, config, seed=7))

    assert [variant.id for variant in expand_variants(matrix)] == ["t0.5_bt0.25_src_s7"]
    assert matrix.dev_max_lines == 7
    assert matrix.seeds == [7]


def test_the_matrix_is_the_one_the_prototype_wrote_but_for_the_base_and_the_steps(
    tmp_path: Path,
) -> None:
    written = yaml.safe_load(write_matrix(tmp_path).read_text(encoding="utf-8"))
    recorded = recorded_yaml("matrix.yaml")

    assert recorded["base"] == "BASE_CONFIG"
    for key in ("name", "axes", "seeds", "dev_max_lines"):
        assert written[key] == recorded[key], key
    assert sorted(set(written) - set(recorded)) == ["overrides"]


def test_the_matrix_names_the_base_it_was_given(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\n", encoding="utf-8")
    path = tmp_path / "matrix.yaml"

    commands.write_matrix(path, shipped_config().train, base, 1234)

    assert load_ablation(path).base == base


def test_the_matrix_refuses_a_relative_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        commands.write_matrix(
            tmp_path / "matrix.yaml",
            shipped_config().train,
            Path("configs/eole/tiny.yaml"),
            1,
        )

    assert names_in(tmp_path) == []


# ---- the synthetic-data config -----------------------------------------------------------------------


def write_synth(tmp_path: Path, **changes: Any) -> Path:
    arguments: dict[str, Any] = {
        "job": "bt_eng_khk",
        "base_url": RECORDED_URL,
        "shard_size": 32,
        "n": 1,
    }
    arguments.update(changes)
    path = tmp_path / "synth.yaml"
    commands.write_synth_config(path, **arguments)
    return path


def test_the_synth_config_is_byte_for_byte_the_one_the_prototype_wrote(
    tmp_path: Path,
) -> None:
    assert (
        write_synth(tmp_path).read_bytes() == (RECORDINGS / "synth.yaml").read_bytes()
    )


def test_the_synth_config_loads_through_the_real_loader(tmp_path: Path) -> None:
    config = load_synth_config(
        write_synth(tmp_path, base_url="http://127.0.0.1:5/v1", n=3)
    )

    generator = config.generators["tiny_oracle"]
    assert isinstance(generator, VllmConfig)
    assert generator.base_url == "http://127.0.0.1:5/v1"
    assert generator.model == "tiny-oracle"
    assert generator.n == 3
    assert generator.temperature == 0.0
    assert generator.top_p == 1.0
    assert generator.retries == 0
    assert generator.timeout_s == 20
    job = config.jobs["bt_eng_khk"]
    assert (job.kind, job.mono_lang, job.other_lang) == ("bt", "khk_Cyrl", "eng_Latn")
    assert job.generator == "tiny_oracle"
    assert job.shard_size == 32
    assert list(config.jobs) == ["bt_eng_khk"]


def test_the_synth_config_names_the_job_and_the_shard_size_it_was_given(
    tmp_path: Path,
) -> None:
    config = load_synth_config(write_synth(tmp_path, job="bt_other", shard_size=16))

    assert list(config.jobs) == ["bt_other"]
    assert config.jobs["bt_other"].shard_size == 16


@pytest.mark.parametrize("port", [1, 80, 41309, 65535])
def test_the_synth_config_accepts_every_port_of_the_loopback_listener(
    tmp_path: Path, port: int
) -> None:
    config = load_synth_config(
        write_synth(tmp_path, base_url=f"http://127.0.0.1:{port}/v1")
    )

    generator = config.generators["tiny_oracle"]
    assert isinstance(generator, VllmConfig)
    assert generator.base_url == f"http://127.0.0.1:{port}/v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:5/v1",
        "http://0.0.0.0:5/v1",
        "http://127.0.0.2:5/v1",
        "http://example.org:5/v1",
        "https://127.0.0.1:5/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:0/v1",
        "http://127.0.0.1:65536/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1:100000/v1",
        "http://127.0.0.1:5/v2",
        "http://127.0.0.1:5/v1/",
        "http://user@127.0.0.1:5/v1",
        "http://127.0.0.1:5/v1?x=1",
        "127.0.0.1:5/v1",
        "",
    ],
)
def test_the_synth_config_refuses_a_server_that_is_not_the_loopback_listener(
    tmp_path: Path, url: str
) -> None:
    with pytest.raises(ValueError, match="loopback"):
        write_synth(tmp_path, base_url=url)

    assert names_in(tmp_path) == []


# ---- the writers all write whole files -----------------------------------------------------------------


def test_a_writer_replaces_a_file_whole_and_leaves_no_part_file(tmp_path: Path) -> None:
    path = write_synth(tmp_path)
    path.write_text("old\n", encoding="utf-8")

    write_synth(tmp_path)

    assert names_in(tmp_path) == ["synth.yaml"]
    assert path.read_bytes() == (RECORDINGS / "synth.yaml").read_bytes()
    assert path.read_bytes().endswith(b"\n")
