# tests/unit/test_config_synth.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import rinchen.config.synth as synth_module
from rinchen.config.filters import QeConfig, load_filter_config
from rinchen.config.languages import LanguageEntry
from rinchen.config.loader import REPO_ROOT
from rinchen.config.synth import (
    SYNTH_CONFIG_PATH,
    Ct2Config,
    JobConfig,
    LlamaCppConfig,
    SelectionConfig,
    SynthConfig,
    VllmConfig,
    load_synth_config,
)
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

_CT2_MODEL_DIR = "models/ct2/nllb-200-3.3B"
_CT2_SPM_MODEL = "models/ct2/nllb-200-3.3B/sentencepiece.bpe.model"


def _write(tmp_path: Path, document: Any) -> Path:
    """Write `document` as YAML (or verbatim when it is already text)."""
    path = tmp_path / "synth.yaml"
    text = document if isinstance(document, str) else yaml.safe_dump(document)
    path.write_text(text, encoding="utf-8")
    return path


def _violations(error: ConfigError, depth: int) -> list[tuple[Any, ...]]:
    """What the chained pydantic error reports, minus the first `depth` path
    parts (the sections above the QE section)."""
    cause = error.__cause__
    assert isinstance(cause, ValidationError)
    return [(e["loc"][depth:], e["type"], e["msg"]) for e in cause.errors()]


def _job(**overrides: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "kind": "bt",
        "mono_lang": "khk_Cyrl",
        "other_lang": "eng_Latn",
        "generator": "g",
    }
    return {**job, **overrides}


def _document(
    generator: dict[str, Any] | None = None, job: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The smallest valid synth config: one generator `g` and one job `j`."""
    return {
        "generators": {"g": {"kind": "vllm"} if generator is None else generator},
        "jobs": {"j": _job() if job is None else job},
    }


def _ct2(**overrides: Any) -> dict[str, Any]:
    generator: dict[str, Any] = {
        "kind": "ct2",
        "model_dir": _CT2_MODEL_DIR,
        "spm_model": _CT2_SPM_MODEL,
        "family": "nllb",
    }
    return {**generator, **overrides}


def test_default_generators_are_one_of_each_kind():
    config = SynthConfig()

    assert config.generators == {
        "vllm_default": VllmConfig(
            n=4,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            base_url="http://127.0.0.1:8000/v1",
            model="Hy-MT2-7B",
            concurrency=32,
            timeout_s=120,
            retries=4,
            backoff_s=1.0,
        ),
        "llamacpp_default": LlamaCppConfig(
            n=4,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            base_url="http://127.0.0.1:8080/v1",
            model="Hy-MT2-7B",
            concurrency=8,
            timeout_s=120,
            retries=4,
            backoff_s=1.0,
        ),
        "ct2_nllb_default": Ct2Config(
            n=4,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            model_dir=_CT2_MODEL_DIR,
            spm_model=_CT2_SPM_MODEL,
            family="nllb",
            device="cpu",
            beam_size=4,
            batch_size=16,
        ),
    }
    assert [type(g) for g in config.generators.values()] == [
        VllmConfig,
        LlamaCppConfig,
        Ct2Config,
    ]
    assert [g.kind for g in config.generators.values()] == ["vllm", "llamacpp", "ct2"]


def test_the_default_ct2_paths_are_placeholders_anchored_at_the_repo_root():
    generator = SynthConfig().generators["ct2_nllb_default"]

    assert isinstance(generator, Ct2Config)
    assert generator.model_dir == REPO_ROOT / "models" / "ct2" / "nllb-200-3.3B"
    assert (
        generator.spm_model
        == REPO_ROOT / "models" / "ct2" / "nllb-200-3.3B" / "sentencepiece.bpe.model"
    )


def test_default_jobs_are_one_backward_and_one_forward_translation():
    assert SynthConfig().jobs == {
        "bt_eng_khk": JobConfig(
            kind="bt",
            mono_lang="khk_Cyrl",
            other_lang="eng_Latn",
            generator="vllm_default",
            shard_size=512,
        ),
        "ft_khk_eng": JobConfig(
            kind="ft",
            mono_lang="khk_Cyrl",
            other_lang="eng_Latn",
            generator="vllm_default",
            shard_size=512,
        ),
    }


def test_default_selection_is_the_documented_priors():
    selection = SynthConfig().selection

    assert selection == SelectionConfig(
        min_script_share=0.9,
        qe=QeConfig(
            scorer="metricx_qe",
            max_metricx_qe=8.0,
            min_cometkiwi=0.55,
            batch_size=32,
        ),
        mbr_utility="chrf",
    )


def test_shipped_synth_yaml_equals_the_defaults():
    loaded = load_synth_config()

    assert SYNTH_CONFIG_PATH == REPO_ROOT / "configs" / "data" / "synth.yaml"
    assert loaded.model_dump(mode="json") == SynthConfig().model_dump(mode="json")
    assert loaded == SynthConfig()


def test_generator_kind_picks_the_model_class(tmp_path: Path):
    document = {
        "generators": {
            "a": {"kind": "vllm"},
            "b": {"kind": "llamacpp"},
            "c": _ct2(),
        },
        "jobs": {},
    }

    config = load_synth_config(_write(tmp_path, document))

    assert [type(g) for g in config.generators.values()] == [
        VllmConfig,
        LlamaCppConfig,
        Ct2Config,
    ]


def test_server_generators_differ_only_in_their_endpoint_and_concurrency(
    tmp_path: Path,
):
    document = {
        "generators": {"a": {"kind": "vllm"}, "b": {"kind": "llamacpp"}},
        "jobs": {},
    }

    generators = load_synth_config(_write(tmp_path, document)).generators

    assert (
        generators["a"].model_dump()
        | {
            "kind": "llamacpp",
            "base_url": "http://127.0.0.1:8080/v1",
            "concurrency": 8,
        }
        == generators["b"].model_dump()
    )


def test_a_supplied_generators_table_replaces_the_default_one(tmp_path: Path):
    config = load_synth_config(_write(tmp_path, _document()))

    assert list(config.generators) == ["g"]
    assert list(config.jobs) == ["j"]


def test_each_config_gets_its_own_default_tables():
    first, second = SynthConfig(), SynthConfig()

    first.generators.clear()
    first.jobs.clear()

    assert len(second.generators) == 3
    assert len(second.jobs) == 2
    assert len(SynthConfig().generators) == 3


def test_empty_sections_use_the_defaults(tmp_path: Path):
    config = load_synth_config(
        _write(tmp_path, {"selection": {"min_script_share": 0.5}})
    )

    assert config.selection.min_script_share == 0.5
    assert config.selection.qe == QeConfig()
    assert config.selection.mbr_utility == "chrf"
    assert config.generators == SynthConfig().generators
    assert config.jobs == SynthConfig().jobs


def test_a_supplied_value_overrides_its_default(tmp_path: Path):
    document = {
        "generators": {
            "g": {
                "kind": "vllm",
                "n": 8,
                "temperature": 1,
                "top_p": 1,
                "max_tokens": 64,
                "base_url": "http://gpu-box:9000/v1",
                "model": "other-model",
                "concurrency": 2,
                "timeout_s": 30,
                "retries": 0,
                "backoff_s": 0,
            },
            "c": _ct2(
                family="madlad",
                device="cuda",
                beam_size=2,
                batch_size=3,
                n=2,
            ),
        },
        "jobs": {
            "j": {
                "kind": "ft",
                "mono_lang": "rus_Cyrl",
                "other_lang": "khk_Cyrl",
                "generator": "c",
                "shard_size": 7,
            }
        },
        "selection": {
            "min_script_share": 0.75,
            "qe": {"scorer": "cometkiwi", "min_cometkiwi": 0.6, "batch_size": 4},
            "mbr_utility": "chrf",
        },
    }

    config = load_synth_config(_write(tmp_path, document))

    assert config.generators["g"] == VllmConfig(
        n=8,
        temperature=1,
        top_p=1,
        max_tokens=64,
        base_url="http://gpu-box:9000/v1",
        model="other-model",
        concurrency=2,
        timeout_s=30,
        retries=0,
        backoff_s=0,
    )
    assert config.generators["c"] == Ct2Config(
        model_dir=_CT2_MODEL_DIR,
        spm_model=_CT2_SPM_MODEL,
        family="madlad",
        device="cuda",
        beam_size=2,
        batch_size=3,
        n=2,
    )
    assert config.jobs["j"] == JobConfig(
        kind="ft",
        mono_lang="rus_Cyrl",
        other_lang="khk_Cyrl",
        generator="c",
        shard_size=7,
    )
    assert config.selection == SelectionConfig(
        min_script_share=0.75,
        qe=QeConfig(scorer="cometkiwi", min_cometkiwi=0.6, batch_size=4),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_ct2_runs_on_cpu_cuda_or_auto(tmp_path: Path, device: str):
    document = {"generators": {"c": _ct2(device=device)}, "jobs": {}}

    config = load_synth_config(_write(tmp_path, document))

    generator = config.generators["c"]
    assert isinstance(generator, Ct2Config)
    assert generator.device == device


def test_ct2_keeps_the_sampling_fields_for_interface_symmetry():
    generator = Ct2Config(
        model_dir=_CT2_MODEL_DIR,
        spm_model=_CT2_SPM_MODEL,
        family="nllb",
        temperature=1.5,
        top_p=0.5,
        max_tokens=32,
    )

    assert (generator.temperature, generator.top_p, generator.max_tokens) == (
        1.5,
        0.5,
        32,
    )
    assert generator.n == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 1),
        ("n", 64),
        ("temperature", 0),
        ("temperature", 2),
        ("top_p", 1),
        ("top_p", 0.001),
        ("max_tokens", 1),
        ("concurrency", 1),
        ("timeout_s", 0.5),
        ("retries", 0),
        ("backoff_s", 0),
    ],
)
def test_server_generator_bounds_include_their_edges(
    tmp_path: Path, field: str, value: float
):
    config = load_synth_config(
        _write(tmp_path, _document(generator={"kind": "vllm", field: value}))
    )

    assert getattr(config.generators["g"], field) == value


def test_a_shard_holds_at_least_one_row(tmp_path: Path):
    config = load_synth_config(_write(tmp_path, _document(job=_job(shard_size=1))))

    assert config.jobs["j"].shard_size == 1


def test_min_script_share_includes_one(tmp_path: Path):
    config = load_synth_config(_write(tmp_path, {"selection": {"min_script_share": 1}}))

    assert config.selection.min_script_share == 1.0


@pytest.mark.parametrize(
    ("job", "mono_lang", "other_lang"),
    [
        pytest.param(
            {"mono_lang": "khk_Cyrl", "other_lang": "rus_Cyrl"},
            "khk_Cyrl",
            "rus_Cyrl",
            id="mongolian-first",
        ),
        pytest.param(
            {"mono_lang": "eng_Latn", "other_lang": "khk_Cyrl"},
            "eng_Latn",
            "khk_Cyrl",
            id="mongolian-second",
        ),
    ],
)
def test_a_job_pairs_khk_cyrl_with_another_registry_language(
    tmp_path: Path, job: dict[str, str], mono_lang: str, other_lang: str
):
    config = load_synth_config(_write(tmp_path, _document(job=_job(**job))))

    assert config.jobs["j"].mono_lang == mono_lang
    assert config.jobs["j"].other_lang == other_lang


def test_relative_ct2_paths_resolve_against_repo_root_and_absolute_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute_model_dir = tmp_path / "abs" / "nllb"
    absolute_spm_model = tmp_path / "abs" / "nllb" / "spm.model"
    document = {
        "generators": {
            "relative": _ct2(
                model_dir="models/ct2/other", spm_model="models/ct2/other/spm.model"
            ),
            "absolute": _ct2(
                model_dir=str(absolute_model_dir), spm_model=str(absolute_spm_model)
            ),
        }
    }

    config = load_synth_config(_write(tmp_path, document))

    relative = config.generators["relative"]
    absolute = config.generators["absolute"]
    assert isinstance(relative, Ct2Config)
    assert isinstance(absolute, Ct2Config)
    assert relative.model_dir == REPO_ROOT / "models" / "ct2" / "other"
    assert relative.spm_model == REPO_ROOT / "models" / "ct2" / "other" / "spm.model"
    assert absolute.model_dir == absolute_model_dir
    assert absolute.spm_model == absolute_spm_model


def test_a_relative_ct2_path_means_the_same_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write(tmp_path, {"generators": {"c": _ct2()}})
    from_repo = load_synth_config(path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_synth_config(path) == from_repo
    assert Ct2Config(
        model_dir=_CT2_MODEL_DIR, spm_model=_CT2_SPM_MODEL, family="nllb"
    ).model_dir == (REPO_ROOT / "models" / "ct2" / "nllb-200-3.3B")


@pytest.mark.parametrize(("n", "beam_size"), [(4, 4), (1, 1), (5, 5), (2, 8), (1, 4)])
def test_ct2_hypotheses_may_equal_the_beam_or_be_fewer(
    tmp_path: Path, n: int, beam_size: int
):
    document = {"generators": {"c": _ct2(n=n, beam_size=beam_size)}}

    generator = load_synth_config(_write(tmp_path, document)).generators["c"]

    assert isinstance(generator, Ct2Config)
    assert (generator.n, generator.beam_size) == (n, beam_size)


@pytest.mark.parametrize(("n", "beam_size"), [(5, 4), (2, 1), (9, 8), (64, 63)])
def test_ct2_cannot_return_more_hypotheses_than_the_beam_holds(n: int, beam_size: int):
    with pytest.raises(
        ValidationError,
        match=rf"n \({n}\) cannot be greater than beam_size \({beam_size}\)",
    ):
        Ct2Config(
            model_dir=_CT2_MODEL_DIR,
            spm_model=_CT2_SPM_MODEL,
            family="nllb",
            n=n,
            beam_size=beam_size,
        )


def test_supplied_generators_leave_the_jobs_empty_instead_of_dangling(tmp_path: Path):
    document = {"generators": {"g": {"kind": "vllm"}}}

    config = load_synth_config(_write(tmp_path, document))

    assert list(config.generators) == ["g"]
    assert config.jobs == {}
    assert SynthConfig(generators={"g": VllmConfig()}).jobs == {}


def test_supplied_jobs_keep_the_default_generators(tmp_path: Path):
    document = {"jobs": {"j": _job(generator="llamacpp_default")}}

    config = load_synth_config(_write(tmp_path, document))

    assert list(config.jobs) == ["j"]
    assert config.generators == SynthConfig().generators


def test_supplied_generators_and_jobs_replace_both_tables(tmp_path: Path):
    document = {
        "generators": {"g": {"kind": "vllm"}},
        "jobs": {
            "j": _job(),
            "k": _job(kind="ft", mono_lang="rus_Cyrl", other_lang="khk_Cyrl"),
        },
    }

    config = load_synth_config(_write(tmp_path, document))

    assert list(config.generators) == ["g"]
    assert list(config.jobs) == ["j", "k"]


def test_an_explicit_empty_jobs_table_stays_empty(tmp_path: Path):
    config = load_synth_config(_write(tmp_path, {"jobs": {}}))

    assert config.jobs == {}
    assert config.generators == SynthConfig().generators


def _registry(*entries: tuple[str, list[Script]]) -> list[LanguageEntry]:
    return [
        LanguageEntry(code=code, scripts=scripts, direction_tag=f"<2{code}>")
        for code, scripts in entries
    ]


def test_a_job_language_is_refused_by_its_registry_script_not_by_its_code(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry(
        ("khk_Cyrl", [Script.CYRILLIC]),
        ("eng_Latn", [Script.LATIN]),
        ("zzz_Mong", [Script.MONGOLIAN]),
        ("mixed_Mong", [Script.CYRILLIC, Script.MONGOLIAN]),
        ("mn_Mong", [Script.LATIN]),
    )
    monkeypatch.setattr(synth_module, "load_languages", lambda: registry)

    accepted = JobConfig(
        kind="bt", mono_lang="khk_Cyrl", other_lang="mn_Mong", generator="g"
    )

    assert accepted.other_lang == "mn_Mong"
    for code in ("zzz_Mong", "mixed_Mong"):
        with pytest.raises(ValidationError, match=f"'{code}' is not a supported"):
            JobConfig(kind="bt", mono_lang=code, other_lang="khk_Cyrl", generator="g")
        with pytest.raises(ValidationError, match=f"'{code}' is not a supported"):
            JobConfig(kind="bt", mono_lang="khk_Cyrl", other_lang=code, generator="g")


@pytest.mark.parametrize(
    "languages",
    [
        pytest.param({"mono_lang": "mn_Mong", "other_lang": "khk_Cyrl"}, id="mono"),
        pytest.param({"mono_lang": "khk_Cyrl", "other_lang": "mn_Mong"}, id="other"),
        pytest.param({"mono_lang": "xxx_Latn", "other_lang": "khk_Cyrl"}, id="unknown"),
    ],
)
def test_the_supported_languages_in_the_message_leave_out_mn_mong(
    languages: dict[str, str],
):
    with pytest.raises(ValidationError) as excinfo:
        JobConfig(kind="bt", generator="g", **languages)

    listing = re.search(r"supported: \[([^\]]*)\]", str(excinfo.value))
    assert listing is not None
    assert sorted(re.findall(r"'(\w+)'", listing.group(1))) == [
        "deu_Latn",
        "eng_Latn",
        "jpn_Jpan",
        "kaz_Cyrl",
        "khk_Cyrl",
        "kor_Hang",
        "rus_Cyrl",
        "tur_Latn",
        "zho_Hans",
    ]


def test_a_job_alone_does_not_know_the_generators():
    job = JobConfig(
        kind="ft",
        mono_lang="khk_Cyrl",
        other_lang="eng_Latn",
        generator="not-configured-anywhere",
    )

    assert job.generator == "not-configured-anywhere"


def test_loaded_config_is_immutable():
    config = SynthConfig()

    with pytest.raises(ValidationError):
        config.selection = SelectionConfig()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.selection.min_script_share = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.jobs["bt_eng_khk"].shard_size = 1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.generators["vllm_default"].n = 1  # type: ignore[misc]


def test_selection_qe_is_the_d4_qe_config_itself():
    qe_annotation = SelectionConfig.model_fields["qe"].annotation

    assert qe_annotation is QeConfig
    assert type(SynthConfig().selection.qe) is QeConfig


@pytest.mark.parametrize(
    "qe",
    [
        pytest.param({"scorer": "bleurt"}, id="unknown-scorer"),
        pytest.param({"max_metricx_qe": -1}, id="negative-max-metricx-qe"),
        pytest.param({"min_cometkiwi": "0.55"}, id="quoted-min-cometkiwi"),
        pytest.param({"batch_size": 0}, id="zero-batch-size"),
        pytest.param({"max_comet": 1}, id="key-outside-qe-config"),
    ],
)
def test_an_invalid_qe_section_is_rejected_exactly_as_d4_rejects_it(
    tmp_path: Path, qe: dict[str, Any]
):
    with pytest.raises(ConfigError) as d4:
        load_filter_config(_write(tmp_path, {"qe": qe}))
    with pytest.raises(ConfigError) as d6:
        load_synth_config(_write(tmp_path, {"selection": {"qe": qe}}))

    # The same violations, one level deeper in the D6 document.
    assert _violations(d6.value, depth=2) == _violations(d4.value, depth=1)
    assert _violations(d6.value, depth=2)


_MUST_BE_FINITE = "Input should be a finite number"
_NO_KIND = "kind must be one of vllm, llamacpp, ct2"


def _at(location: str, message: str) -> str:
    """The pydantic error report for `location`: its path on one line, its
    message on the next, so a fragment cannot match some other field."""
    return rf"{re.escape(location)}\n +{message}"


@pytest.mark.parametrize(
    ("document", "message_fragment"),
    [
        pytest.param(
            {"bogus": 1}, _at("bogus", "Extra inputs are not permitted"), id="top-key"
        ),
        pytest.param(
            {"selection": {"bogus": 1}},
            _at("selection.bogus", "Extra inputs are not permitted"),
            id="selection-key",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "bogus": 1}),
            _at("generators.g.vllm.bogus", "Extra inputs are not permitted"),
            id="generator-key",
        ),
        pytest.param(
            _document(job=_job(bogus=1)),
            _at("jobs.j.bogus", "Extra inputs are not permitted"),
            id="job-key",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "model_dir": "m"}),
            _at("generators.g.vllm.model_dir", "Extra inputs are not permitted"),
            id="ct2-key-on-vllm",
        ),
        pytest.param(
            _document(generator=_ct2(base_url="http://x/v1")),
            _at("generators.g.ct2.base_url", "Extra inputs are not permitted"),
            id="server-key-on-ct2",
        ),
        pytest.param(
            _document(generator={"kind": "grpc"}),
            _at("generators.g", _NO_KIND) + r".*input_value=\{'kind': 'grpc'\}",
            id="unknown-generator-kind",
        ),
        pytest.param(
            _document(generator={"n": 4}),
            _at("generators.g", _NO_KIND),
            id="missing-generator-kind",
        ),
        pytest.param(
            _document(generator={"kind": ["vllm"]}),
            _at("generators.g", _NO_KIND),
            id="list-generator-kind",
        ),
        pytest.param(
            _document(generator={"kind": 7}),
            _at("generators.g", _NO_KIND),
            id="numeric-generator-kind",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": 0}),
            _at("generators.g.vllm.n", "Input should be greater than or equal to 1"),
            id="zero-n",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": -2}),
            _at("generators.g.vllm.n", "Input should be greater than or equal to 1"),
            id="negative-n",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": 65}),
            _at("generators.g.vllm.n", "Input should be less than or equal to 64"),
            id="n-above-64",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": "4"}),
            _at("generators.g.vllm.n", "Input should be a valid integer"),
            id="quoted-n",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": True}),
            _at("generators.g.vllm.n", "Input should be a valid integer"),
            id="boolean-n",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "n": 2.5}),
            _at("generators.g.vllm.n", "Input should be a valid integer"),
            id="fractional-n",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "temperature": -0.1}),
            _at(
                "generators.g.vllm.temperature",
                "Input should be greater than or equal to 0",
            ),
            id="negative-temperature",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "temperature": 2.1}),
            _at(
                "generators.g.vllm.temperature",
                "Input should be less than or equal to 2",
            ),
            id="temperature-above-2",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "temperature": "0.7"}),
            _at("generators.g.vllm.temperature", "Input should be a valid number"),
            id="quoted-temperature",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "temperature": float("nan")}),
            _at("generators.g.vllm.temperature", _MUST_BE_FINITE),
            id="nan-temperature",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "temperature": float("inf")}),
            _at("generators.g.vllm.temperature", _MUST_BE_FINITE),
            id="infinite-temperature",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "top_p": 0}),
            _at("generators.g.vllm.top_p", "Input should be greater than 0"),
            id="zero-top-p",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "top_p": 1.01}),
            _at("generators.g.vllm.top_p", "Input should be less than or equal to 1"),
            id="top-p-above-1",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "top_p": True}),
            _at("generators.g.vllm.top_p", "Input should be a valid number"),
            id="boolean-top-p",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "top_p": float("nan")}),
            _at("generators.g.vllm.top_p", _MUST_BE_FINITE),
            id="nan-top-p",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "top_p": float("inf")}),
            _at("generators.g.vllm.top_p", _MUST_BE_FINITE),
            id="infinite-top-p",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "max_tokens": 0}),
            _at("generators.g.vllm.max_tokens", "Input should be greater than 0"),
            id="zero-max-tokens",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "base_url": ""}),
            _at("generators.g.vllm.base_url", "Value error, must not be blank"),
            id="empty-base-url",
        ),
        pytest.param(
            _document(generator={"kind": "llamacpp", "base_url": "   "}),
            _at("generators.g.llamacpp.base_url", "Value error, must not be blank"),
            id="blank-base-url",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "model": ""}),
            _at("generators.g.vllm.model", "Value error, must not be blank"),
            id="empty-model",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "model": " "}),
            _at("generators.g.vllm.model", "Value error, must not be blank"),
            id="blank-model",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "model": 7}),
            _at("generators.g.vllm.model", "Input should be a valid string"),
            id="numeric-model",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "base_url": None}),
            _at("generators.g.vllm.base_url", "Input should be a valid string"),
            id="null-base-url",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "concurrency": 0}),
            _at("generators.g.vllm.concurrency", "Input should be greater than 0"),
            id="zero-concurrency",
        ),
        pytest.param(
            _document(generator={"kind": "llamacpp", "concurrency": "8"}),
            _at("generators.g.llamacpp.concurrency", "Input should be a valid integer"),
            id="quoted-concurrency",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "timeout_s": 0}),
            _at("generators.g.vllm.timeout_s", "Input should be greater than 0"),
            id="zero-timeout",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "timeout_s": "120"}),
            _at("generators.g.vllm.timeout_s", "Input should be a valid number"),
            id="quoted-timeout",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "timeout_s": float("nan")}),
            _at("generators.g.vllm.timeout_s", _MUST_BE_FINITE),
            id="nan-timeout",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "timeout_s": float("inf")}),
            _at("generators.g.vllm.timeout_s", _MUST_BE_FINITE),
            id="infinite-timeout",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "retries": -1}),
            _at(
                "generators.g.vllm.retries",
                "Input should be greater than or equal to 0",
            ),
            id="negative-retries",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "retries": False}),
            _at("generators.g.vllm.retries", "Input should be a valid integer"),
            id="boolean-retries",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "backoff_s": -0.5}),
            _at(
                "generators.g.vllm.backoff_s",
                "Input should be greater than or equal to 0",
            ),
            id="negative-backoff",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "backoff_s": float("nan")}),
            _at("generators.g.vllm.backoff_s", _MUST_BE_FINITE),
            id="nan-backoff",
        ),
        pytest.param(
            _document(generator={"kind": "vllm", "backoff_s": float("inf")}),
            _at("generators.g.vllm.backoff_s", _MUST_BE_FINITE),
            id="infinite-backoff",
        ),
        pytest.param(
            _document(generator=_ct2(model_dir=None)),
            _at("generators.g.ct2.model_dir", "Input is not a valid path"),
            id="null-model-dir",
        ),
        pytest.param(
            _document(generator=_ct2(model_dir=7)),
            _at("generators.g.ct2.model_dir", "Input is not a valid path"),
            id="numeric-model-dir",
        ),
        pytest.param(
            _document(
                generator={"kind": "ct2", "spm_model": _CT2_SPM_MODEL, "family": "nllb"}
            ),
            _at("generators.g.ct2.model_dir", "Field required"),
            id="ct2-without-model-dir",
        ),
        pytest.param(
            _document(
                generator={"kind": "ct2", "model_dir": _CT2_MODEL_DIR, "family": "nllb"}
            ),
            _at("generators.g.ct2.spm_model", "Field required"),
            id="ct2-without-spm-model",
        ),
        pytest.param(
            _document(
                generator={
                    "kind": "ct2",
                    "model_dir": _CT2_MODEL_DIR,
                    "spm_model": _CT2_SPM_MODEL,
                }
            ),
            _at("generators.g.ct2.family", "Field required"),
            id="ct2-without-family",
        ),
        pytest.param(
            _document(generator=_ct2(model_dir="")),
            _at(
                "generators.g.ct2.model_dir",
                "Value error, model_dir must not be empty or blank",
            ),
            id="empty-model-dir",
        ),
        pytest.param(
            _document(generator=_ct2(spm_model="  ")),
            _at(
                "generators.g.ct2.spm_model",
                "Value error, spm_model must not be empty or blank",
            ),
            id="blank-spm-model",
        ),
        pytest.param(
            _document(generator=_ct2(family="marian")),
            _at("generators.g.ct2.family", "Input should be 'nllb' or 'madlad'"),
            id="unknown-family",
        ),
        pytest.param(
            _document(generator=_ct2(device="tpu")),
            _at("generators.g.ct2.device", "Input should be 'cpu', 'cuda' or 'auto'"),
            id="unknown-device",
        ),
        pytest.param(
            _document(generator=_ct2(beam_size=0)),
            _at("generators.g.ct2.beam_size", "Input should be greater than 0"),
            id="zero-beam-size",
        ),
        pytest.param(
            _document(generator=_ct2(batch_size=0)),
            _at("generators.g.ct2.batch_size", "Input should be greater than 0"),
            id="zero-ct2-batch-size",
        ),
        pytest.param(
            _document(generator=_ct2(n=0)),
            _at("generators.g.ct2.n", "Input should be greater than or equal to 1"),
            id="ct2-zero-n",
        ),
        pytest.param(
            _document(generator=_ct2(n=5, beam_size=4)),
            _at(
                "generators.g.ct2",
                r"Value error, n \(5\) cannot be greater than beam_size \(4\)",
            ),
            id="ct2-more-hypotheses-than-beam",
        ),
        pytest.param(
            _document(generator=_ct2(n=64)),
            _at(
                "generators.g.ct2",
                r"Value error, n \(64\) cannot be greater than beam_size \(4\)",
            ),
            id="ct2-default-beam-with-large-n",
        ),
        pytest.param(
            _document(job=_job(kind="rt")),
            _at("jobs.j.kind", "Input should be 'bt' or 'ft'"),
            id="unknown-job-kind",
        ),
        pytest.param(
            _document(job=_job(shard_size=0)),
            _at("jobs.j.shard_size", "Input should be greater than 0"),
            id="zero-shard-size",
        ),
        pytest.param(
            _document(job=_job(shard_size="512")),
            _at("jobs.j.shard_size", "Input should be a valid integer"),
            id="quoted-shard-size",
        ),
        pytest.param(
            _document(job=_job(generator="")),
            _at("jobs.j.generator", "String should have at least 1 character"),
            id="empty-job-generator",
        ),
        pytest.param(
            _document(job=_job(mono_lang="xxx_Latn", other_lang="khk_Cyrl")),
            _at(
                "jobs.j.mono_lang",
                "Value error, 'xxx_Latn' is not a supported job language",
            ),
            id="unknown-mono-lang",
        ),
        pytest.param(
            _document(job=_job(mono_lang="khk_Cyrl", other_lang="xxx_Latn")),
            _at(
                "jobs.j.other_lang",
                "Value error, 'xxx_Latn' is not a supported job language",
            ),
            id="unknown-other-lang",
        ),
        pytest.param(
            _document(job=_job(mono_lang="khk", other_lang="khk_Cyrl")),
            _at(
                "jobs.j.mono_lang", "Value error, 'khk' is not a supported job language"
            ),
            id="language-without-script",
        ),
        pytest.param(
            _document(job=_job(mono_lang="mn_Mong", other_lang="khk_Cyrl")),
            _at(
                "jobs.j.mono_lang",
                "Value error, 'mn_Mong' is not a supported job language",
            ),
            id="mongolian-script-mono-lang",
        ),
        pytest.param(
            _document(job=_job(mono_lang="khk_Cyrl", other_lang="mn_Mong")),
            _at(
                "jobs.j.other_lang",
                "Value error, 'mn_Mong' is not a supported job language",
            ),
            id="mongolian-script-other-lang",
        ),
        pytest.param(
            _document(job=_job(mono_lang="eng_Latn", other_lang="rus_Cyrl")),
            _at(
                "jobs.j",
                "Value error, one of mono_lang and other_lang must be khk_Cyrl",
            ),
            id="neither-language-is-khk-cyrl",
        ),
        pytest.param(
            _document(job=_job(mono_lang="khk_Cyrl", other_lang="khk_Cyrl")),
            _at("jobs.j", "Value error, mono_lang and other_lang must differ"),
            id="both-languages-are-khk-cyrl",
        ),
        pytest.param(
            _document(job=_job(generator="nope")),
            r"SynthConfig\n +Value error, job 'j' names unknown generator 'nope'",
            id="unknown-generator",
        ),
        pytest.param(
            {
                "generators": {"g": {"kind": "vllm"}},
                "jobs": {"j": _job(), "k": _job(generator="vllm_default")},
            },
            r"SynthConfig\n +Value error, job 'k' names unknown generator 'vllm_default'",
            id="supplied-jobs-may-not-name-a-replaced-generator",
        ),
        pytest.param(
            {"selection": {"min_script_share": 0}},
            _at("selection.min_script_share", "Input should be greater than 0"),
            id="zero-min-script-share",
        ),
        pytest.param(
            {"selection": {"min_script_share": 1.5}},
            _at(
                "selection.min_script_share", "Input should be less than or equal to 1"
            ),
            id="min-script-share-above-1",
        ),
        pytest.param(
            {"selection": {"min_script_share": "0.9"}},
            _at("selection.min_script_share", "Input should be a valid number"),
            id="quoted-min-script-share",
        ),
        pytest.param(
            {"selection": {"min_script_share": float("nan")}},
            _at("selection.min_script_share", _MUST_BE_FINITE),
            id="nan-min-script-share",
        ),
        pytest.param(
            {"selection": {"min_script_share": float("inf")}},
            _at("selection.min_script_share", _MUST_BE_FINITE),
            id="infinite-min-script-share",
        ),
        pytest.param(
            {"selection": {"mbr_utility": "comet"}},
            _at("selection.mbr_utility", "Input should be 'chrf'"),
            id="unknown-mbr-utility",
        ),
        pytest.param(["jobs"], "top level must be a mapping", id="top-level-list"),
        pytest.param("", "top level must be a mapping", id="empty-file"),
    ],
)
def test_malformed_config_raises_config_error(
    tmp_path: Path, document: Any, message_fragment: str
):
    path = _write(tmp_path, document)

    # (?s): pydantic puts the field name and its message on separate lines.
    with pytest.raises(ConfigError, match=f"(?s){message_fragment}") as excinfo:
        load_synth_config(path)

    assert path.name in str(excinfo.value)


def test_a_job_with_a_default_generator_name_loads_when_the_defaults_are_kept(
    tmp_path: Path,
):
    document = {"jobs": {"j": _job(generator="ct2_nllb_default")}}

    config = load_synth_config(_write(tmp_path, document))

    assert config.jobs["j"].generator == "ct2_nllb_default"


def test_field_violation_chains_the_validation_error(tmp_path: Path):
    path = _write(tmp_path, {"selection": {"min_script_share": 0}})

    with pytest.raises(ConfigError) as excinfo:
        load_synth_config(path)

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_yaml_syntax_error_raises_config_error_with_cause(tmp_path: Path):
    path = _write(tmp_path, "generators: [vllm\n")

    with pytest.raises(ConfigError, match="synth.yaml") as excinfo:
        load_synth_config(path)

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_non_utf8_config_raises_config_error_with_cause(tmp_path: Path):
    path = tmp_path / "synth.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="synth.yaml") as excinfo:
        load_synth_config(path)

    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_unreadable_config_raises_config_error_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write(tmp_path, {})

    def deny(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(ConfigError, match="synth.yaml") as excinfo:
        load_synth_config(path)

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found") as excinfo:
        load_synth_config(tmp_path / "nonexistent.yaml")

    assert "nonexistent.yaml" in str(excinfo.value)
