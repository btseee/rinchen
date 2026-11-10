"""The systems `rinchen eval run` builds (D13, Task 7, spec T5): the engine seam, the baselines of
D6's generator configuration with the evaluation's own sampling settings and the vocabulary guard
(F14: D6's `CT2Translator` is silent about a family whose language tokens are not in the
vocabulary, so the toy export's vocabulary is the negative control), the file system and the
engine system.

Every model is a fake behind the seams of `rinchen.cli.eval_systems`: no CTranslate2 model, no
server and no socket.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import eval_systems
from rinchen.cli.eval_robustness_plan import RunOptions
from rinchen.config.synth import Ct2Config, LlamaCppConfig, VllmConfig
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.eval.outputs import OutputKey
from rinchen.eval.systems import SuiteRequest
from rinchen.infer.engine_types import DecodeOptions, LoadOptions
from tests.support.eval_cli import (
    MARKER,
    SPM_BYTES,
    EvalWorkspace,
    alpha_engine,
    family_tokens,
    install,
    make_workspace,
    write_family_directory,
)
from tests.support.eval_fakes import (
    FakeCandidateTranslator,
    engine_info,
    golden_outputs,
    golden_requests,
    golden_suite,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

ALL_LANGUAGES = (
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
    "khk_Cyrl",
)


@pytest.fixture
def workspace(tmp_path: Path) -> EvalWorkspace:
    return make_workspace(tmp_path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the seam of the engine -----------------------------------------------------------------------


def test_open_engine_is_engine_open_over_the_directory_and_the_load_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Any, ...]] = []
    sentinel = object()

    def fake_open(directory: Path, load: LoadOptions, **keywords: Any) -> object:
        calls.append((directory, load, keywords))
        return sentinel

    monkeypatch.setattr("rinchen.infer.engine.Engine.open", fake_open)
    load = LoadOptions(device="cpu", intra_threads=3)

    opened = eval_systems.open_engine(tmp_path, load)

    assert opened is sentinel
    assert calls == [(tmp_path, load, {})]


# --- the vocabulary guard (F14) -------------------------------------------------------------------


@pytest.mark.parametrize("family", ["nllb", "madlad"])
def test_a_vocabulary_that_holds_the_family_tokens_of_every_language_passes(
    tmp_path: Path, family: str
) -> None:
    directory = write_family_directory(tmp_path / "m", family_tokens(family))

    eval_systems.vocabulary_guard(directory, family, ALL_LANGUAGES)


@pytest.mark.parametrize("family", ["nllb", "madlad"])
def test_the_vocabulary_of_the_toy_export_is_refused_naming_the_family_and_a_language(
    workspace: EvalWorkspace, family: str
) -> None:
    # The negative control: a real vocabulary that has `<2eng_Latn>` and no NLLB or MADLAD token.
    with pytest.raises(ConfigError) as refused:
        eval_systems.vocabulary_guard(
            workspace.export.directory, family, ("khk_Cyrl", "eng_Latn")
        )

    message = str(refused.value)
    assert family in message
    assert "khk_Cyrl" in message
    assert "eng_Latn" not in message, "only the first language that is missing is named"


def test_a_vocabulary_that_lacks_one_language_names_that_language_only(
    tmp_path: Path,
) -> None:
    tokens = [token for token in family_tokens("nllb") if token != "khk_Cyrl"]
    directory = write_family_directory(tmp_path / "m", tokens)

    with pytest.raises(ConfigError) as refused:
        eval_systems.vocabulary_guard(directory, "nllb", ALL_LANGUAGES)

    assert "khk_Cyrl" in str(refused.value)
    assert "rus_Cyrl" not in str(refused.value)


def test_madlad_needs_the_target_tag_of_every_language_and_the_message_says_which(
    tmp_path: Path,
) -> None:
    tokens = [token for token in family_tokens("madlad") if token != "<2mn>"]
    directory = write_family_directory(tmp_path / "m", tokens)

    with pytest.raises(ConfigError) as refused:
        eval_systems.vocabulary_guard(directory, "madlad", ("eng_Latn", "khk_Cyrl"))

    message = str(refused.value)
    assert "madlad" in message
    assert "khk_Cyrl" in message
    assert "<2mn>" in message


def test_the_message_of_a_refusal_holds_no_entry_of_the_vocabulary(
    tmp_path: Path,
) -> None:
    directory = write_family_directory(tmp_path / "m", [MARKER, "<unk>"])

    with pytest.raises(ConfigError) as refused:
        eval_systems.vocabulary_guard(directory, "nllb", ("khk_Cyrl",))

    assert MARKER not in str(refused.value)


def test_a_directory_without_the_vocabulary_file_is_a_config_error(
    tmp_path: Path,
) -> None:
    directory = write_family_directory(tmp_path / "m", family_tokens("nllb"))
    (directory / "shared_vocabulary.json").unlink()

    with pytest.raises(ConfigError, match="shared_vocabulary.json"):
        eval_systems.vocabulary_guard(directory, "nllb", ("khk_Cyrl",))


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"khk_Cyrl": 1}', b'["khk_Cyrl", 7]', b'"khk_Cyrl"', b"[]x"],
)
def test_a_vocabulary_that_is_not_a_list_of_strings_is_a_config_error(
    tmp_path: Path, content: bytes
) -> None:
    directory = write_family_directory(tmp_path / "m", family_tokens("nllb"))
    (directory / "shared_vocabulary.json").write_bytes(content)

    with pytest.raises(ConfigError, match="list of strings"):
        eval_systems.vocabulary_guard(directory, "nllb", ("khk_Cyrl",))


def test_the_vocabulary_is_hashed_first_and_parsed_from_the_bytes_that_were_hashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = write_family_directory(tmp_path / "m", family_tokens("nllb"))
    file = directory / "shared_vocabulary.json"
    real = eval_systems.verified_content
    events: list[str] = []

    def swapping(path: Path, digest: str) -> bytes:
        events.append("read")
        file.write_text(json.dumps([MARKER]), "utf-8")
        return real(path, digest)

    monkeypatch.setattr(eval_systems, "verified_content", swapping)

    with pytest.raises(ConfigError, match="changed after it was hashed"):
        eval_systems.vocabulary_guard(directory, "nllb", ("khk_Cyrl",))
    assert events == ["read"]


# --- the generator system ---------------------------------------------------------------------------


def config_bytes(workspace: EvalWorkspace) -> bytes:
    return workspace.generators.read_bytes()


def test_an_unknown_generator_is_a_config_error_that_lists_the_configured_ones(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    with pytest.raises(ConfigError) as refused:
        eval_systems.generator_system(config_bytes(workspace), "nope")

    message = str(refused.value)
    for name in ("llamacpp_toy", "madlad_toy", "nllb_toy", "vllm_toy"):
        assert name in message
    assert seams.builds == []


def test_a_ct2_baseline_runs_with_one_hypothesis_and_the_record_says_so(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    system = eval_systems.generator_system(config_bytes(workspace), "nllb_toy")

    (built,) = seams.builds
    assert isinstance(built, Ct2Config)
    assert (built.n, built.beam_size, built.max_tokens, built.device) == (
        1,
        4,
        128,
        "cpu",
    )
    assert built.family == "nllb"
    record = system.record
    assert (record.kind, record.name) == ("baseline", "nllb_toy")
    assert (record.model_sha256, record.manifest_sha256, record.outputs_sha256) == (
        None,
        None,
        None,
    )
    assert dict(record.options) == {
        "generator": "nllb_toy",
        "kind": "ct2",
        "family": "nllb",
        "device": "cpu",
        "beam_size": 4,
        "batch_size": 16,
        "n": 1,
        "max_tokens": 128,
    }


@pytest.mark.parametrize(
    ("name", "kind", "config_class", "extra"),
    [
        ("vllm_toy", "vllm", VllmConfig, {"model": "Hy-MT2-7B", "max_tokens": 200}),
        (
            "llamacpp_toy",
            "llamacpp",
            LlamaCppConfig,
            {"model": "Hy-MT2-7B", "max_tokens": 256},
        ),
    ],
)
def test_a_server_baseline_runs_greedy_with_one_candidate_and_the_record_says_so(
    workspace: EvalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kind: str,
    config_class: type,
    extra: dict[str, Any],
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    system = eval_systems.generator_system(config_bytes(workspace), name)

    (built,) = seams.builds
    assert isinstance(built, config_class)
    assert (built.n, built.temperature, built.top_p) == (1, 0.0, 1.0)
    assert dict(system.record.options) == {
        "generator": name,
        "kind": kind,
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        **extra,
    }


def test_the_options_record_holds_no_path_and_no_address(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())

    texts = [
        json.dumps(
            dict(
                eval_systems.generator_system(
                    config_bytes(workspace), name
                ).record.options
            )
        )
        for name in ("nllb_toy", "vllm_toy")
    ]

    for text in texts:
        assert str(workspace.root) not in text
        assert "127.0.0.1" not in text
        assert "http" not in text


def test_the_system_name_is_the_generators_unless_one_is_given(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())

    default = eval_systems.generator_system(config_bytes(workspace), "vllm_toy")
    named = eval_systems.generator_system(
        config_bytes(workspace), "vllm_toy", system_name="hy-mt2"
    )
    ct2_default = eval_systems.generator_system(config_bytes(workspace), "nllb_toy")
    ct2_named = eval_systems.generator_system(
        config_bytes(workspace), "nllb_toy", system_name="nllb-3.3b"
    )

    assert (default.record.name, named.record.name) == ("vllm_toy", "hy-mt2")
    assert named.record.options["generator"] == "vllm_toy"
    assert (ct2_default.record.name, ct2_named.record.name) == ("nllb_toy", "nllb-3.3b")
    assert ct2_named.record.options["generator"] == "nllb_toy"


def test_a_ct2_baseline_on_auto_is_a_config_error_and_nothing_is_built(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    with pytest.raises(ConfigError) as refused:
        eval_systems.generator_system(config_bytes(workspace), "nllb_auto")

    assert "auto" in str(refused.value)
    assert "nllb_auto" in str(refused.value)
    assert seams.builds == []


def test_a_family_whose_tokens_are_missing_is_refused_before_the_translator_is_built(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())

    with pytest.raises(ConfigError) as refused:
        eval_systems.generator_system(config_bytes(workspace), "nllb_bad_vocabulary")

    assert "nllb" in str(refused.value)
    assert seams.builds == []
    assert seams.events == []


def test_the_vocabulary_guard_is_asked_about_the_nine_languages_of_the_family(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())
    asked: list[tuple[str, tuple[str, ...]]] = []
    real = eval_systems.vocabulary_guard

    def recording(
        directory: Path, family: str, languages: Any, **keywords: Any
    ) -> None:
        asked.append((family, tuple(languages)))
        real(directory, family, languages, **keywords)

    monkeypatch.setattr(eval_systems, "vocabulary_guard", recording)

    eval_systems.generator_system(config_bytes(workspace), "madlad_toy")

    assert [family for family, _ in asked] == ["madlad"]
    assert sorted(asked[0][1]) == sorted(ALL_LANGUAGES)


def test_a_ct2_baseline_lists_the_files_it_hashed_before_it_loaded_anything(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())
    directory = workspace.root / "nllb"

    system = eval_systems.generator_system(config_bytes(workspace), "nllb_toy")

    entries = {Path(entry.path).name: entry for entry in system.inputs}
    assert sorted(entries) == [
        "config.json",
        "model.bin",
        "sentencepiece.bpe.model",
        "shared_vocabulary.json",
    ]
    for name, entry in entries.items():
        assert entry.path == str(directory / name)
        assert entry.sha256 == sha(directory / name)
        assert entry.bytes == (directory / name).stat().st_size
    assert (
        entries["sentencepiece.bpe.model"].sha256
        == hashlib.sha256(SPM_BYTES).hexdigest()
    )
    assert seams.events == ["build"]


def test_a_server_baseline_has_no_file_to_hash(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())

    assert (
        eval_systems.generator_system(config_bytes(workspace), "vllm_toy").inputs == ()
    )


@pytest.mark.parametrize(
    "missing", ["model.bin", "config.json", "sentencepiece.bpe.model"]
)
def test_a_baseline_directory_that_lacks_a_file_is_a_config_error_naming_it(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    seams = install(monkeypatch, None, FakeCandidateTranslator())
    (workspace.root / "nllb" / missing).unlink()

    with pytest.raises(ConfigError, match=missing):
        eval_systems.generator_system(config_bytes(workspace), "nllb_toy")

    assert seams.builds == []


def test_a_ct2_baseline_knows_the_nine_languages_and_a_server_only_khalkha_pairs(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())
    ct2 = eval_systems.generator_system(config_bytes(workspace), "nllb_toy")
    server = eval_systems.generator_system(config_bytes(workspace), "vllm_toy")

    assert ct2.verdict("khk_Cyrl-eng_Latn").state == "run"
    assert ct2.verdict("eng_Latn-rus_Cyrl").state == "run"
    assert ct2.verdict("mn_Mong-eng_Latn").state == "unsupported_language"
    assert server.verdict("khk_Cyrl-eng_Latn").state == "run"
    assert server.verdict("tur_Latn-khk_Cyrl").state == "run"
    # D6's prompts exist only for Khalkha to and from a partner: any other pair would raise
    # in the middle of the run, so it is a state of the direction instead.
    refused = server.verdict("eng_Latn-rus_Cyrl")
    assert refused.state == "unsupported_language"
    assert refused.reason
    assert server.verdict("mn_Mong-eng_Latn").state == "unsupported_language"


def test_a_generator_system_translates_per_direction_through_the_built_translator(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    translator = FakeCandidateTranslator()
    install(monkeypatch, None, translator)
    system = eval_systems.generator_system(config_bytes(workspace), "nllb_toy")
    requests = golden_requests()

    outputs = system.translate(requests)

    assert [(call.src, call.tgt, len(call.texts)) for call in translator.calls] == [
        ("khk_Cyrl", "eng_Latn", 10),
        ("eng_Latn", "khk_Cyrl", 10),
    ]
    assert [output.text for output in outputs] == [r.text for r in requests]


@pytest.mark.parametrize(
    "content",
    [
        b"\xff\xfe",
        b"generators: [",
        b"- a list",
        b"generators: {x: {kind: nope}}",
        b"extra: 1",
    ],
)
def test_a_generators_config_that_is_not_usable_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())

    with pytest.raises(ConfigError, match="generators config"):
        eval_systems.generator_system(content, "x")


def test_the_message_about_a_generators_config_quotes_no_value_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, None, FakeCandidateTranslator())
    content = (
        f"generators:\n  x:\n    kind: vllm\n    model: {MARKER}\n    n: {MARKER}\n"
    ).encode()

    with pytest.raises(ConfigError) as refused:
        eval_systems.generator_system(content, "x")

    assert MARKER not in str(refused.value)


def test_an_error_of_the_translator_is_not_caught_here(
    workspace: EvalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, None, None, build_error=UpstreamError("down"))

    with pytest.raises(UpstreamError):
        eval_systems.generator_system(config_bytes(workspace), "vllm_toy")


# --- the engine system -----------------------------------------------------------------------------


def test_the_engine_system_names_the_model_and_runs_only_its_trained_directions(
    workspace: EvalWorkspace,
) -> None:
    engine = alpha_engine(engine_info(name="toy"))
    options = RunOptions(2, 64, "always", "cpu", 3)

    system = eval_systems.engine_system(engine, options, "toy")

    record = system.record
    assert (record.kind, record.name) == ("rinchen", "toy")
    assert record.model_sha256 == engine.info.model_sha256
    assert record.manifest_sha256 == engine.info.manifest_sha256
    assert dict(record.options) == {
        "beam_size": 2,
        "max_decoding_length": 64,
        "script_mask": "always",
        "device": "cpu",
        "threads": 3,
    }
    assert system.verdict("khk_Cyrl-eng_Latn").state == "run"
    refused = system.verdict("eng_Latn-rus_Cyrl")
    assert refused.state == "not_trained"
    assert refused.reason


def test_the_engine_system_makes_one_call_with_the_decode_options(
    workspace: EvalWorkspace,
) -> None:
    engine = alpha_engine()
    system = eval_systems.engine_system(
        engine, RunOptions(2, 64, "always", "cpu", 1), "toy"
    )
    requests = golden_requests()

    outputs = system.translate(requests)

    (call,) = engine.calls
    assert [row.text for row in call.rows] == [r.text for r in requests]
    assert call.options == DecodeOptions(
        beam_size=2, max_decoding_length=64, script_mask="always"
    )
    assert [output.text for output in outputs] == [
        output.text for output in golden_outputs("alpha").values()
    ]


# --- the file system ---------------------------------------------------------------------------------


def test_a_file_system_is_read_from_the_bytes_that_were_hashed(
    workspace: EvalWorkspace,
) -> None:
    data = workspace.alpha.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    system = eval_systems.file_system(data, digest, golden_suite(), "alpha")

    assert (system.record.kind, system.record.name) == ("hypotheses", "alpha")
    assert system.record.outputs_sha256 == digest
    assert system.verdict("khk_Cyrl-eng_Latn").state == "run"
    requests = golden_requests()
    assert [o.text for o in system.translate(requests)] == [
        o.text for o in golden_outputs("alpha").values()
    ]


def test_a_file_that_holds_one_direction_leaves_the_other_not_in_the_file(
    tmp_path: Path,
) -> None:
    from tests.support.eval_cli import FIRST, SECOND, one_direction_file

    data = one_direction_file(tmp_path / "one.jsonl").read_bytes()

    system = eval_systems.file_system(
        data, hashlib.sha256(data).hexdigest(), golden_suite(), "one"
    )

    assert system.verdict(FIRST).state == "run"
    assert system.verdict(SECOND).state == "not_in_file"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lines: lines[:-1],
        lambda lines: [*lines, lines[0].replace('"g01"', '"g99"')],
        lambda lines: [*lines, "not json"],
    ],
    ids=["missing-row", "unknown-id", "not-json"],
)
def test_a_file_that_is_not_outputs_of_the_suite_is_a_data_error(
    workspace: EvalWorkspace, mutate: Any
) -> None:
    lines = workspace.alpha.read_text("utf-8").splitlines()
    data = ("\n".join(mutate(lines)) + "\n").encode()

    with pytest.raises(DataError):
        eval_systems.file_system(
            data, hashlib.sha256(data).hexdigest(), golden_suite(), "x"
        )


def test_the_golden_outputs_key_every_request_of_both_directions() -> None:
    # A guard of the helpers above: `golden_requests` and `golden_outputs` agree key for key.
    requests = golden_requests()
    assert [OutputKey(r.direction, r.id) for r in requests] == list(
        golden_outputs("alpha")
    )
    assert all(isinstance(r, SuiteRequest) for r in requests)
