"""`rinchen eval robustness`: what its manifest says and the order in which it hashes, reads and
writes (D12, Task 7). The manifest (stage `eval_robustness`) names every input with the digest
taken before it was read, every output with the digest of the bytes on disk and the measurements,
and is written last.

The suite and the hypotheses are the golden files, the export a fabricated one that
`verify_export` vouches for, the engine and the converter fakes
(`tests.support.robust_cli`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import eval_robustness
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from tests.support.engines import FakeConverter
from tests.support.robust_cli import (
    CONVERTER_FILES,
    RobustWorkspace,
    echo_answers,
    install,
    invoke,
    make_workspace,
    sha256_of,
)
from tests.support.robust_golden import GOLDEN_HYPOTHESES_SHA256, GOLDEN_SUITE_SHA256
from tests.support.robust_suite_documents import document, encode

PACKAGES = ("ctranslate2", "sentencepiece", "sacrebleu", "regex")
GOLDEN_METRICS = {
    "numbers_acc": 0.5,
    "placeholder_acc": 0.4,
    "term_acc": 0.75,
    "script_purity": 0.75,
    "entity_acc": 0.5714285714285714,
    "mong_source_retention": 0.5017570743156547,
    "rows.numbers": 8.0,
    "rows.placeholders": 6.0,
    "rows.entities": 4.0,
    "rows.terms": 3.0,
    "rows.script": 6.0,
    "rows.mong": 3.0,
    # What each value is a share of, in the unit of its section: rows, or mentions for the
    # entities and the terms (hand-counted from the golden suite).
    "scored.numbers": 6.0,
    "scored.placeholders": 5.0,
    "scored.entities": 7.0,
    "scored.terms": 4.0,
    "scored.script": 5.0,
    "scored.mong": 3.0,
    "excluded.numbers": 2.0,
    "excluded.placeholders": 1.0,
    "excluded.entities": 0.0,
    "excluded.terms": 1.0,
    "excluded.script": 1.0,
    "excluded.mong": 0.0,
}


@pytest.fixture
def workspace(tmp_path: Path) -> RobustWorkspace:
    return make_workspace(tmp_path)


def manifest_of(workspace: RobustWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def entry(path: Path, *, rows: int | None = None) -> ManifestFile:
    """The entry a manifest holds for the file `path` as it is on disk."""
    return ManifestFile(
        path=str(path), sha256=sha256_of(path), bytes=path.stat().st_size, rows=rows
    )


def canonical_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- a hypotheses run --------------------------------------------------------------------------


def test_the_manifest_of_a_hypotheses_run_names_the_stage_the_inputs_and_the_outputs(
    workspace: RobustWorkspace,
) -> None:
    assert invoke(workspace.hypotheses_args("--system", "golden")).exit_code == 0

    manifest = manifest_of(workspace)

    assert manifest.stage == "eval_robustness"
    assert manifest.seed == -1
    assert [(f.path, f.sha256, f.rows) for f in manifest.inputs] == [
        (str(workspace.suite), GOLDEN_SUITE_SHA256, None),
        (str(workspace.hypotheses), GOLDEN_HYPOTHESES_SHA256, None),
    ]
    assert manifest.outputs == [entry(workspace.result), entry(workspace.report)]
    assert manifest.created != ""
    assert manifest.code_version != ""


def test_the_manifest_records_the_four_tools_and_their_installed_versions(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    assert manifest_of(workspace).tool_versions == {
        package: importlib.metadata.version(package) for package in PACKAGES
    }


def test_the_status_of_a_hypotheses_run_says_ok_and_the_mode(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    assert manifest_of(workspace).status == {"robustness": "ok", "mode": "hypotheses"}


def test_the_metrics_are_the_six_values_and_the_rows_of_every_section(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    assert manifest_of(workspace).metrics == GOLDEN_METRICS


def test_a_value_that_is_null_is_not_a_metric_and_an_absent_section_has_no_rows(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    doc = document()
    doc["sections"] = {"numbers": doc["sections"]["numbers"]}
    doc["sections"]["numbers"][0]["source"] = "It costs a lot."
    suite = tmp_path / "one.json"
    suite.write_bytes(encode(doc))
    hypotheses = tmp_path / "one.jsonl"
    hypotheses.write_text(
        '{"arm": "main", "id": "n1", "section": "numbers", "text": "x"}\n',
        encoding="utf-8",
    )

    result = invoke(
        workspace.hypotheses_args("--system", "x", suite=suite, hypotheses=hypotheses)
    )

    assert result.exit_code == 0, result.output
    assert manifest_of(workspace).metrics == {
        "rows.numbers": 1.0,
        "scored.numbers": 0.0,
        "excluded.numbers": 1.0,
    }


def test_the_config_digest_is_that_of_the_canonical_json_of_the_options(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    assert manifest_of(workspace).config_sha256 == canonical_sha256(
        '{"mode":"hypotheses"}'
    )


def test_two_runs_differ_in_the_manifest_only_by_the_time_it_was_made(
    workspace: RobustWorkspace, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    invoke(workspace.hypotheses_args("--system", "golden"))
    invoke(workspace.hypotheses_args("--system", "golden", out_dir=again))

    first = manifest_of(workspace).model_dump()
    second = store.read(again)
    assert second is not None
    other = second.model_dump()
    # The outputs are named by path: the second run wrote them elsewhere.
    for part in (first, other):
        part.pop("created")
        part["outputs"] = [
            {**item, "path": Path(item["path"]).name} for item in part["outputs"]
        ]

    assert first == other


def test_the_outputs_the_manifest_lists_are_the_bytes_on_disk(
    workspace: RobustWorkspace,
) -> None:
    invoke(workspace.hypotheses_args("--system", "golden"))

    for item in manifest_of(workspace).outputs:
        assert hashlib.sha256(Path(item.path).read_bytes()).hexdigest() == item.sha256
        assert item.bytes == Path(item.path).stat().st_size


# --- a model run -------------------------------------------------------------------------------


def export_entries(workspace: RobustWorkspace) -> list[ManifestFile]:
    """The five files of the export in the order of `EXPORT_FILES` and its manifest, with
    the digests the export's manifest vouches for."""
    export = workspace.export
    files = [
        ManifestFile(
            path=str(export.directory / name),
            sha256=digest,
            bytes=(export.directory / name).stat().st_size,
        )
        for name, digest in export.sha256.items()
    ]
    manifest = export.directory / "manifest.json"
    files.append(
        ManifestFile(
            path=str(manifest),
            sha256=export.manifest_sha256,
            bytes=manifest.stat().st_size,
        )
    )
    return files


def test_the_manifest_of_a_model_run_names_the_suite_and_the_export_with_its_digests(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    assert invoke(workspace.model_args()).exit_code == 0

    manifest = manifest_of(workspace)
    assert manifest.inputs == [
        ManifestFile(
            path=str(workspace.suite),
            sha256=GOLDEN_SUITE_SHA256,
            bytes=workspace.suite.stat().st_size,
        ),
        *export_entries(workspace),
    ]
    assert [name.path.rsplit("/", 1)[1] for name in manifest.inputs[1:]] == [
        "model.bin",
        "shared_vocabulary.json",
        "config.json",
        "spm.model",
        "rinchen.json",
        "manifest.json",
    ]


def test_the_outputs_of_a_model_run_are_the_hypotheses_with_their_rows_the_result_and_the_report(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    invoke(workspace.model_args())

    assert manifest_of(workspace).outputs == [
        entry(workspace.hypotheses_out, rows=33),
        entry(workspace.result),
        entry(workspace.report),
    ]


def test_the_status_of_a_model_run_says_the_mode_the_script_mask_and_the_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path, quantization="int8")
    install(monkeypatch, workspace.engine())

    invoke(workspace.model_args("--script-mask", "off"))

    assert manifest_of(workspace).status == {
        "robustness": "ok",
        "mode": "model",
        "script_mask": "off",
        "quantization": "int8",
    }


def test_the_config_digest_of_a_model_run_follows_every_option_that_decides_the_numbers(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())

    invoke(workspace.model_args())

    default = (
        '{"beam_size":4,"converter":false,"device":"cpu","max_decoding_length":256,'
        '"mode":"model","script_mask":"retry","threads":1}'
    )
    assert manifest_of(workspace).config_sha256 == canonical_sha256(default)
    seen = {manifest_of(workspace).config_sha256}
    for option in (
        ("--beam-size", "2"),
        ("--max-decoding-length", "64"),
        ("--script-mask", "always"),
        ("--device", "cuda"),
        ("--threads", "2"),
    ):
        install(monkeypatch, workspace.engine())
        invoke(workspace.model_args(*option))
        seen.add(manifest_of(workspace).config_sha256)
    assert len(seen) == 6


def test_the_system_name_does_not_change_the_config_digest(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    invoke(workspace.model_args())
    first = manifest_of(workspace).config_sha256

    install(monkeypatch, workspace.engine())
    invoke(workspace.model_args("--system", "other"))

    assert manifest_of(workspace).config_sha256 == first


def test_the_manifest_of_a_converter_run_names_the_four_files_of_the_converter(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        workspace.engine(answer=echo_answers),
        converter=FakeConverter(),
    )

    assert invoke(workspace.model_args(*workspace.converter_args())).exit_code == 0

    manifest = manifest_of(workspace)
    converter_inputs = manifest.inputs[-4:]
    assert converter_inputs == [
        *(entry(workspace.converter_ct2 / name) for name in CONVERTER_FILES),
        entry(workspace.converter_spm),
    ]
    assert len(manifest.inputs) == 1 + 6 + 4
    options = json.loads(workspace.result.read_text(encoding="utf-8"))["system"][
        "options"
    ]
    assert options["converter"] is True
    assert manifest.config_sha256 == canonical_sha256(
        '{"beam_size":4,"converter":true,"device":"cpu","max_decoding_length":256,'
        '"mode":"model","script_mask":"retry","threads":1}'
    )
    assert manifest.outputs[0].rows == 36


# --- hash, then read, then write ---------------------------------------------------------------


def record_io(
    monkeypatch: pytest.MonkeyPatch,
    files: set[Path],
    events: list[str] | None = None,
) -> list[str]:
    """Record, in order, `hash:<name>` when the command hashes one of `files` and
    `read:<name>` when it opens one for anything but that hash, in `events` (a list the
    engine and the seams write to too, when one is given)."""
    events = [] if events is None else events
    hashing = False
    real_hash = eval_robustness.hash_input
    real_open = Path.open

    def hash_input(path: Path, **kwargs: Any) -> Any:
        nonlocal hashing
        if Path(path) in files:
            events.append(f"hash:{Path(path).name}")
        hashing = True
        try:
            return real_hash(path, **kwargs)
        finally:
            hashing = False

    def opening(self: Path, *args: Any, **kwargs: Any) -> Any:
        if not hashing and self in files:
            events.append(f"read:{self.name}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(eval_robustness, "hash_input", hash_input)
    monkeypatch.setattr(Path, "open", opening)
    return events


def test_the_suite_and_the_hypotheses_are_each_hashed_before_they_are_read_and_read_once(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = record_io(monkeypatch, {workspace.suite, workspace.hypotheses})

    assert invoke(workspace.hypotheses_args("--system", "golden")).exit_code == 0

    assert events == [
        f"hash:{workspace.suite.name}",
        f"hash:{workspace.hypotheses.name}",
        f"read:{workspace.suite.name}",
        f"read:{workspace.hypotheses.name}",
    ]


def test_a_model_run_hashes_the_suite_and_the_converter_before_it_reads_the_suite(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        workspace.suite,
        workspace.converter_spm,
        workspace.converter_ct2 / "model.bin",
    }
    install(
        monkeypatch, workspace.engine(answer=echo_answers), converter=FakeConverter()
    )
    events = record_io(monkeypatch, files)

    assert invoke(workspace.model_args(*workspace.converter_args())).exit_code == 0

    assert events.index(f"hash:{workspace.suite.name}") < events.index(
        f"read:{workspace.suite.name}"
    )
    assert events.count(f"read:{workspace.suite.name}") == 1
    assert f"hash:{workspace.converter_spm.name}" in events
    # Only the converter's `model.bin`: `verify_export` hashes the export's own.
    assert events.count("hash:model.bin") == 1


def test_the_inputs_are_hashed_and_read_before_the_engine_is_opened_and_asked(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(monkeypatch, workspace.engine())
    events = record_io(monkeypatch, {workspace.suite}, seams.events)

    assert invoke(workspace.model_args()).exit_code == 0

    assert events == [
        f"hash:{workspace.suite.name}",
        f"read:{workspace.suite.name}",
        "open",
        "translate",
        "close",
    ]
