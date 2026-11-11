"""The release record and the checks of `rinchen.export.release` (D13, Task 9, spec T17): the nine
names a release uploads, the repository id and system name rules, the binding of a report to an
export by digests and the document `release.json`.

The record is a function of the export's record, the report and the facts of the staged files:
sorted keys, two-space indent, one final LF, no timestamp and no path (it is uploaded). The binding
is the one that stops a card from describing another model than the one it ships with: the report
must hold a system of that name of kind `rinchen` whose `model_sha256` and `manifest_sha256` are the
export's, and at least one result of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.export.model_card import FileFacts
from rinchen.export.release import (
    RELEASE_SCHEMA,
    STAGE_ONLY,
    UPLOAD_NAMES,
    check_repo_id,
    check_system_name,
    release_record,
    release_to_bytes,
    require_report_for_export,
)
from rinchen.export.verified import verify_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    REPO_ID,
    SYSTEM,
    as_benchmark,
    bound_document,
    view_of,
    with_a_run_in_another_suite,
    with_the_note_of,
    without_results_of,
)
from tests.support.release_fixture import Release, fabricated_release, sha256_of

pytestmark = pytest.mark.usefixtures("no_network")

NINE = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
    "README.md",
    "report.json",
    "report.md",
    "release.json",
)


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return fabricated_release(tmp_path)


def facts_of(release: Release, *, readme: bytes = b"# a card\n") -> list[FileFacts]:
    """The facts of the eight uploaded files release.json lists (every one but itself)."""
    contents = {
        **{
            name: (release.export_dir / name).read_bytes()
            for name in release.export.sha256
        },
        "README.md": readme,
        "report.json": release.report_bytes,
        "report.md": release.markdown_bytes,
    }
    return [
        FileFacts(name, len(contents[name]), sha256_of(contents[name]))
        for name in NINE
        if name in contents
    ]


def record_of(release: Release, **changes: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model_sha256": release.model_sha256,
        "export_manifest_sha256": release.manifest_sha256,
        "report_sha256": sha256_of(release.report_bytes),
    }
    options.update(changes)
    return release_record(
        release.export.record,
        release.view,
        release.system,
        facts_of(release),
        release.repo_id,
        **options,
    )


# ---- the names ---------------------------------------------------------------------------------


def test_the_upload_names_are_the_nine_names_in_this_order() -> None:
    assert UPLOAD_NAMES == NINE


def test_the_stage_manifest_is_stage_only_and_never_an_upload_name() -> None:
    assert STAGE_ONLY == ("manifest.json",)
    assert "manifest.json" not in UPLOAD_NAMES
    assert not set(STAGE_ONLY) & set(UPLOAD_NAMES)


def test_the_schema_is_the_first_version_of_the_release_record() -> None:
    assert RELEASE_SCHEMA == "rinchen.release/1"


# ---- the repository id and the system name ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "org/name",
        "a/b",
        "rinchen-test/rinchen-tiny",
        "Org.x_y/n-1.2",
        "a" * 96 + "/" + "b" * 96,
        "0/1",
    ],
)
def test_a_repository_id_of_two_parts_of_the_allowed_characters_is_accepted(
    value: str,
) -> None:
    assert check_repo_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "a",
        "",
        "/",
        "a/b/c",
        "a b/c",
        "a/b c",
        "/b",
        "a/",
        "a/..",
        "../b",
        "./b",
        "a/.",
        "a" * 97 + "/b",
        "a/" + "b" * 97,
        "a\\b",
        f"{chr(0xE9)}/x",
        "a/b\n",
        "a//b",
        "a:b/c",
    ],
)
def test_anything_else_is_not_a_repository_id(value: str) -> None:
    with pytest.raises(ConfigError, match="ORG/NAME"):
        check_repo_id(value)


def test_the_message_about_a_repository_id_does_not_repeat_it() -> None:
    marker = "SECRET-MARKER-9/x y"

    with pytest.raises(ConfigError) as caught:
        check_repo_id(marker)

    assert "SECRET" not in str(caught.value)


@pytest.mark.parametrize("value", ["alpha", "a.b-c_d", "A" * 64, "1"])
def test_a_system_name_is_an_id_of_the_report_format(value: str) -> None:
    assert check_system_name(value) == value


@pytest.mark.parametrize("value", ["a b", "", "A" * 65, "a/b", "a:b", chr(0xE9), "a\n"])
def test_a_system_name_that_is_not_an_id_is_refused(value: str) -> None:
    with pytest.raises(ConfigError, match="--system"):
        check_system_name(value)


# ---- the binding of a report to an export -------------------------------------------------------


def test_a_report_with_a_rinchen_system_of_the_export_is_accepted(
    release: Release,
) -> None:
    verified = verify_export(release.export_dir)

    require_report_for_export(release.view, SYSTEM, verified)


def refusal(release: Release, document: dict[str, Any], system: str = SYSTEM) -> str:
    verified = verify_export(release.export_dir)
    with pytest.raises(DataError) as caught:
        require_report_for_export(view_of(document), system, verified)
    return str(caught.value)


def test_a_report_without_a_system_of_that_name_is_refused_naming_the_system(
    release: Release,
) -> None:
    message = refusal(release, release.document, system="gamma")

    assert "gamma" in message


def test_a_system_of_kind_hypotheses_is_not_a_rinchen_export(release: Release) -> None:
    message = refusal(release, release.document, system="beta")

    assert "beta" in message
    assert "hypotheses" in message


def test_a_system_of_kind_baseline_is_not_a_rinchen_export(release: Release) -> None:
    document = release.document
    for entry in document["systems"]:
        if entry["name"] == SYSTEM:
            entry["kind"] = "baseline"

    assert "baseline" in refusal(release, document)


def test_a_report_measured_on_another_model_is_refused(release: Release) -> None:
    document = bound_document("a" * 64, release.manifest_sha256)

    message = refusal(release, document)

    assert SYSTEM in message
    assert "model" in message
    assert "a" * 64 not in message


def test_a_report_measured_on_another_export_manifest_is_refused(
    release: Release,
) -> None:
    document = bound_document(release.model_sha256, "b" * 64)

    message = refusal(release, document)

    assert SYSTEM in message
    assert "manifest" in message


def test_a_system_with_no_result_in_the_report_is_refused(release: Release) -> None:
    document = without_results_of(release.document, SYSTEM)

    message = refusal(release, document)

    assert SYSTEM in message
    assert "result" in message


def test_a_system_with_a_result_in_only_one_direction_is_accepted(
    release: Release,
) -> None:
    document = release.document
    document["results"] = [
        r for r in document["results"] if r["direction"] != "eng_Latn-khk_Cyrl"
    ]
    document["coverage"] = [
        c for c in document["coverage"] if c["direction"] != "eng_Latn-khk_Cyrl"
    ]
    document["comparisons"] = [
        c for c in document["comparisons"] if c["direction"] != "eng_Latn-khk_Cyrl"
    ]
    verified = verify_export(release.export_dir)

    require_report_for_export(view_of(with_the_note_of(document)), SYSTEM, verified)


# ---- the record ---------------------------------------------------------------------------------


def test_the_record_holds_exactly_the_fields_of_the_spec(release: Release) -> None:
    assert sorted(record_of(release)) == sorted(
        [
            "schema",
            "repo_id",
            "name",
            "rinchen_version",
            "quantization",
            "tag_placement",
            "directions",
            "export",
            "report",
            "evidence",
            "files",
        ]
    )


def test_the_record_says_what_the_export_and_the_report_are(release: Release) -> None:
    document = record_of(release)
    record = release.export.record

    assert document["schema"] == RELEASE_SCHEMA
    assert document["repo_id"] == REPO_ID
    assert document["name"] == record.name
    assert document["rinchen_version"] == record.rinchen_version == "0.0.0"
    assert document["quantization"] == "float32"
    assert document["tag_placement"] == "src"
    assert document["directions"] == list(record.directions)
    assert document["evidence"] == "fixture"
    assert document["report"] == {
        "sha256": sha256_of(release.report_bytes),
        "system": SYSTEM,
    }


def test_the_export_block_holds_the_digests_of_the_export_and_the_checkpoint(
    release: Release,
) -> None:
    export = record_of(release)["export"]
    checkpoint = release.export.record.checkpoint

    assert export["model_sha256"] == release.model_sha256
    assert export["manifest_sha256"] == release.manifest_sha256
    assert export["spm_sha256"] == release.export.record.spm_sha256
    assert export["checkpoint"] == {
        "kind": checkpoint.kind,
        "step": checkpoint.step,
        "weights_sha256": checkpoint.weights_sha256,
        "config_sha256": checkpoint.config_sha256,
        "vocab_sha256": checkpoint.vocab_sha256,
        "manifest_sha256": checkpoint.manifest_sha256,
        "manifest_stage": checkpoint.manifest_stage,
    }
    assert sorted(export) == [
        "checkpoint",
        "manifest_sha256",
        "model_sha256",
        "spm_sha256",
    ]


def test_the_files_are_every_uploaded_file_but_the_record_itself(
    release: Release,
) -> None:
    files = record_of(release)["files"]

    assert sorted(files) == sorted(name for name in NINE if name != "release.json")
    for name, entry in files.items():
        assert sorted(entry) == ["bytes", "sha256"]
        if name == "README.md":
            assert entry == {"bytes": 9, "sha256": sha256_of(b"# a card\n")}


def test_the_record_refuses_a_file_that_is_not_an_upload_name_or_is_itself(
    release: Release,
) -> None:
    known = facts_of(release)
    for name in ("release.json", "manifest.json", "notes.txt"):
        with pytest.raises(ValueError, match="upload"):
            release_record(
                release.export.record,
                release.view,
                SYSTEM,
                [*known, FileFacts(name, 1, "0" * 64)],
                REPO_ID,
                model_sha256=release.model_sha256,
                export_manifest_sha256=release.manifest_sha256,
                report_sha256=sha256_of(release.report_bytes),
            )


def test_the_record_refuses_a_file_listed_twice(release: Release) -> None:
    known = facts_of(release)

    with pytest.raises(ValueError, match="each once"):
        release_record(
            release.export.record,
            release.view,
            SYSTEM,
            [*known, known[0]],
            REPO_ID,
            model_sha256=release.model_sha256,
            export_manifest_sha256=release.manifest_sha256,
            report_sha256=sha256_of(release.report_bytes),
        )


def test_the_evidence_of_a_report_that_is_not_all_fixture_is_measured_or_mixed(
    release: Release,
) -> None:
    measured = release_record(
        release.export.record,
        view_of(
            as_benchmark(bound_document(release.model_sha256, release.manifest_sha256))
        ),
        SYSTEM,
        facts_of(release),
        REPO_ID,
        model_sha256=release.model_sha256,
        export_manifest_sha256=release.manifest_sha256,
        report_sha256="0" * 64,
    )
    mixed = release_record(
        release.export.record,
        view_of(
            with_a_run_in_another_suite(
                bound_document(release.model_sha256, release.manifest_sha256),
                SYSTEM,
                name="golden-human",
                kind="human",
            )
        ),
        SYSTEM,
        facts_of(release),
        REPO_ID,
        model_sha256=release.model_sha256,
        export_manifest_sha256=release.manifest_sha256,
        report_sha256="0" * 64,
    )

    assert measured["evidence"] == "measured"
    assert mixed["evidence"] == "mixed"


def test_the_bytes_are_sorted_two_space_indented_json_with_one_final_line_feed(
    release: Release,
) -> None:
    data = release_to_bytes(record_of(release))
    text = data.decode("utf-8")

    assert text.endswith("}\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text
    assert json.loads(text) == record_of(release)
    assert (
        text
        == json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    )
    assert text.startswith('{\n  "directions": [')


def test_the_bytes_are_equal_across_calls(release: Release) -> None:
    assert release_to_bytes(record_of(release)) == release_to_bytes(record_of(release))


def test_the_record_holds_no_timestamp_and_no_path(release: Release) -> None:
    text = release_to_bytes(record_of(release)).decode("utf-8")

    assert "created" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert not re.search(r'"/|\\\\|[A-Za-z]:[\\/]|/home|/tmp', text)
    assert str(release.root) not in text
    assert "http" not in text
