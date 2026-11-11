"""What `rinchen release stage` refuses (D13, Task 9, spec T15 and T17): every input is hashed before it
is read and parsed from the bytes that were hashed, the export must be vouched for, the report must
be the one of THIS export, the options must be usable, the out directory must not be an input's or
hold another stage's manifest, and a file that cannot be written or whose copy differs from its
source ends the command without a manifest.

Exit codes: 2 for a bad option, an export that is not vouched for and a file that changed after it was
hashed; 3 for a report that is missing, not what its manifest vouches for, of another system or
another model, and a foreign manifest in the out directory; 1 when the output cannot be written; 130
when interrupted. A message names places, options, counts and digests, never what a file held.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

import rinchen.cli.release_stage_inputs as inputs
from rinchen.export.verified import verify_export
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    OTHER,
    SYSTEM,
    bound_document,
    markdown_bytes,
    without_results_of,
)
from tests.support.release_fixture import (
    Release,
    another_report_dir,
    fabricated_release,
    invoke_stage,
    listing,
    write_report_dir,
)
from tests.support.robust_cli import fail_replace

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-93"


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return fabricated_release(tmp_path)


def refused(release: Release, code: int, **options: Any) -> Result:
    """Run the stage, expect `code`, and expect that no stage directory was made."""
    existed = release.stage.exists()
    made = invoke_stage(release, **options)

    assert made.exit_code == code, made.output
    assert made.stderr.startswith("error: ")
    assert MARKER not in made.output
    if not existed and "out_dir" not in options:
        assert not release.stage.exists(), "the stage directory was made by a refusal"
    return made


def with_report_dir(release: Release, document: dict[str, Any], **options: Any) -> Path:
    return another_report_dir(release, document, "other-report", **options)


# ---- the options (2) ----------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["a", "a/b/c", "a b/c", "", "/b", "a/..", "a/b c"])
def test_a_repository_id_that_is_not_org_slash_name_is_refused_with_2(
    release: Release, value: str
) -> None:
    made = refused(release, 2, repo_id=value)

    assert "ORG/NAME" in made.stderr


@pytest.mark.parametrize("value", ["a b", "", "a/b", "A" * 65, chr(0xE9)])
def test_a_system_name_that_is_not_an_id_is_refused_with_2(
    release: Release, value: str
) -> None:
    made = refused(release, 2, system=value)

    assert "--system" in made.stderr


def test_a_malformed_option_is_refused_before_anything_is_read(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[str] = []
    monkeypatch.setattr(
        "rinchen.cli.release_stage.verify_export",
        lambda *a, **k: read.append("verified"),
    )

    refused(release, 2, repo_id="a")

    assert read == []


def test_the_marker_in_a_bad_repository_id_or_system_name_is_not_repeated(
    release: Release,
) -> None:
    refused(release, 2, repo_id=f"{MARKER}/x y")
    refused(release, 2, system=f"{MARKER} y")


def test_an_out_directory_that_is_the_directory_of_the_export_is_refused_with_2(
    release: Release,
) -> None:
    made = refused(release, 2, out_dir=release.export_dir)

    assert "--out-dir" in made.stderr
    assert listing(release.export_dir) == sorted(
        [
            "model.bin",
            "shared_vocabulary.json",
            "config.json",
            "spm.model",
            "rinchen.json",
            "manifest.json",
        ]
    )


def test_an_out_directory_that_is_the_directory_of_the_report_is_refused_with_2(
    release: Release,
) -> None:
    before = {p.name: p.read_bytes() for p in release.report_dir.iterdir()}

    made = refused(release, 2, out_dir=release.report_dir)

    assert "--out-dir" in made.stderr
    assert {p.name: p.read_bytes() for p in release.report_dir.iterdir()} == before


def test_an_out_directory_that_is_a_link_to_an_input_directory_is_refused_with_2(
    release: Release,
) -> None:
    link = release.root / "link"
    link.symlink_to(release.export_dir, target_is_directory=True)

    refused(release, 2, out_dir=link)


# ---- the export (2) -----------------------------------------------------------------------------


def test_an_export_without_a_manifest_is_refused_with_2(release: Release) -> None:
    (release.export_dir / "manifest.json").unlink()

    made = refused(release, 2)

    assert "manifest" in made.stderr


def test_an_export_with_a_flipped_byte_is_refused_with_2(release: Release) -> None:
    path = release.export_dir / "model.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))

    made = refused(release, 2)

    assert "model.bin" in made.stderr


def test_an_export_directory_that_does_not_exist_is_refused_with_2(
    release: Release,
) -> None:
    refused(release, 2, export=release.root / "missing")


def test_the_export_is_vouched_for_before_the_report_is_read(release: Release) -> None:
    (release.export_dir / "manifest.json").unlink()
    shutil.rmtree(release.report_dir)

    refused(release, 2)


def test_an_export_that_changes_after_it_was_vouched_for_is_refused_with_2_and_nothing_is_written(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    def verify_then_change(directory: Path) -> Any:
        verified = verify_export(directory)
        (directory / "model.bin").write_bytes(b"swapped after the hash")
        return verified

    monkeypatch.setattr("rinchen.cli.release_stage.verify_export", verify_then_change)

    made = refused(release, 2)

    assert "changed" in made.stderr


# ---- the report directory (3) --------------------------------------------------------------------


def test_a_report_directory_without_a_manifest_is_refused_with_3(
    release: Release,
) -> None:
    (release.report_dir / "manifest.json").unlink()

    made = refused(release, 3)

    assert "manifest" in made.stderr


def test_a_report_directory_that_does_not_exist_is_refused_with_3(
    release: Release,
) -> None:
    refused(release, 3, report_dir=release.root / "missing")


def test_a_report_manifest_of_another_stage_is_refused_with_3(release: Release) -> None:
    other = with_report_dir(
        release, release.document, stage="eval_run", status={"eval": "ok"}
    )

    made = refused(release, 3, report_dir=other)

    assert "eval_report" in made.stderr


def test_a_report_manifest_of_a_stage_named_by_a_marker_is_refused_without_repeating_it(
    release: Release,
) -> None:
    other = with_report_dir(release, release.document, stage=MARKER)

    made = refused(release, 3, report_dir=other)

    assert "eval_report" in made.stderr


def test_a_report_manifest_that_does_not_say_the_report_finished_is_refused_with_3(
    release: Release,
) -> None:
    other = with_report_dir(release, release.document, status={"report": "partial: x"})

    refused(release, 3, report_dir=other)


def test_a_report_manifest_that_does_not_list_report_md_is_refused_with_3(
    release: Release,
) -> None:
    other = with_report_dir(release, release.document, listed=("report.json",))

    made = refused(release, 3, report_dir=other)

    assert "report.md" in made.stderr


def test_a_report_manifest_that_does_not_list_report_json_is_refused_with_3(
    release: Release,
) -> None:
    other = with_report_dir(release, release.document, listed=("report.md",))

    made = refused(release, 3, report_dir=other)

    assert "report.json" in made.stderr


def test_a_report_manifest_that_lists_a_file_twice_is_refused_with_3(
    release: Release,
) -> None:
    other = with_report_dir(release, release.document)
    manifest = store.read(other)
    assert manifest is not None
    manifest.outputs = [*manifest.outputs, manifest.outputs[0]]
    store.write(other, manifest)

    refused(release, 3, report_dir=other)


@pytest.mark.parametrize("name", ["report.json", "report.md"])
def test_a_report_file_that_changed_since_its_manifest_was_written_is_refused_with_3(
    release: Release, name: str
) -> None:
    path = release.report_dir / name
    path.write_bytes(path.read_bytes() + b" ")

    made = refused(release, 3)

    assert name in made.stderr


@pytest.mark.parametrize("name", ["report.json", "report.md"])
def test_a_report_file_that_is_missing_is_refused_with_3(
    release: Release, name: str
) -> None:
    (release.report_dir / name).unlink()

    refused(release, 3)


def test_an_unreadable_report_manifest_is_refused_with_3_without_quoting_it(
    release: Release,
) -> None:
    (release.report_dir / "manifest.json").write_text(
        f'{{"stage": "{MARKER}", "not": ', encoding="utf-8"
    )

    refused(release, 3)


def edited_report(release: Release, change: Callable[[dict[str, Any]], None]) -> Path:
    document = json.loads(release.report_bytes)
    change(document)
    data = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return write_report_dir(release.root / "edited", data, release.markdown_bytes)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema="rinchen.report/2"),
        lambda d: d.update({MARKER: 1}),
        lambda d: d.pop("targets"),
        lambda d: d["systems"][0].update(name=f"/etc/{MARKER}"),
    ],
    ids=["another-schema", "unknown-key", "missing-key", "absolute-path"],
)
def test_a_report_that_the_strict_reader_refuses_is_refused_with_3_without_quoting_it(
    release: Release, change: Callable[[dict[str, Any]], None]
) -> None:
    refused(release, 3, report_dir=edited_report(release, change))


def test_a_report_that_is_not_json_is_refused_with_3_without_quoting_it(
    release: Release,
) -> None:
    other = write_report_dir(
        release.root / "garbage", f"not json {MARKER}".encode(), release.markdown_bytes
    )

    refused(release, 3, report_dir=other)


# ---- the binding of the report to the export (3) ------------------------------------------------


def test_a_report_without_a_system_of_that_name_is_refused_with_3_naming_the_system(
    release: Release,
) -> None:
    made = refused(release, 3, system="gamma")

    assert "gamma" in made.stderr


def test_a_system_of_kind_hypotheses_is_refused_with_3(release: Release) -> None:
    made = refused(release, 3, system="beta")

    assert "beta" in made.stderr
    assert "hypotheses" in made.stderr


def test_a_system_of_kind_baseline_is_refused_with_3(release: Release) -> None:
    def change(document: dict[str, Any]) -> None:
        document["systems"][0]["kind"] = "baseline"

    made = refused(release, 3, report_dir=edited_report(release, change))

    assert "baseline" in made.stderr


def test_a_report_of_another_model_is_refused_with_3_and_nothing_is_written(
    release: Release,
) -> None:
    document = bound_document("a" * 64, release.manifest_sha256)

    made = refused(release, 3, report_dir=with_report_dir(release, document))

    assert SYSTEM in made.stderr
    assert "model" in made.stderr


def test_a_report_of_another_export_manifest_is_refused_with_3(
    release: Release,
) -> None:
    document = bound_document(release.model_sha256, "b" * 64)

    made = refused(release, 3, report_dir=with_report_dir(release, document))

    assert "manifest" in made.stderr


def test_a_system_with_no_result_is_refused_with_3_naming_only_the_system(
    release: Release,
) -> None:
    document = without_results_of(release.document, SYSTEM)

    made = refused(release, 3, report_dir=with_report_dir(release, document))

    assert SYSTEM in made.stderr
    assert "result" in made.stderr


def test_the_report_of_the_other_export_is_refused_even_when_both_are_vouched_for(
    release: Release, tmp_path: Path
) -> None:
    other = fabricated_release(tmp_path / "second", directions=("eng_Latn-khk_Cyrl",))
    assert (
        other.model_sha256 != release.model_sha256
        or other.manifest_sha256 != release.manifest_sha256
    )

    refused(release, 3, report_dir=other.report_dir)


# ---- the out directory (3) ------------------------------------------------------------------------


def test_a_foreign_manifest_in_the_out_directory_is_refused_with_3_and_kept(
    release: Release,
) -> None:
    release.stage.mkdir()
    foreign = Manifest(
        stage="export",
        created="2026-11-10T10:00:00+00:00",
        code_version="0000000",
        config_sha256="0" * 64,
        seed=-1,
    )
    store.write(release.stage, foreign)
    before = (release.stage / "manifest.json").read_bytes()

    made = refused(release, 3)

    assert "export" in made.stderr
    assert listing(release.stage) == ["manifest.json"]
    assert (release.stage / "manifest.json").read_bytes() == before


# ---- the output cannot be written (1), an interrupt (130) -------------------------------------------


def test_a_file_that_cannot_be_written_is_exit_1_with_no_manifest(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, "README.md")

    made = invoke_stage(release)

    assert made.exit_code == 1, made.output
    assert made.stderr.startswith("error: ")
    assert "disk full" in made.stderr
    assert not (release.stage / "manifest.json").exists()


def test_a_manifest_that_cannot_be_written_is_exit_1(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, "manifest.json")

    made = invoke_stage(release)

    assert made.exit_code == 1, made.output
    assert not (release.stage / "manifest.json").exists()


def test_an_out_directory_that_cannot_be_made_is_exit_1(release: Release) -> None:
    blocker = release.root / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")

    made = invoke_stage(release, out_dir=blocker / "stage")

    assert made.exit_code == 1, made.output


@pytest.mark.parametrize("name", ["model.bin", "spm.model", "report.md"])
def test_a_copy_that_differs_from_its_source_is_exit_1_with_no_manifest(
    release: Release, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    real = Path.replace

    def corrupting(self: Path, target: Any) -> Any:
        done = real(self, target)
        if Path(target).name == name:
            Path(target).write_bytes(b"corrupted after the copy")
        return done

    monkeypatch.setattr(Path, "replace", corrupting)

    made = invoke_stage(release)

    assert made.exit_code == 1, made.output
    assert name in made.stderr
    assert not (release.stage / "manifest.json").exists()


def test_an_interrupt_while_writing_is_exit_130_with_no_manifest(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Path.replace

    def interrupting(self: Path, target: Any) -> Any:
        if Path(target).name == "README.md":
            raise KeyboardInterrupt
        return real(self, target)

    monkeypatch.setattr(Path, "replace", interrupting)

    made = invoke_stage(release)

    assert made.exit_code == 130, made.output
    assert not (release.stage / "manifest.json").exists()


def test_the_old_manifest_goes_before_the_new_files_so_a_failure_leaves_none(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke_stage(release).exit_code == 0
    fail_replace(monkeypatch, "report.md")

    made = invoke_stage(release)

    assert made.exit_code == 1
    assert not (release.stage / "manifest.json").exists()


# ---- the files are hashed before they are read -----------------------------------------------------


def test_a_report_file_that_changes_after_it_was_hashed_is_refused_with_2(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = inputs.hash_input

    def hash_then_change(path: Path, **options: Any) -> Any:
        entry = real(path, **options)
        if path.name == "report.json":
            path.write_bytes(path.read_bytes() + b" ")
        return entry

    monkeypatch.setattr(inputs, "hash_input", hash_then_change)

    made = refused(release, 2)

    assert "changed" in made.stderr


# ---- report.md is the rendering of report.json (3) ------------------------------------------------------


def report_dir_with_markdown(release: Release, name: str, markdown: bytes) -> Path:
    """A report directory holding the release's `report.json` and this `report.md`, whose manifest
    vouches for both exactly as they are (every digest holds)."""
    return write_report_dir(release.root / name, release.report_bytes, markdown)


def test_a_report_md_that_is_not_the_rendering_of_report_json_is_refused_with_3_naming_the_file(
    release: Release,
) -> None:
    stale = report_dir_with_markdown(
        release, "stale", release.markdown_bytes + b"\nAn edited line.\n"
    )

    result = refused(release, 3, report_dir=stale)

    assert (
        f"error: report.md of the report directory {stale} is not the rendering of its "
        "report.json"
    ) in result.stderr
    assert "An edited line" not in result.output


def test_a_report_md_with_one_changed_digit_is_refused(release: Release) -> None:
    markdown = release.markdown_bytes
    index = markdown.index(b"73.08")
    edited = markdown[:index] + b"83.08" + markdown[index + 5 :]
    assert edited != markdown

    refused(release, 3, report_dir=report_dir_with_markdown(release, "digit", edited))


def test_the_report_md_of_another_report_is_refused(release: Release) -> None:
    other = without_results_of(
        bound_document(release.model_sha256, release.manifest_sha256, system=SYSTEM),
        OTHER,
    )

    refused(
        release,
        3,
        report_dir=report_dir_with_markdown(release, "other", markdown_bytes(other)),
    )


def test_an_empty_report_md_is_refused(release: Release) -> None:
    refused(release, 3, report_dir=report_dir_with_markdown(release, "empty", b""))


def test_the_rendering_of_the_report_is_accepted_whatever_the_manifest_lists_last(
    release: Release,
) -> None:
    again = report_dir_with_markdown(release, "same", release.markdown_bytes)

    made = invoke_stage(release, report_dir=again)

    assert made.exit_code == 0, made.output
