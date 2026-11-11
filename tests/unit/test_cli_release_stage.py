"""`rinchen release stage` over a fabricated release (D13, Task 9, spec T17): what it writes, prints and
records when it succeeds. What it refuses is in `test_cli_release_stage_refusals.py`.

The out directory is the whole of what a release uploads, nine files, and the manifest that is not
uploaded (it holds local paths): copies of the export's five files, `report.json` and `report.md`
as they were hashed, the card `README.md`, the release record `release.json` and, last,
`manifest.json` of the stage `release_stage`. The card and the record are functions of the inputs
(the card is `render_card`, the record `release_record`), so the test rebuilds both from the disk
and compares bytes. Nothing loads a model: the export is fabricated and the report is the golden
one with `alpha` as a Rinchen system bound to it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.export.model_card import FileFacts, render_card
from rinchen.export.release import (
    STAGE_ONLY,
    UPLOAD_NAMES,
    release_record,
    release_to_bytes,
)
from rinchen.manifest import store
from tests.support.eval_cli import invoke
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import REPO_ID, SYSTEM
from tests.support.release_fixture import (
    Release,
    disk_facts,
    fabricated_release,
    invoke_stage,
    listing,
    sha256_of,
)

pytestmark = pytest.mark.usefixtures("no_network")

EXPORT_FILES = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
)
# The order the files are written in, and so listed as the manifest's outputs.
WRITTEN = (*EXPORT_FILES, "report.json", "report.md", "README.md", "release.json")


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return fabricated_release(tmp_path)


def staged(release: Release) -> Path:
    made = invoke_stage(release)
    assert made.exit_code == 0, made.output
    return release.stage


def rebuilt_card(release: Release) -> str:
    return render_card(
        release.export.record,
        release.view,
        SYSTEM,
        disk_facts(release),
        REPO_ID,
        report_sha256=sha256_of(release.report_bytes),
        export_manifest_sha256=release.manifest_sha256,
    )


def facts_of_the_eight(release: Release, card: str) -> list[FileFacts]:
    facts = {fact.name: fact for fact in disk_facts(release)}
    readme = card.encode("utf-8")
    facts["README.md"] = FileFacts("README.md", len(readme), sha256_of(readme))
    return [facts[name] for name in UPLOAD_NAMES if name in facts]


def rebuilt_record(release: Release, card: str) -> bytes:
    return release_to_bytes(
        release_record(
            release.export.record,
            release.view,
            SYSTEM,
            facts_of_the_eight(release, card),
            REPO_ID,
            model_sha256=release.model_sha256,
            export_manifest_sha256=release.manifest_sha256,
            report_sha256=sha256_of(release.report_bytes),
        )
    )


# ---- what it prints -----------------------------------------------------------------------------


def test_it_prints_the_stage_a_line_for_each_file_and_the_manifest(
    release: Release,
) -> None:
    made = invoke_stage(release)

    assert made.exit_code == 0, made.output
    assert made.stderr == ""
    out = release.stage
    expected = [f"stage: {out}"]
    for name in UPLOAD_NAMES:
        data = (out / name).read_bytes()
        expected.append(f"{name} {len(data)} {sha256_of(data)[:12]}")
    expected.append(f"manifest: {out / 'manifest.json'}")
    assert made.stdout.splitlines() == expected
    assert len(expected) == 11


# ---- what it writes -----------------------------------------------------------------------------


def test_the_stage_holds_the_nine_upload_names_and_the_manifest_and_nothing_else(
    release: Release,
) -> None:
    out = staged(release)

    assert listing(out) == sorted([*UPLOAD_NAMES, *STAGE_ONLY])
    assert all(
        (out / name).is_file() and not (out / name).is_symlink()
        for name in listing(out)
    )


def test_each_copy_is_byte_equal_to_its_source(release: Release) -> None:
    out = staged(release)

    for name in EXPORT_FILES:
        assert (out / name).read_bytes() == (release.export_dir / name).read_bytes()
    assert (out / "report.json").read_bytes() == release.report_bytes
    assert (out / "report.md").read_bytes() == release.markdown_bytes


def test_the_readme_is_the_card_of_the_system(release: Release) -> None:
    out = staged(release)

    assert (out / "README.md").read_bytes() == rebuilt_card(release).encode("utf-8")


def test_the_release_record_is_the_one_the_library_makes_and_holds_the_digests_of_the_disk(
    release: Release,
) -> None:
    out = staged(release)
    card = rebuilt_card(release)

    assert (out / "release.json").read_bytes() == rebuilt_record(release, card)
    document = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert sorted(document["files"]) == sorted(
        n for n in UPLOAD_NAMES if n != "release.json"
    )
    for name, entry in document["files"].items():
        data = (out / name).read_bytes()
        assert entry == {"bytes": len(data), "sha256": sha256_of(data)}
    assert document["export"]["model_sha256"] == release.model_sha256
    assert document["export"]["manifest_sha256"] == release.manifest_sha256
    assert document["export"]["checkpoint"]["manifest_sha256"] == (
        release.export.record.checkpoint.manifest_sha256
    )
    assert document["export"]["spm_sha256"] == release.export.record.spm_sha256
    assert document["report"] == {
        "sha256": sha256_of(release.report_bytes),
        "system": SYSTEM,
    }
    assert document["repo_id"] == REPO_ID
    assert document["evidence"] == "fixture"


def test_the_export_manifest_is_never_copied_and_the_stage_manifest_is_the_stages(
    release: Release,
) -> None:
    out = staged(release)

    assert (out / "manifest.json").read_bytes() != (
        release.export_dir / "manifest.json"
    ).read_bytes()
    manifest = store.read(out)
    assert manifest is not None
    assert manifest.stage == "release_stage"
    assert "manifest.json" not in UPLOAD_NAMES


def test_no_uploaded_file_holds_a_created_key_a_timestamp_or_a_local_path(
    release: Release,
) -> None:
    out = staged(release)

    for name in (n for n in UPLOAD_NAMES if n.endswith((".json", ".md"))):
        text = (out / name).read_text(encoding="utf-8")
        assert '"created"' not in text, name
        assert str(release.root) not in text, name
        assert "/tmp/" not in text and "/home/" not in text, name
    for name in ("README.md", "release.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}", text), name


# ---- the manifest -------------------------------------------------------------------------------


def manifest_of(out: Path) -> Any:
    manifest = store.read(out)
    assert manifest is not None
    return manifest


def test_the_manifest_lists_what_was_hashed_before_it_was_read_and_what_was_written(
    release: Release,
) -> None:
    out = staged(release)
    manifest = manifest_of(out)

    inputs = [(Path(item.path), item.sha256) for item in manifest.inputs]
    expected = [
        *(
            (release.export_dir / name, release.export.sha256[name])
            for name in EXPORT_FILES
        ),
        (release.export_dir / "manifest.json", release.manifest_sha256),
        (
            release.report_dir / "manifest.json",
            sha256_of((release.report_dir / "manifest.json").read_bytes()),
        ),
        (release.report_dir / "report.json", sha256_of(release.report_bytes)),
        (release.report_dir / "report.md", sha256_of(release.markdown_bytes)),
    ]
    assert inputs == expected
    assert all(item.bytes == Path(item.path).stat().st_size for item in manifest.inputs)
    assert [(Path(item.path), item.sha256) for item in manifest.outputs] == [
        (out / name, sha256_of((out / name).read_bytes())) for name in WRITTEN
    ]
    assert all(
        item.bytes == Path(item.path).stat().st_size for item in manifest.outputs
    )
    assert "manifest.json" not in [Path(item.path).name for item in manifest.outputs]


def test_the_manifest_says_the_release_is_ok_and_names_the_repository_and_the_evidence(
    release: Release,
) -> None:
    manifest = manifest_of(staged(release))

    assert manifest.status == {
        "release": "ok",
        "repo_id": REPO_ID,
        "evidence": "fixture",
    }
    assert manifest.seed == -1
    assert manifest.created
    assert len(manifest.config_sha256) == 64


def test_the_manifest_is_written_last(
    release: Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real = Path.replace

    def recording(self: Path, target: Any) -> Any:
        order.append(Path(target).name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", recording)

    assert invoke_stage(release).exit_code == 0

    assert order[-1] == "manifest.json"
    assert order[:-1] == list(WRITTEN)


# ---- twice, again and moved -----------------------------------------------------------------------


def test_two_runs_give_equal_files_and_manifests_that_differ_only_in_created_and_paths(
    release: Release, tmp_path: Path
) -> None:
    first = staged(release)
    second = tmp_path / "again"
    assert invoke_stage(release, out_dir=second).exit_code == 0

    for name in UPLOAD_NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def normalised(out: Path) -> dict[str, Any]:
        document = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        document.pop("created")
        return json.loads(json.dumps(document).replace(str(out), "<out>"))

    assert normalised(first) == normalised(second)


def test_running_it_again_into_the_same_directory_replaces_the_files(
    release: Release,
) -> None:
    out = staged(release)
    first = {
        name: (out / name).read_bytes()
        for name in listing(out)
        if name != "manifest.json"
    }

    again = invoke_stage(release)

    assert again.exit_code == 0, again.output
    assert listing(out) == sorted([*UPLOAD_NAMES, *STAGE_ONLY])
    assert {name: (out / name).read_bytes() for name in first} == first


def test_only_the_files_the_old_manifest_lists_are_replaced_a_stray_file_stays(
    release: Release,
) -> None:
    out = staged(release)
    (out / "notes.txt").write_text("kept", encoding="utf-8")

    assert invoke_stage(release).exit_code == 0

    assert (out / "notes.txt").read_text(encoding="utf-8") == "kept"
    assert "notes.txt" not in [Path(i.path).name for i in manifest_of(out).outputs]


def test_a_release_that_was_moved_still_stages(
    release: Release, tmp_path: Path
) -> None:
    moved_export = tmp_path / "elsewhere" / "export"
    moved_report = tmp_path / "elsewhere" / "report"
    shutil.copytree(release.export_dir, moved_export)
    shutil.copytree(release.report_dir, moved_report)

    made = invoke_stage(release, export=moved_export, report_dir=moved_report)

    assert made.exit_code == 0, made.output
    assert (release.stage / "model.bin").read_bytes() == (
        moved_export / "model.bin"
    ).read_bytes()


def test_a_staged_release_is_what_the_fixture_hands_to_a_later_task(
    tmp_path: Path,
) -> None:
    staged_release = fabricated_release(tmp_path, staged=True)

    assert listing(staged_release.stage) == sorted([*UPLOAD_NAMES, *STAGE_ONLY])


def test_the_stage_command_is_registered_and_answers_help() -> None:
    for arguments in (["release", "--help"], ["release", "stage", "--help"]):
        made = invoke(arguments)

        assert made.exit_code == 0, made.output
    assert "--repo-id" in made.stdout
    assert "--report-dir" in made.stdout
