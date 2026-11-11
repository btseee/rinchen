"""The upload plan of a staged release (D13, Task 10, spec T18): `plan_upload` lists exactly the nine
files of `UPLOAD_NAMES` with the size and SHA-256 each has on the disk, leaves `manifest.json` out,
and refuses, naming the FILE and the kind of problem and never a text, a stray file, a directory, a
link or another non-regular file, a file that is not what `release.json` lists, a name that
`release.json` lists and the stage lacks, a text file that looks like a token or holds an absolute
local path (the export's own manifest does), and a file over the size limit.

Every stage here is the one `rinchen release stage` wrote over a fabricated export; a test damages
it in ONE way (`tests.support.hub_stage.rewrite` keeps the documents that vouch for a file in step,
so that a content refusal is not a digest refusal as well). The strings that look like a token or
a local path are put together from pieces when the test runs: no file of the project holds one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.export import hub
from rinchen.export.hub import UploadPlan, plan_upload
from rinchen.export.release import UPLOAD_NAMES, release_to_bytes
from tests.support.hub_stage import (
    COMMIT_FILE,
    MANIFEST,
    release_document,
    rewrite,
    sha256_of,
    staged_release,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release

pytestmark = pytest.mark.usefixtures("no_network")

BACKSLASH: Final = chr(0x5C)
SLASH: Final = chr(0x2F)


def under(root: str, rest: str) -> str:
    """A POSIX path under the root directory `root`, put together from pieces."""
    return SLASH + root + SLASH + rest


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


@pytest.fixture
def stage(release: Release) -> Path:
    return release.stage


def planned(stage: Path, repo_id: str = "rinchen-test/rinchen-tiny") -> UploadPlan:
    return plan_upload(stage, repo_id)


def refusal(stage: Path) -> str:
    """The message of the `DataError` the plan of `stage` raises."""
    with pytest.raises(DataError) as raised:
        planned(stage)
    return str(raised.value)


# ---- what the plan holds ---------------------------------------------------------------------------


def test_the_plan_lists_the_nine_names_in_order_with_the_size_and_digest_of_the_disk(
    stage: Path, release: Release
) -> None:
    plan = planned(stage)

    assert [file.name for file in plan.files] == list(UPLOAD_NAMES)
    assert len(plan.files) == 9
    for file in plan.files:
        data = (stage / file.name).read_bytes()
        assert file.bytes == len(data)
        assert file.sha256 == hashlib.sha256(data).hexdigest()
    assert plan.repo_id == release.repo_id


def test_the_total_is_the_sum_of_the_sizes_and_the_release_digest_is_that_of_release_json(
    stage: Path,
) -> None:
    plan = planned(stage)

    assert plan.total_bytes == sum(
        (stage / name).stat().st_size for name in UPLOAD_NAMES
    )
    assert plan.release_sha256 == sha256_of((stage / "release.json").read_bytes())
    assert plan.release_sha256 == plan.files[-1].sha256


def test_the_stage_manifest_is_known_and_neither_uploaded_nor_refused(
    stage: Path,
) -> None:
    assert (stage / MANIFEST).is_file()

    plan = planned(stage)

    assert MANIFEST not in [file.name for file in plan.files]


def test_a_stage_without_its_manifest_still_has_a_plan_of_the_nine_files(
    stage: Path,
) -> None:
    (stage / MANIFEST).unlink()

    assert [file.name for file in planned(stage).files] == list(UPLOAD_NAMES)


def test_the_files_are_values_a_plan_cannot_change() -> None:
    file = hub.PlannedFile("model.bin", 3, "0" * 64)

    with pytest.raises(AttributeError):
        file.bytes = 4  # type: ignore[misc]


# ---- hub_plan.json ----------------------------------------------------------------------------------


def test_the_document_of_the_plan_is_sorted_json_with_one_final_line_feed(
    stage: Path,
) -> None:
    plan = planned(stage)

    data = plan.to_bytes()
    text = data.decode("utf-8")
    document = json.loads(text)

    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert (
        text
        == json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    assert document == {
        "schema": "rinchen.hub_plan/1",
        "repo_id": plan.repo_id,
        "release_sha256": plan.release_sha256,
        "total_bytes": plan.total_bytes,
        "files": [
            {
                "name": name,
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_of((stage / name).read_bytes()),
            }
            for name in UPLOAD_NAMES
        ],
    }


def test_the_document_is_the_same_bytes_every_time_and_holds_no_time_path_or_token(
    stage: Path, tmp_path: Path
) -> None:
    first = planned(stage).to_bytes()
    second = planned(stage).to_bytes()
    text = first.decode("utf-8")

    assert first == second
    for forbidden in ("created", "time", str(tmp_path), SLASH + "tmp", "token", "hf_"):
        assert forbidden not in text.lower(), forbidden


def test_the_document_is_the_same_for_a_copy_of_the_stage_elsewhere(
    stage: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "somewhere" / "else"
    shutil.copytree(stage, elsewhere)

    assert planned(elsewhere).to_bytes() == planned(stage).to_bytes()


# ---- the options -----------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["a", "a/b/c", "a b/c", "", "/b", "a/.."])
def test_a_repository_id_that_is_not_org_slash_name_is_a_config_error(
    stage: Path, value: str
) -> None:
    with pytest.raises(ConfigError, match="ORG/NAME"):
        planned(stage, value)


def test_a_repository_the_stage_was_not_made_for_is_refused_naming_neither_id(
    stage: Path,
) -> None:
    with pytest.raises(ConfigError) as raised:
        planned(stage, "other-org/other-name")

    message = str(raised.value)
    assert "--repo-id" in message and "release.json" in message
    assert "other-org" not in message and "rinchen-test" not in message


def test_a_stage_that_is_not_a_directory_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not a directory"):
        planned(tmp_path / "missing")


# ---- what is not an uploadable file --------------------------------------------------------------------


def test_a_stray_file_is_refused_by_name(stage: Path) -> None:
    (stage / "notes.txt").write_bytes(b"x")

    assert "notes.txt (not an uploadable name)" in refusal(stage)


def test_a_checkpoint_file_is_refused_by_name(stage: Path) -> None:
    (stage / "x.pt").write_bytes(b"x")

    assert "x.pt (not an uploadable name)" in refusal(stage)


def test_a_file_of_a_name_that_only_looks_like_an_uploadable_one_is_refused(
    stage: Path,
) -> None:
    (stage / "README.md.bak").write_bytes(b"x")
    (stage / "Model.bin").write_bytes(b"x")

    message = refusal(stage)

    assert "README.md.bak (not an uploadable name)" in message
    assert "Model.bin (not an uploadable name)" in message


def test_a_subdirectory_is_refused_by_name_empty_or_not(stage: Path) -> None:
    (stage / "extra").mkdir()
    (stage / "extra" / "inner.json").write_bytes(b"{}")
    (stage / "data").mkdir()

    message = refusal(stage)

    assert "extra (not a regular file)" in message
    assert "data (not a regular file)" in message
    assert "inner.json" not in message


def test_a_symlink_to_a_regular_file_is_refused_by_name(stage: Path) -> None:
    (stage / "alias.json").symlink_to(stage / "config.json")

    assert "alias.json (not a regular file)" in refusal(stage)


def test_an_uploadable_name_that_is_a_symlink_is_refused_though_the_bytes_agree(
    stage: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "elsewhere.md"
    shutil.copyfile(stage / "report.md", copy)
    (stage / "report.md").unlink()
    (stage / "report.md").symlink_to(copy)

    assert "report.md (not a regular file)" in refusal(stage)


def test_a_manifest_that_is_a_link_is_refused_though_it_is_not_uploaded(
    stage: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "moved-manifest.json"
    shutil.move(stage / MANIFEST, moved)
    (stage / MANIFEST).symlink_to(moved)

    assert "manifest.json (not a regular file)" in refusal(stage)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="this platform has no FIFO")
def test_a_named_pipe_is_refused_by_name_and_never_opened(stage: Path) -> None:
    (stage / "report.md").unlink()
    os.mkfifo(stage / "report.md")
    os.mkfifo(stage / "pipe")

    message = refusal(stage)

    assert "report.md (not a regular file)" in message
    assert "pipe (not a regular file)" in message


# ---- what is not the stage release.json lists -------------------------------------------------------------


def test_one_flipped_byte_of_the_model_is_a_refusal_by_name(stage: Path) -> None:
    model = bytearray((stage / "model.bin").read_bytes())
    model[len(model) // 2] ^= 1
    rewrite(stage, "model.bin", bytes(model), stale=True)

    message = refusal(stage)

    assert "model.bin (differs from what release.json lists)" in message


def test_a_file_that_grew_is_refused_even_when_the_digest_of_the_old_bytes_is_listed(
    stage: Path,
) -> None:
    rewrite(stage, "report.md", (stage / "report.md").read_bytes() + b"\n", stale=True)

    assert "report.md (differs from what release.json lists)" in refusal(stage)


def test_a_size_that_release_json_lists_wrongly_is_a_refusal_though_the_digest_agrees(
    stage: Path,
) -> None:
    document = release_document(stage)
    document["files"]["config.json"]["bytes"] += 1
    rewrite(stage, "release.json", _release_bytes(document))

    assert "config.json (differs from what release.json lists)" in refusal(stage)


def test_a_name_that_release_json_lists_and_the_stage_lacks_is_refused(
    stage: Path,
) -> None:
    (stage / "spm.model").unlink()

    message = refusal(stage)

    assert "spm.model (listed in release.json but absent)" in message


def test_a_file_that_release_json_does_not_list_is_refused(stage: Path) -> None:
    document = release_document(stage)
    del document["files"]["config.json"]
    rewrite(stage, "release.json", _release_bytes(document))

    assert "config.json (not listed in release.json)" in refusal(stage)


def test_a_file_that_is_neither_listed_nor_present_is_refused_as_absent(
    stage: Path,
) -> None:
    document = release_document(stage)
    del document["files"]["report.md"]
    rewrite(stage, "release.json", _release_bytes(document))
    (stage / "report.md").unlink()

    assert "report.md (absent from the stage)" in refusal(stage)


def test_the_release_record_may_list_only_the_files_it_vouches_for(
    stage: Path,
) -> None:
    document = release_document(stage)
    document["files"]["release.json"] = {"bytes": 1, "sha256": "0" * 64}
    document["files"]["manifest.json"] = {"bytes": 1, "sha256": "0" * 64}
    document["files"]["notes.txt"] = {"bytes": 1, "sha256": "0" * 64}
    rewrite(stage, "release.json", _release_bytes(document))

    message = refusal(stage)

    for name in ("release.json", "manifest.json", "notes.txt"):
        assert f"{name} (not a name release.json may list)" in message


def test_without_release_json_there_is_no_plan(stage: Path) -> None:
    (stage / "release.json").unlink()

    assert "release.json (absent from the stage)" in refusal(stage)


def _break(
    change: Callable[[dict[str, Any]], object],
) -> Callable[[dict[str, Any]], None]:
    def broken(document: dict[str, Any]) -> None:
        change(document)

    return broken


@pytest.mark.parametrize(
    "mutate, field",
    [
        (_break(lambda d: d.update(schema="rinchen.release/2")), "schema"),
        (_break(lambda d: d.update(files=[])), "files"),
        (_break(lambda d: d.pop("files")), "files"),
        (_break(lambda d: d.update(repo_id=5)), "repo_id"),
        (_break(lambda d: d.update(name="")), "name"),
        (_break(lambda d: d.update(name="a b")), "name"),
        (
            _break(lambda d: d["files"]["config.json"].update(bytes=True)),
            "files.config.json",
        ),
        (
            _break(lambda d: d["files"]["config.json"].update(bytes=-1)),
            "files.config.json",
        ),
        (
            _break(lambda d: d["files"]["config.json"].update(sha256="ABC")),
            "files.config.json",
        ),
        (
            _break(lambda d: d["files"]["config.json"].update(extra=1)),
            "files.config.json",
        ),
        (
            _break(lambda d: d["files"].update({"config.json": "x"})),
            "files.config.json",
        ),
    ],
)
def test_a_release_record_that_is_not_one_is_refused_naming_the_field_only(
    stage: Path, mutate: Callable[[dict[str, Any]], None], field: str
) -> None:
    document = release_document(stage)
    mutate(document)
    rewrite(stage, "release.json", _release_bytes(document))

    message = refusal(stage)

    assert f"release.json (not a release record: field '{field}')" in message


@pytest.mark.parametrize("data", [b"", b"[]", b"{", b"\xff\xfe", b"null"])
def test_a_release_record_that_is_not_json_is_refused(stage: Path, data: bytes) -> None:
    rewrite(stage, "release.json", data)

    assert "release.json (not a release record" in refusal(stage)


def test_every_problem_of_the_stage_is_named_not_only_the_first(stage: Path) -> None:
    (stage / "notes.txt").write_bytes(b"x")
    (stage / "spm.model").unlink()
    rewrite(stage, "model.bin", b"changed", stale=True)

    message = refusal(stage)

    for fragment in (
        "notes.txt (not an uploadable name)",
        "spm.model (listed in release.json but absent)",
        "model.bin (differs from what release.json lists)",
    ):
        assert fragment in message


def test_a_stray_name_is_shown_escaped_and_cut(stage: Path) -> None:
    (stage / ("x" + chr(0xE9) + "y" * 200)).write_bytes(b"x")

    message = refusal(stage)

    assert "\\xe9" in message
    assert "y" * 100 not in message


# ---- the size limit ---------------------------------------------------------------------------------


def test_the_limit_is_ten_gibibytes() -> None:
    assert hub.MAX_FILE_BYTES == 10 * 2**30


def test_a_file_over_the_limit_is_refused_by_name_before_it_is_read(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = (stage / "model.bin").stat().st_size
    monkeypatch.setattr(hub, "MAX_FILE_BYTES", size - 1)

    message = refusal(stage)

    assert "model.bin (over the size limit)" in message


def test_a_file_of_exactly_the_limit_is_accepted(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = [(stage / name).stat().st_size for name in UPLOAD_NAMES]
    monkeypatch.setattr(hub, "MAX_FILE_BYTES", max(sizes))
    monkeypatch.setattr(hub, "MAX_TEXT_BYTES", max(sizes))

    assert [file.name for file in planned(stage).files] == list(UPLOAD_NAMES)


def test_a_text_file_too_large_to_scan_is_refused_by_name(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = (stage / "report.json").stat().st_size
    monkeypatch.setattr(hub, "MAX_TEXT_BYTES", size - 1)

    message = refusal(stage)

    assert "report.json (too large to scan)" in message
    assert "model.bin" not in message


# ---- what looks like a token or a local path ---------------------------------------------------------------


def with_text(stage: Path, name: str, text: str) -> None:
    """`name` of the stage with `text` appended, vouched for as it is."""
    rewrite(stage, name, (stage / name).read_bytes() + text.encode("utf-8"))


def test_a_card_that_holds_a_hub_token_is_refused_and_the_message_does_not_repeat_it(
    stage: Path,
) -> None:
    token = "hf" + "_" + "a" * 30
    with_text(stage, "README.md", f"\nuse {token} to log in\n")

    message = refusal(stage)

    assert "README.md (looks like a token)" in message
    assert token not in message and "a" * 20 not in message


def test_a_report_that_holds_a_bearer_credential_is_refused(stage: Path) -> None:
    secret = "b" * 20
    with_text(stage, "report.md", f"\nAuthorization: Bearer {secret}\n")

    message = refusal(stage)

    assert "report.md (looks like a token)" in message
    assert secret not in message


def test_a_config_that_names_the_token_variable_is_refused(stage: Path) -> None:
    with_text(stage, "config.json", "\n" + "HF_" + "TOKEN" + "=x\n")

    message = refusal(stage)

    assert "config.json (looks like a token)" in message
    assert "=x" not in message


LOCAL_PATHS: Final = (
    under("ho" + "me", "someone/x"),
    under("tm" + "p", "run/x"),
    under("va" + "r", "lib/x"),
    under("ro" + "ot", "x"),
    under("Us" + "ers", "x/y"),
    under("mn" + "t", "disk/x"),
    under("op" + "t", "tool/x"),
    under("us" + "r", "local/x"),
    under("sr" + "v", "data/x"),
    "C:" + BACKSLASH + "Users" + BACKSLASH + "x",
    "d:" + BACKSLASH + "x",
)


@pytest.mark.parametrize(
    "path",
    LOCAL_PATHS,
    ids=lambda path: path.split(SLASH)[1] if path.startswith(SLASH) else "drive",
)
def test_a_report_that_holds_an_absolute_local_path_is_refused_without_the_path(
    stage: Path, path: str
) -> None:
    with_text(stage, "report.md", f"\nsee {path} for the model\n")

    message = refusal(stage)

    assert "report.md (holds an absolute local path)" in message
    assert path not in message and "someone" not in message


def test_a_path_inside_json_is_found_as_the_json_writes_it(stage: Path) -> None:
    path = "C:" + BACKSLASH + "Users" + BACKSLASH + "x"
    document = json.loads((stage / "report.json").read_text(encoding="utf-8"))
    document["a_note"] = path
    rewrite(stage, "report.json", json.dumps(document).encode("utf-8"))

    assert "report.json (holds an absolute local path)" in refusal(stage)


def test_the_export_manifest_with_its_local_paths_is_refused_under_any_name(
    stage: Path, release: Release
) -> None:
    manifest = (release.export_dir / "manifest.json").read_bytes()
    rewrite(stage, "report.json", manifest)

    assert "report.json (holds an absolute local path)" in refusal(stage)


def test_a_token_in_a_binary_file_is_not_looked_for(stage: Path) -> None:
    token = "hf" + "_" + "a" * 30
    rewrite(stage, "spm.model", b"\x00\x01" + token.encode() + b"\x02")

    assert [file.name for file in planned(stage).files] == list(UPLOAD_NAMES)


def test_the_scan_of_a_stray_file_is_not_needed_it_is_refused_for_its_name(
    stage: Path,
) -> None:
    (stage / "notes.md").write_text("fine\n", encoding="utf-8")

    message = refusal(stage)

    assert "notes.md (not an uploadable name)" in message
    assert "token" not in message and "path" not in message


def test_the_plan_leaves_the_stage_as_it_found_it(stage: Path) -> None:
    before = sorted(
        (path.name, path.read_bytes(), path.stat().st_mtime_ns)
        for path in stage.iterdir()
    )

    planned(stage)

    after = sorted(
        (path.name, path.read_bytes(), path.stat().st_mtime_ns)
        for path in stage.iterdir()
    )
    assert before == after
    assert not (stage / COMMIT_FILE).exists()


def _release_bytes(document: dict[str, Any]) -> bytes:
    return release_to_bytes(document)


def test_more_problems_than_a_message_shows_are_counted_not_dropped(
    stage: Path,
) -> None:
    for number in range(15):
        (stage / f"stray-{number:02d}.txt").write_bytes(b"x")

    message = refusal(stage)

    assert message.count("(not an uploadable name)") == 12
    assert message.endswith("; and 3 more")
    assert "stray-00.txt" in message and "stray-11.txt" in message
    assert "stray-12.txt" not in message


def test_a_file_over_the_limit_is_not_read_at_all(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rinchen.export.hub_files as files

    size = (stage / "report.json").stat().st_size
    monkeypatch.setattr(hub, "MAX_FILE_BYTES", size - 1)
    read: list[str] = []
    real = files.read_hashed

    def watching(path: Path, **options: object) -> object:
        read.append(path.name)
        return real(path, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(files, "read_hashed", watching)

    refusal(stage)

    assert "report.json" not in read
    assert "report.md" in read
