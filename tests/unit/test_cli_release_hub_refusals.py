"""What `rinchen release hub` refuses (D13, Task 10, spec T15 and T18): a stage that is not the one
`rinchen release stage` vouches for, every refusal of the upload plan, an out directory that is the
stage or holds another stage's manifest, a repository id that is not `ORG/NAME` or not the one the
stage was made for, and `--upload` without a usable `HF_TOKEN`.

Exit codes: 2 for a bad option, an out directory that is the stage and a missing or blank `HF_TOKEN`
for `--upload`; 3 for a stage or a plan that is refused and another stage's manifest in the out
directory. A refusal writes nothing and its message names a file and a kind of problem, never a text.
`--upload` without a token ends before anything is hashed or read.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
from click.testing import Result

from rinchen.export.release import UPLOAD_NAMES, release_to_bytes
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.hub_fakes import (
    COMMIT_OID,
    RecordingHub,
    fake_token,
    install_open_client,
)
from tests.support.hub_stage import (
    COMMIT_FILE,
    MANIFEST,
    PLAN_FILE,
    bytes_under,
    holds,
    invoke_hub,
    release_document,
    rewrite,
    staged_release,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release
from tests.support.robust_cli import fail_replace, written

pytestmark = pytest.mark.usefixtures("no_network")

MARKER: Final = "SECRET-MARKER-95"
BACKSLASH: Final = chr(0x5C)


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "hub-out"


def refused(
    release: Release, out: Path, code: int, *extra: str, **options: Any
) -> Result:
    """Run the command, expect `code`, an error line, no marker and nothing written."""
    existed = out.exists()
    made = invoke_hub(release, out, *extra, **options)

    assert made.exit_code == code, made.output
    assert made.stderr.startswith("error: ")
    assert made.stdout == ""
    assert MARKER not in made.output
    if not existed:
        assert not out.exists(), "the out directory was made by a refusal"
    return made


def damage_manifest(stage: Path, change: Callable[[Manifest], None]) -> None:
    manifest = store.read(stage)
    assert manifest is not None
    change(manifest)
    store.write(stage, manifest)


# ---- the stage and its manifest (3) --------------------------------------------------------------------------------


def test_a_stage_without_its_manifest_is_refused_with_3(
    release: Release, out: Path
) -> None:
    (release.stage / MANIFEST).unlink()

    made = refused(release, out, 3)

    assert "manifest" in made.stderr


def test_a_stage_that_is_not_there_is_refused_with_3(
    release: Release, out: Path, tmp_path: Path
) -> None:
    refused(release, out, 3, stage=tmp_path / "missing-stage")


def test_a_manifest_that_is_not_json_is_refused_with_3(
    release: Release, out: Path
) -> None:
    (release.stage / MANIFEST).write_text(MARKER, encoding="utf-8")

    refused(release, out, 3)


def test_the_manifest_of_another_stage_is_refused_with_3(
    release: Release, out: Path
) -> None:
    damage_manifest(
        release.stage, lambda manifest: setattr(manifest, "stage", "export")
    )

    made = refused(release, out, 3)

    assert "release_stage" in made.stderr


def test_a_stage_that_did_not_finish_is_refused_with_3(
    release: Release, out: Path
) -> None:
    damage_manifest(
        release.stage, lambda manifest: manifest.status.update(release="partial")
    )

    refused(release, out, 3)


@pytest.mark.parametrize("name", UPLOAD_NAMES)
def test_a_manifest_that_does_not_list_a_file_of_the_upload_is_refused_naming_it(
    release: Release, out: Path, name: str
) -> None:
    damage_manifest(
        release.stage,
        lambda manifest: setattr(
            manifest,
            "outputs",
            [entry for entry in manifest.outputs if Path(entry.path).name != name],
        ),
    )

    made = refused(release, out, 3)

    assert f"{name} (not listed by the manifest of the stage)" in made.stderr


def test_a_manifest_that_lists_a_file_twice_does_not_vouch_for_it(
    release: Release, out: Path
) -> None:
    damage_manifest(
        release.stage,
        lambda manifest: manifest.outputs.append(manifest.outputs[0].model_copy()),
    )

    made = refused(release, out, 3)

    assert f"{UPLOAD_NAMES[0]} (not listed by the manifest of the stage)" in made.stderr


@pytest.mark.parametrize("name", UPLOAD_NAMES)
def test_a_file_that_is_not_the_one_the_manifest_vouches_for_is_refused_naming_it(
    release: Release, out: Path, name: str
) -> None:
    # release.json is kept in step with the new bytes, so that only the manifest, which
    # `rinchen release stage` wrote, still says what the file was.
    data = (release.stage / name).read_bytes()
    changed = data + (b"\n" if name.endswith((".json", ".md")) else b"\x00")
    rewrite(release.stage, name, changed, in_manifest=False)

    made = refused(release, out, 3)

    assert f"{name} (differs from what the manifest of the stage lists)" in made.stderr


# ---- every refusal of the plan, through the command (3) ----------------------------------------------------------------


def stray_file(stage: Path) -> None:
    (stage / "notes.txt").write_bytes(b"x")


def a_directory(stage: Path) -> None:
    (stage / "extra").mkdir()


def a_symlink(stage: Path) -> None:
    (stage / "alias.json").symlink_to(stage / "config.json")


def a_flipped_byte(stage: Path) -> None:
    model = bytearray((stage / "model.bin").read_bytes())
    model[0] ^= 1
    rewrite(stage, "model.bin", bytes(model), stale=True)


def a_listed_file_that_is_gone(stage: Path) -> None:
    (stage / "spm.model").unlink()


def an_unlisted_file(stage: Path) -> None:
    document = release_document(stage)
    del document["files"]["config.json"]
    rewrite(stage, "release.json", release_to_bytes(document))


def a_card_with_a_token(stage: Path) -> None:
    text = "use " + "hf" + "_" + "z" * 30 + " here"
    rewrite(stage, "README.md", (stage / "README.md").read_bytes() + text.encode())


def a_report_with_a_bearer(stage: Path) -> None:
    text = "Bearer " + "q" * 20
    rewrite(stage, "report.md", (stage / "report.md").read_bytes() + text.encode())


def a_config_with_the_variable(stage: Path) -> None:
    text = "HF_" + "TOKEN" + "=x"
    rewrite(stage, "config.json", (stage / "config.json").read_bytes() + text.encode())


def a_report_with_a_home_path(stage: Path) -> None:
    text = "see /" + "ho" + "me/someone/x"
    rewrite(stage, "report.md", (stage / "report.md").read_bytes() + text.encode())


def a_report_with_a_drive_path(stage: Path) -> None:
    text = "see C:" + BACKSLASH + "Users" + BACKSLASH + "x"
    rewrite(stage, "report.md", (stage / "report.md").read_bytes() + text.encode())


def the_export_manifest(stage: Path, release: Release) -> None:
    rewrite(stage, "report.json", (release.export_dir / "manifest.json").read_bytes())


PLAN_REFUSALS: Final[dict[str, tuple[Callable[[Path], None], str]]] = {
    "stray": (stray_file, "notes.txt (not an uploadable name)"),
    "directory": (a_directory, "extra (not a regular file)"),
    "symlink": (a_symlink, "alias.json (not a regular file)"),
    "flipped byte": (
        a_flipped_byte,
        "model.bin (differs from what release.json lists)",
    ),
    "gone": (
        a_listed_file_that_is_gone,
        "spm.model (listed in release.json but absent)",
    ),
    "unlisted": (an_unlisted_file, "config.json (not listed in release.json)"),
    "hub token": (a_card_with_a_token, "README.md (looks like a token)"),
    "bearer": (a_report_with_a_bearer, "report.md (looks like a token)"),
    "variable": (a_config_with_the_variable, "config.json (looks like a token)"),
    "home path": (
        a_report_with_a_home_path,
        "report.md (holds an absolute local path)",
    ),
    "drive path": (
        a_report_with_a_drive_path,
        "report.md (holds an absolute local path)",
    ),
}


@pytest.mark.parametrize("case", sorted(PLAN_REFUSALS))
def test_every_refusal_of_the_plan_is_exit_3_with_the_file_named_and_nothing_written(
    release: Release, out: Path, case: str
) -> None:
    damage, expected = PLAN_REFUSALS[case]
    damage(release.stage)

    made = refused(release, out, 3)

    assert expected in made.stderr


def test_the_export_manifest_with_its_paths_is_refused_through_the_command(
    release: Release, out: Path
) -> None:
    the_export_manifest(release.stage, release)

    made = refused(release, out, 3)

    assert "report.json (holds an absolute local path)" in made.stderr
    assert str(release.export_dir) not in made.stderr


def test_a_refusal_message_holds_no_text_of_the_files(
    release: Release, out: Path
) -> None:
    secret = "hf" + "_" + "m" * 25
    rewrite(
        release.stage,
        "README.md",
        (release.stage / "README.md").read_bytes() + f"\n{secret}\n".encode(),
    )

    made = refused(release, out, 3)

    assert secret not in made.stderr and "m" * 15 not in made.stderr


def test_a_file_that_changed_after_the_manifest_was_written_is_refused_with_3(
    release: Release, out: Path
) -> None:
    rewrite(
        release.stage,
        "report.md",
        (release.stage / "report.md").read_bytes() + b"\nmore\n",
        stale=True,
    )

    made = refused(release, out, 3)

    assert "report.md" in made.stderr


# ---- the out directory ----------------------------------------------------------------------------------------------------


def test_an_out_directory_that_is_the_stage_is_refused_with_2(release: Release) -> None:
    before = written(release.stage)

    made = invoke_hub(release, release.stage)

    assert made.exit_code == 2, made.output
    assert "--out-dir" in made.stderr
    assert written(release.stage) == before


def test_an_out_directory_that_is_the_stage_by_another_spelling_is_refused_with_2(
    release: Release, tmp_path: Path
) -> None:
    link = tmp_path / "link-to-stage"
    link.symlink_to(release.stage, target_is_directory=True)
    before = written(release.stage)

    made = invoke_hub(release, link)

    assert made.exit_code == 2, made.output
    assert written(release.stage) == before


def test_another_stages_manifest_in_the_out_directory_is_refused_with_3_and_left_alone(
    release: Release, out: Path
) -> None:
    out.mkdir()
    shutil.copyfile(release.report_dir / MANIFEST, out / MANIFEST)
    (out / "keep.txt").write_bytes(b"kept")
    before = written(out)

    made = invoke_hub(release, out)

    assert made.exit_code == 3, made.output
    assert made.stderr.startswith("error: ")
    assert written(out) == before


def test_a_dry_run_does_not_erase_the_record_of_an_upload_made_into_the_directory(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_open_client(monkeypatch, RecordingHub())
    token = fake_token()
    assert invoke_hub(release, out, "--upload", env={"HF_TOKEN": token}).exit_code == 0
    before = written(out)
    assert COMMIT_FILE in before

    made = invoke_hub(release, out)

    assert made.exit_code == 3, made.output
    assert "upload" in made.stderr
    assert written(out) == before


# ---- the options (2) ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["a", "a/b/c", "a b/c", "", "/b", "a/..", "a/b c"])
def test_a_repository_id_that_is_not_org_slash_name_is_refused_with_2(
    release: Release, out: Path, value: str
) -> None:
    made = refused(release, out, 2, repo_id=value)

    assert "ORG/NAME" in made.stderr


def test_a_repository_the_stage_was_not_made_for_is_refused_with_2(
    release: Release, out: Path
) -> None:
    made = refused(release, out, 2, repo_id="other-org/other-name")

    assert "release.json" in made.stderr
    assert "other-org" not in made.stderr


def test_a_malformed_option_is_refused_before_anything_is_read(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rinchen.cli.release_hub.hash_input",
        lambda *a, **k: pytest.fail("a file was hashed"),
    )

    refused(release, out, 2, repo_id=f"{MARKER}/x y")


# ---- --upload needs HF_TOKEN (2) ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env", [{"HF_TOKEN": None}, {"HF_TOKEN": ""}, {"HF_TOKEN": "   "}]
)
def test_an_upload_without_a_usable_token_is_refused_with_2_before_anything_is_hashed(
    release: Release,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str | None],
) -> None:
    opened = install_open_client(monkeypatch, None)
    monkeypatch.setattr(
        "rinchen.cli.release_hub.hash_input",
        lambda *a, **k: pytest.fail("a file was hashed"),
    )
    monkeypatch.setattr(
        "rinchen.cli.release_hub.plan_upload",
        lambda *a, **k: pytest.fail("a plan was made"),
    )
    reads: list[str] = []
    real_open = os.open

    def watching(path: Any, *args: Any, **kwargs: Any) -> int:
        reads.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", watching)

    made = refused(release, out, 2, "--upload", env=env)

    assert "HF_TOKEN" in made.stderr
    assert opened == []
    assert [path for path in reads if str(release.stage) in path] == []
    assert not (release.stage / PLAN_FILE).exists()


def test_an_upload_with_no_token_leaves_an_earlier_dry_run_as_it_was(
    release: Release, out: Path
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    before = written(out)

    made = invoke_hub(release, out, "--upload", env={"HF_TOKEN": None})

    assert made.exit_code == 2
    assert written(out) == before


def test_the_token_variable_is_not_looked_for_when_the_options_are_wrong(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        "rinchen.cli.release_hub.token_from_environment",
        lambda: asked.append("asked") or fake_token(),
    )
    token = fake_token()

    made = invoke_hub(
        release, out, "--upload", repo_id="not a repo", env={"HF_TOKEN": token}
    )

    assert made.exit_code == 2
    assert not holds(token, [made.stdout, made.stderr, *bytes_under(out.parent)])
    assert asked == []


def test_a_refused_upload_never_prints_nor_writes_the_token(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = install_open_client(monkeypatch, None)
    token = fake_token()
    (release.stage / "notes.txt").write_bytes(b"x")

    made = invoke_hub(release, out, "--upload", env={"HF_TOKEN": token})

    assert made.exit_code == 3
    assert opened == []
    assert not holds(token, [made.stdout, made.stderr, *bytes_under(release.root)])
    assert not out.exists()


@pytest.mark.parametrize("case", sorted(PLAN_REFUSALS))
def test_no_refusal_of_an_upload_shows_or_writes_the_token_of_the_environment(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    opened = install_open_client(monkeypatch, None)
    damage, expected = PLAN_REFUSALS[case]
    damage(release.stage)
    token = fake_token()

    made = invoke_hub(release, out, "--upload", env={"HF_TOKEN": token})

    assert made.exit_code == 3, made.output
    assert expected in made.stderr
    assert opened == []
    assert not holds(token, [made.stdout, made.stderr, *bytes_under(release.root)])


# ---- the output cannot be written (1) ------------------------------------------------------------------------------------------


def test_a_plan_that_cannot_be_written_is_exit_1_with_no_manifest(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, PLAN_FILE)

    made = invoke_hub(release, out)

    assert made.exit_code == 1, made.output
    assert made.stderr.startswith("error: ")
    assert not (out / MANIFEST).exists()


def test_an_out_directory_that_is_a_file_is_exit_1(
    release: Release, tmp_path: Path
) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_bytes(b"x")

    made = invoke_hub(release, blocked)

    assert made.exit_code == 1, made.output
    assert blocked.read_bytes() == b"x"


def test_a_commit_record_that_cannot_be_written_still_tells_the_commit_and_leaves_no_manifest(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_open_client(monkeypatch, RecordingHub())
    fail_replace(monkeypatch, COMMIT_FILE)

    made = invoke_hub(release, out, "--upload", env={"HF_TOKEN": fake_token()})

    assert made.exit_code == 1, made.output
    assert f"commit: {COMMIT_OID}" in made.stdout
    assert not (out / MANIFEST).exists()


def test_a_failed_rerun_leaves_no_manifest_that_vouches_for_a_mix_of_old_and_new_files(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke_hub(release, out).exit_code == 0
    assert (out / MANIFEST).is_file()
    fail_replace(monkeypatch, PLAN_FILE)

    made = invoke_hub(release, out)

    assert made.exit_code == 1, made.output
    assert not (out / MANIFEST).exists()


def test_a_copy_that_differs_from_its_source_is_a_failed_write_with_no_manifest(
    release: Release, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rinchen.cli.release_hub_output as output

    real = output.replace_file
    monkeypatch.setattr(
        output, "replace_file", lambda path, data: real(path, data + b" ")
    )

    made = invoke_hub(release, out)

    assert made.exit_code == 1, made.output
    assert "differs from its source" in made.stderr
    assert not (out / MANIFEST).exists()
