"""The dry run of the Hub upload through the real `rinchen` binary (D13, Task 10, spec T18 and T19): a
fabricated release is staged by `rinchen release stage` and planned by `rinchen release hub`, each a
process of its own, with the Hub library in offline mode and every device hidden. The command lists
the nine files of the upload, writes `hub_plan.json` and a manifest, and refuses a stray file with
exit 3 and nothing written. This module is what the release workflow runs after the build
(`uv run pytest tests/integration/test_release_dry_run.py -q`).

`--upload` is NOT run here, with or without a token: nothing in this module opens a client, reads a
token or reaches the Hub. The export is fabricated (no model is loaded) and the report is the golden
report of D13 bound to it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Final

import pytest

from rinchen.export.release import UPLOAD_NAMES
from tests.support.distill_gate import Invocation, run_rinchen
from tests.support.hub_stage import rewrite
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release, fabricated_release

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.timeout(120),
    pytest.mark.usefixtures("d11_gpu_hidden", "no_network"),
]

PLAN_FILE: Final = "hub_plan.json"


@pytest.fixture(scope="module", autouse=True)
def offline_hub(
    d11_gpu_hidden: None, tmp_path_factory: pytest.TempPathFactory
) -> object:
    """The library's offline switch on, its home an empty directory and no token in the
    environment of every process this module starts."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("HF_HOME", str(tmp_path_factory.mktemp("hf-home")))
        patch.delenv("HF_TOKEN", raising=False)
        yield None


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Release:
    """A fabricated release staged by the real `rinchen release stage`."""
    made = fabricated_release(tmp_path_factory.mktemp("d13_dry_run"))
    staged = run_rinchen(*made.args())
    assert staged.returncode == 0, staged.stderr
    return made


def hub(release: Release, out_dir: Path, stage: Path | None = None) -> Invocation:
    return run_rinchen(
        "release",
        "hub",
        "--stage",
        release.stage if stage is None else stage,
        "--repo-id",
        release.repo_id,
        "--out-dir",
        out_dir,
    )


def expected_plan(stage: Path, repo_id: str) -> bytes:
    """`hub_plan.json` as the disk gives it, written out here without the library under test."""
    contents = {name: (stage / name).read_bytes() for name in UPLOAD_NAMES}
    document = {
        "schema": "rinchen.hub_plan/1",
        "repo_id": repo_id,
        "release_sha256": hashlib.sha256(contents["release.json"]).hexdigest(),
        "total_bytes": sum(len(data) for data in contents.values()),
        "files": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in contents.items()
        ],
    }
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    return f"{text}\n".encode()


def test_the_stage_the_real_binary_wrote_holds_the_nine_names_and_its_manifest(
    release: Release,
) -> None:
    assert sorted(path.name for path in release.stage.iterdir()) == sorted(
        [*UPLOAD_NAMES, "manifest.json"]
    )


def test_the_real_binary_plans_the_nine_files_and_writes_what_the_disk_says(
    release: Release, tmp_path: Path
) -> None:
    out = tmp_path / "plan"

    planned = hub(release, out)

    assert planned.returncode == 0, planned.stderr
    assert planned.stderr == ""
    assert planned.stdout[:2] == ["mode: dry-run", f"repo: {release.repo_id}"]
    assert [line.split()[0] for line in planned.stdout[2:11]] == list(UPLOAD_NAMES)
    for line in planned.stdout[2:11]:
        name, size, digest = line.split()
        data = (release.stage / name).read_bytes()
        assert size == str(len(data))
        assert digest == hashlib.sha256(data).hexdigest()[:12]
    assert planned.stdout[11] == (
        f"total: {sum((release.stage / name).stat().st_size for name in UPLOAD_NAMES)}"
    )
    assert (out / PLAN_FILE).read_bytes() == expected_plan(
        release.stage, release.repo_id
    )
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["manifest.json", PLAN_FILE]
    )


def test_two_runs_of_the_real_binary_write_the_same_plan(
    release: Release, tmp_path: Path
) -> None:
    first, second = tmp_path / "one", tmp_path / "two"

    one, two = hub(release, first), hub(release, second)

    assert one.returncode == two.returncode == 0, (one.stderr, two.stderr)
    assert (first / PLAN_FILE).read_bytes() == (second / PLAN_FILE).read_bytes()


def test_the_real_binary_refuses_a_stray_file_with_3_and_writes_nothing(
    release: Release, tmp_path: Path
) -> None:
    damaged = tmp_path / "damaged-stage"
    shutil.copytree(release.stage, damaged)
    (damaged / "notes.txt").write_bytes(b"x")
    out = tmp_path / "refused"

    refused = hub(release, out, stage=damaged)

    assert refused.returncode == 3, refused.stderr
    assert refused.stderr.startswith("error: ")
    assert "notes.txt (not an uploadable name)" in refused.stderr
    assert refused.stdout == []
    assert not out.exists()


def test_the_real_binary_names_the_files_and_never_the_text_it_refused(
    release: Release, tmp_path: Path
) -> None:
    damaged = tmp_path / "damaged-stage"
    shutil.copytree(release.stage, damaged)
    # The export's own manifest holds the paths of this machine. It is put in as the report with
    # every document that vouches for the file kept in step, so that only the scan can refuse it.
    rewrite(damaged, "report.json", (release.export_dir / "manifest.json").read_bytes())
    out = tmp_path / "refused-too"

    refused = hub(release, out, stage=damaged)

    assert refused.returncode == 3, refused.stderr
    assert "report.json (holds an absolute local path)" in refused.stderr
    assert "differs" not in refused.stderr
    assert str(release.export_dir) not in refused.stderr
    assert not out.exists()
