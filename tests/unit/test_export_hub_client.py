"""The upload of a planned release and the adapter to the Hub library (D13, Task 10, spec T18).

`upload` talks to the Hub only through a `HubClient`: it re-hashes every file just before the
operations are built and refuses a change (before any call), creates the repository private unless
`public` is given, makes ONE commit of the nine files in the order of `UPLOAD_NAMES`, and turns an
error of the client into an `UpstreamError` that names the class of the error and nothing else (an
error text may hold the token). The tests here use `RecordingHub`.

`HfHubClient` is the one place that imports `huggingface_hub`, inside its constructor. It is tested
three ways, none of which reaches the Hub: with `huggingface_hub.HfApi` replaced by `FakeHfApi` (the
calls and the real `CommitOperationAdd` objects it receives; nothing that logs in or reads a token
is touched), in a fresh process with `HF_HUB_OFFLINE=1` and every socket refused, where the REAL
library raises its own `OfflineModeIsEnabled` before it opens any socket (spike finding F8), and in
a fresh process that imports `rinchen.export.hub` and finds the library not loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Final

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.export.hub import (
    CommitResult,
    HfHubClient,
    UploadOperation,
    plan_upload,
    upload,
)
from rinchen.export.release import UPLOAD_NAMES, release_to_bytes
from tests.support.hub_fakes import (
    COMMIT_OID,
    FakeHfApi,
    RecordingHub,
    TokenInError,
    fake_token,
    forbid_token_handling,
)
from tests.support.hub_stage import release_document, rewrite, staged_release
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release

pytestmark = pytest.mark.usefixtures("no_network")

SOURCE_ROOT: Final = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


def run_upload(release: Release, hub: RecordingHub, **options: bool) -> CommitResult:
    plan = plan_upload(release.stage, release.repo_id)
    return upload(plan, release.stage, hub, fake_token(), **options)


# ---- upload: the calls it makes ---------------------------------------------------------------------------


def test_the_repository_is_created_private_and_one_commit_holds_the_nine_files_in_order(
    release: Release,
) -> None:
    hub = RecordingHub()
    token = fake_token()

    result = run_upload(release, hub)

    assert result == CommitResult(COMMIT_OID)
    assert [type(call).__name__ for call in hub.calls] == ["RepoCall", "CommitCall"]
    repo, commit = hub.repo_calls[0], hub.commit_calls[0]
    assert (repo.repo_id, repo.private, repo.exist_ok, repo.token) == (
        release.repo_id,
        True,
        True,
        token,
    )
    assert commit.repo_id == release.repo_id and commit.token == token
    assert [name for name, _ in commit.operations] == list(UPLOAD_NAMES)
    assert [path for _, path in commit.operations] == [
        release.stage / name for name in UPLOAD_NAMES
    ]
    assert "manifest.json" not in [name for name, _ in commit.operations]


def test_the_answer_is_the_commit_the_client_made(release: Release) -> None:
    hub = RecordingHub(oid="f" * 40)

    result = run_upload(release, hub)

    assert result.commit_oid == "f" * 40


def test_the_commit_message_is_rinchen_and_the_name_of_the_model(
    release: Release,
) -> None:
    hub = RecordingHub()

    run_upload(release, hub)

    name = release_document(release.stage)["name"]
    assert hub.commit_calls[0].commit_message == f"Rinchen {name}"
    assert name == release.export.record.name


def test_public_is_explicit_and_only_then_the_repository_is_not_private(
    release: Release,
) -> None:
    private, public = RecordingHub(), RecordingHub()

    run_upload(release, private)
    run_upload(release, public, public=True)

    assert private.repo_calls[0].private is True
    assert public.repo_calls[0].private is False
    assert len(public.commit_calls) == 1


def test_only_one_commit_is_made_however_often_the_plan_is_read(
    release: Release,
) -> None:
    hub = RecordingHub()

    run_upload(release, hub)

    assert len(hub.repo_calls) == 1 and len(hub.commit_calls) == 1


def test_a_blank_token_is_refused_before_any_call(release: Release) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)

    for token in ("", "   "):
        with pytest.raises(ConfigError, match="token"):
            upload(plan, release.stage, hub, token)

    assert hub.calls == []


# ---- upload: a stage that changed after it was planned ----------------------------------------------------------


def test_a_file_changed_between_the_plan_and_the_upload_is_refused_before_any_call(
    release: Release,
) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    model = bytearray((release.stage / "model.bin").read_bytes())
    model[0] ^= 1
    (release.stage / "model.bin").write_bytes(bytes(model))

    with pytest.raises(ConfigError) as raised:
        upload(plan, release.stage, hub, fake_token())

    assert "model.bin" in str(raised.value) and "changed" in str(raised.value)
    assert hub.calls == []


def test_a_file_that_grew_is_refused_before_any_call(release: Release) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    (release.stage / "report.md").write_bytes(
        (release.stage / "report.md").read_bytes() + b"x"
    )

    with pytest.raises(ConfigError, match="report.md"):
        upload(plan, release.stage, hub, fake_token())

    assert hub.calls == []


def test_a_file_that_became_a_link_to_the_same_bytes_is_refused_before_any_call(
    release: Release, tmp_path: Path
) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    # the link leads to a copy of the very bytes that were planned: only the link is wrong
    copy = tmp_path / "copy-of-the-report.md"
    copy.write_bytes((release.stage / "report.md").read_bytes())
    (release.stage / "report.md").unlink()
    (release.stage / "report.md").symlink_to(copy)

    with pytest.raises(ConfigError, match="report.md"):
        upload(plan, release.stage, hub, fake_token())

    assert hub.calls == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="this platform has no FIFO")
@pytest.mark.timeout(30)
def test_a_file_that_became_a_pipe_is_refused_and_never_waited_for(
    release: Release,
) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    (release.stage / "report.md").unlink()
    os.mkfifo(release.stage / "report.md")

    with pytest.raises(ConfigError, match="report.md"):
        upload(plan, release.stage, hub, fake_token())

    assert hub.calls == []


def test_a_file_that_is_gone_is_refused_before_any_call(release: Release) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    (release.stage / "spm.model").unlink()

    with pytest.raises(ConfigError, match="spm.model"):
        upload(plan, release.stage, hub, fake_token())

    assert hub.calls == []


def test_a_change_of_the_release_record_is_refused_too(release: Release) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    document = release_document(release.stage)
    document["name"] = "another-name"
    rewrite(release.stage, "release.json", release_to_bytes(document))

    with pytest.raises(ConfigError, match="release.json"):
        upload(plan, release.stage, hub, fake_token())

    assert hub.calls == []


def test_the_message_of_a_changed_file_holds_no_text_of_the_file(
    release: Release,
) -> None:
    hub = RecordingHub()
    plan = plan_upload(release.stage, release.repo_id)
    secret = "SECRET-MARKER-77"
    (release.stage / "README.md").write_text(secret, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        upload(plan, release.stage, hub, fake_token())

    assert secret not in str(raised.value)


# ---- upload: what the Hub does wrong ----------------------------------------------------------------------------


def test_an_error_of_the_client_that_holds_the_token_becomes_an_upstream_error_by_class(
    release: Release,
) -> None:
    token = fake_token()
    hub = RecordingHub(commit_error=TokenInError(f"401 for Bearer {token}"))
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(UpstreamError) as raised:
        upload(plan, release.stage, hub, token)

    error = raised.value
    assert str(error) == "the Hub refused the upload (TokenInError)"
    assert token not in str(error) and token not in repr(error)
    assert error.__cause__ is None and error.__context__ is None
    assert token not in "".join(traceback.format_exception(error))


def test_a_repository_that_cannot_be_created_stops_before_the_commit(
    release: Release,
) -> None:
    token = fake_token()
    hub = RecordingHub(repo_error=TokenInError(f"token {token} is not allowed"))
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(UpstreamError) as raised:
        upload(plan, release.stage, hub, token)

    assert str(raised.value) == "the Hub refused the repository (TokenInError)"
    assert token not in str(raised.value)
    assert [type(call).__name__ for call in hub.calls] == ["RepoCall"]


def test_an_interrupt_is_not_turned_into_an_upstream_error(release: Release) -> None:
    hub = RecordingHub(commit_error=KeyboardInterrupt())
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(KeyboardInterrupt):
        upload(plan, release.stage, hub, fake_token())


def test_an_error_of_ours_from_the_client_passes_through_unchanged(
    release: Release,
) -> None:
    hub = RecordingHub(commit_error=ConfigError("a refusal of the adapter"))
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(ConfigError, match="a refusal of the adapter"):
        upload(plan, release.stage, hub, fake_token())


@pytest.mark.parametrize("oid", ["", "abc", "Z" * 40, "a" * 41, "A" * 40])
def test_a_commit_id_that_is_not_one_is_an_upstream_error_that_does_not_repeat_it(
    release: Release, oid: str
) -> None:
    hub = RecordingHub(oid=oid)
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(UpstreamError) as raised:
        upload(plan, release.stage, hub, fake_token())

    assert "commit id" in str(raised.value)
    if oid:
        assert oid not in str(raised.value)


# ---- the adapter, over a stand-in for the library's client ---------------------------------------------------------


@pytest.fixture
def fake_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[FakeHfApi, list[str]]:
    import huggingface_hub

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    return api, forbid_token_handling(monkeypatch)


def operations_of(release: Release) -> list[UploadOperation]:
    return [UploadOperation(name, release.stage / name) for name in UPLOAD_NAMES]


def test_the_adapter_builds_the_library_client_with_the_token_and_nothing_else(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, touched = fake_api

    HfHubClient("tok")

    assert api.built_with == ["tok"]
    assert api.calls == [] and touched == []


def test_the_adapter_creates_a_model_repository_with_the_flags_it_is_given(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, touched = fake_api
    client = HfHubClient("tok")

    client.create_repo("org/name", private=True, exist_ok=True, token="tok")
    client.create_repo("org/other", private=False, exist_ok=True, token="tok")

    assert api.calls == [
        (
            "create_repo",
            ("org/name",),
            {"private": True, "exist_ok": True, "repo_type": "model"},
        ),
        (
            "create_repo",
            ("org/other",),
            {"private": False, "exist_ok": True, "repo_type": "model"},
        ),
    ]
    assert touched == []


def test_the_adapter_commits_real_add_operations_for_the_files_in_order(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    from huggingface_hub import CommitOperationAdd

    api, touched = fake_api
    client = HfHubClient("tok")

    result = client.create_commit(
        "org/name",
        operations_of(release),
        commit_message="Rinchen x",
        token="tok",
    )

    ((call, args, kwargs),) = api.calls
    assert call == "create_commit" and args == ()
    assert set(kwargs) == {"repo_id", "operations", "commit_message", "repo_type"}
    assert kwargs["repo_id"] == "org/name"
    assert kwargs["commit_message"] == "Rinchen x" and kwargs["repo_type"] == "model"
    operations = kwargs["operations"]
    assert all(type(operation) is CommitOperationAdd for operation in operations)
    assert [operation.path_in_repo for operation in operations] == list(UPLOAD_NAMES)
    assert [operation.upload_info.size for operation in operations] == [
        (release.stage / name).stat().st_size for name in UPLOAD_NAMES
    ]
    assert result.commit_oid == COMMIT_OID
    assert touched == []


def test_the_adapter_never_logs_in_nor_reads_a_token_file(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, touched = fake_api
    client = HfHubClient("tok")

    client.create_repo("org/name", private=True, exist_ok=True, token="tok")
    client.create_commit(
        "org/name", operations_of(release), commit_message="m", token="tok"
    )

    assert touched == []
    assert [call for call, *_ in api.calls] == ["create_repo", "create_commit"]


def test_the_adapter_refuses_a_token_other_than_the_one_it_was_built_with(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, _ = fake_api
    client = HfHubClient("t0k3n-built")

    with pytest.raises(ConfigError) as raised:
        client.create_repo("org/name", private=True, exist_ok=True, token="t0k3n-call")

    assert "t0k3n" not in str(raised.value)
    assert api.calls == []


@pytest.mark.parametrize("token", ["", " ", "\n"])
def test_the_adapter_refuses_a_blank_token(
    fake_api: tuple[FakeHfApi, list[str]], token: str
) -> None:
    api, _ = fake_api

    with pytest.raises(ConfigError):
        HfHubClient(token)

    assert api.built_with == []


def test_the_commit_of_the_adapter_goes_through_upload_as_one_commit(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, _ = fake_api
    token = fake_token()
    plan = plan_upload(release.stage, release.repo_id)

    result = upload(plan, release.stage, HfHubClient(token), token)

    assert result.commit_oid == COMMIT_OID
    assert [call for call, *_ in api.calls] == ["create_repo", "create_commit"]
    assert api.calls[0][2]["private"] is True


# ---- the real library, offline ------------------------------------------------------------------------------------


OFFLINE_PROBE: Final = """
import json, os, socket, sys
sys.path.insert(0, sys.argv[1])
attempts = []

def refuse(*args, **kwargs):
    attempts.append(1)
    raise AssertionError("socket")

socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
from pathlib import Path
from rinchen.export.hub import HfHubClient, UploadOperation
token = sys.argv[3]
client = HfHubClient(token)
found = {}
for step in ("create_repo", "create_commit"):
    try:
        if step == "create_repo":
            client.create_repo("org/name", private=True, exist_ok=True, token=token)
        else:
            client.create_commit(
                "org/name",
                [UploadOperation("model.bin", Path(sys.argv[2]))],
                commit_message="Rinchen x",
                token=token,
            )
        found[step] = None
    except BaseException as error:
        found[step] = type(error).__module__ + "." + type(error).__name__
print(json.dumps({"errors": found, "attempts": len(attempts)}))
"""


def hub_free_environment(home: Path) -> dict[str, str]:
    """The environment of a child process: no Hub variable of the parent, the library's home an
    empty directory (so no token file is ever looked at) and the offline switch on."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HF_", "HUGGING"))
    }
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(home), CUDA_VISIBLE_DEVICES="")
    return env


def test_the_real_library_refuses_offline_before_any_socket_is_opened(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"weights")
    token = fake_token()

    completed = subprocess.run(
        [sys.executable, "-I", "-c", OFFLINE_PROBE, SOURCE_ROOT, str(model), token],
        capture_output=True,
        text=True,
        env=hub_free_environment(tmp_path / "hf-home"),
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["attempts"] == 0
    assert report["errors"] == {
        "create_repo": "huggingface_hub.errors.OfflineModeIsEnabled",
        "create_commit": "huggingface_hub.errors.OfflineModeIsEnabled",
    }
    assert token not in completed.stdout and token not in completed.stderr
    assert not (tmp_path / "hf-home").exists() or not any(
        (tmp_path / "hf-home").rglob("*token*")
    )


# ---- the library is imported only when the adapter is built ----------------------------------------------------------


def test_importing_the_hub_module_does_not_load_the_library_and_building_the_adapter_does(
    tmp_path: Path,
) -> None:
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.export.hub, rinchen.cli.release_hub\n"
        "before = [m for m in sys.modules if m == 'huggingface_hub' "
        "or m.startswith('huggingface_hub.')]\n"
        "from rinchen.export.hub import HfHubClient\n"
        "HfHubClient('tok')\n"
        "after = 'huggingface_hub' in sys.modules\n"
        "print(before, after)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, SOURCE_ROOT],
        capture_output=True,
        text=True,
        env=hub_free_environment(tmp_path / "hf-home"),
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[] True"
