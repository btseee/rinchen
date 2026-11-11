"""A repository is private unless `--public` says otherwise, also when it already exists (D13 fix
wave 1, finding I5).

`HfApi.create_repo(private=..., exist_ok=True)` ignores `private` when the repository exists, so a
repository an operator made public in the web page stays public and the upload would go into it
while the local record said private. `upload` therefore asks the Hub (`HubClient.repo_is_private`)
what the repository IS, after `create_repo` and BEFORE `create_commit`: a public repository without
`public=True` is a `ConfigError` and no commit is made; the visibility that goes into the result is
the one the Hub reports, never the one that was asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.export.hub import HfHubClient, plan_upload, upload
from tests.support.hub_fakes import (
    FakeHfApi,
    RecordingHub,
    TokenInError,
    fake_token,
    forbid_token_handling,
)
from tests.support.hub_stage import staged_release
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_fixture import Release

pytestmark = pytest.mark.usefixtures("no_network")

REFUSAL = "the repository exists and is public: pass --public to upload into it"


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return staged_release(tmp_path)


def run_upload(release: Release, hub: RecordingHub, **options: bool) -> object:
    plan = plan_upload(release.stage, release.repo_id)
    return upload(plan, release.stage, hub, fake_token(), **options)


# ---- the refusal -----------------------------------------------------------------------------------


def test_an_existing_public_repository_is_refused_without_public_and_nothing_is_committed(
    release: Release,
) -> None:
    hub = RecordingHub(reports_private=False)

    with pytest.raises(ConfigError, match=REFUSAL):
        run_upload(release, hub)

    assert hub.commit_calls == []


def test_the_refusal_is_a_config_error_that_names_neither_the_token_nor_a_file(
    release: Release,
) -> None:
    hub = RecordingHub(reports_private=False)

    with pytest.raises(ConfigError) as raised:
        run_upload(release, hub)

    assert str(raised.value) == REFUSAL
    assert fake_token() not in str(raised.value)


def test_an_existing_public_repository_is_accepted_when_public_is_said(
    release: Release,
) -> None:
    hub = RecordingHub(reports_private=False)

    result = run_upload(release, hub, public=True)

    assert len(hub.commit_calls) == 1
    assert result.visibility == "public"  # type: ignore[attr-defined]


# ---- the order -------------------------------------------------------------------------------------


def test_the_visibility_is_asked_after_the_repository_is_created_and_before_the_commit(
    release: Release,
) -> None:
    hub = RecordingHub()

    run_upload(release, hub)

    assert [type(call).__name__ for call in hub.calls] == [
        "RepoCall",
        "VisibilityCall",
        "CommitCall",
    ]


def test_the_visibility_is_asked_about_the_repository_of_the_plan_with_the_token(
    release: Release,
) -> None:
    hub = RecordingHub()

    run_upload(release, hub)

    ((call,)) = hub.visibility_calls
    assert (call.repo_id, call.token) == (release.repo_id, fake_token())


# ---- what is reported ------------------------------------------------------------------------------


def test_a_private_repository_is_committed_to_and_reported_private(
    release: Release,
) -> None:
    hub = RecordingHub(reports_private=True)

    result = run_upload(release, hub)

    assert len(hub.commit_calls) == 1
    assert result.visibility == "private"  # type: ignore[attr-defined]


def test_the_visibility_recorded_is_the_one_the_hub_reports_not_the_one_asked_for(
    release: Release,
) -> None:
    # `--public` was asked for, but the repository already existed and the Hub says it is private:
    # `create_repo` does not change an existing repository, so the record must say private.
    hub = RecordingHub(reports_private=True)

    result = run_upload(release, hub, public=True)

    assert hub.repo_calls[0].private is False
    assert result.visibility == "private"  # type: ignore[attr-defined]


# ---- a client that cannot say ----------------------------------------------------------------------


def test_an_error_of_the_visibility_question_is_an_upstream_error_by_class_and_no_commit(
    release: Release,
) -> None:
    token = fake_token()
    hub = RecordingHub(visibility_error=TokenInError(f"rejected {token}"))

    with pytest.raises(UpstreamError) as raised:
        run_upload(release, hub)

    assert "TokenInError" in str(raised.value)
    assert token not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert hub.commit_calls == []


@pytest.mark.parametrize("answer", [None, "private", 1, 0])
def test_an_answer_that_is_not_a_boolean_is_an_upstream_error_and_no_commit(
    release: Release, answer: object
) -> None:
    hub = RecordingHub(reports_private=answer)

    with pytest.raises(UpstreamError, match="whether the repository is private"):
        run_upload(release, hub)

    assert hub.commit_calls == []


def test_an_error_of_ours_from_the_visibility_question_passes_through(
    release: Release,
) -> None:
    hub = RecordingHub(visibility_error=ConfigError("a message of ours"))

    with pytest.raises(ConfigError, match="a message of ours"):
        run_upload(release, hub)


# ---- the adapter -----------------------------------------------------------------------------------


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeHfApi, list[str]]:
    import huggingface_hub

    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    return api, forbid_token_handling(monkeypatch)


@pytest.mark.parametrize("private", [True, False])
def test_the_adapter_reads_the_visibility_from_the_repository_info_of_a_model_repo(
    fake_api: tuple[FakeHfApi, list[str]], private: bool
) -> None:
    api, touched = fake_api
    api.private = private
    client = HfHubClient("tok")

    answer = client.repo_is_private("org/name", token="tok")

    assert answer is private
    assert api.calls == [("repo_info", ("org/name",), {"repo_type": "model"})]
    assert touched == []


def test_the_adapter_does_not_guess_when_the_library_has_no_visibility(
    fake_api: tuple[FakeHfApi, list[str]],
) -> None:
    api, _ = fake_api
    api.private = None
    client = HfHubClient("tok")

    with pytest.raises(UpstreamError, match="whether the repository is private"):
        client.repo_is_private("org/name", token="tok")


def test_the_adapter_refuses_a_token_other_than_the_one_it_was_built_with_for_the_question(
    fake_api: tuple[FakeHfApi, list[str]],
) -> None:
    api, _ = fake_api
    client = HfHubClient("t0k3n-built")

    with pytest.raises(ConfigError) as raised:
        client.repo_is_private("org/name", token="t0k3n-call")

    assert "t0k3n" not in str(raised.value)
    assert api.calls == []


def test_the_adapter_asks_before_it_commits_through_upload(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, _ = fake_api
    token = fake_token()
    plan = plan_upload(release.stage, release.repo_id)

    result = upload(plan, release.stage, HfHubClient(token), token)

    assert [call for call, *_ in api.calls] == [
        "create_repo",
        "repo_info",
        "create_commit",
    ]
    assert result.visibility == "private"  # type: ignore[attr-defined]


def test_the_adapter_stops_before_the_commit_for_a_public_repository(
    release: Release, fake_api: tuple[FakeHfApi, list[str]]
) -> None:
    api, _ = fake_api
    api.private = False
    token = fake_token()
    plan = plan_upload(release.stage, release.repo_id)

    with pytest.raises(ConfigError, match=REFUSAL):
        upload(plan, release.stage, HfHubClient(token), token)

    assert [call for call, *_ in api.calls] == ["create_repo", "repo_info"]
