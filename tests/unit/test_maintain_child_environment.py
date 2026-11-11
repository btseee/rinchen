"""The environment of a child of `rinchen maintain check` holds none of the parent's secrets
(D14 fix wave 1, finding M7).

The check starts `uv`, a pytest rerun and the parity command, and used to hand each a copy of its
own environment, tokens included: a rerun runs the project's tests, which a changed dependency
may have changed. `child_environment` now leaves out `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` and
every variable whose name ends in `_TOKEN`, `_SECRET`, `_PASSWORD` or `_API_KEY`, in addition to
the switches it already sets, and removes nothing else. The match is on the upper-case spelling
only (a variable written in lower case is kept): the names above are the convention of the tools
the project uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.maintain.commands import (
    HIDDEN_ENVIRONMENT,
    SECRET_SUFFIXES,
    UV_ENVIRONMENT,
    child_environment,
)
from tests.support.maintain_check import execute, golden_runner, make_seams
from tests.support.maintain_cli import Workspace, install, invoke, make_workspace
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

DROPPED = [
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "CI_TOKEN",
    "UV_PUBLISH_TOKEN",
    "AWS_SECRET",
    "CLIENT_SECRET",
    "DB_PASSWORD",
    "UV_INDEX_PRIVATE_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "X_API_KEY",
    "_TOKEN",
]
KEPT = [
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "UV_OFFLINE",
    "UV_CACHE_DIR",
    "UV_INDEX_URL",
    "VIRTUAL_ENV",
    "TOKENIZERS_PARALLELISM",
    "SECRETS_DIR",
    "PASSWORD_FILE",
    "API_KEY_FILE",
    "MYTOKEN",
    "MYSECRET",
    "my_token",
    "hf_token",
    "x_api_key",
    "WORKTOKENS",
]


def test_the_suffixes_are_the_four_of_the_ruling() -> None:
    assert SECRET_SUFFIXES == ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")


@pytest.mark.parametrize("uv", [False, True])
@pytest.mark.parametrize("name", DROPPED)
def test_a_secret_of_the_parent_is_not_in_the_child(name: str, uv: bool) -> None:
    child = child_environment({"PATH": "/usr/bin", name: "swordfish-0451"}, uv=uv)

    assert name not in child
    assert "swordfish-0451" not in child.values()
    assert child["PATH"] == "/usr/bin"


@pytest.mark.parametrize("uv", [False, True])
@pytest.mark.parametrize("name", KEPT)
def test_any_other_variable_of_the_parent_is_kept(name: str, uv: bool) -> None:
    child = child_environment({name: "kept-value"}, uv=uv)

    assert child[name] == "kept-value"


def test_the_child_is_the_parent_less_its_secrets_with_the_switches_on_top() -> None:
    parent = {name: f"value of {name}" for name in [*DROPPED, *KEPT]}

    plain = child_environment(parent)
    for_uv = child_environment(parent, uv=True)

    kept = {name: parent[name] for name in KEPT}
    assert plain == {**kept, **HIDDEN_ENVIRONMENT}
    assert for_uv == {**kept, **HIDDEN_ENVIRONMENT, **UV_ENVIRONMENT}


def test_the_parent_is_never_modified() -> None:
    parent = {"HF_TOKEN": "x", "KEEP": "me"}

    child_environment(parent)

    assert parent == {"HF_TOKEN": "x", "KEEP": "me"}


def test_a_secret_cannot_be_put_back_by_the_switches() -> None:
    # The hidden switches and the uv switches are fixed names; none of them is a secret.
    assert not [
        name
        for name in (*HIDDEN_ENVIRONMENT, *UV_ENVIRONMENT)
        if name.endswith(SECRET_SUFFIXES)
    ]


# ---- every command of a run ------------------------------------------------------------------------


def test_no_command_of_a_run_is_given_a_secret(tmp_path: Path) -> None:
    runner = golden_runner()
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "UV_OFFLINE": "1",
        "HF_TOKEN": "hf-secret-1",
        "HUGGING_FACE_HUB_TOKEN": "hf-secret-2",
        "SOME_API_KEY": "key-3",
    }

    execute(tmp_path, runner, seams=make_seams(runner, parent_env=parent))

    assert len(runner.calls) >= 4  # the lock check, the dry run and the two reruns
    for call in runner.calls:
        assert not [name for name in call.env if name.endswith(SECRET_SUFFIXES)]
        assert (call.env["PATH"], call.env["HOME"], call.env["UV_OFFLINE"]) == (
            "/usr/bin",
            "/home/someone",
            "1",
        )


def test_the_command_hands_none_of_the_secrets_of_its_own_environment_to_any_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws: Workspace = make_workspace(tmp_path)
    runner = install(monkeypatch)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CI_TOKEN", "X_PASSWORD"):
        monkeypatch.setenv(name, "swordfish-0451")
    monkeypatch.setenv("UV_OFFLINE", "1")

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert (
        len(runner.calls) >= 5
    )  # the lock check, the dry run, two reruns and uv --version
    for call in runner.calls:
        assert "swordfish-0451" not in call.env.values()
        assert call.env["UV_OFFLINE"] == "1"
