"""Tests for `rinchen.train.checkpoints`: which step directories are complete, the
identity of a run and the record of its launch.

The step directories are fabricated by `tests.support.step_dirs` with real tiny
safetensors files and a real zip archive, so `is_complete` runs its real checks. Each
damage is made to one file of a complete directory, and the problem string must name
that file.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import random
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.train import checkpoints
from rinchen.train.checkpoints import (
    LAUNCH_FILE,
    ArchitectureMismatch,
    ContinuedFrom,
    RunIdentity,
    StepDirectory,
    architecture_mismatch,
    identity_of,
    is_complete,
    latest_complete,
    read_launch_record,
    step_directories,
    step_of,
    write_launch_record,
)
from tests.support import step_dirs
from tests.support.eole_train_config import make_config
from tests.support.step_dirs import (
    CONFIG_FILE,
    OPTIMIZER_FILE,
    REQUIRED_FILES,
    SPM_ARTIFACT,
    VOCAB_FILE,
    WEIGHTS_FILE,
    eole_model_section,
    make_run,
    make_step_dir,
)

HEX = "ab" * 32
NOW = "2026-11-08T12:00:00+00:00"


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "ckpt"


@pytest.fixture
def step_dir(model_path: Path) -> Path:
    return make_step_dir(model_path, 100)


# ---- is_complete: a complete directory ------------------------------------------


def test_a_complete_step_directory_has_no_problem(step_dir: Path) -> None:
    assert is_complete(step_dir) is None


def test_the_spm_artifact_is_not_needed_for_a_directory_to_be_complete(
    step_dir: Path,
) -> None:
    step_dirs.delete(step_dir, SPM_ARTIFACT)

    assert is_complete(step_dir) is None


def test_extra_files_in_a_complete_directory_are_not_a_problem(step_dir: Path) -> None:
    (step_dir / "notes.txt").write_text("x", encoding="utf-8")

    assert is_complete(step_dir) is None


def test_is_complete_changes_nothing_in_the_directory(step_dir: Path) -> None:
    before = {p.name: p.read_bytes() for p in step_dir.iterdir()}

    is_complete(step_dir)

    assert {p.name: p.read_bytes() for p in step_dir.iterdir()} == before


def test_the_module_is_light_to_import_and_checking_a_directory_loads_no_framework(
    step_dir: Path,
) -> None:
    import subprocess
    import sys

    heavy = "('torch', 'eole', 'polars', 'safetensors')"
    code = (
        "import sys; from pathlib import Path; "
        "from rinchen.train.checkpoints import is_complete; "
        f"print(sorted(m for m in {heavy} if m in sys.modules)); "
        f"print(is_complete(Path({str(step_dir)!r}))); "
        f"print(sorted(m for m in {heavy} if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    # Nothing heavy on import; the weights are opened through safetensors, lazily, and
    # no framework is loaded to do it.
    assert done.stdout.split("\n")[:3] == ["[]", "None", "['safetensors']"]


# ---- is_complete: a missing file ------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_a_missing_file_is_named(step_dir: Path, name: str) -> None:
    step_dirs.delete(step_dir, name)

    assert is_complete(step_dir) == f"{name}: is missing"


def test_a_directory_that_does_not_exist_is_a_problem_not_an_error(
    model_path: Path,
) -> None:
    assert is_complete(model_path / "step_7") == "step_7: does not exist"


def test_a_file_where_the_directory_should_be_is_a_problem(model_path: Path) -> None:
    model_path.mkdir()
    (model_path / "step_7").write_text("x", encoding="utf-8")

    assert is_complete(model_path / "step_7") == "step_7: is not a directory"


# ---- is_complete: damaged files -------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_an_empty_file_is_named(step_dir: Path, name: str) -> None:
    step_dirs.truncate(step_dir, name, 0)

    assert is_complete(step_dir) == f"{name}: is empty"


@pytest.mark.parametrize(
    "name", [WEIGHTS_FILE, OPTIMIZER_FILE, CONFIG_FILE, VOCAB_FILE]
)
def test_a_file_cut_short_at_any_length_is_named(model_path: Path, name: str) -> None:
    size = (make_step_dir(model_path, 1) / name).stat().st_size
    for length in step_dirs.lengths_to_truncate_to(size):
        step_dir = make_step_dir(model_path, 10 + length)
        step_dirs.truncate(step_dir, name, length)

        problem = is_complete(step_dir)

        assert problem is not None, (name, length)
        assert problem.startswith(f"{name}: "), (name, length, problem)


def test_weights_cut_short_are_not_a_complete_safetensors_file(step_dir: Path) -> None:
    size = (step_dir / WEIGHTS_FILE).stat().st_size
    step_dirs.truncate(step_dir, WEIGHTS_FILE, size - 1)

    problem = is_complete(step_dir)

    assert problem is not None
    assert problem.startswith(f"{WEIGHTS_FILE}: is not a complete safetensors file (")


def test_weights_with_bytes_added_are_not_a_complete_safetensors_file(
    step_dir: Path,
) -> None:
    with (step_dir / WEIGHTS_FILE).open("ab") as handle:
        handle.write(b"\0" * 16)

    problem = is_complete(step_dir)

    assert problem is not None and problem.startswith(f"{WEIGHTS_FILE}: is not a ")


def test_weights_that_are_not_safetensors_at_all_are_named(step_dir: Path) -> None:
    step_dirs.overwrite(step_dir, WEIGHTS_FILE, b"\xff" * 300)

    problem = is_complete(step_dir)

    assert problem is not None and problem.startswith(f"{WEIGHTS_FILE}: ")


def test_weights_that_hold_no_tensor_are_named(step_dir: Path) -> None:
    from safetensors.numpy import save_file

    save_file({}, str(step_dir / WEIGHTS_FILE))

    assert is_complete(step_dir) == f"{WEIGHTS_FILE}: holds no tensor"


def test_an_optimizer_file_that_is_not_a_zip_archive_is_named(step_dir: Path) -> None:
    step_dirs.overwrite(step_dir, OPTIMIZER_FILE, b"this is not a zip archive at all")

    problem = is_complete(step_dir)

    assert problem is not None
    assert problem.startswith(f"{OPTIMIZER_FILE}: is not a complete zip archive (")


def test_an_optimizer_archive_with_no_entry_is_named(step_dir: Path) -> None:
    with zipfile.ZipFile(step_dir / OPTIMIZER_FILE, "w"):
        pass

    assert is_complete(step_dir) == f"{OPTIMIZER_FILE}: is a zip archive with no entry"


def _region_positions(optimizer: Path) -> dict[str, list[int]]:
    """The byte positions of the parts of the zip archive `optimizer`: the local headers
    (with their names), the central directory and the end-of-central-directory record
    (the archive has no comment, so it is the last 22 bytes)."""
    content = optimizer.read_bytes()
    with zipfile.ZipFile(optimizer) as archive:
        central_start = archive.start_dir
        offsets = [info.header_offset for info in archive.infolist()]
    local: list[int] = []
    for offset in offsets:
        name_length, extra_length = struct.unpack_from("<HH", content, offset + 26)
        local += range(offset, offset + 30 + name_length + extra_length)
    return {
        "local headers": local,
        "central directory": list(range(central_start, len(content) - 22)),
        "end record": list(range(len(content) - 22, len(content))),
    }


FLIPS_PER_REGION = 1000


@pytest.mark.parametrize("region", ["local headers", "central directory", "end record"])
def test_a_flipped_bit_in_the_zip_structure_is_a_problem_never_an_exception(
    step_dir: Path, region: str
) -> None:
    # `is_complete` is documented never to raise: a damaged zip reader error of any
    # kind (a name that is not UTF-8, a version it cannot read) is a named problem.
    optimizer = step_dir / OPTIMIZER_FILE
    original = optimizer.read_bytes()
    positions = _region_positions(optimizer)[region]
    random_flips = random.Random(20261108)
    problems = 0

    for _ in range(FLIPS_PER_REGION):
        damaged = bytearray(original)
        damaged[random_flips.choice(positions)] ^= 1 << random_flips.randrange(8)
        optimizer.write_bytes(bytes(damaged))

        problem = is_complete(step_dir)

        assert problem is None or problem.startswith(f"{OPTIMIZER_FILE}: ")
        problems += problem is not None

    # The matrix does reach the code that reads the damage: most flips in the
    # central directory and the end record are found out.
    if region != "local headers":
        assert problems > FLIPS_PER_REGION // 10


def test_the_optimizer_archive_has_a_member_name_stored_as_utf8(step_dir: Path) -> None:
    # The fabricated archive must hold what lets a flip in the central directory break
    # the decoding of a name (the flag says: decode as UTF-8), beside the plain names.
    with zipfile.ZipFile(step_dir / OPTIMIZER_FILE) as archive:
        members = archive.infolist()

    assert [m.filename for m in members if m.flag_bits & 0x800] == [
        step_dirs.UTF8_MEMBER
    ]
    assert [m.filename for m in members if not m.flag_bits & 0x800] == [
        "optimizer/data.pkl",
        "optimizer/byteorder",
    ]


def test_a_member_name_that_is_not_valid_utf8_is_named_as_a_problem(
    step_dir: Path,
) -> None:
    optimizer = step_dir / OPTIMIZER_FILE
    content = bytearray(optimizer.read_bytes())
    # The second copy of the name is the one in the central directory.
    name = step_dirs.UTF8_MEMBER.encode("utf-8")
    central_copy = content.rindex(name)
    content[central_copy + 11] = 0xFD  # inside a two-byte letter: no UTF-8 sequence
    optimizer.write_bytes(bytes(content))

    problem = is_complete(step_dir)

    assert problem is not None
    assert problem.startswith(f"{OPTIMIZER_FILE}: is not a readable zip archive (")


def test_a_zip_version_the_reader_does_not_know_is_named_as_a_problem(
    step_dir: Path,
) -> None:
    optimizer = step_dir / OPTIMIZER_FILE
    content = bytearray(optimizer.read_bytes())
    # "version needed to extract" of the first central directory header.
    content[content.index(b"PK\x01\x02") + 6] = 0x7F
    optimizer.write_bytes(bytes(content))

    problem = is_complete(step_dir)

    assert problem is not None
    assert problem.startswith(f"{OPTIMIZER_FILE}: is not a readable zip archive (")


@pytest.mark.parametrize(
    "error",
    [
        OverflowError("Python int too large to convert to C long"),
        EOFError("Ran out of input"),
    ],
    ids=["overflow", "end-of-file"],
)
def test_the_other_errors_a_zip_reader_can_raise_are_problems_too(
    step_dir: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    # Not reproduced by flipping bits (the fuzz above finds only the three kinds), but
    # offset arithmetic and short reads of a damaged archive can end this way.
    def failing_reader(*_: object, **__: object) -> zipfile.ZipFile:
        raise error

    monkeypatch.setattr(checkpoints.zipfile, "ZipFile", failing_reader)

    problem = is_complete(step_dir)

    assert problem == (
        f"{OPTIMIZER_FILE}: is not a readable zip archive "
        f"({type(error).__name__}: {error})"
    )


@pytest.mark.parametrize("name", [CONFIG_FILE, VOCAB_FILE])
@pytest.mark.parametrize(
    "content",
    [b"{", b'{"a": ', b"not json", b"\xff\xfe\x00", b"[" * 100_000],
    ids=["open-brace", "cut-value", "words", "bad-utf8", "deeply-nested"],
)
def test_json_that_does_not_parse_is_named(
    step_dir: Path, name: str, content: bytes
) -> None:
    step_dirs.overwrite(step_dir, name, content)

    problem = is_complete(step_dir)

    assert problem is not None
    assert problem.startswith(f"{name}: is not valid JSON (")


@pytest.mark.parametrize("name", [CONFIG_FILE, VOCAB_FILE])
@pytest.mark.parametrize("content", [b"[]", b"123", b'"text"', b"null", b"true"])
def test_json_that_is_not_an_object_is_named(
    step_dir: Path, name: str, content: bytes
) -> None:
    step_dirs.overwrite(step_dir, name, content)

    assert is_complete(step_dir) == f"{name}: is not a JSON object"


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_a_directory_in_place_of_a_file_is_named(step_dir: Path, name: str) -> None:
    step_dirs.replace_with_directory(step_dir, name)

    assert is_complete(step_dir) == f"{name}: is not a regular file"


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_a_symlink_to_nothing_is_named(step_dir: Path, name: str) -> None:
    step_dirs.replace_with_dangling_symlink(step_dir, name)

    assert is_complete(step_dir) == (
        f"{name}: is a symbolic link, not a file Eole wrote"
    )


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_a_symlink_to_a_good_file_elsewhere_is_named_too(
    tmp_path: Path, step_dir: Path, name: str
) -> None:
    other = make_step_dir(tmp_path / "other", 100)
    step_dirs.replace_with_symlink_to(step_dir, name, other / name)

    assert is_complete(step_dir) == (
        f"{name}: is a symbolic link, not a file Eole wrote"
    )


def test_a_symlink_loop_in_place_of_a_file_is_a_problem_not_an_error(
    step_dir: Path,
) -> None:
    (step_dir / WEIGHTS_FILE).unlink()
    os.symlink(WEIGHTS_FILE, step_dir / WEIGHTS_FILE)

    assert is_complete(step_dir) == (
        f"{WEIGHTS_FILE}: is a symbolic link, not a file Eole wrote"
    )


def test_a_file_that_cannot_be_read_is_a_problem_not_an_error(
    step_dir: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads files whatever their mode")
    (step_dir / CONFIG_FILE).chmod(0)
    try:
        problem = is_complete(step_dir)
    finally:
        (step_dir / CONFIG_FILE).chmod(0o644)

    assert problem is not None
    assert problem.startswith(f"{CONFIG_FILE}: cannot be read (")


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_a_file_that_is_not_allowed_to_be_read_is_named_with_its_real_error(
    step_dir: Path, name: str
) -> None:
    # safetensors words a refused open as FileNotFoundError ("No such file"): the
    # problem has to say what happened, not what the library says.
    if os.geteuid() == 0:
        pytest.skip("root reads files whatever their mode")
    (step_dir / name).chmod(0)
    try:
        problem = is_complete(step_dir)
    finally:
        (step_dir / name).chmod(0o644)

    assert problem == f"{name}: cannot be read (PermissionError: Permission denied)"


def test_an_os_error_while_opening_the_weights_is_reported_with_its_class(
    monkeypatch: pytest.MonkeyPatch, step_dir: Path
) -> None:
    def failing_open(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr("safetensors.safe_open", failing_open)

    assert is_complete(step_dir) == (
        f"{WEIGHTS_FILE}: cannot be read (OSError: Input/output error)"
    )


def test_an_os_error_without_a_message_of_its_own_is_reported_with_its_class(
    monkeypatch: pytest.MonkeyPatch, step_dir: Path
) -> None:
    def failing_open(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("no such thing")

    monkeypatch.setattr("safetensors.safe_open", failing_open)

    assert is_complete(step_dir) == (
        f"{WEIGHTS_FILE}: cannot be read (FileNotFoundError: no such thing)"
    )


def test_the_first_problem_in_the_order_eole_saves_them_backwards_is_reported(
    step_dir: Path,
) -> None:
    # config.json is written last, so it is looked at first; then vocab, optimizer,
    # weights.
    step_dirs.truncate(step_dir, WEIGHTS_FILE, 5)
    step_dirs.truncate(step_dir, OPTIMIZER_FILE, 5)
    assert is_complete(step_dir) is not None
    assert str(is_complete(step_dir)).startswith(f"{OPTIMIZER_FILE}: ")

    step_dirs.delete(step_dir, VOCAB_FILE)
    assert is_complete(step_dir) == f"{VOCAB_FILE}: is missing"

    step_dirs.delete(step_dir, CONFIG_FILE)
    assert is_complete(step_dir) == f"{CONFIG_FILE}: is missing"


def test_a_bfloat16_checkpoint_is_complete(step_dir: Path) -> None:
    # The checks read the header only, so a dtype numpy lacks is no obstacle.
    import struct

    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    ).encode()
    (step_dir / WEIGHTS_FILE).write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 4
    )

    assert is_complete(step_dir) is None


# ---- step_directories -----------------------------------------------------------


def test_step_directories_are_sorted_by_step_number_not_by_name(
    model_path: Path,
) -> None:
    make_run(model_path, (100, 20, 9, 1000))

    listed = step_directories(model_path)

    assert [d.step for d in listed] == [9, 20, 100, 1000]
    assert [d.path for d in listed] == [
        model_path / "step_9",
        model_path / "step_20",
        model_path / "step_100",
        model_path / "step_1000",
    ]
    assert all(d.complete and d.problem is None for d in listed)


def test_each_step_directory_carries_its_own_problem(model_path: Path) -> None:
    make_run(model_path, (50, 100, 150))
    step_dirs.truncate(model_path / "step_100", WEIGHTS_FILE, 10)
    step_dirs.delete(model_path / "step_150", CONFIG_FILE)

    listed = step_directories(model_path)

    assert [(d.step, d.complete) for d in listed] == [
        (50, True),
        (100, False),
        (150, False),
    ]
    assert listed[0].problem is None
    assert str(listed[1].problem).startswith(f"{WEIGHTS_FILE}: ")
    assert listed[2].problem == f"{CONFIG_FILE}: is missing"


def test_a_step_directory_value_is_immutable_and_knows_if_it_is_complete() -> None:
    complete = StepDirectory(5, Path("x/step_5"), None)
    broken = StepDirectory(6, Path("x/step_6"), "config.json: is missing")

    assert complete.complete is True
    assert broken.complete is False
    with pytest.raises(AttributeError):
        complete.step = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    [
        "step_",
        "step_x",
        "step_5x",
        "step_x5",
        "step_0",
        "step_00",
        "step_007",
        "step_-5",
        "step_+5",
        "step_5.0",
        "step_5 ",
        " step_5",
        "Step_5",
        "step5",
        "step_5_old",
        "old_step_5",
        "step_" + chr(0x665),  # an Arabic-Indic digit five
        "step_" + chr(0xFF15),  # a full-width digit five
        "step_1" + chr(0x665),  # one, then an Arabic-Indic digit five
        "step_2" + chr(0xFF15),  # two, then a full-width digit five
    ],
)
def test_a_directory_whose_name_is_not_exactly_step_and_a_canonical_number_is_left_out(
    model_path: Path, name: str
) -> None:
    make_step_dir(model_path, 5)
    (model_path / name).mkdir()

    assert [d.step for d in step_directories(model_path)] == [5]


def test_step_007_is_not_step_7_and_a_real_step_7_is_still_found(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 7)
    copy_of = model_path / "step_007"
    copy_of.mkdir()

    listed = step_directories(model_path)

    assert [(d.step, d.path.name) for d in listed] == [(7, "step_7")]


def test_a_file_named_like_a_step_directory_is_left_out(model_path: Path) -> None:
    make_step_dir(model_path, 5)
    (model_path / "step_9").write_text("x", encoding="utf-8")

    assert [d.step for d in step_directories(model_path)] == [5]


def test_a_symlink_to_a_directory_named_like_a_step_directory_is_left_out(
    tmp_path: Path, model_path: Path
) -> None:
    make_step_dir(model_path, 5)
    elsewhere = make_step_dir(tmp_path / "foreign", 9)
    (model_path / "step_9").symlink_to(elsewhere, target_is_directory=True)

    assert [d.step for d in step_directories(model_path)] == [5]


def test_a_symlink_to_nothing_named_like_a_step_directory_is_left_out(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 5)
    (model_path / "step_9").symlink_to("deleted", target_is_directory=True)

    assert [d.step for d in step_directories(model_path)] == [5]


def test_the_links_eole_leaves_in_the_checkpoint_directory_are_not_step_directories(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100))
    step_dirs.link_newest_step(model_path, 100)

    assert [d.step for d in step_directories(model_path)] == [50, 100]


def test_the_root_links_pointing_at_a_deleted_directory_do_not_matter(
    model_path: Path,
) -> None:
    import shutil

    make_run(model_path, (50, 100))
    step_dirs.link_newest_step(model_path, 100)
    shutil.rmtree(model_path / "step_100")

    listed = step_directories(model_path)

    assert [d.step for d in listed] == [50]
    assert latest_complete(model_path).directory == listed[0]


def test_unrelated_files_in_the_checkpoint_directory_are_left_out(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 5)
    (model_path / "notes.txt").write_text("x", encoding="utf-8")
    (model_path / "extra").mkdir()

    assert [d.step for d in step_directories(model_path)] == [5]


def test_a_huge_step_number_is_an_ordinary_number(model_path: Path) -> None:
    huge = 10**30
    make_step_dir(model_path, huge)
    make_step_dir(model_path, 5)

    assert [d.step for d in step_directories(model_path)] == [5, huge]


def test_an_absent_checkpoint_directory_holds_no_step_directory(tmp_path: Path) -> None:
    assert step_directories(tmp_path / "never-made") == []


def test_an_empty_checkpoint_directory_holds_no_step_directory(
    model_path: Path,
) -> None:
    model_path.mkdir()

    assert step_directories(model_path) == []


def test_a_checkpoint_directory_that_is_a_file_is_a_data_error(
    model_path: Path,
) -> None:
    model_path.write_text("x", encoding="utf-8")

    with pytest.raises(DataError, match="is not a directory"):
        step_directories(model_path)


def test_a_checkpoint_directory_that_is_a_link_to_nothing_is_a_data_error(
    model_path: Path,
) -> None:
    model_path.symlink_to("deleted", target_is_directory=True)

    with pytest.raises(DataError, match="is not a directory"):
        step_directories(model_path)


def test_a_checkpoint_directory_reached_through_a_symlink_is_listed(
    tmp_path: Path, model_path: Path
) -> None:
    make_run(tmp_path / "real", (5,))
    model_path.symlink_to(tmp_path / "real", target_is_directory=True)

    assert [d.step for d in step_directories(model_path)] == [5]


def test_a_checkpoint_directory_that_cannot_be_listed_is_a_data_error(
    model_path: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root lists any directory")
    make_step_dir(model_path, 5)
    model_path.chmod(0)
    try:
        with pytest.raises(DataError, match="cannot list the checkpoint directory"):
            step_directories(model_path)
    finally:
        model_path.chmod(0o755)


# ---- latest_complete ------------------------------------------------------------


def test_the_latest_complete_directory_is_the_highest_complete_one(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100, 150))

    point = latest_complete(model_path)

    assert point.directory == StepDirectory(150, model_path / "step_150", None)
    assert point.incomplete_newer == ()


def test_incomplete_directories_above_the_latest_complete_one_are_reported(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100, 150, 200))
    step_dirs.truncate(model_path / "step_150", WEIGHTS_FILE, 10)
    step_dirs.delete(model_path / "step_200", CONFIG_FILE)

    point = latest_complete(model_path)

    assert point.directory is not None and point.directory.step == 100
    assert [(d.step, d.complete) for d in point.incomplete_newer] == [
        (150, False),
        (200, False),
    ]
    assert str(point.incomplete_newer[0].problem).startswith(f"{WEIGHTS_FILE}: ")
    assert point.incomplete_newer[1].problem == f"{CONFIG_FILE}: is missing"


def test_an_incomplete_directory_below_the_latest_complete_one_is_not_reported(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100))
    step_dirs.truncate(model_path / "step_50", OPTIMIZER_FILE, 3)

    point = latest_complete(model_path)

    assert point.directory is not None and point.directory.step == 100
    assert point.incomplete_newer == ()


def test_a_half_written_newest_directory_falls_back_to_the_one_before(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100))
    for name in (CONFIG_FILE, VOCAB_FILE):
        step_dirs.delete(model_path / "step_100", name)

    point = latest_complete(model_path)

    assert point.directory is not None and point.directory.step == 50
    assert [d.step for d in point.incomplete_newer] == [100]


def test_no_complete_directory_gives_none_and_every_incomplete_one(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100))
    step_dirs.truncate(model_path / "step_50", WEIGHTS_FILE, 0)
    step_dirs.truncate(model_path / "step_100", WEIGHTS_FILE, 0)

    point = latest_complete(model_path)

    assert point.directory is None
    assert [d.step for d in point.incomplete_newer] == [50, 100]


def test_an_absent_or_empty_checkpoint_directory_has_no_latest(
    tmp_path: Path, model_path: Path
) -> None:
    assert latest_complete(tmp_path / "never-made") == checkpoints.ResumePoint(None, ())
    model_path.mkdir()
    assert latest_complete(model_path) == checkpoints.ResumePoint(None, ())


def test_a_checkpoint_directory_that_is_a_file_is_a_data_error_for_latest_too(
    model_path: Path,
) -> None:
    model_path.write_text("x", encoding="utf-8")

    with pytest.raises(DataError):
        latest_complete(model_path)


def test_the_latest_complete_directory_ignores_step_names_that_are_not_canonical(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 50)
    (model_path / "step_0100").mkdir()
    (model_path / "step_x").mkdir()

    assert latest_complete(model_path).directory == StepDirectory(
        50, model_path / "step_50", None
    )


def test_latest_complete_stops_examining_at_the_first_complete_directory(
    model_path: Path,
) -> None:
    make_run(model_path, (50, 100))
    # Below the answer and unreadable: it must not even be looked at.
    (model_path / "step_50" / CONFIG_FILE).write_bytes(b"{")

    assert latest_complete(model_path).directory is not None


# ---- the model of a checkpoint --------------------------------------------------
#
# Eole ignores the `model` section of the config when it resumes and builds the model of
# the checkpoint's own saved config, so a step directory copied in from another run would
# be trained on as if it were this one's. `architecture_mismatch` compares what the two
# sides both say. The saved sections here are the real layout (`eole_model_section`).


def wanted_model() -> dict[str, Any]:
    """The `model` section of the config the real tiny run was started from."""
    model = make_config()["model"]
    assert isinstance(model, dict)
    return model


def test_the_model_section_of_a_real_checkpoint_matches_the_config_it_came_from() -> (
    None
):
    saved = {"model": eole_model_section()}

    assert architecture_mismatch(saved, wanted_model()) is None


@pytest.mark.parametrize(
    ("path", "other", "saved_value"),
    [
        (("architecture",), "transformer_lm", "transformer"),
        (("hidden_size",), 128, 64),
        (("heads",), 8, 4),
        (("transformer_ff",), 256, 128),
        (("embeddings", "word_vec_size"), 128, 64),
        (
            ("embeddings", "position_encoding_type"),
            "Rotary",
            "SinusoidalInterleaved",
        ),
        (("encoder", "layers"), 6, 2),
        (("decoder", "layers"), 3, 2),
        (("encoder", "hidden_size"), 128, 64),
        (("encoder", "heads"), 8, 4),
        (("encoder", "transformer_ff"), 256, 128),
        (("encoder", "position_encoding_type"), "Rotary", "SinusoidalInterleaved"),
        (("decoder", "hidden_size"), 128, 64),
        (("decoder", "heads"), 8, 4),
        (("decoder", "transformer_ff"), 256, 128),
        (("decoder", "position_encoding_type"), "Rotary", "SinusoidalInterleaved"),
    ],
    ids=lambda value: ".".join(value) if isinstance(value, tuple) else str(value),
)
def test_a_key_that_differs_is_named_with_both_values(
    path: tuple[str, ...], other: object, saved_value: object
) -> None:
    wanted = wanted_model()
    section = wanted
    for key in path[:-1]:
        section = section.setdefault(key, {})
    section[path[-1]] = other

    mismatch = architecture_mismatch({"model": eole_model_section()}, wanted)

    assert mismatch == ArchitectureMismatch(".".join(path), saved_value, other)


def test_numbers_written_as_text_in_the_config_are_the_numbers_eole_saved() -> None:
    # Eole accepts `hidden_size: "64"` and saves 64: a run whose config quotes its
    # numbers must not see its own checkpoint taken for another model's.
    wanted = wanted_model()
    wanted["hidden_size"] = str(wanted["hidden_size"])
    wanted["encoder"]["layers"] = str(wanted["encoder"]["layers"])
    wanted["heads"] = " " + str(wanted["heads"]) + " "

    assert architecture_mismatch({"model": eole_model_section()}, wanted) is None


@pytest.mark.parametrize("other", ["128", "sixty-four", "6e1", "6_4", "-64", ""])
def test_text_that_is_not_the_saved_number_is_still_a_mismatch(other: str) -> None:
    wanted = wanted_model()
    wanted["hidden_size"] = other

    mismatch = architecture_mismatch({"model": eole_model_section()}, wanted)

    assert mismatch is not None
    assert mismatch.key == "hidden_size"
    assert mismatch.wanted == other


def test_the_mismatch_reads_as_a_sentence_with_both_values() -> None:
    wanted = wanted_model()
    wanted["encoder"]["layers"] = 24

    mismatch = architecture_mismatch({"model": eole_model_section()}, wanted)

    assert str(mismatch) == (
        "model.encoder.layers is 2 in the checkpoint but 24 in the config"
    )


def test_the_first_key_that_differs_is_the_one_reported() -> None:
    wanted = wanted_model()
    wanted["decoder"]["layers"] = 12
    wanted["hidden_size"] = 128

    mismatch = architecture_mismatch({"model": eole_model_section()}, wanted)

    assert mismatch is not None
    assert mismatch.key == "hidden_size"


def test_a_key_the_config_does_not_set_is_not_compared() -> None:
    wanted = wanted_model()
    del wanted["heads"]
    del wanted["encoder"]
    saved = {"model": eole_model_section(heads=16, encoder_layers=24)}

    assert architecture_mismatch(saved, wanted) is None


@pytest.mark.parametrize(
    "saved_config",
    [
        pytest.param({}, id="no model section"),
        pytest.param({"model": None}, id="a null model section"),
        pytest.param({"model": "transformer"}, id="a text model section"),
        pytest.param({"model": {}}, id="an empty model section"),
        pytest.param({"model": {"transformer_ff": 999}}, id="a key the config lacks"),
        pytest.param({"model": {"encoder": "x", "embeddings": 3}}, id="odd stacks"),
        pytest.param({"training": {"model_path": "x"}}, id="only a training section"),
    ],
)
def test_a_saved_config_without_the_keys_is_not_comparable(
    saved_config: dict[str, Any],
) -> None:
    wanted = {"hidden_size": 64, "encoder": {"layers": 2}, "heads": 4}

    assert architecture_mismatch(saved_config, wanted) is None


def test_a_saved_value_of_another_kind_is_a_mismatch_not_an_error() -> None:
    saved = {"model": eole_model_section()}
    saved["model"]["hidden_size"] = [64]  # not "64": decimal text is the number

    mismatch = architecture_mismatch(saved, wanted_model())

    assert mismatch == ArchitectureMismatch("hidden_size", [64], 64)


def test_the_comparison_leaves_both_sections_as_they_were() -> None:
    saved = {"model": eole_model_section()}
    wanted = wanted_model()
    wanted["heads"] = 8
    before = copy.deepcopy((saved, wanted))

    architecture_mismatch(saved, wanted)

    assert (saved, wanted) == before


def test_latest_complete_without_a_model_compares_nothing(model_path: Path) -> None:
    make_step_dir(model_path, 100, model=eole_model_section(hidden_size=128))

    point = latest_complete(model_path)

    assert point.directory is not None
    assert point.directory.step == 100


def test_latest_complete_resumes_from_a_checkpoint_of_the_config_s_model(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 100, model=eole_model_section())

    point = latest_complete(model_path, model=wanted_model())

    assert point.directory == StepDirectory(100, model_path / "step_100", None)
    assert point.incomplete_newer == ()


def test_latest_complete_skips_a_newer_checkpoint_of_another_model(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 100, model=eole_model_section())
    make_step_dir(model_path, 200, model=eole_model_section(hidden_size=128))

    point = latest_complete(model_path, model=wanted_model())

    assert point.directory is not None
    assert point.directory.step == 100
    (skipped,) = point.incomplete_newer
    assert skipped.step == 200
    assert not skipped.complete
    assert skipped.of_another_model
    assert skipped.problem == (
        "config.json: model.hidden_size is 128 in the checkpoint but 64 in the config"
    )


def test_a_directory_that_is_damaged_is_not_called_a_checkpoint_of_another_model(
    model_path: Path,
) -> None:
    step_dir = make_step_dir(model_path, 200, model=eole_model_section(hidden_size=128))
    step_dirs.truncate(step_dir, WEIGHTS_FILE, 5)

    point = latest_complete(model_path, model=wanted_model())

    assert point.directory is None
    (skipped,) = point.incomplete_newer
    assert not skipped.of_another_model
    assert skipped.problem is not None
    assert skipped.problem.startswith(f"{WEIGHTS_FILE}:")


def test_with_only_checkpoints_of_another_model_there_is_no_resume_point(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 100, model=eole_model_section(encoder_layers=6))
    make_step_dir(model_path, 200, model=eole_model_section(hidden_size=128))

    point = latest_complete(model_path, model=wanted_model())

    assert point.directory is None
    assert [(d.step, d.of_another_model) for d in point.incomplete_newer] == [
        (100, True),
        (200, True),
    ]


def test_a_checkpoint_whose_config_has_no_model_section_is_resumable(
    model_path: Path,
) -> None:
    make_step_dir(model_path, 100)

    point = latest_complete(model_path, model=wanted_model())

    assert point.directory is not None
    assert point.directory.step == 100


def test_step_directories_do_not_compare_the_model(model_path: Path) -> None:
    make_step_dir(model_path, 100, model=eole_model_section(hidden_size=128))

    (listed,) = step_directories(model_path)

    assert listed.complete


# ---- identity_of ----------------------------------------------------------------


def config() -> dict[str, Any]:
    return {
        "seed": 1234,
        "share_vocab": True,
        "data": {"en-mn.auth": {"path_src": "d/a.src", "weight": 3}},
        "model": {"hidden_size": 64, "encoder": {"layers": 2}},
        "training": {
            "model_path": "run/ckpt",
            "train_steps": 200,
            "learning_rate": 0.5,
            "gpu_ranks": [],
            "world_size": 1,
        },
    }


def test_the_identity_is_64_lowercase_hex_characters() -> None:
    identity = identity_of(config())

    assert len(identity) == 64
    assert identity == identity.lower()
    assert int(identity, 16) >= 0


def test_the_identity_is_the_sha256_of_the_canonical_json() -> None:
    import hashlib

    accent = chr(0xE9)  # text that is not ASCII stays as it is in the canonical JSON
    cfg = {"b": 1, "a": {"y": [1, 2], "x": accent}, "training": {"z": 1}}
    canonical = f'{{"a":{{"x":"{accent}","y":[1,2]}},"b":1,"training":{{"z":1}}}}'

    assert identity_of(cfg) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_the_identity_ignores_train_steps_and_train_from() -> None:
    base = config()
    longer = copy.deepcopy(base)
    longer["training"]["train_steps"] = 5000
    resumed = copy.deepcopy(base)
    resumed["training"]["train_from"] = "run/ckpt/step_100"
    neither = copy.deepcopy(base)
    del neither["training"]["train_steps"]

    assert identity_of(longer) == identity_of(base)
    assert identity_of(resumed) == identity_of(base)
    assert identity_of(neither) == identity_of(base)


def test_the_identity_does_not_depend_on_the_order_of_the_keys() -> None:
    base = config()
    reordered = {
        "training": dict(reversed(list(base["training"].items()))),
        "model": base["model"],
        "data": base["data"],
        "share_vocab": True,
        "seed": 1234,
    }

    assert identity_of(reordered) == identity_of(base)


def test_the_identity_does_depend_on_the_order_of_a_list() -> None:
    first = config()
    first["training"]["gpu_ranks"] = [0, 1]
    second = config()
    second["training"]["gpu_ranks"] = [1, 0]

    assert identity_of(first) != identity_of(second)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), 1235),
        (("share_vocab",), False),
        (("data", "en-mn.auth", "path_src"), "d/a.src2"),
        (("data", "en-mn.auth", "weight"), 4),
        (("model", "hidden_size"), 65),
        (("model", "encoder", "layers"), 3),
        (("training", "model_path"), "run/ckpt2"),
        (("training", "learning_rate"), 0.50001),
        (("training", "learning_rate"), 1),
        (("training", "world_size"), 2),
        (("training", "gpu_ranks"), [0]),
        (("training", "extra_key"), 1),
        (("extra_section",), {}),
    ],
)
def test_the_identity_changes_with_any_other_value(
    path: tuple[str, ...], value: object
) -> None:
    changed = config()
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    assert identity_of(changed) != identity_of(config())


def test_one_changed_character_of_a_path_changes_the_identity() -> None:
    changed = config()
    changed["data"]["en-mn.auth"]["path_src"] = "d/a.sre"

    assert identity_of(changed) != identity_of(config())


def test_the_identity_does_not_modify_the_config() -> None:
    cfg = config()
    before = copy.deepcopy(cfg)

    identity_of(cfg)

    assert cfg == before
    assert cfg["training"]["train_steps"] == 200


def test_a_config_without_a_training_section_has_an_identity() -> None:
    assert len(identity_of({"seed": 1})) == 64


@pytest.mark.parametrize("bad", [{1, 2}, Path("x"), float("nan"), float("inf"), b"x"])
def test_a_config_that_is_not_plain_data_is_a_config_error(bad: object) -> None:
    with pytest.raises(ConfigError, match="cannot take the identity"):
        identity_of({"training": {"x": bad}})


def test_keys_that_cannot_be_ordered_are_a_config_error() -> None:
    with pytest.raises(ConfigError, match="cannot take the identity"):
        identity_of({"a": 1, 2: 3})


# ---- the launch record ----------------------------------------------------------


def record(**changes: Any) -> RunIdentity:
    fields: dict[str, Any] = {
        "identity": HEX,
        "world_size": 8,
        "gpu_ranks": (0, 1, 2, 3, 4, 5, 6, 7),
        "created": NOW,
    }
    return RunIdentity(**{**fields, **changes})


def test_the_launch_file_is_launch_json() -> None:
    assert LAUNCH_FILE == "launch.json"


def test_a_launch_record_round_trips(tmp_path: Path) -> None:
    written = write_launch_record(tmp_path, record())

    assert written == tmp_path / "launch.json"
    assert read_launch_record(tmp_path) == record()


def test_a_cpu_run_records_empty_ranks(tmp_path: Path) -> None:
    write_launch_record(tmp_path, record(world_size=1, gpu_ranks=()))

    assert read_launch_record(tmp_path) == record(world_size=1, gpu_ranks=())


def test_the_launch_file_is_plain_readable_json_with_these_keys(tmp_path: Path) -> None:
    write_launch_record(tmp_path, record(world_size=2, gpu_ranks=(0, 1)))

    assert json.loads((tmp_path / LAUNCH_FILE).read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "identity": HEX,
        "world_size": 2,
        "gpu_ranks": [0, 1],
        "created": NOW,
    }
    assert (tmp_path / LAUNCH_FILE).read_text(encoding="utf-8").endswith("}\n")


def test_no_launch_file_is_no_record(tmp_path: Path) -> None:
    assert read_launch_record(tmp_path) is None


def test_writing_replaces_an_earlier_record_whole(tmp_path: Path) -> None:
    write_launch_record(tmp_path, record())
    write_launch_record(tmp_path, record(identity="cd" * 32))

    reread = read_launch_record(tmp_path)
    assert reread is not None and reread.identity == "cd" * 32


def test_writing_leaves_no_temporary_file_and_no_other_file(tmp_path: Path) -> None:
    write_launch_record(tmp_path, record())

    assert [p.name for p in tmp_path.iterdir()] == [LAUNCH_FILE]


def test_a_failed_write_keeps_the_earlier_record_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_launch_record(tmp_path, record())

    def fail(self: Path, target: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_launch_record(tmp_path, record(identity="cd" * 32))
    monkeypatch.undo()

    assert read_launch_record(tmp_path) == record()
    assert [p.name for p in tmp_path.iterdir()] == [LAUNCH_FILE]


def test_writing_into_a_directory_that_does_not_exist_is_an_os_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(OSError):
        write_launch_record(tmp_path / "no" / "such", record())


@pytest.mark.parametrize(
    "changes",
    [
        {"identity": "ab" * 31},
        {"identity": "AB" * 32},
        {"identity": "zz" * 32},
        {"identity": ""},
        {"world_size": 0},
        {"world_size": -1},
        {"gpu_ranks": (-1,)},
        {"created": "yesterday"},
        {"created": "2026-11-08T12:00:00"},
        {"created": ""},
        {"schema_version": 1},
        {"note": "x"},
    ],
)
def test_a_record_that_could_not_be_read_back_cannot_be_made(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        record(**changes)


@pytest.mark.parametrize("world_size", [True, 1.0, "1"])
def test_a_record_is_strictly_typed(world_size: object) -> None:
    with pytest.raises(ValueError):
        record(world_size=world_size)


def test_a_record_is_immutable() -> None:
    with pytest.raises(ValueError):
        record().world_size = 2  # type: ignore[misc]


def good_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": HEX,
        "world_size": 1,
        "gpu_ranks": [],
        "created": NOW,
    }


def write_raw(tmp_path: Path, content: bytes) -> None:
    (tmp_path / LAUNCH_FILE).write_bytes(content)


def test_a_good_hand_written_document_reads(tmp_path: Path) -> None:
    write_raw(tmp_path, json.dumps(good_document()).encode())

    assert read_launch_record(tmp_path) == record(world_size=1, gpu_ranks=())


@pytest.mark.parametrize(
    "content",
    [b"", b"{", b"not json", b"[]", b"null", b'"launch"', b"\xff\xfe", b"{}"],
    ids=["empty", "cut", "words", "list", "null", "string", "bad-utf8", "empty-object"],
)
def test_a_launch_file_that_is_not_a_record_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    write_raw(tmp_path, content)

    with pytest.raises(DataError, match="is not a Rinchen launch record"):
        read_launch_record(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("schema_version", "1"),
        ("schema_version", True),
        ("identity", "ab" * 31),
        ("identity", 5),
        ("identity", None),
        ("world_size", 0),
        ("world_size", "1"),
        ("world_size", 1.5),
        ("world_size", True),
        ("gpu_ranks", "0"),
        ("gpu_ranks", [-1]),
        ("gpu_ranks", [1.5]),
        ("gpu_ranks", ["0"]),
        ("gpu_ranks", [True]),
        ("gpu_ranks", None),
        ("created", 5),
        ("created", "not a time"),
        ("created", "2026-11-08T12:00:00"),
    ],
)
def test_a_record_with_a_wrong_field_is_a_data_error_naming_the_field(
    tmp_path: Path, key: str, value: object
) -> None:
    document = good_document()
    document[key] = value
    write_raw(tmp_path, json.dumps(document).encode())

    with pytest.raises(DataError) as raised:
        read_launch_record(tmp_path)

    assert key in str(raised.value)
    assert str(tmp_path / LAUNCH_FILE) in str(raised.value)


@pytest.mark.parametrize("key", list(good_document()))
def test_a_record_missing_a_field_is_a_data_error_naming_it(
    tmp_path: Path, key: str
) -> None:
    document = good_document()
    del document[key]
    write_raw(tmp_path, json.dumps(document).encode())

    with pytest.raises(DataError) as raised:
        read_launch_record(tmp_path)

    assert key in str(raised.value)


def test_a_record_with_an_extra_key_is_a_data_error(tmp_path: Path) -> None:
    write_raw(tmp_path, json.dumps({**good_document(), "note": "x"}).encode())

    with pytest.raises(DataError, match="note"):
        read_launch_record(tmp_path)


def test_a_foreign_launch_file_is_a_data_error(tmp_path: Path) -> None:
    write_raw(tmp_path, json.dumps({"command": "torchrun", "nodes": 2}).encode())

    with pytest.raises(DataError, match="is not a Rinchen launch record"):
        read_launch_record(tmp_path)


def test_the_data_error_does_not_echo_the_values_it_refused(tmp_path: Path) -> None:
    secret = "s3cr3t-token-value"
    write_raw(tmp_path, json.dumps({**good_document(), "identity": secret}).encode())

    with pytest.raises(DataError) as raised:
        read_launch_record(tmp_path)

    assert secret not in str(raised.value)


def test_a_launch_file_over_64_kib_is_a_data_error_without_reading_it_all(
    tmp_path: Path,
) -> None:
    write_raw(tmp_path, b" " * (64 * 1024 + 1))

    with pytest.raises(DataError, match="too large"):
        read_launch_record(tmp_path)


def test_a_launch_file_that_is_a_directory_is_a_data_error(tmp_path: Path) -> None:
    (tmp_path / LAUNCH_FILE).mkdir()

    with pytest.raises(DataError, match="cannot read the launch record"):
        read_launch_record(tmp_path)


def test_a_launch_file_that_is_a_link_to_nothing_is_a_data_error(
    tmp_path: Path,
) -> None:
    (tmp_path / LAUNCH_FILE).symlink_to("deleted")

    with pytest.raises(DataError, match="cannot read the launch record"):
        read_launch_record(tmp_path)


def test_a_timestamp_with_z_is_accepted() -> None:
    assert record(created="2026-11-08T12:00:00Z").created == "2026-11-08T12:00:00Z"


# ---- step_of --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "step"),
    [("step_1", 1), ("step_200", 200), ("step_9" + "0" * 20, 9 * 10**20)],
)
def test_step_of_reads_the_number_of_a_step_directory_name(
    name: str, step: int
) -> None:
    assert step_of(Path("/runs/a/ckpt") / name) == step


@pytest.mark.parametrize(
    "name",
    [
        "step_0",
        "step_007",
        "step_",
        "step_x",
        "step_-3",
        "step_5.1",
        "step_5 ",
        "Step_5",
        "xstep_5",
        "step_5_old",
        "step_" + chr(0x0663),
        "step_1" + chr(0x0663),
        "averaged",
        "",
    ],
)
def test_step_of_is_none_for_any_other_name(name: str) -> None:
    assert step_of(Path("/runs/a/ckpt") / name) is None


def test_step_of_agrees_with_what_step_directories_lists(model_path: Path) -> None:
    for name in ("step_5", "step_007", "step_x", "averaged"):
        (model_path / name).mkdir(parents=True)

    listed = {d.path.name for d in step_directories(model_path)}

    assert listed == {
        path.name for path in model_path.iterdir() if step_of(path) is not None
    }


# ---- a run that continues one of our own checkpoints (D9) ------------------------------


def continued(**changes: Any) -> ContinuedFrom:
    fields: dict[str, Any] = {
        "kind": "step",
        "path": "/runs/teacher/ckpt/step_200",
        "weights_sha256": "01" * 32,
        "optimizer_sha256": "02" * 32,
        "config_sha256": "03" * 32,
        "vocab_sha256": "04" * 32,
        "manifest_sha256": "05" * 32,
        "manifest_stage": "train_launch",
        "step": 200,
        "source_run": "/runs/teacher",
    }
    return ContinuedFrom(**{**fields, **changes})


def averaged_source(**changes: Any) -> ContinuedFrom:
    return continued(
        kind="averaged",
        path="/runs/teacher/averaged",
        optimizer_sha256=None,
        manifest_stage="train_average",
        step=None,
        source_run=None,
        **changes,
    )


def test_a_source_record_of_each_kind_can_be_made() -> None:
    assert continued().kind == "step"
    assert averaged_source().kind == "averaged"


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "checkpoint"},
        {"path": ""},
        {"weights_sha256": "ab" * 31},
        {"weights_sha256": "AB" * 32},
        {"config_sha256": ""},
        {"vocab_sha256": None},
        {"manifest_sha256": "zz" * 32},
        {"optimizer_sha256": "ab"},
        {"manifest_stage": ""},
        {"step": 0},
        {"step": True},
        {"step": "200"},
        {"note": "x"},
        # a step source has a step, an optimizer and a run; its manifest is a launch's
        {"step": None},
        {"optimizer_sha256": None},
        {"source_run": None},
    ],
)
def test_a_step_source_record_that_could_not_be_read_back_cannot_be_made(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        continued(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"step": 200},
        {"optimizer_sha256": "02" * 32},
        {"source_run": "/runs/teacher"},
    ],
)
def test_an_averaged_source_record_has_no_step_optimizer_or_run(
    changes: dict[str, Any],
) -> None:
    fields = {
        "kind": "averaged",
        "path": "/runs/teacher/averaged",
        "optimizer_sha256": None,
        "manifest_stage": "train_average",
        "step": None,
        "source_run": None,
        **changes,
    }

    with pytest.raises(ValueError):
        continued(**fields)


def test_a_source_record_is_immutable() -> None:
    with pytest.raises(ValueError):
        continued().step = 1  # type: ignore[misc]


def test_a_launch_record_without_a_source_has_none() -> None:
    assert record().continued_from is None


def test_a_launch_record_of_a_continuation_round_trips_as_version_2(
    tmp_path: Path,
) -> None:
    for source in (continued(), averaged_source()):
        write_launch_record(tmp_path, record(continued_from=source))

        assert read_launch_record(tmp_path) == record(continued_from=source)


def test_a_continuation_record_file_is_version_2_and_holds_the_source(
    tmp_path: Path,
) -> None:
    write_launch_record(tmp_path, record(continued_from=continued()))

    assert json.loads((tmp_path / LAUNCH_FILE).read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "identity": HEX,
        "world_size": 8,
        "gpu_ranks": [0, 1, 2, 3, 4, 5, 6, 7],
        "created": NOW,
        "continued_from": {
            "kind": "step",
            "path": "/runs/teacher/ckpt/step_200",
            "weights_sha256": "01" * 32,
            "optimizer_sha256": "02" * 32,
            "config_sha256": "03" * 32,
            "vocab_sha256": "04" * 32,
            "manifest_sha256": "05" * 32,
            "manifest_stage": "train_launch",
            "step": 200,
            "source_run": "/runs/teacher",
        },
    }


def test_a_record_without_a_source_is_still_written_as_version_1_without_the_key(
    tmp_path: Path,
) -> None:
    write_launch_record(tmp_path, record())

    document = json.loads((tmp_path / LAUNCH_FILE).read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert "continued_from" not in document


def test_a_version_1_record_is_read_with_no_source(tmp_path: Path) -> None:
    write_raw(tmp_path, json.dumps(good_document()).encode())

    assert read_launch_record(tmp_path) == record(world_size=1, gpu_ranks=())


def test_a_version_1_record_cannot_hold_a_source(tmp_path: Path) -> None:
    document = good_document()
    document["continued_from"] = json.loads(continued().model_dump_json())
    write_raw(tmp_path, json.dumps(document).encode())

    with pytest.raises(DataError, match="schema_version"):
        read_launch_record(tmp_path)


def test_a_version_2_record_must_hold_a_source(tmp_path: Path) -> None:
    document = good_document()
    document["schema_version"] = 2
    write_raw(tmp_path, json.dumps(document).encode())

    with pytest.raises(DataError, match="schema_version"):
        read_launch_record(tmp_path)


def test_a_version_3_record_is_refused(tmp_path: Path) -> None:
    document = good_document()
    document["schema_version"] = 3
    write_raw(tmp_path, json.dumps(document).encode())

    with pytest.raises(DataError, match="schema_version"):
        read_launch_record(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("weights_sha256", "ab" * 31),
        ("kind", "other"),
        ("step", 0),
        ("step", "200"),
        ("path", 5),
        ("extra", 1),
    ],
)
def test_a_source_with_a_wrong_field_is_a_data_error_naming_the_field(
    tmp_path: Path, key: str, value: object
) -> None:
    source = json.loads(continued().model_dump_json())
    source[key] = value
    write_raw(
        tmp_path,
        json.dumps(
            {**good_document(), "schema_version": 2, "continued_from": source}
        ).encode(),
    )

    with pytest.raises(DataError) as raised:
        read_launch_record(tmp_path)

    assert key in str(raised.value)
    assert "continued_from" in str(raised.value)


def test_a_source_missing_a_field_is_a_data_error_naming_it(tmp_path: Path) -> None:
    source = json.loads(continued().model_dump_json())
    del source["manifest_sha256"]
    write_raw(
        tmp_path,
        json.dumps(
            {**good_document(), "schema_version": 2, "continued_from": source}
        ).encode(),
    )

    with pytest.raises(DataError, match="manifest_sha256"):
        read_launch_record(tmp_path)


def test_the_old_check_path_reads_the_identity_of_a_continuation_record(
    tmp_path: Path,
) -> None:
    # What a D8 caller does with a record: compare `identity`; the new field is beside it.
    write_launch_record(tmp_path, record(continued_from=continued()))

    reread = read_launch_record(tmp_path)

    assert reread is not None
    assert (reread.identity, reread.world_size, reread.gpu_ranks) == (
        HEX,
        8,
        tuple(range(8)),
    )


def test_the_identity_of_a_plain_config_is_not_changed_by_the_source_argument() -> None:
    assert identity_of(config(), continued_from=None) == identity_of(config())


def test_the_identity_of_a_continuation_differs_from_the_plain_config_s() -> None:
    assert identity_of(config(), continued_from=continued()) != identity_of(config())


def test_the_identity_of_a_continuation_is_64_hex_characters() -> None:
    identity = identity_of(config(), continued_from=continued())

    assert len(identity) == 64
    assert int(identity, 16) >= 0


def test_another_source_weights_file_is_another_identity() -> None:
    first = identity_of(config(), continued_from=continued())
    second = identity_of(config(), continued_from=continued(weights_sha256="09" * 32))

    assert first != second


def test_another_kind_of_source_is_another_identity() -> None:
    step = identity_of(config(), continued_from=continued())
    averaged = identity_of(config(), continued_from=averaged_source())

    assert step != averaged


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "/elsewhere/ckpt/step_200"},
        {"manifest_sha256": "0a" * 32},
        {"optimizer_sha256": "0d" * 32},
        {"source_run": "/elsewhere"},
    ],
)
def test_where_the_source_is_and_what_else_it_holds_do_not_change_the_identity(
    changes: dict[str, Any],
) -> None:
    # The same files moved to another place, or vouched for by a manifest written again,
    # are the same source.
    assert identity_of(config(), continued_from=continued(**changes)) == identity_of(
        config(), continued_from=continued()
    )


@pytest.mark.parametrize("field", ["config_sha256", "vocab_sha256"])
def test_the_config_and_the_vocabulary_of_the_source_are_part_of_the_identity(
    field: str,
) -> None:
    # The same weights read through another vocabulary are another starting point.
    assert identity_of(config(), continued_from=continued(**{field: "0e" * 32})) != (
        identity_of(config(), continued_from=continued())
    )


def test_a_continuation_identity_still_ignores_train_steps_and_train_from() -> None:
    longer = config()
    longer["training"]["train_steps"] = 5000
    resumed = config()
    resumed["training"]["train_from"] = "run/ckpt/step_100"

    expected = identity_of(config(), continued_from=continued())
    assert identity_of(longer, continued_from=continued()) == expected
    assert identity_of(resumed, continued_from=continued()) == expected


def test_a_continuation_identity_changes_with_the_config_too() -> None:
    changed = config()
    changed["training"]["learning_rate"] = 0.25

    assert identity_of(changed, continued_from=continued()) != identity_of(
        config(), continued_from=continued()
    )
