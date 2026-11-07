"""`rinchen mine align`: the end-to-end run on small literal files and on the
planted alignments, the corpus and manifest it writes, skipping, and the exit
code of each failure.

The LaBSE embedder comes from a module-level factory in the CLI; tests replace
it with the hash embedder, so no model is downloaded. Everything lands under
pytest's temporary directories.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import Result
from numpy.typing import NDArray
from typer.testing import CliRunner

from rinchen.cli import mine as mine_cli
from rinchen.cli.main import app
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import CORPUS_SCHEMA, make_row_id
from rinchen.errors import UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import sha256_of
from rinchen.mine.align import align
from rinchen.mine.embed import SentenceTransformerEmbedder
from tests.support.embedders import HashEmbedder
from tests.support.planted_alignments import Planted, build_planted_alignments

runner = CliRunner()
# The real factory, before the fixture replaces it.
REAL_BUILD_EMBEDDER = mine_cli.build_embedder

OPTIONS = [
    "--src",
    "--tgt",
    "--src-lang",
    "--tgt-lang",
    "--corpus",
    "--out",
    "--max-k",
    "--skip-cost",
    "--merge-penalty",
    "--force",
]
TOOL_PACKAGES = ["polars", "numpy", "sentence-transformers", "torch"]
SRC_LANG = "khk_Cyrl"
TGT_LANG = "eng_Latn"
CORPUS = "unit"

# Source sentences 1 and 2 are one 2-1 group, sentence 3 and target sentence 3
# have no counterpart; the hash embedder aligns the rest 1-1.
SRC_SENTENCES = [
    "kor tam vel nis ruk",
    "dap sol wen",
    "lur gim hat",
    "extra lonely sentence here now",
    "bor fen zik tul qua",
]
TGT_SENTENCES = [
    "kor tam vel nis ruk end",
    "dap sol wen lur gim hat end",
    "bor fen zik tul qua end",
    "tail only target words",
]
EXPECTED_PAIRS = [
    ("kor tam vel nis ruk", "kor tam vel nis ruk end"),
    ("dap sol wen lur gim hat", "dap sol wen lur gim hat end"),
    ("bor fen zik tul qua", "bor fen zik tul qua end"),
]
EXPECTED_METRICS = {
    "aligned": 3.0,
    "unaligned_src": 1.0,
    "unaligned_tgt": 1.0,
    "src_sentences": 5.0,
    "tgt_sentences": 4.0,
    "blank_lines_dropped": 0.0,
    "param.max_k": 4.0,
    "param.skip_cost": 0.9,
    "param.merge_penalty": 0.3,
}
# What the config hash is taken over: the parameters as canonical JSON.
CANONICAL_PARAMETERS = (
    '{"corpus":"unit","margin_k":4,"max_k":4,"merge_penalty":0.3,'
    '"skip_cost":0.9,"src_lang":"khk_Cyrl","tgt_lang":"eng_Latn"}'
)


class WrongShapeEmbedder:
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.zeros((len(texts) + 1, 4), dtype=np.float32)


class UnavailableEmbedder:
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        raise UpstreamError("embedding model unavailable")


@dataclass(frozen=True, slots=True)
class Workspace:
    src: Path
    tgt: Path
    out: Path

    @property
    def manifest(self) -> Path:
        return self.out.parent / "manifest.json"

    def args(self, *extra: str, **replace: str | None) -> list[str]:
        """The command line; `replace` swaps an option's value by name with
        underscores for dashes (`src_lang="x"`), and None drops the option."""
        options: dict[str, str | None] = {
            "src": str(self.src),
            "tgt": str(self.tgt),
            "src_lang": SRC_LANG,
            "tgt_lang": TGT_LANG,
            "corpus": CORPUS,
            "out": str(self.out),
        }
        options.update(replace)
        given = [
            part
            for name, value in options.items()
            if value is not None
            for part in (f"--{name.replace('_', '-')}", value)
        ]
        return ["mine", "align", *given, *extra]


def _write_lines(path: Path, lines: Sequence[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mine_cli, "build_embedder", HashEmbedder)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        src=_write_lines(tmp_path / "news_mn.txt", SRC_SENTENCES),
        tgt=_write_lines(tmp_path / "news_en.txt", TGT_SENTENCES),
        out=tmp_path / "out" / "aligned.parquet",
    )


def _run_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _manifest(workspace: Workspace) -> Manifest:
    manifest = store.read(workspace.out.parent)
    assert manifest is not None
    return manifest


def _aligned(path: Path) -> pl.DataFrame:
    return scan_corpus(path).collect()


def _text_pairs(frame: pl.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["src"].to_list(), frame["tgt"].to_list(), strict=True))


def _f1(predicted: Sequence[tuple[str, str]], gold: Sequence[tuple[str, str]]) -> float:
    matched = len(set(predicted) & set(gold))
    if matched == 0:
        return 0.0
    precision, recall = matched / len(predicted), matched / len(gold)
    return 2 * precision * recall / (precision + recall)


# --- a run on the literal files ---------------------------------------------


def test_the_literal_files_are_aligned_into_one_row_per_group(
    workspace: Workspace,
) -> None:
    result = _run_ok(workspace.args())

    frame = _aligned(workspace.out)
    assert _text_pairs(frame) == EXPECTED_PAIRS
    assert result.stdout.splitlines() == [
        "aligned 3 rows (unaligned: 1 source, 1 target)",
        f"manifest: {workspace.manifest}",
    ]


def test_every_row_carries_the_run_fields_and_a_content_id(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())

    frame = _aligned(workspace.out)
    assert frame.schema == CORPUS_SCHEMA
    assert frame["origin"].to_list() == ["mined"] * 3
    assert frame["split"].to_list() == ["train"] * 3
    assert frame["corpus"].to_list() == [CORPUS] * 3
    assert frame["src_lang"].to_list() == [SRC_LANG] * 3
    assert frame["tgt_lang"].to_list() == [TGT_LANG] * 3
    assert frame["doc_id"].to_list() == ["news_mn"] * 3
    assert frame["seg_idx"].to_list() == [0, 1, 2]
    assert frame["id"].to_list() == [
        make_row_id(SRC_LANG, TGT_LANG, src, tgt) for src, tgt in EXPECTED_PAIRS
    ]
    assert frame["filter_reason"].null_count() == 3
    assert frame["scores"].null_count() == 3


def test_the_manifest_records_inputs_output_parameters_and_environment(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())

    manifest = _manifest(workspace)
    assert manifest.stage == "mine_align"
    assert manifest.seed == 0
    assert manifest.metrics == EXPECTED_METRICS
    assert manifest.code_version
    assert datetime.fromisoformat(manifest.created).utcoffset() == timedelta(0)
    assert [(item.path, item.sha256, item.bytes) for item in manifest.inputs] == [
        (str(path), sha256_of(path), path.stat().st_size)
        for path in (workspace.src, workspace.tgt)
    ]
    assert [
        (item.path, item.rows, item.bytes, item.sha256) for item in manifest.outputs
    ] == [
        (str(workspace.out), 3, workspace.out.stat().st_size, sha256_of(workspace.out))
    ]
    assert list(manifest.tool_versions) == TOOL_PACKAGES
    assert manifest.tool_versions["polars"] == pl.__version__
    assert manifest.tool_versions["numpy"] == np.__version__


def test_the_config_hash_is_taken_over_the_canonical_json_of_the_parameters(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())

    expected = hashlib.sha256(CANONICAL_PARAMETERS.encode("utf-8")).hexdigest()
    assert _manifest(workspace).config_sha256 == expected


def test_the_options_default_to_the_aligners_own_parameters(
    workspace: Workspace,
) -> None:
    parameters = inspect.signature(align).parameters
    _run_ok(workspace.args())

    metrics = _manifest(workspace).metrics
    assert metrics["param.max_k"] == parameters["max_k"].default
    assert metrics["param.skip_cost"] == parameters["skip_cost"].default
    assert metrics["param.merge_penalty"] == parameters["merge_penalty"].default


def test_the_alignment_options_reach_the_aligner_and_the_manifest(
    workspace: Workspace,
) -> None:
    _run_ok(
        workspace.args("--max-k", "2", "--skip-cost", "0.5", "--merge-penalty", "0.0")
    )

    metrics = _manifest(workspace).metrics
    assert (
        metrics["param.max_k"],
        metrics["param.skip_cost"],
        metrics["param.merge_penalty"],
    ) == (2.0, 0.5, 0.0)
    # With max_k 2 only 1-1 pairs and deletions exist: no row joins sentences.
    assert all(
        src in SRC_SENTENCES and tgt in TGT_SENTENCES
        for src, tgt in _text_pairs(_aligned(workspace.out))
    )
    assert (
        _manifest(workspace).config_sha256
        != hashlib.sha256(CANONICAL_PARAMETERS.encode("utf-8")).hexdigest()
    )


def test_a_zero_skip_cost_makes_leaving_sentences_out_cheaper_than_any_pair(
    workspace: Workspace,
) -> None:
    result = _run_ok(workspace.args("--skip-cost", "0"))

    assert _aligned(workspace.out).height == 0
    assert result.stdout.splitlines()[0] == (
        "aligned 0 rows (unaligned: 5 source, 4 target)"
    )


def test_a_high_merge_penalty_stops_sentences_from_being_joined(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args("--merge-penalty", "5"))

    assert _text_pairs(_aligned(workspace.out)) == [
        (SRC_SENTENCES[0], TGT_SENTENCES[0]),
        (SRC_SENTENCES[2], TGT_SENTENCES[1]),
        (SRC_SENTENCES[4], TGT_SENTENCES[2]),
    ]
    assert _manifest(workspace).metrics["unaligned_src"] == 2.0


def test_the_default_out_is_the_corpus_name_under_the_default_root(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mined_root"
    monkeypatch.setattr(mine_cli, "DEFAULT_OUT_ROOT", root)
    expected = root / CORPUS / "aligned.parquet"

    result = _run_ok(workspace.args(out=None))

    assert (
        result.stdout.splitlines()[-1]
        == f"manifest: {expected.parent / 'manifest.json'}"
    )
    assert sorted(path.name for path in expected.parent.iterdir()) == [
        "aligned.parquet",
        "manifest.json",
    ]
    assert not workspace.out.exists()


def test_lines_end_with_lf_or_crlf_and_a_byte_order_mark_is_ignored(
    workspace: Workspace,
) -> None:
    workspace.src.write_bytes(
        b"\xef\xbb\xbf" + "\r\n".join(SRC_SENTENCES).encode("utf-8") + b"\r\n"
    )

    _run_ok(workspace.args())

    assert _text_pairs(_aligned(workspace.out)) == EXPECTED_PAIRS


def test_sentences_are_stripped_and_split_on_line_feeds_only(
    workspace: Workspace,
) -> None:
    # U+2028 and U+0085 are not line ends here; a sentence may contain them.
    separator = chr(0x2028)
    _write_lines(
        workspace.src, [f"  kor tam vel{separator}nis ruk\t", "dap sol wen lur gim"]
    )
    _write_lines(workspace.tgt, ["kor tam vel nis ruk end", "dap sol wen lur gim end"])

    _run_ok(workspace.args())

    assert _aligned(workspace.out)["src"].to_list() == [
        f"kor tam vel{separator}nis ruk",
        "dap sol wen lur gim",
    ]


def test_blank_lines_are_dropped_and_counted(workspace: Workspace) -> None:
    workspace.src.write_text(
        "kor tam vel nis ruk\n\n   \ndap sol wen lur gim\n", encoding="utf-8"
    )
    workspace.tgt.write_text(
        "\nkor tam vel nis ruk end\n\n\ndap sol wen lur gim end", encoding="utf-8"
    )

    _run_ok(workspace.args())

    metrics = _manifest(workspace).metrics
    # Two blank lines in the source file and three in the target file.
    assert metrics["blank_lines_dropped"] == 5.0
    assert (metrics["src_sentences"], metrics["tgt_sentences"]) == (2.0, 2.0)
    assert _aligned(workspace.out)["seg_idx"].to_list() == [0, 1]


@pytest.mark.parametrize(
    ("src_text", "tgt_text", "unaligned_src", "blank_lines"),
    [
        pytest.param("\n".join(SRC_SENTENCES), "", 5.0, 0.0, id="empty-target"),
        pytest.param("\n  \n\n", "", 0.0, 3.0, id="blank-lines-only"),
    ],
)
def test_a_side_without_sentences_gives_an_empty_corpus_and_no_embedding_work(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    src_text: str,
    tgt_text: str,
    unaligned_src: float,
    blank_lines: float,
) -> None:
    monkeypatch.setattr(mine_cli, "build_embedder", UnavailableEmbedder)
    workspace.src.write_text(src_text, encoding="utf-8")
    workspace.tgt.write_text(tgt_text, encoding="utf-8")

    result = _run_ok(workspace.args())

    assert _aligned(workspace.out).height == 0
    manifest = _manifest(workspace)
    assert manifest.metrics["aligned"] == 0.0
    assert manifest.metrics["unaligned_src"] == unaligned_src
    assert manifest.metrics["unaligned_tgt"] == 0.0
    assert manifest.metrics["blank_lines_dropped"] == blank_lines
    assert [item.rows for item in manifest.outputs] == [0]
    assert result.stdout.splitlines()[0] == (
        f"aligned 0 rows (unaligned: {int(unaligned_src)} source, 0 target)"
    )


# --- the planted alignments through the whole command -----------------------


@dataclass(frozen=True, slots=True)
class PlantedRun:
    planted: Planted
    result: Result
    out: Path
    manifest: Manifest


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory: pytest.TempPathFactory) -> PlantedRun:
    planted = build_planted_alignments()
    root = tmp_path_factory.mktemp("mine_planted")
    src = _write_lines(root / "planted_src.txt", planted.src)
    tgt = _write_lines(root / "planted_tgt.txt", planted.tgt)
    out = root / "mined" / "aligned.parquet"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mine_cli, "build_embedder", HashEmbedder)
        result = runner.invoke(
            app,
            [
                "mine",
                "align",
                "--src",
                str(src),
                "--tgt",
                str(tgt),
                "--src-lang",
                SRC_LANG,
                "--tgt-lang",
                TGT_LANG,
                "--corpus",
                "planted",
                "--out",
                str(out),
            ],
        )
    manifest = store.read(out.parent)
    assert manifest is not None
    return PlantedRun(planted, result, out, manifest)


def test_the_planted_alignments_are_recovered_through_the_command(
    planted_run: PlantedRun,
) -> None:
    assert planted_run.result.exit_code == 0, planted_run.result.output
    planted = planted_run.planted
    gold = [
        (
            " ".join(planted.src[i] for i in src_indexes),
            " ".join(planted.tgt[j] for j in tgt_indexes),
        )
        for src_indexes, tgt_indexes in planted.gold
        if src_indexes and tgt_indexes
    ]

    predicted = _text_pairs(_aligned(planted_run.out))

    assert _f1(predicted, gold) >= 0.95


def test_the_planted_run_writes_a_valid_corpus_in_alignment_order(
    planted_run: PlantedRun,
) -> None:
    frame = _aligned(planted_run.out)

    assert frame.schema == CORPUS_SCHEMA
    assert frame.height > 0
    assert frame["seg_idx"].to_list() == list(range(frame.height))
    assert frame["doc_id"].unique().to_list() == ["planted_src"]
    assert frame["origin"].unique().to_list() == ["mined"]
    assert frame["split"].unique().to_list() == ["train"]
    assert frame["id"].to_list() == [
        make_row_id(SRC_LANG, TGT_LANG, src, tgt) for src, tgt in _text_pairs(frame)
    ]


def test_the_planted_run_equals_the_aligner_and_its_metrics_add_up(
    planted_run: PlantedRun,
) -> None:
    planted = planted_run.planted
    alignments = align(planted.src, planted.tgt, HashEmbedder())
    expected = [
        (
            " ".join(planted.src[i] for i in alignment.src),
            " ".join(planted.tgt[j] for j in alignment.tgt),
        )
        for alignment in alignments
        if alignment.src and alignment.tgt
    ]
    metrics = planted_run.manifest.metrics

    assert _text_pairs(_aligned(planted_run.out)) == expected
    assert metrics["aligned"] == float(len(expected))
    assert metrics["unaligned_src"] == float(
        sum(1 for a in alignments if a.src and not a.tgt)
    )
    assert metrics["unaligned_tgt"] == float(
        sum(1 for a in alignments if a.tgt and not a.src)
    )
    assert (metrics["src_sentences"], metrics["tgt_sentences"]) == (
        float(len(planted.src)),
        float(len(planted.tgt)),
    )
    assert planted_run.manifest.outputs[0].rows == len(expected)
    assert planted_run.result.stdout.splitlines()[0] == (
        f"aligned {len(expected)} rows (unaligned: "
        f"{int(metrics['unaligned_src'])} source, {int(metrics['unaligned_tgt'])} target)"
    )


# --- skipping, forcing and commit safety ------------------------------------


def test_a_rerun_with_unchanged_inputs_and_parameters_skips(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(workspace.args())
    manifest_before = workspace.manifest.read_bytes()
    output_before = workspace.out.stat().st_mtime_ns
    monkeypatch.setattr(mine_cli, "build_embedder", UnavailableEmbedder)

    result = _run_ok(workspace.args())

    lines = result.stdout.splitlines()
    assert lines[0].startswith("skipped:")
    assert lines[-1] == f"manifest: {workspace.manifest}"
    assert workspace.manifest.read_bytes() == manifest_before
    assert workspace.out.stat().st_mtime_ns == output_before


class RewritingEmbedder(HashEmbedder):
    """Changes the source file in the middle of a run, as another process
    might."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._path.write_text("changed during the run\n", encoding="utf-8")
        return super().embed(texts)


def test_the_manifest_vouches_for_the_bytes_that_were_read_not_for_later_ones(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_hash = sha256_of(workspace.src)
    monkeypatch.setattr(
        mine_cli, "build_embedder", lambda: RewritingEmbedder(workspace.src)
    )
    _run_ok(workspace.args())
    monkeypatch.setattr(mine_cli, "build_embedder", HashEmbedder)

    recorded = {item.path: item.sha256 for item in _manifest(workspace).inputs}
    assert recorded[str(workspace.src)] == read_hash
    assert read_hash != sha256_of(workspace.src)
    rerun = _run_ok(workspace.args())

    assert not rerun.stdout.startswith("skipped:")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
@pytest.mark.parametrize("first_run", [True, False])
def test_an_unreadable_input_file_is_a_data_error(
    workspace: Workspace, first_run: bool
) -> None:
    if not first_run:
        _run_ok(workspace.args())
    workspace.tgt.chmod(0)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith(f"error: cannot read {workspace.tgt}")


def test_an_input_that_cannot_be_hashed_is_a_data_error_not_a_write_failure(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_hash(paths: list[Path]) -> list[ManifestFile]:
        raise PermissionError(13, "Permission denied", str(paths[0]))

    monkeypatch.setattr(mine_cli, "build_inputs", cannot_hash)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: cannot read the input files: ")
    assert not workspace.out.exists()


def test_force_reruns_even_when_nothing_changed(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    before = workspace.out.stat()

    result = _run_ok(workspace.args("--force"))

    after = workspace.out.stat()
    assert not result.stdout.startswith("skipped:")
    assert (after.st_ino, after.st_mtime_ns) != (before.st_ino, before.st_mtime_ns)


def test_a_deleted_output_is_rebuilt_instead_of_skipped(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    workspace.out.unlink()

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    assert _aligned(workspace.out).height == 3


@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_changed_input_file_reruns(workspace: Workspace, side: str) -> None:
    _run_ok(workspace.args())
    path: Path = getattr(workspace, side)
    path.write_text(
        path.read_text(encoding="utf-8") + "one more sentence\n", encoding="utf-8"
    )

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    recorded = {item.path: item.sha256 for item in _manifest(workspace).inputs}
    assert recorded[str(path)] == sha256_of(path)


@pytest.mark.parametrize(
    "change",
    [
        {"src_lang": "rus_Cyrl"},
        {"tgt_lang": "rus_Cyrl"},
        {"corpus": "other"},
    ],
)
def test_a_changed_label_reruns(workspace: Workspace, change: dict[str, str]) -> None:
    _run_ok(workspace.args())

    result = _run_ok(workspace.args(**change))

    assert not result.stdout.startswith("skipped:")


@pytest.mark.parametrize(
    "option", [["--max-k", "3"], ["--skip-cost", "0.8"], ["--merge-penalty", "0.2"]]
)
def test_a_changed_alignment_parameter_reruns(
    workspace: Workspace, option: list[str]
) -> None:
    _run_ok(workspace.args())

    result = _run_ok(workspace.args(*option))

    assert not result.stdout.startswith("skipped:")


def test_a_run_for_another_output_in_the_same_directory_is_not_skipped(
    workspace: Workspace,
) -> None:
    other = workspace.out.with_name("other.parquet")
    _run_ok(workspace.args())
    _run_ok(workspace.args(out=str(other)))

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    assert [item.path for item in _manifest(workspace).outputs] == [str(workspace.out)]


def test_a_run_that_fails_part_way_leaves_no_manifest_for_a_later_run_to_trust(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(workspace.args())

    def fail_to_write(frame: pl.DataFrame, path: Path) -> int:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(mine_cli, "write_corpus", fail_to_write)
        failed = runner.invoke(app, workspace.args("--force"))

    assert failed.exit_code == 1, failed.output
    assert failed.stderr.startswith("error: cannot write the alignment outputs under ")
    assert "disk full" in failed.stderr
    assert not workspace.manifest.exists()

    rerun = _run_ok(workspace.args())

    assert not rerun.stdout.startswith("skipped:")
    assert _manifest(workspace).outputs[0].sha256 == sha256_of(workspace.out)


def test_an_output_directory_that_cannot_be_created_is_a_plain_failure(
    workspace: Workspace, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should go", encoding="utf-8")

    result = runner.invoke(app, workspace.args(out=str(blocker / "out" / "a.parquet")))

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: cannot write the alignment outputs under ")


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_missing_input_file_is_a_data_error(workspace: Workspace, side: str) -> None:
    path: Path = getattr(workspace, side)
    path.unlink()

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr == f"error: sentence file not found: {path}\n"
    assert result.stdout == ""
    assert not workspace.out.exists()


def test_a_directory_in_place_of_an_input_file_is_a_data_error(
    workspace: Workspace, tmp_path: Path
) -> None:
    result = runner.invoke(app, workspace.args(src=str(tmp_path)))

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: sentence file not found: ")


def test_an_input_file_that_is_not_utf8_is_a_data_error(workspace: Workspace) -> None:
    workspace.tgt.write_bytes(b"caf\xe9 au lait\n")

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith(f"error: {workspace.tgt} is not valid UTF-8")
    assert not workspace.out.exists()


@pytest.mark.parametrize("max_k", ["1", "-3"])
def test_max_k_below_two_is_a_config_error_before_any_file_is_read(
    workspace: Workspace, max_k: str
) -> None:
    workspace.src.unlink()

    result = runner.invoke(app, workspace.args("--max-k", max_k))

    assert result.exit_code == 2, result.output
    assert result.stderr == f"error: --max-k must be at least 2, got {max_k}\n"


@pytest.mark.parametrize(
    ("option", "language", "code"),
    [("src_lang", "xxx_Latn", "xxx_Latn"), ("tgt_lang", "mongolian", "mongolian")],
)
def test_a_language_outside_the_registry_is_a_config_error(
    workspace: Workspace, option: str, language: str, code: str
) -> None:
    workspace.tgt.unlink()

    result = runner.invoke(app, workspace.args(**{option: language}))

    assert result.exit_code == 2, result.output
    flag = f"--{option.replace('_', '-')}"
    assert result.stderr == (
        f"error: {flag} {code!r} is not in the language registry "
        "(configs/languages.yaml)\n"
    )


@pytest.mark.parametrize(
    "option",
    [
        "--skip-cost=-0.5",
        "--skip-cost=nan",
        "--skip-cost=inf",
        "--merge-penalty=-1",
        "--merge-penalty=nan",
    ],
)
def test_a_negative_or_non_finite_cost_is_a_config_error(
    workspace: Workspace, option: str
) -> None:
    workspace.src.unlink()
    name = option.split("=")[0]

    result = runner.invoke(app, workspace.args(option))

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith(f"error: {name} must be finite and non-negative")


@pytest.mark.parametrize("corpus", ["", "  ", "a/b", "..", "."])
def test_a_corpus_name_that_is_not_a_plain_name_is_a_config_error(
    workspace: Workspace, corpus: str
) -> None:
    result = runner.invoke(app, workspace.args(corpus=corpus, out=None))

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: --corpus must be a plain name")


def test_an_embedder_that_cannot_load_is_an_upstream_error_that_writes_nothing(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mine_cli, "build_embedder", UnavailableEmbedder)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 4, result.output
    assert result.stderr == "error: embedding model unavailable\n"
    assert not workspace.out.exists()
    assert not workspace.manifest.exists()


def test_an_embedder_that_returns_the_wrong_shape_is_an_upstream_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mine_cli, "build_embedder", WrongShapeEmbedder)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 4, result.output
    assert result.stderr.startswith("error: alignment failed")
    assert "almost certainly bad embedder output, otherwise an internal error" in (
        result.stderr
    )
    assert not workspace.manifest.exists()


# --- the surface ------------------------------------------------------------


def test_the_embedder_factory_builds_a_lazy_sentence_transformer() -> None:
    assert isinstance(REAL_BUILD_EMBEDDER(), SentenceTransformerEmbedder)


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["mine", "align", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout


def test_the_mine_group_lists_the_align_command() -> None:
    result = runner.invoke(app, ["mine", "--help"])

    assert result.exit_code == 0
    assert "align" in result.stdout
