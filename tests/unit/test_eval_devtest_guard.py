"""The devtest guard of D13 (spec T2, part c): no module of `src/rinchen` outside `rinchen/eval/`
and the commands `rinchen/cli/eval*.py` reaches the readers of the suite files
(`rinchen.eval.suites`, `rinchen.eval.suite_rules`, `rinchen.eval.suite_sources`), so training,
synthetic-data, distillation, filtering, mining and serving code cannot read a benchmark, devtest
included. The structural guard has two more parts that are not scans: `Suite.selection_view`
accepts only a `dev` suite and `rinchen eval report` refuses a `dev` run.

Two scans make the first part. The readers' own (`violations`) covers every file outside the exempt
ones. It cannot see a one-hop re-export (`rinchen.eval.runner.Suite` IS `rinchen.eval.suites.Suite`),
so the second scan is the inverse (`eval_violations`): NO module outside the exempt ones imports
anything of `rinchen.eval`, or any of the eval commands `rinchen.cli.eval*` (which import the suite
readers themselves), at all unless it is in `ALLOWED_EVAL_IMPORTERS`, each with its reason; a new
importer is a decision somebody takes in this file, with the reason written down. The modules of
the allowlist are still held by the first scan. A transitive import (an allowed module that imports
a module of the package that imports a suite reader) is not seen by either scan.

What the scan reads: `import` and `from ... import` statements (relative ones are resolved
against the package of the file, and `from rinchen.eval import suites` names the submodule it
takes), calls of `importlib.import_module` and `__import__`, also under another name (`from importlib
import import_module as im`; a literal name, or a relative one with its literal package; a name the
scan cannot read is itself a violation), and attribute chains such as
`rinchen.eval.suites.read_suite`, also through an alias (`import rinchen.eval as ev` then
`ev.suites`). It cannot see a computed `getattr`, `exec` or a plugin loader, which
nothing in the project uses. A mention in a docstring, a comment or a string is not an import.

Every function of the scan is tested on sources that violate it and on sources that do not, and
the scan of the real tree runs over every file that is not exempt."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

import rinchen
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

PACKAGE_ROOT: Final = Path(rinchen.__file__).resolve().parent
GUARDED: Final = (
    "rinchen.eval.suites",
    "rinchen.eval.suite_rules",
    "rinchen.eval.suite_sources",
)
# The commands `rinchen/cli/eval*.py` (a trailing `*` is a prefix) read suites too: from outside the
# exempt files, reaching one of them is reaching a suite reader through it.
EVAL_COMMANDS: Final = "rinchen.cli.eval*"
EVAL_PACKAGE: Final = ("rinchen.eval", EVAL_COMMANDS)
UNREADABLE: Final = "<an import the scan cannot read>"
DYNAMIC_IMPORTS: Final = ("import_module", "__import__")
# The modules outside `rinchen/eval/` and `rinchen/cli/eval*.py` that import `rinchen.eval` or one of
# the eval commands, each with the reason. They read a PARSED report (`report_read`, `report_view`,
# `report_format`, the cells of the Markdown) or the result types the report holds, or take a helper
# of the commands that reads no suite, never a suite file: the first scan still holds them to that.
ALLOWED_EVAL_IMPORTERS: Final = {
    "export/model_card.py": "renders the model card from the parsed report (report_read, report_view)",
    "export/model_card_evaluation.py": "the evaluation, robustness and target sections of the card, whose figures are the parsed report's",
    "export/model_card_notes.py": "the limits and provenance sections of the card, from the parsed report",
    "export/release.py": "the release record and the report-to-export binding, through the parsed report and the evidence label",
    "cli/release_stage.py": "`rinchen release stage` builds the card and the release record from the parsed report",
    "cli/release_stage_inputs.py": "`rinchen release stage` hashes and parses the report directory the card is bound to, and names its files and stage through cli.eval_report_output",
    "cli/release_stage_output.py": "writes the stage outputs with the manifest-led helpers begin_outputs and options_digest of cli.eval_suite_output, which read no suite",
    "cli/release_hub_output.py": "writes the plan outputs with the manifest-led helpers begin_outputs and options_digest of cli.eval_suite_output, which read no suite",
    "cli/main.py": "registers the `eval` command group (`from rinchen.cli import eval`) and takes nothing of a suite",
}


def inside(name: str, package: str) -> bool:
    if package.endswith("*"):
        return name.startswith(package.removesuffix("*"))
    return name == package or name.startswith(f"{package}.")


def guarded_of(name: str | None, guarded: Iterable[str] = GUARDED) -> str | None:
    if name is None:
        return None
    return next((module for module in guarded if inside(name, module)), None)


def absolute_base(node: ast.ImportFrom, package: str) -> str | None:
    """The module a `from` statement names, a relative one resolved against `package`."""
    if node.level == 0:
        return node.module
    parts = package.split(".")
    assert node.level - 1 < len(parts), "a relative import that leaves the package"
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def dotted(node: ast.expr, aliases: dict[str, str]) -> str | None:
    """`a.b.c` for a chain of names and attributes, its first name replaced by what an
    import bound it to."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(aliases.get(node.id, node.id))
    return ".".join(reversed(parts))


def dynamic_target(node: ast.Call, aliases: dict[str, str]) -> list[str]:
    """What an `import_module` or `__import__` call imports, as names (the function may be
    called by the name an `import ... as` or `from ... import ... as` gave it); `UNREADABLE` when
    the scan cannot tell."""
    function = node.func
    name = (
        aliases.get(function.id, function.id).rpartition(".")[2]
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else None
    )
    if name not in DYNAMIC_IMPORTS:
        return []
    package_argument = next(
        (k.value for k in node.keywords if k.arg == "package"), None
    )
    if package_argument is None and len(node.args) > 1:
        package_argument = node.args[1]
    first = node.args[0] if node.args else None
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return [UNREADABLE]
    if not first.value.startswith("."):
        return [first.value]
    if isinstance(package_argument, ast.Constant) and isinstance(
        package_argument.value, str
    ):
        return [importlib.util.resolve_name(first.value, package_argument.value)]
    return [UNREADABLE]


def aliases_of(tree: ast.AST, package: str) -> dict[str, str]:
    """The names an `import ... as` or `from ... import ... as` binds to a module."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            base = absolute_base(node, package)
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{base}.{alias.name}"
    return bound


def reached(
    source: str,
    module: str,
    *,
    is_package: bool = False,
    guarded: Iterable[str] = GUARDED,
) -> list[str]:
    """The `guarded` modules (the suite readers by default) `source`, the code of `module`,
    reaches, sorted; `UNREADABLE` stands for a dynamic import the scan cannot read."""
    package = module if is_package else module.rpartition(".")[0]
    tree = ast.parse(source)
    aliases = aliases_of(tree, package)
    found: set[str] = set()
    for node in ast.walk(tree):
        names: Iterable[str | None]
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = absolute_base(node, package)
            names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Call):
            names = dynamic_target(node, aliases)
        elif isinstance(node, ast.Attribute):
            names = [dotted(node, aliases)]
        else:
            continue
        for name in names:
            if name == UNREADABLE:
                found.add(UNREADABLE)
            elif (hit := guarded_of(name, guarded)) is not None:
                found.add(hit)
    return sorted(found)


def module_of(path: Path, root: Path) -> tuple[str, bool]:
    """The dotted name of the file under `root` (the folder that holds `rinchen`) and whether
    it is a package's `__init__`."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def is_exempt(path: Path, package_root: Path) -> bool:
    """The files that may read a suite: everything of `rinchen/eval/` and the commands
    `rinchen/cli/eval*.py`."""
    relative = path.relative_to(package_root)
    if relative.parts[0] == "eval":
        return True
    return (
        relative.parts[0] == "cli"
        and len(relative.parts) == 2
        and relative.name.startswith("eval")
    )


def scanned_files(package_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts and not is_exempt(path, package_root)
    ]


def violations(package_root: Path) -> dict[str, list[str]]:
    """The files outside the exempt ones that reach a guarded module, with what they reach."""
    root = package_root.parent
    found: dict[str, list[str]] = {}
    for path in scanned_files(package_root):
        module, is_package = module_of(path, root)
        hits = reached(path.read_text(encoding="utf-8"), module, is_package=is_package)
        if hits:
            found[path.relative_to(package_root).as_posix()] = hits
    return found


def eval_violations(
    package_root: Path, allowed: Iterable[str] = ALLOWED_EVAL_IMPORTERS
) -> dict[str, list[str]]:
    """The files outside the exempt ones and outside `allowed` that import anything of
    `rinchen.eval` (or a dynamic import the scan cannot read), with what they reach."""
    root = package_root.parent
    permitted = set(allowed)
    found: dict[str, list[str]] = {}
    for path in scanned_files(package_root):
        relative = path.relative_to(package_root).as_posix()
        if relative in permitted:
            continue
        module, is_package = module_of(path, root)
        hits = reached(
            path.read_text(encoding="utf-8"),
            module,
            is_package=is_package,
            guarded=EVAL_PACKAGE,
        )
        if hits:
            found[relative] = hits
    return found


# ---- the real tree ------------------------------------------------------------------------


def test_no_module_outside_eval_and_the_eval_commands_reaches_a_suite_reader() -> None:
    assert violations(PACKAGE_ROOT) == {}


def test_no_module_outside_eval_and_the_eval_commands_imports_rinchen_eval_but_the_allowed_ones() -> (
    None
):
    assert eval_violations(PACKAGE_ROOT) == {}


def test_every_allowed_importer_exists_has_a_reason_and_still_imports_rinchen_eval() -> (
    None
):
    for relative, reason in ALLOWED_EVAL_IMPORTERS.items():
        path = PACKAGE_ROOT / relative
        assert path.is_file(), relative
        assert not is_exempt(path, PACKAGE_ROOT), relative
        assert len(reason) > 20, relative
        module, is_package = module_of(path, PACKAGE_ROOT.parent)
        assert reached(
            path.read_text(encoding="utf-8"),
            module,
            is_package=is_package,
            guarded=EVAL_PACKAGE,
        ), f"{relative} no longer imports rinchen.eval: take it off the allowlist"


def test_the_allowed_importers_are_still_held_to_the_suite_readers_scan() -> None:
    scanned = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in scanned_files(PACKAGE_ROOT)
    }

    assert set(ALLOWED_EVAL_IMPORTERS) <= scanned
    assert not {
        name: hits
        for name, hits in violations(PACKAGE_ROOT).items()
        if name in ALLOWED_EVAL_IMPORTERS
    }


def test_the_scan_of_the_real_tree_reads_the_modules_that_must_not_read_a_suite() -> (
    None
):
    scanned = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in scanned_files(PACKAGE_ROOT)
    }

    assert len(scanned) > 150
    assert {
        "cli/main.py",
        "cli/train.py",
        "cli/synth.py",
        "cli/translate.py",
        "train/eole_run.py",
        "train/mixture.py",
        "synth/run.py",
        "serve/__init__.py",
        "filter/pipeline.py",
        "mine/pairs.py",
        "export/record.py",
        "infer/engine.py",
        "metrics/segment_stats.py",
        "config/synth.py",
        "acquire/hf.py",
        "__init__.py",
    } <= scanned


def test_the_scan_of_the_real_tree_skips_what_may_read_a_suite() -> None:
    scanned = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in scanned_files(PACKAGE_ROOT)
    }

    assert not {name for name in scanned if name.startswith("eval/")}
    assert not {
        name for name in scanned if name.startswith("cli/eval") and name.count("/") == 1
    }
    assert "cli/eval.py" not in scanned
    assert "cli/eval_robustness.py" not in scanned


def test_the_modules_the_guard_names_exist_where_the_guard_says() -> None:
    for module in ("suites", "suite_rules"):
        assert (PACKAGE_ROOT / "eval" / f"{module}.py").is_file()
    assert all(name.startswith("rinchen.eval.suite") for name in GUARDED)


# ---- what the scan finds ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "module", "expected"),
    [
        ("import rinchen.eval.suites\n", "rinchen.export.x", ["rinchen.eval.suites"]),
        (
            "import rinchen.eval.suite_rules as rules\n",
            "rinchen.train.x",
            ["rinchen.eval.suite_rules"],
        ),
        (
            "from rinchen.eval.suites import Suite\n",
            "rinchen.synth.x",
            ["rinchen.eval.suites"],
        ),
        (
            "from rinchen.eval import suites\n",
            "rinchen.serve.x",
            ["rinchen.eval.suites"],
        ),
        (
            "from rinchen.eval import suite_sources as sources, robust_suite\n",
            "rinchen.filter.x",
            ["rinchen.eval.suite_sources"],
        ),
        (
            "from rinchen.eval.suite_rules import checked_language, refusal\n",
            "rinchen.mine.x",
            ["rinchen.eval.suite_rules"],
        ),
        (
            "def late():\n    from rinchen.eval.suites import read_suite\n",
            "rinchen.cli.main",
            ["rinchen.eval.suites"],
        ),
        (
            "import rinchen.eval.suites, rinchen.eval.suite_sources\n",
            "rinchen.cli.train",
            ["rinchen.eval.suite_sources", "rinchen.eval.suites"],
        ),
    ],
)
def test_an_import_of_a_suite_reader_is_found(
    source: str, module: str, expected: list[str]
) -> None:
    assert reached(source, module) == expected


@pytest.mark.parametrize(
    ("source", "module", "is_package", "expected"),
    [
        (
            "from ..eval.suites import Suite\n",
            "rinchen.export.x",
            False,
            ["rinchen.eval.suites"],
        ),
        (
            "from ..eval import suite_rules\n",
            "rinchen.train.x",
            False,
            ["rinchen.eval.suite_rules"],
        ),
        (
            "from ...eval import suites\n",
            "rinchen.train.sub.x",
            False,
            ["rinchen.eval.suites"],
        ),
        ("from .. import eval\n", "rinchen.synth", True, []),
        (
            "from ..eval import suite_sources\n",
            "rinchen.synth",
            True,
            ["rinchen.eval.suite_sources"],
        ),
        ("from . import suites\n", "rinchen.eval.x", False, ["rinchen.eval.suites"]),
        ("from .suites import Suite\n", "rinchen.eval", True, ["rinchen.eval.suites"]),
        (
            "from .suites import Suite\n",
            "rinchen.eval.x",
            False,
            ["rinchen.eval.suites"],
        ),
    ],
)
def test_a_relative_import_is_resolved_against_the_package_of_the_file(
    source: str, module: str, is_package: bool, expected: list[str]
) -> None:
    assert reached(source, module, is_package=is_package) == expected


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nimportlib.import_module("rinchen.eval.suites")\n',
        'from importlib import import_module\nimport_module("rinchen.eval.suite_rules")\n',
        '__import__("rinchen.eval.suite_sources")\n',
        'import importlib\nimportlib.import_module(".suites", "rinchen.eval")\n',
        'import importlib\nimportlib.import_module(".suites", package="rinchen.eval")\n',
        'import importlib\nimportlib.import_module("rinchen.eval.suites.Suite")\n',
        # an alias of the function or of the module does not hide it
        'from importlib import import_module as im\nim("rinchen.eval.suites")\n',
        'import importlib as il\nil.import_module("rinchen.eval.suite_rules")\n',
        'from builtins import __import__ as imp\nimp("rinchen.eval.suite_sources")\n',
        'from importlib import import_module as im\nim(".suites", "rinchen.eval")\n',
        'from importlib import import_module as im\nim(".suites", package="rinchen.eval")\n',
        'def late():\n    from importlib import import_module as im\n    return im("rinchen.eval.suites")\n',
    ],
)
def test_a_dynamic_import_of_a_suite_reader_is_found(source: str) -> None:
    hits = reached(source, "rinchen.export.x")

    assert len(hits) == 1
    assert hits[0].startswith("rinchen.eval.suite")


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nimportlib.import_module(name)\n",
        'import importlib\nimportlib.import_module("." + name, "rinchen.eval")\n',
        'import importlib\nimportlib.import_module(".suites", package)\n',
        "__import__(name)\n",
        "import importlib\nimportlib.import_module()\n",
        "from importlib import import_module as im\nim(name)\n",
        "from builtins import __import__ as imp\nimp(name)\n",
    ],
)
def test_a_dynamic_import_the_scan_cannot_read_is_a_violation_too(source: str) -> None:
    assert reached(source, "rinchen.export.x") == [UNREADABLE]


@pytest.mark.parametrize(
    "source",
    [
        "import rinchen.eval\nrinchen.eval.suites.read_suite(path)\n",
        "import rinchen.eval as ev\nev.suites.read_suite(path)\n",
        "from rinchen import eval as ev\nev.suite_rules.refusal\n",
        "from rinchen import eval\neval.suite_sources.read_rows\n",
        "import rinchen\nrinchen.eval.suites\n",
    ],
)
def test_a_use_of_a_suite_reader_through_an_attribute_chain_is_found(
    source: str,
) -> None:
    hits = reached(source, "rinchen.export.x")

    assert len(hits) == 1
    assert hits[0].startswith("rinchen.eval.suite")


@pytest.mark.parametrize(
    "source",
    [
        "import rinchen.eval.robust_suite\n",
        "from rinchen.eval import robust_suite, significance, bootstrap\n",
        "from rinchen.eval.robust_suite_rules import is_id\n",
        "from rinchen.eval.report_read import read_report\n",
        "from rinchen.eval.suite import x\n",
        "from rinchen.eval.suites_of_others import x\n",
        "from rinchen.export.record import ModelRecord\n",
        "from rinchen import eval\n",
        "import rinchen.eval\n",
        'import importlib\nimportlib.import_module("eole")\n',
        'from importlib import import_module as im\nim("eole")\n',
        # an alias of something else than an import function is not a dynamic import
        'from json import loads as im\nim("rinchen.eval.suites")\n',
        "import rinchen.metrics.segment_stats as stats\nstats.statistics_metric\n",
        "from . import suites\n",
        "from .suites import Suite\n",
        "from ..export import suites\n",
        '"""Reads rinchen.eval.suites and rinchen.eval.suite_rules."""\n# rinchen.eval.suite_sources\n',
        'MESSAGE = "see rinchen.eval.suites"\nprint(MESSAGE)\n',
        "suites = 1\nprint(suites)\n",
        "def eval(x):\n    return x.suites\n",
    ],
)
def test_what_is_not_an_import_of_a_suite_reader_is_not_flagged(source: str) -> None:
    assert reached(source, "rinchen.export.x") == []


# ---- the inverse scan: any import of rinchen.eval ----------------------------------------


@pytest.mark.parametrize(
    ("source", "module", "is_package"),
    [
        # the one-hop re-export of the review: the class IS the one of rinchen.eval.suites
        ("from rinchen.eval.runner import Suite\n", "rinchen.train.x", False),
        ("from rinchen.eval.runner import Suite as S\n", "rinchen.synth.x", False),
        ("import rinchen.eval.runner\n", "rinchen.export.x", False),
        ("import rinchen.eval.runner as runner\n", "rinchen.serve.x", False),
        ("from rinchen.eval import runner\n", "rinchen.filter.x", False),
        ("from rinchen.eval import runner as r, bootstrap\n", "rinchen.mine.x", False),
        ("import rinchen.eval\n", "rinchen.cli.train", False),
        ("import rinchen.eval as ev\n", "rinchen.cli.synth", False),
        ("from rinchen import eval\n", "rinchen.export.x", False),
        ("from rinchen import eval as ev\n", "rinchen.export.x", False),
        ("from rinchen import config, eval\n", "rinchen.train.x", False),
        ("from rinchen.eval import *\n", "rinchen.train.x", False),
        (
            "def late():\n    from rinchen.eval.runner import Suite\n",
            "rinchen.cli.main",
            False,
        ),
        ("from ..eval.runner import Suite\n", "rinchen.train.x", False),
        ("from ..eval import runner\n", "rinchen.train.x", False),
        ("from .. import eval\n", "rinchen.train.x", False),
        ("from ...eval import runner\n", "rinchen.train.sub.x", False),
        ("from .. import eval\n", "rinchen.synth", True),
        ("import rinchen\nrinchen.eval.runner.Suite\n", "rinchen.export.x", False),
        (
            'import importlib\nimportlib.import_module("rinchen.eval.runner")\n',
            "rinchen.train.x",
            False,
        ),
        ('__import__("rinchen.eval")\n', "rinchen.train.x", False),
        (
            'import importlib\nimportlib.import_module(".runner", "rinchen.eval")\n',
            "rinchen.train.x",
            False,
        ),
    ],
)
def test_any_import_of_rinchen_eval_is_found_by_the_inverse_scan_the_one_hop_ones_too(
    source: str, module: str, is_package: bool
) -> None:
    assert reached(source, module, is_package=is_package, guarded=EVAL_PACKAGE) == [
        "rinchen.eval"
    ]


def test_the_suite_readers_scan_does_not_see_the_one_hop_re_export_and_the_inverse_one_does() -> (
    None
):
    source = "from rinchen.eval.runner import Suite\n"

    assert reached(source, "rinchen.train.x") == []
    assert reached(source, "rinchen.train.x", guarded=EVAL_PACKAGE) == ["rinchen.eval"]


def test_an_import_the_inverse_scan_cannot_read_is_a_violation_too() -> None:
    assert reached(
        "import importlib\nimportlib.import_module(name)\n",
        "rinchen.train.x",
        guarded=EVAL_PACKAGE,
    ) == [UNREADABLE]


@pytest.mark.parametrize(
    ("source", "module"),
    [
        # the review's probe: the exempt eval commands import the suite readers themselves
        ("from rinchen.cli.eval_report import Suite\n", "rinchen.train.x"),
        ("from rinchen.cli.eval_suite_output import begin_outputs\n", "rinchen.cli.x"),
        ("from rinchen.cli import eval_report\n", "rinchen.synth.x"),
        ("from rinchen.cli import eval_run as run\n", "rinchen.serve.x"),
        ("import rinchen.cli.eval_suite\n", "rinchen.filter.x"),
        # a submodule of a command that is a package
        ("import rinchen.cli.eval_report.sub\n", "rinchen.filter.x"),
        ("import rinchen.cli.eval.groups as groups\n", "rinchen.filter.x"),
        ("import rinchen.cli.eval_suite as suite_command\n", "rinchen.mine.x"),
        ("from rinchen.cli import eval as eval_cli\n", "rinchen.cli.main"),
        ("from rinchen.cli import (\n    eval,\n    train,\n)\n", "rinchen.cli.main"),
        ("import rinchen.cli.eval\n", "rinchen.cli.main"),
        ("from rinchen.cli.eval import app\n", "rinchen.cli.main"),
        ("from rinchen.cli.eval_report_output import STAGE\n", "rinchen.export.x"),
        ("from . import eval\n", "rinchen.cli.main"),
        ("from .eval_report import Suite\n", "rinchen.cli.x"),
        ("from . import eval_run\n", "rinchen.cli.x"),
        ("from .. import cli\ncli.eval_report.Suite\n", "rinchen.train.x"),
        ("from ..cli import eval_report\n", "rinchen.train.x"),
        ("from ..cli.eval_report import Suite\n", "rinchen.train.x"),
        ("import rinchen\nrinchen.cli.eval_suite.build\n", "rinchen.export.x"),
        ("import rinchen.cli as c\nc.eval_report.report_runs\n", "rinchen.train.x"),
        (
            "def late():\n    from rinchen.cli.eval_report import Suite\n",
            "rinchen.train.x",
        ),
        (
            'import importlib\nimportlib.import_module("rinchen.cli.eval_report")\n',
            "rinchen.train.x",
        ),
        (
            'from importlib import import_module as im\nim("rinchen.cli.eval_suite")\n',
            "rinchen.train.x",
        ),
    ],
)
def test_an_import_of_an_eval_command_is_found_by_the_inverse_scan(
    source: str, module: str
) -> None:
    # The commands `rinchen/cli/eval*.py` are exempt because they read suites, so reaching one
    # from outside is reaching a suite reader through it.
    assert reached(source, module, guarded=EVAL_PACKAGE) == ["rinchen.cli.eval*"]


def test_the_suite_readers_scan_does_not_see_an_eval_command_and_the_inverse_one_does() -> (
    None
):
    source = "from rinchen.cli.eval_report import Suite\n"

    assert reached(source, "rinchen.train.x") == []
    assert reached(source, "rinchen.train.x", guarded=EVAL_PACKAGE) == [
        "rinchen.cli.eval*"
    ]


def test_an_eval_package_import_and_an_eval_command_import_are_told_apart() -> None:
    source = "from rinchen.eval.runner import Suite\nfrom rinchen.cli import eval_run\n"

    assert reached(source, "rinchen.train.x", guarded=EVAL_PACKAGE) == [
        "rinchen.cli.eval*",
        "rinchen.eval",
    ]


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("import rinchen.evaluation\n", "rinchen.train.x"),
        # a command that is not one of the eval commands
        ("from rinchen.cli import train, synth\n", "rinchen.cli.main"),
        ("from rinchen.cli.train_common import write_manifest\n", "rinchen.cli.x"),
        ("from . import train\n", "rinchen.cli.main"),
        ("import rinchen.cli.translate\n", "rinchen.train.x"),
        ("from rinchen.cli import e\n", "rinchen.train.x"),
        ("from rinchen.evaluator import x\n", "rinchen.train.x"),
        ("from rinchen.export.record import ModelRecord\n", "rinchen.train.x"),
        ("from rinchen.metrics import bleu_chrf\n", "rinchen.train.x"),
        ("import evaluate\n", "rinchen.train.x"),
        ('"""Reads rinchen.eval.runner."""\n# rinchen.eval\n', "rinchen.train.x"),
        ('MESSAGE = "see rinchen.eval"\nprint(MESSAGE)\n', "rinchen.train.x"),
        ("def eval(x):\n    return x\n", "rinchen.train.x"),
        ("import importlib\nimportlib.import_module('eole')\n", "rinchen.train.x"),
    ],
)
def test_what_is_not_an_import_of_rinchen_eval_is_not_flagged_by_the_inverse_scan(
    source: str, module: str
) -> None:
    assert reached(source, module, guarded=EVAL_PACKAGE) == []


# ---- which files are looked at and which are exempt --------------------------------------


def build_tree(root: Path, files: dict[str, str]) -> Path:
    package_root = root / "rinchen"
    for relative, source in files.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


LEAK = "from rinchen.eval.suites import read_suite\n"
RELATIVE_LEAK = "from ..eval import suite_rules\n"


def test_the_scan_finds_a_leak_in_any_package_outside_the_exempt_files(
    tmp_path: Path,
) -> None:
    package_root = build_tree(
        tmp_path,
        {
            "__init__.py": "",
            "export/__init__.py": "",
            "export/leak.py": LEAK,
            "train/__init__.py": RELATIVE_LEAK,
            "train/ok.py": "from rinchen.export.record import ModelRecord\n",
            "synth/deep/leak.py": "import rinchen.eval.suite_sources\n",
            "cli/main.py": "def run():\n    from rinchen.eval import suites\n",
            "cli/robust_eval.py": LEAK,
            "cli/sub/eval_x.py": LEAK,
            "evaluate/x.py": LEAK,
            "metrics/dyn.py": "import importlib\nimportlib.import_module(name)\n",
        },
    )

    assert violations(package_root) == {
        "export/leak.py": ["rinchen.eval.suites"],
        "train/__init__.py": ["rinchen.eval.suite_rules"],
        "synth/deep/leak.py": ["rinchen.eval.suite_sources"],
        "cli/main.py": ["rinchen.eval.suites"],
        "cli/robust_eval.py": ["rinchen.eval.suites"],
        "cli/sub/eval_x.py": ["rinchen.eval.suites"],
        "evaluate/x.py": ["rinchen.eval.suites"],
        "metrics/dyn.py": [UNREADABLE],
    }


def test_the_scan_exempts_the_eval_package_and_the_eval_commands_only(
    tmp_path: Path,
) -> None:
    package_root = build_tree(
        tmp_path,
        {
            "__init__.py": "",
            "eval/__init__.py": "",
            "eval/report.py": LEAK,
            "eval/suite_sources.py": "from .suites import Suite\n",
            "cli/eval.py": LEAK,
            "cli/eval_suite.py": LEAK,
            "cli/eval_robustness_plan.py": LEAK,
        },
    )

    assert violations(package_root) == {}
    assert [
        p.relative_to(package_root).as_posix() for p in scanned_files(package_root)
    ] == ["__init__.py"]


def test_a_file_that_is_not_python_source_is_not_read(tmp_path: Path) -> None:
    package_root = build_tree(
        tmp_path,
        {"export/__init__.py": "", "export/notes.txt": LEAK, "export/data.json": LEAK},
    )

    assert violations(package_root) == {}


def test_the_inverse_scan_finds_the_one_hop_leak_in_any_package_but_the_exempt_and_allowed(
    tmp_path: Path,
) -> None:
    one_hop = "from rinchen.eval.runner import Suite\n"
    package_root = build_tree(
        tmp_path,
        {
            "__init__.py": "",
            "train/mixture.py": one_hop,
            "train/ok.py": "from rinchen.export.record import ModelRecord\n",
            "synth/deep/leak.py": "import rinchen.eval\n",
            "serve/__init__.py": "from .. import eval\n",
            "train/zz_probe.py": "from rinchen.cli.eval_report import Suite\n",
            "cli/main.py": "from rinchen.cli import eval as eval_cli\n",
            "cli/release_hub_output.py": "from rinchen.cli.eval_suite_output import x\n",
            "cli/release_stage.py": "from rinchen.eval.report_read import ReportView\n",
            "cli/robust_eval.py": one_hop,
            "export/model_card.py": "from rinchen.eval.report_read import ReportView\n",
            "export/model_card_extra.py": one_hop,
            "eval/report.py": one_hop,
            "cli/eval_run.py": one_hop,
        },
    )
    allowed = ["cli/release_stage.py", "export/model_card.py"]

    assert eval_violations(package_root, allowed) == {
        "train/mixture.py": ["rinchen.eval"],
        "synth/deep/leak.py": ["rinchen.eval"],
        "serve/__init__.py": ["rinchen.eval"],
        "train/zz_probe.py": ["rinchen.cli.eval*"],
        "cli/main.py": ["rinchen.cli.eval*"],
        "cli/release_hub_output.py": ["rinchen.cli.eval*"],
        "cli/robust_eval.py": ["rinchen.eval"],
        "export/model_card_extra.py": ["rinchen.eval"],
    }


def test_an_allowed_file_is_exempt_by_its_exact_path_only(tmp_path: Path) -> None:
    package_root = build_tree(
        tmp_path,
        {
            "__init__.py": "",
            "export/release.py": "from rinchen.eval.report_read import ReportView\n",
            "export/release_extra.py": "from rinchen.eval.report_read import ReportView\n",
            "synth/release.py": "from rinchen.eval.report_read import ReportView\n",
        },
    )

    assert eval_violations(package_root, ["export/release.py"]) == {
        "export/release_extra.py": ["rinchen.eval"],
        "synth/release.py": ["rinchen.eval"],
    }
