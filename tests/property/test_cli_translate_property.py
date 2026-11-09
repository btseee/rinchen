"""Property test of the line contract of `rinchen translate` (D11, Task 5): for any list of
lines (any character but LF, whatever its length, empty ones included), with or without a final
LF and for any chunk size, the output has exactly one line per input line, in the same order,
whatever statuses the engine gives them.

The engine is a `FakeEngine` whose status depends on the length of the text (so the statuses
are spread over all five); the output line of a row is the text of its answer. The hypothesis
profile is loaded in `tests/conftest.py`.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.infer.engine_types import Row, Status, Translation
from tests.support.translate_cli import (
    FakeEngine,
    install,
    invoke,
    translate_args,
    translation_of,
)

STATUSES: tuple[Status, ...] = ("ok", "empty", "too_long", "truncated", "no_output")
# Any character but LF; no lone surrogate (it is no text) and no CR at the end of a line (a
# trailing CR belongs to the line break and is dropped: a rule the unit tests pin).
LINE = st.text(
    st.characters(blacklist_categories=("Cs",), blacklist_characters="\n"), max_size=12
).filter(lambda line: not line.endswith("\r"))


def by_length(row: Row) -> Translation:
    """The status is drawn from the length of the text; only `ok` and `truncated` say a text."""
    status = STATUSES[len(row.text) % len(STATUSES)]
    text = f"<{len(row.text)}>" if status in ("ok", "truncated") else ""
    return translation_of(row, text, status)


@settings(max_examples=80, deadline=None)
@given(
    lines=st.lists(LINE, max_size=30),
    batch_size=st.integers(min_value=1, max_value=7),
    final_lf=st.booleans(),
)
def test_the_output_has_a_line_per_input_line_in_order(
    lines: list[str], batch_size: int, final_lf: bool
) -> None:
    content = "\n".join(lines) + "\n" if lines else ""
    if lines and lines[-1] and not final_lf:
        content = content.removesuffix("\n")
    engine = FakeEngine(answer=by_length)
    with pytest.MonkeyPatch.context() as patch:
        install(patch, engine)

        result = invoke(
            translate_args("--batch-size", str(batch_size)), stdin=content.encode()
        )

    assert result.exit_code == 0, result.stderr
    rows = [row for call in engine.calls for row in call.rows]
    assert [row.text for row in rows] == lines
    answers = [by_length(row) for row in rows]
    output = result.stdout_bytes.decode().split("\n")
    assert output.pop() == ""
    assert output == [answer.text for answer in answers]
    sizes = [len(call.rows) for call in engine.calls]
    assert all(size == batch_size for size in sizes[:-1])
    assert sizes == [] or 1 <= sizes[-1] <= batch_size
    counts = {status: [a.status for a in answers].count(status) for status in STATUSES}
    assert result.stderr.splitlines()[0] == (
        f"translated {len(lines)} lines: ok {counts['ok']}, empty {counts['empty']}, "
        f"too_long {counts['too_long']}, truncated {counts['truncated']}, "
        f"no_output {counts['no_output']}"
    )
