"""Agreement statistics: pairwise chrF and Krippendorff's alpha.

Krippendorff's alpha is hand-implemented (not a new dependency) per
DECISIONS.md's build-ourselves precedent, established during D1's
pyonmttok/fasttext-wheel work. D2 produces exactly one review per item, so
`review_score_alpha` operates on a single-rater vector -- Krippendorff's
alpha is formally undefined (NaN) for that shape until a later milestone
captures 2+ independent reviews per item. The function itself is still
the real, tested formula (not a stub), ready for that future data shape.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from itertools import permutations

from rinchen.metrics.bleu_chrf import chrf


def krippendorff_alpha(reliability_data: list[list[float | None]]) -> float:
    """Krippendorff's alpha for interval-level data.

    `reliability_data` is a list of rater rows, each row a list of values
    per unit (same length across rows), with None for a missing rating.
    """
    n_raters = len(reliability_data)
    n_units = len(reliability_data[0]) if n_raters else 0

    units: list[list[float]] = []
    for u in range(n_units):
        vals: list[float] = []
        for r in range(n_raters):
            value = reliability_data[r][u]
            if value is not None:
                vals.append(value)
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan")

    coincidence: dict[tuple[float, float], float] = defaultdict(float)
    for vals in units:
        weight = 1.0 / (len(vals) - 1)
        for v1, v2 in permutations(vals, 2):
            coincidence[(v1, v2)] += weight

    n = sum(coincidence.values())

    marginal: dict[float, float] = defaultdict(float)
    for (v1, _v2), w in coincidence.items():
        marginal[v1] += w

    def metric(v1: float, v2: float) -> float:
        return (v1 - v2) ** 2

    d_o = sum(w * metric(v1, v2) for (v1, v2), w in coincidence.items()) / n

    d_e_sum = 0.0
    for v1, n_v1 in marginal.items():
        for v2, n_v2 in marginal.items():
            d_e_sum += n_v1 * n_v2 * metric(v1, v2)
    d_e = d_e_sum / (n * (n - 1))

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def pairwise_chrf_mean(conn: sqlite3.Connection, set_name: str) -> float:
    conn.row_factory = sqlite3.Row
    items = conn.execute(
        "SELECT id FROM items WHERE set_name = ?", (set_name,)
    ).fetchall()
    scores = []
    for item in items:
        rows = conn.execute(
            "SELECT tr.text FROM tasks t JOIN translations tr ON tr.task_id = t.id "
            "WHERE t.item_id = ? AND t.kind = 'translate' AND t.status = 'done' "
            "ORDER BY t.id",
            (item["id"],),
        ).fetchall()
        if len(rows) == 2:
            scores.append(chrf(rows[0]["text"], rows[1]["text"]))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def review_score_alpha(conn: sqlite3.Connection, set_name: str) -> float:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT r.score FROM reviews r "
        "JOIN tasks t ON t.id = r.task_id "
        "JOIN items i ON i.id = t.item_id "
        "WHERE i.set_name = ? AND t.status = 'done'",
        (set_name,),
    ).fetchall()
    scores: list[float | None] = [float(r["score"]) for r in rows]
    return krippendorff_alpha([scores])


def agreement_report(conn: sqlite3.Connection, set_name: str) -> dict[str, float]:
    return {
        "pairwise_chrf_mean": pairwise_chrf_mean(conn, set_name),
        "krippendorff_alpha": review_score_alpha(conn, set_name),
    }
