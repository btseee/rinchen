from __future__ import annotations

import math
from pathlib import Path

from rinchen.annotate import db
from rinchen.annotate.agreement import (
    agreement_report,
    krippendorff_alpha,
    pairwise_chrf_mean,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def test_krippendorff_alpha_perfect_agreement_is_one():
    data = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert krippendorff_alpha(data) == 1.0


def test_krippendorff_alpha_matches_hayes_krippendorff_2007_worked_example():
    # Canonical worked example from Hayes & Krippendorff (2007),
    # "Answering the Call for a Standard Reliability Measure for Coding
    # Data", Table 1 (interval metric). Cross-checked directly against the
    # real `krippendorff` PyPI package for this exact dataset, which gives
    # 0.8621041879468846 (level_of_measurement="interval").
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    ]
    alpha = krippendorff_alpha(data)
    assert math.isclose(alpha, 0.8621, abs_tol=0.001)


def test_pairwise_chrf_mean_on_identical_translations(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    db.import_items(conn, FIXTURES_DIR / "import_sample.jsonl", "demo")
    item_id = conn.execute("SELECT id FROM items LIMIT 1").fetchone()["id"]
    translate_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE item_id=? AND kind='translate'", (item_id,)
        ).fetchall()
    ]
    db.submit_translate(conn, translate_ids[0], "same text", 1.0, [])
    db.submit_translate(conn, translate_ids[1], "same text", 1.0, [])

    mean = pairwise_chrf_mean(conn, "demo")

    assert mean == 100.0


def test_agreement_report_returns_both_keys(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    db.init_db(conn)
    report = agreement_report(conn, "demo")
    assert set(report.keys()) == {"pairwise_chrf_mean", "krippendorff_alpha"}
    # No data yet in this fresh DB -- both are NaN, not an exception.
    assert math.isnan(report["pairwise_chrf_mean"])
    assert math.isnan(report["krippendorff_alpha"])
