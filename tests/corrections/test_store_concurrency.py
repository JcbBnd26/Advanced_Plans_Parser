"""Concurrency regression tests for CorrectionStore.

This suite specifically targets the multi-process scenario (GUI + CLI/batch)
writing to the same SQLite WAL database.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from plancheck.corrections.store import CorrectionStore


@dataclass
class _DummyCandidate:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    trigger_methods: list[str]
    predicted_symbol: str = ""
    found_symbol: str = ""
    outcome: str = "hit"
    confidence: float = 0.9


def _writer_worker(db_path: str, n_rows: int, page_base: int) -> None:
    store = CorrectionStore(db_path=Path(db_path))
    candidates = []
    for i in range(n_rows):
        candidates.append(
            _DummyCandidate(
                page=page_base + i,
                x0=float(i),
                y0=0.0,
                x1=float(i) + 1.0,
                y1=1.0,
                trigger_methods=["test"],
            )
        )
    store.save_candidate_outcomes_batch(candidates, run_id="test_run")
    store.close()


def test_multi_process_candidate_outcome_writes(tmp_path: Path) -> None:
    """Two+ processes writing concurrently should not error or lose rows."""

    db_path = tmp_path / "shared.db"

    ctx = mp.get_context("spawn")
    n_procs = 4
    n_rows = 25

    procs = []
    for p in range(n_procs):
        proc = ctx.Process(
            target=_writer_worker,
            args=(str(db_path), n_rows, p * n_rows),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(30)
        assert proc.exitcode == 0

    store = CorrectionStore(db_path=db_path)
    total = store._conn.execute(
        "SELECT COUNT(*) AS n FROM candidate_outcomes"
    ).fetchone()["n"]
    store.close()

    assert total == n_procs * n_rows
