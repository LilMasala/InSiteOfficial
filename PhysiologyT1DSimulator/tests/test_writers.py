from pathlib import Path
import sqlite3

from t1d_sim import simulate_population


def test_sqlite_writer(tmp_path: Path):
    db = tmp_path / "sim.db"
    simulate_population(outdb=str(db), n_patients=2, days=2, jobs=1)
    con = sqlite3.connect(db)
    n = con.execute("select count(*) from patients").fetchone()[0]
    assert n == 2
