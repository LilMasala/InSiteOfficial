import sqlite3
import subprocess
import sys


def test_cli(tmp_path):
    db = tmp_path / "cli.db"
    subprocess.check_call([sys.executable, "-m", "t1d_sim", "--outdb", str(db), "--n_patients", "2", "--days", "2"])
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from bg_hourly").fetchone()[0] > 0
