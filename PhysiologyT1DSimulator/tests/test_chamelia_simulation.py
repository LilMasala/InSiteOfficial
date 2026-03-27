import json
import sqlite3
import subprocess
import sys


def test_run_chamelia_simulation_cli(tmp_path):
    db = tmp_path / "chamelia.db"
    report = tmp_path / "chamelia_report.json"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "chamelia.run_simulation",
            "--n-patients",
            "3",
            "--days",
            "10",
            "--seed",
            "7",
            "--outdb",
            str(db),
            "--report",
            str(report),
            "--quiet",
        ]
    )

    summary = json.loads(report.read_text())
    assert summary["n_patients"] == 3
    assert summary["n_days"] == 10
    assert "did_mean_tir_improve" in summary
    assert "did_burnout_remain_acceptable" in summary
    assert len(summary["tir_by_30_day_bucket"]) >= 1

    con = sqlite3.connect(db)
    assert con.execute("select count(*) from patients").fetchone()[0] == 3
    assert con.execute("select count(*) from patient_run_summary").fetchone()[0] == 3
    assert con.execute("select count(*) from simulation_runs").fetchone()[0] == 1
    assert con.execute("select count(*) from evaluation_snapshots").fetchone()[0] >= 1
