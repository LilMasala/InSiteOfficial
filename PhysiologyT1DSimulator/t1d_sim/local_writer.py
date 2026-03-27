"""Local filesystem artifact writer for no-Firebase simulation runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from t1d_sim.population import PatientConfig
from t1d_sim.simulate import DailySimResult
from t1d_sim.therapy import TherapySchedule, TherapySegment


class LocalArtifactWriter:
    def __init__(self, uid: str, root: str | Path, namespace: str = "dev-sim") -> None:
        self.uid = uid
        self.namespace = namespace
        self.root = Path(root).expanduser().resolve() / namespace / uid
        self.root.mkdir(parents=True, exist_ok=True)

    def write_user_profile(self, cfg: PatientConfig, email: str | None = None, namespace: str | None = None) -> None:
        payload: dict[str, Any] = {
            "uid": self.uid,
            "email": email,
            "persona": cfg.persona,
            "isFemale": cfg.is_female,
            "isSim": True,
            "simPersona": cfg.persona,
            "simSeed": cfg.seed,
            "simNamespace": namespace or self.namespace,
            "updatedAt": _iso_now(),
        }
        self._write_json(self.root / "user_profile.json", payload)

    def write_daily_result(self, result: DailySimResult) -> None:
        date_id = result.decision_frame.hour_start_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")
        payload = {
            "decision_frame": result.decision_frame.to_firebase_dict(),
            "feature_frames": [frame.to_firebase_dict() for frame in result.feature_frames],
            "bg_hourly": result.bg_hourly,
            "bg_average_hourly": result.bg_average_hourly,
            "bg_percent_hourly": result.bg_percent_hourly,
            "bg_uroc_hourly": result.bg_uroc_hourly,
            "hr_hourly": result.hr_hourly,
            "hr_daily_average": result.hr_daily_average,
            "energy_hourly": result.energy_hourly,
            "energy_daily_average": result.energy_daily_average,
            "exercise_hourly": result.exercise_hourly,
            "exercise_daily_average": result.exercise_daily_average,
            "sleep_daily": result.sleep_daily,
            "menstrual_daily": result.menstrual_daily,
            "site_daily": result.site_daily,
            "site_change_event": result.site_change_event,
            "therapy_hourly": result.therapy_hourly,
            "mood_events": result.mood_events,
            "mood_hourly": result.mood_hourly,
            "true_bg": result.true_bg.tolist(),
            "observed_cgm": result.observed_cgm.tolist(),
        }
        self._write_json(self.root / "daily_results" / f"{date_id}.json", payload)

    def write_therapy_snapshot(
        self,
        schedule: TherapySchedule,
        profile_id: str = "profile_default",
        profile_name: str = "Default",
        timestamp: datetime | None = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        payload = {
            "timestamp": ts.isoformat(),
            "profileId": profile_id,
            "profileName": profile_name,
            "hourRanges": [self._hour_range_payload(seg) for seg in schedule.segments],
            "therapyFunctionV2": {
                "version": 2,
                "tzIdentifier": schedule.tz_identifier,
                "resolutionMin": 30,
                "knots": [
                    {
                        "offsetMin": seg.start_min,
                        "basalRate": seg.basal,
                        "insulinSensitivity": seg.isf,
                        "carbRatio": seg.cr,
                    }
                    for seg in schedule.segments
                ],
            },
        }
        self._write_json(self.root / "therapy_snapshots" / f"{ts.strftime('%Y-%m-%dT%H-%M-%SZ')}.json", payload)

    def load_latest_therapy_schedule(self) -> TherapySchedule | None:
        snapshots_dir = self.root / "therapy_snapshots"
        if not snapshots_dir.exists():
            return None
        files = sorted(snapshots_dir.glob("*.json"))
        if not files:
            return None
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        ranges = data.get("hourRanges") or []
        if not ranges:
            return None
        segments = [
            TherapySegment(
                segment_id=str(item.get("id") or f"seg_{int(item['startMinute'])}_{int(item['endMinute'])}"),
                start_min=int(item["startMinute"]),
                end_min=int(item["endMinute"]),
                isf=float(item["insulinSensitivity"]),
                cr=float(item["carbRatio"]),
                basal=float(item["basalRate"]),
            )
            for item in ranges
        ]
        tz_identifier = data.get("therapyFunctionV2", {}).get("tzIdentifier", "UTC")
        return TherapySchedule(segments=segments, tz_identifier=str(tz_identifier))

    def write_sim_log(self, entries: list[dict[str, Any]]) -> None:
        log_dir = self.root / "sim_log"
        for entry in entries:
            doc_id = f"day_{int(entry['day']):04d}.json"
            self._write_json(log_dir / doc_id, entry)

    def load_sim_log_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        log_dir = self.root / "sim_log"
        if not log_dir.exists():
            return []
        files = sorted(log_dir.glob("day_*.json"))
        entries = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        return entries if limit is None else entries[:limit]

    def load_latest_sim_log_entry(self) -> dict[str, Any] | None:
        entries = self.load_sim_log_entries()
        return entries[-1] if entries else None

    def write_run_report(self, report: dict[str, Any]) -> None:
        self._write_json(self.root / "sim_reports" / "latest.json", report)
        run_id = str(report.get("run_id") or "latest")
        self._write_json(self.root / "sim_reports" / "history" / f"{run_id}.json", report)

    def load_latest_report(self) -> dict[str, Any] | None:
        path = self.root / "sim_reports" / "latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_all_user_data(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.root.exists():
            self.root.rmdir()

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _hour_range_payload(seg: TherapySegment) -> dict[str, Any]:
        return {
            "id": seg.segment_id,
            "startMinute": seg.start_min,
            "endMinute": seg.end_min,
            "carbRatio": seg.cr,
            "basalRate": seg.basal,
            "insulinSensitivity": seg.isf,
        }


def list_local_users(root: str | Path, namespace: str) -> list[dict[str, Any]]:
    base = Path(root).expanduser().resolve() / namespace
    if not base.exists():
        return []
    users: list[dict[str, Any]] = []
    for user_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        profile = user_dir / "user_profile.json"
        data = json.loads(profile.read_text(encoding="utf-8")) if profile.exists() else {}
        users.append({
            "uid": user_dir.name,
            "email": data.get("email"),
            "persona": data.get("persona"),
        })
    return users


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if is_dataclass(value):
        return _sanitize(asdict(value))
    return str(value)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
