"""Firestore/Firebase writer aligned with the current InSite app schema."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from firebase_admin import firestore, storage

from t1d_sim.population import PatientConfig
from t1d_sim.simulate import DailySimResult
from t1d_sim.therapy import TherapySchedule, TherapySegment


def _iso_hour(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00:00Z")


def _iso_day(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


class FirebaseWriter:
    def __init__(self, uid: str) -> None:
        self.uid = uid
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def write_user_profile(
        self,
        cfg: PatientConfig,
        email: str | None = None,
        namespace: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "persona": cfg.persona,
            "isFemale": cfg.is_female,
            "isSim": True,
            "simPersona": cfg.persona,
            "simSeed": cfg.seed,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        if email:
            payload["email"] = email
            payload["displayName"] = email.split("@")[0]
        if namespace:
            payload["simNamespace"] = namespace
        self._user_doc().set(payload, merge=True)

    def write_daily_result(self, result: DailySimResult) -> None:
        for row in result.bg_hourly:
            self._set_items_doc("blood_glucose", "hourly", row["hourUtc"], row)
        for row in result.bg_average_hourly:
            self._set_items_doc("blood_glucose", "average", row["hourUtc"], row)
        for row in result.bg_percent_hourly:
            self._set_items_doc("blood_glucose", "percent", row["hourUtc"], row)
        for row in result.bg_uroc_hourly:
            self._set_items_doc("blood_glucose", "uROC", row["hourUtc"], row)

        for row in result.hr_hourly:
            self._set_items_doc("heart_rate", "hourly", row["hourUtc"], row)
        self._set_items_doc("heart_rate", "daily_average", result.hr_daily_average["dateUtc"], result.hr_daily_average)

        for row in result.energy_hourly:
            self._set_items_doc("energy", "hourly", row["hourUtc"], row)
        self._set_items_doc("energy", "daily_average", result.energy_daily_average["dateUtc"], result.energy_daily_average)

        for row in result.exercise_hourly:
            self._set_items_doc("exercise", "hourly", row["hourUtc"], row)
        self._set_items_doc("exercise", "daily_average", result.exercise_daily_average["dateUtc"], result.exercise_daily_average)

        if result.sleep_daily is not None:
            self._set_items_doc("sleep", "daily", result.sleep_daily["dateUtc"], result.sleep_daily)
        if result.menstrual_daily is not None:
            self._set_items_doc("menstrual", "daily", result.menstrual_daily["dateUtc"], result.menstrual_daily)

        self._set_items_doc("site_changes", "daily", result.site_daily["dateUtc"], result.site_daily)
        if result.site_change_event is not None:
            event = dict(result.site_change_event)
            event["clientTimestamp"] = result.site_change_event["timestamp"].isoformat()
            self._set_items_doc("site_changes", "events", event["id"], event)

        for event in result.mood_events:
            payload = {
                "clientTs": event["timestamp"],
                "serverTs": firestore.SERVER_TIMESTAMP,
                "valence": event["valence"],
                "arousal": event["arousal"],
            }
            self._set_items_doc("mood", "events", event["id"], payload)
        for row in result.mood_hourly:
            self._set_items_doc("mood", "hourly", row["hourUtc"], row)

        for frame in result.feature_frames:
            payload = frame.to_firebase_dict()
            self._set_items_doc("features", "ml_feature_frames", payload["hourStartUtc"], payload)

        for row in result.therapy_hourly:
            self._set_items_doc("therapy_settings", "hourly", row["hourStartUtc"], row)

    def write_therapy_snapshot(
        self,
        schedule: TherapySchedule,
        profile_id: str = "profile_default",
        profile_name: str = "Default",
        timestamp: datetime | None = None,
    ) -> None:
        ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        hour_ranges = [self._hour_range_payload(seg) for seg in schedule.segments]
        payload = {
            "timestamp": ts,
            "profileId": profile_id,
            "profileName": profile_name,
            "hourRanges": hour_ranges,
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
        doc_id = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._user_doc().collection("therapy_settings_log").document(doc_id).set(payload, merge=True)

    def load_latest_therapy_schedule(self) -> TherapySchedule | None:
        docs = (
            self._user_doc()
            .collection("therapy_settings_log")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .get()
        )
        if not docs:
            return None
        data = docs[0].to_dict() or {}
        ranges = data.get("hourRanges") or []
        if not ranges:
            return None
        segments = [
            TherapySegment(
                segment_id=str(item.get("id") or f"seg_{int(item['startMinute'])}_{int(item['endMinute'])}" or f"seg_{idx}"),
                start_min=int(item["startMinute"]),
                end_min=int(item["endMinute"]),
                isf=float(item["insulinSensitivity"]),
                cr=float(item["carbRatio"]),
                basal=float(item["basalRate"]),
            )
            for idx, item in enumerate(ranges)
        ]
        tz_identifier = data.get("therapyFunctionV2", {}).get("tzIdentifier", "UTC")
        return TherapySchedule(segments=segments, tz_identifier=str(tz_identifier))

    def write_sim_log(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            doc_id = f"day_{int(entry['day']):04d}"
            self._set_items_doc("sim_log", "entries", doc_id, entry)

    def write_run_report(self, report: dict[str, Any]) -> None:
        reports = self._user_doc().collection("sim_reports")
        reports.document("latest").set(report, merge=True)
        run_id = str(report.get("run_id") or "latest")
        reports.document("history").collection("items").document(run_id).set(report, merge=True)

    def load_latest_report(self) -> dict[str, Any] | None:
        doc = self._user_doc().collection("sim_reports").document("latest").get()
        if not doc.exists:
            return None
        return doc.to_dict() or None

    def load_latest_sim_log_entry(self) -> dict[str, Any] | None:
        docs = (
            self._user_doc()
            .collection("sim_log")
            .document("entries")
            .collection("items")
            .order_by("day", direction=firestore.Query.DESCENDING)
            .limit(1)
            .get()
        )
        if not docs:
            return None
        return docs[0].to_dict() or None

    def load_sim_log_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = (
            self._user_doc()
            .collection("sim_log")
            .document("entries")
            .collection("items")
            .order_by("day", direction=firestore.Query.ASCENDING)
        )
        if limit is not None:
            query = query.limit(limit)
        return [(doc.to_dict() or {}) for doc in query.get()]

    def delete_all_user_data(self) -> None:
        prefix = f"users/{self.uid}/"
        for blob in self.bucket.list_blobs(prefix=prefix):
            blob.delete()

    def _user_doc(self):
        return self.db.collection("users").document(self.uid)

    def _set_items_doc(self, kind: str, subpath: str, doc_id: str, payload: dict[str, Any]) -> None:
        cleaned = {k: v for k, v in payload.items() if v is not None}
        (
            self._user_doc()
            .collection(kind)
            .document(subpath)
            .collection("items")
            .document(doc_id)
            .set(cleaned, merge=True)
        )

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
