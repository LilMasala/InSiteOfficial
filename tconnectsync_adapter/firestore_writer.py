from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import firebase_admin
from firebase_admin import firestore


def _ensure_firebase_app() -> None:
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app()


class CanonicalFirestoreWriter:
    def __init__(self):
        _ensure_firebase_app()
        self.db = firestore.client()

    def write_therapy_snapshot(self, uid: str, snapshot: dict[str, Any]) -> str:
        doc = self.db.collection("users").document(uid).collection("therapy_settings_log").document()
        doc.set(snapshot, merge=True)
        return doc.id

    def write_hourly_insulin_context(self, uid: str, document_id: str, payload: dict[str, Any]) -> None:
        self.db.collection("users").document(uid).collection("insulin_context").document("hourly").collection("items").document(document_id).set(payload, merge=True)

    def write_treatment_events(self, uid: str, events: list[dict[str, Any]]) -> int:
        if not events:
            return 0
        batch = self.db.batch()
        col = self.db.collection("users").document(uid).collection("insulin_context").document("events").collection("items")
        for event in events:
            batch.set(col.document(str(event["eventId"])), event, merge=True)
        batch.commit()
        return len(events)

    def write_connection_status(
        self,
        uid: str,
        *,
        connected: bool,
        needs_reauth: bool,
        region: str,
        source: str = "tandem",
        last_successful_sync: datetime | None = None,
        last_error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "source": source,
            "connected": connected,
            "needsReauth": needs_reauth,
            "region": region,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        if last_successful_sync is not None:
            payload["lastSuccessfulSync"] = last_successful_sync.astimezone(UTC)
        if last_error:
            payload["lastError"] = last_error

        self.db.collection("users").document(uid).collection("telemetry_sources").document("tandem").set(payload, merge=True)

