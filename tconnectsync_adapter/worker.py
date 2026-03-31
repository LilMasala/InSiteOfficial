from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .canonical import (
    CanonicalHourlyInsulinContext,
    CanonicalTherapySnapshot,
    CanonicalTreatmentEvent,
    canonical_event_from_therapy_event,
    canonical_hourly_context,
    canonical_snapshot_from_pump_settings,
)
from .firestore_writer import CanonicalFirestoreWriter
from .store import TandemConnectionRecord, TandemConnectionStore


@dataclass
class TandemConnectionStatus:
    uid: str
    connected: bool
    needs_reauth: bool
    source: str = "tandem"
    region: str = "US"
    last_successful_sync: datetime | None = None
    last_error: str | None = None


class TandemAdapterWorker:
    """
    Service-side scaffold for a future Tandem ingestion worker.

    This deliberately stops at canonical mapping boundaries. It does not yet
    own credential storage, HTTP routing, or Firestore persistence.
    """

    def connect(self, *, uid: str, email: str, password: str, region: str = "US") -> TandemConnectionStatus:
        from tconnectsync.tconnectsync.api.tandemsource import TandemSourceApi

        TandemSourceApi(email, password, region=region)
        return TandemConnectionStatus(
            uid=uid,
            connected=True,
            needs_reauth=False,
            region=region,
            last_successful_sync=None,
            last_error=None,
        )

    def map_profile_snapshot(self, pump_settings: Any, *, synced_at: datetime | None = None) -> CanonicalTherapySnapshot:
        return canonical_snapshot_from_pump_settings(
            pump_settings,
            synced_at=synced_at or datetime.now(UTC),
        )

    def map_hourly_context(
        self,
        *,
        ts: datetime | None = None,
        iob: float | None,
        cob: float | None,
        recent_bolus_count: int,
        recent_carb_entry_count: int,
        recent_temp_basal_count: int,
    ) -> CanonicalHourlyInsulinContext:
        return canonical_hourly_context(
            ts=ts or datetime.now(UTC),
            iob=iob,
            cob=cob,
            recent_bolus_count=recent_bolus_count,
            recent_carb_entry_count=recent_carb_entry_count,
            recent_temp_basal_count=recent_temp_basal_count,
        )

    def map_events(self, events: list[CanonicalTreatmentEvent]) -> list[dict[str, Any]]:
        return [event.to_firestore() for event in events]


class _SecretShim:
    def __init__(self, pump_serial_number: str | None = None):
        self.PUMP_SERIAL_NUMBER = pump_serial_number
        self.FETCH_ALL_EVENT_TYPES = False


class TandemIngestionService:
    def __init__(
        self,
        *,
        store: TandemConnectionStore | None = None,
        writer: CanonicalFirestoreWriter | None = None,
    ):
        self.store = store or TandemConnectionStore()
        self.writer = writer

    def _writer(self) -> CanonicalFirestoreWriter:
        if self.writer is None:
            self.writer = CanonicalFirestoreWriter()
        return self.writer

    def connect(
        self,
        *,
        uid: str,
        email: str,
        password: str,
        region: str = "US",
        pump_serial_number: str | None = None,
    ) -> dict[str, Any]:
        from tconnectsync.tconnectsync.api import TConnectApi

        tconnect = TConnectApi(email, password, region)
        _ = tconnect.tandemsource

        record = TandemConnectionRecord(
            uid=uid,
            email=email,
            password=password,
            region=region,
            pump_serial_number=pump_serial_number,
            connected=True,
            needs_reauth=False,
            last_error=None,
        )
        self.store.save(record)
        self._writer().write_connection_status(uid, connected=True, needs_reauth=False, region=region, last_error=None)
        return {
            "ok": True,
            "connectionState": "connected",
            "source": "tandem",
            "region": region,
        }

    def status(self, *, uid: str) -> dict[str, Any]:
        record = self.store.load(uid)
        if record is None:
            return {
                "ok": True,
                "connected": False,
                "needsReauth": False,
                "source": "tandem",
            }
        return {
            "ok": True,
            "connected": record.connected,
            "needsReauth": record.needs_reauth,
            "lastSuccessfulSync": record.last_successful_sync,
            "lastError": record.last_error,
            "source": "tandem",
            "region": record.region,
        }

    def sync(self, *, uid: str, days: int = 14) -> dict[str, Any]:
        from tconnectsync.tconnectsync.api import TConnectApi
        from tconnectsync.tconnectsync.parser.tconnect import TConnectEntry
        from tconnectsync.tconnectsync.sync.tandemsource.choose_device import ChooseDevice
        from tconnectsync.tconnectsync.domain.tandemsource.pump_settings import PumpSettings

        record = self.store.load(uid)
        if record is None:
            raise ValueError(f"no Tandem connection stored for uid={uid}")

        tconnect = TConnectApi(record.email, record.password, record.region)
        secret = _SecretShim(record.pump_serial_number)
        device = ChooseDevice(secret, tconnect).choose()

        all_metadata = tconnect.tandemsource.pump_event_metadata()
        pump_meta = next((m for m in all_metadata if m["tconnectDeviceId"] == device["tconnectDeviceId"]), None)
        if pump_meta is None:
            raise ValueError("selected Tandem device metadata not found")

        raw_settings = (pump_meta.get("lastUpload") or {}).get("settings")
        if not raw_settings:
            raise ValueError("Tandem lastUpload.settings missing for selected device")

        synced_at = datetime.now(UTC)
        pump_settings = PumpSettings.from_dict(raw_settings)
        snapshot = canonical_snapshot_from_pump_settings(pump_settings, synced_at=synced_at)
        therapy_doc_id = self._writer().write_therapy_snapshot(uid, snapshot.to_firestore())

        time_end = datetime.now()
        from datetime import timedelta
        time_start = time_end - timedelta(days=days)

        therapy_events_raw = tconnect.controliq.therapy_events(time_start, time_end).get("event", [])
        canonical_events: list[CanonicalTreatmentEvent] = []
        recent_bolus_count = 0
        recent_carb_entry_count = 0
        recent_temp_basal_count = 0

        for raw_event in therapy_events_raw:
            try:
                parsed = TConnectEntry.parse_therapy_event(raw_event)
            except Exception:
                continue
            canonical = canonical_event_from_therapy_event(
                parsed,
                pump_serial_number=device.get("serialNumber"),
                device_id=str(device.get("tconnectDeviceId")),
            )
            if canonical is None:
                continue
            canonical_events.append(canonical)
            if canonical.event_type in ("Bolus", "Carb Correction"):
                recent_bolus_count += 1
            if canonical.carbs is not None and canonical.carbs > 0:
                recent_carb_entry_count += 1
            if canonical.event_type == "Temp Basal":
                recent_temp_basal_count += 1

        event_count = self._writer().write_treatment_events(uid, [event.to_firestore() for event in canonical_events])

        csv = tconnect.ws2.therapy_timeline_csv(time_start, time_end)
        iob_rows = csv.get("iobData") or []
        latest_iob = None
        for row in iob_rows:
            try:
                parsed = TConnectEntry.parse_iob_entry(row)
                latest_iob = float(parsed["iob"])
            except Exception:
                continue

        hourly_context = canonical_hourly_context(
            ts=synced_at,
            iob=latest_iob,
            cob=None,
            recent_bolus_count=recent_bolus_count,
            recent_carb_entry_count=recent_carb_entry_count,
            recent_temp_basal_count=recent_temp_basal_count,
            source="tandem",
        )
        self._writer().write_hourly_insulin_context(uid, hourly_context.document_id, hourly_context.to_firestore())

        self.store.update_status(uid, connected=True, needs_reauth=False, last_successful_sync=synced_at, last_error=None)
        self._writer().write_connection_status(
            uid,
            connected=True,
            needs_reauth=False,
            region=record.region,
            last_successful_sync=synced_at,
            last_error=None,
        )

        return {
            "ok": True,
            "source": "tandem",
            "therapySnapshotDocId": therapy_doc_id,
            "therapySnapshotWritten": True,
            "eventCount": event_count,
            "hourlyContextWritten": True,
            "lastSuccessfulSync": synced_at.isoformat().replace("+00:00", "Z"),
        }
