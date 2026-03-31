from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _utc_hour_id(ts: datetime) -> str:
    ts = ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")


def _timestamp_iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _nonnull(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class CanonicalTherapySnapshot:
    timestamp: datetime
    profile_id: str
    profile_name: str
    hour_ranges: list[dict[str, Any]]
    source: str = "tandem"
    source_profile_id: str | None = None

    def to_firestore(self) -> dict[str, Any]:
        return _nonnull(
            {
                "timestamp": self.timestamp,
                "profileId": self.profile_id,
                "profileName": self.profile_name,
                "hourRanges": self.hour_ranges,
                "source": self.source,
                "sourceProfileId": self.source_profile_id,
            }
        )


@dataclass(frozen=True)
class CanonicalTreatmentEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    source: str = "tandem"
    source_event_id: str | None = None
    insulin: float | None = None
    carbs: float | None = None
    rate: float | None = None
    duration_minutes: int | None = None
    entered_by: str | None = None
    notes: str | None = None
    raw_event_type: str | None = None
    pump_serial_number: str | None = None
    device_id: str | None = None
    profile_id: str | None = None

    def to_firestore(self) -> dict[str, Any]:
        return _nonnull(
            {
                "eventId": self.event_id,
                "timestamp": self.timestamp,
                "timestampIso": _timestamp_iso(self.timestamp),
                "eventType": self.event_type,
                "source": self.source,
                "sourceEventId": self.source_event_id,
                "insulin": self.insulin,
                "carbs": self.carbs,
                "rate": self.rate,
                "durationMinutes": self.duration_minutes,
                "enteredBy": self.entered_by,
                "notes": self.notes,
                "rawEventType": self.raw_event_type,
                "pumpSerialNumber": self.pump_serial_number,
                "deviceId": self.device_id,
                "profileId": self.profile_id,
            }
        )


@dataclass(frozen=True)
class CanonicalHourlyInsulinContext:
    hour_start_utc: datetime
    recent_bolus_count: int
    recent_carb_entry_count: int
    recent_temp_basal_count: int
    source: str = "tandem"
    iob: float | None = None
    cob: float | None = None

    @property
    def document_id(self) -> str:
        return _utc_hour_id(self.hour_start_utc)

    def to_firestore(self) -> dict[str, Any]:
        return _nonnull(
            {
                "hourStartUtc": self.document_id,
                "iob": self.iob,
                "cob": self.cob,
                "recentBolusCount": self.recent_bolus_count,
                "recentCarbEntryCount": self.recent_carb_entry_count,
                "recentTempBasalCount": self.recent_temp_basal_count,
                "source": self.source,
            }
        )


def canonical_snapshot_from_pump_settings(
    pump_settings: Any,
    *,
    synced_at: datetime,
) -> CanonicalTherapySnapshot:
    profiles = _safe_get(pump_settings, "profiles")
    active_idp = _safe_get(profiles, "activeIdp")
    all_profiles = list(_safe_get(profiles, "profile", []))
    active = next((p for p in all_profiles if _safe_get(p, "idp") == active_idp), None)
    if active is None:
        raise ValueError("active Tandem profile not found in pump settings")

    segments = list(_safe_get(active, "tDependentSegs", []))
    hour_ranges: list[dict[str, Any]] = []
    for index, seg in enumerate(segments):
        start_minute = int(_safe_get(seg, "startTime", 0))
        next_start = int(_safe_get(segments[index + 1], "startTime", 1440)) if index + 1 < len(segments) else 1440
        hour_ranges.append(
            {
                "startMinute": start_minute,
                "endMinute": next_start,
                "carbRatio": float(_safe_get(seg, "carbRatio")),
                "basalRate": float(_safe_get(seg, "basalRate")) / 1000.0,
                "insulinSensitivity": float(_safe_get(seg, "isf")),
            }
        )

    profile_id = f"tandem_{active_idp}"
    return CanonicalTherapySnapshot(
        timestamp=synced_at,
        profile_id=profile_id,
        profile_name=str(_safe_get(active, "name", "Tandem Imported Profile")),
        hour_ranges=hour_ranges,
        source="tandem",
        source_profile_id=str(active_idp) if active_idp is not None else None,
    )


def canonical_event_from_bolus(
    bolus_event: Any,
    *,
    event_id_prefix: str = "tandem-bolus",
    pump_serial_number: str | None = None,
    device_id: str | None = None,
) -> CanonicalTreatmentEvent:
    ts = _safe_get(bolus_event, "eventDateTime")
    if not isinstance(ts, datetime):
        raise ValueError("bolus eventDateTime must be a datetime")

    source_rec_id = _safe_get(bolus_event, "sourceRecId")
    event_id = f"{event_id_prefix}-{source_rec_id or int(ts.timestamp())}"
    description = _safe_get(bolus_event, "description") or "Bolus"
    raw_type = _safe_get(bolus_event, "type") or "Bolus"
    carbs = _safe_get(bolus_event, "carbs")
    insulin = _safe_get(bolus_event, "insulin")

    if carbs and float(carbs) > 0:
        event_type = "Carb Correction"
    else:
        event_type = "Bolus"

    return CanonicalTreatmentEvent(
        event_id=event_id,
        timestamp=ts,
        event_type=event_type,
        source="tandem",
        source_event_id=str(source_rec_id) if source_rec_id is not None else None,
        insulin=float(insulin) if insulin is not None else None,
        carbs=float(carbs) if carbs is not None else None,
        notes=str(description),
        raw_event_type=str(raw_type),
        pump_serial_number=pump_serial_number,
        device_id=device_id,
    )


def canonical_hourly_context(
    *,
    ts: datetime,
    iob: float | None,
    cob: float | None,
    recent_bolus_count: int,
    recent_carb_entry_count: int,
    recent_temp_basal_count: int,
    source: str = "tandem",
) -> CanonicalHourlyInsulinContext:
    return CanonicalHourlyInsulinContext(
        hour_start_utc=ts,
        iob=iob,
        cob=cob,
        recent_bolus_count=recent_bolus_count,
        recent_carb_entry_count=recent_carb_entry_count,
        recent_temp_basal_count=recent_temp_basal_count,
        source=source,
    )


def canonical_event_from_therapy_event(
    therapy_event: Any,
    *,
    pump_serial_number: str | None = None,
    device_id: str | None = None,
) -> CanonicalTreatmentEvent | None:
    raw_type = _safe_get(therapy_event, "type")
    timestamp = _safe_get(therapy_event, "eventDateTime") or _safe_get(therapy_event, "eventTime")
    if not isinstance(timestamp, datetime):
        return None

    source_rec_id = _safe_get(therapy_event, "sourceRecId")
    event_id = f"tandem-{raw_type or 'event'}-{source_rec_id or int(timestamp.timestamp())}"

    if raw_type == "Bolus":
        carbs = _safe_get(therapy_event, "carbs")
        insulin = _safe_get(therapy_event, "insulin")
        description = _safe_get(therapy_event, "description") or "Bolus"
        event_type = "Carb Correction" if carbs and float(carbs) > 0 else "Bolus"
        return CanonicalTreatmentEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            source="tandem",
            source_event_id=str(source_rec_id) if source_rec_id is not None else None,
            insulin=float(insulin) if insulin is not None else None,
            carbs=float(carbs) if carbs is not None else None,
            notes=str(description),
            raw_event_type=str(raw_type),
            pump_serial_number=pump_serial_number,
            device_id=device_id,
        )

    if raw_type == "Basal":
        rate = _safe_get(therapy_event, "basalRateValue")
        duration = _safe_get(therapy_event, "basalRateDuration")
        event_type = "Temp Basal"
        if rate is not None and float(rate) == 0.0:
            event_type = "Suspend"
        return CanonicalTreatmentEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            source="tandem",
            source_event_id=str(source_rec_id) if source_rec_id is not None else None,
            rate=float(rate) if rate is not None else None,
            duration_minutes=int(duration) if duration is not None else None,
            raw_event_type=str(raw_type),
            pump_serial_number=pump_serial_number,
            device_id=device_id,
        )

    return None
