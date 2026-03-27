"""Therapy schedule surfaces for schedule-aware simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from t1d_sim.population import PatientConfig


@dataclass(frozen=True)
class TherapySegment:
    """One contiguous therapy segment over the local 24h schedule."""

    segment_id: str
    start_min: int
    end_min: int
    isf: float
    cr: float
    basal: float


@dataclass(frozen=True)
class SegmentDelta:
    """Typed Level-1 schedule edit for one segment."""

    segment_id: str
    isf_delta: float = 0.0
    cr_delta: float = 0.0
    basal_delta: float = 0.0


@dataclass(frozen=True)
class StructureEdit:
    """Typed Level-2 schedule structure edit."""

    edit_type: str
    target_segment_id: str
    split_at_minute: int | None = None
    neighbor_segment_id: str | None = None


def _coerce_delta(raw: SegmentDelta | dict) -> SegmentDelta:
    if isinstance(raw, SegmentDelta):
        return raw
    return SegmentDelta(
        segment_id=str(raw["segment_id"]),
        isf_delta=float(raw.get("isf_delta", 0.0)),
        cr_delta=float(raw.get("cr_delta", 0.0)),
        basal_delta=float(raw.get("basal_delta", 0.0)),
    )


def _coerce_edit(raw: StructureEdit | dict) -> StructureEdit:
    if isinstance(raw, StructureEdit):
        return raw
    return StructureEdit(
        edit_type=str(raw["edit_type"]),
        target_segment_id=str(raw["target_segment_id"]),
        split_at_minute=None if raw.get("split_at_minute") is None else int(raw["split_at_minute"]),
        neighbor_segment_id=None if raw.get("neighbor_segment_id") is None else str(raw["neighbor_segment_id"]),
    )


@dataclass
class TherapySchedule:
    """Piecewise therapy schedule covering the 24h day."""

    segments: list[TherapySegment] = field(default_factory=list)
    tz_identifier: str = "UTC"

    def __post_init__(self) -> None:
        self.segments = sorted(self.segments, key=lambda seg: seg.start_min)

    def copy(self) -> "TherapySchedule":
        return TherapySchedule(segments=list(self.segments), tz_identifier=self.tz_identifier)

    def value_at_minute(self, minute: int) -> TherapySegment:
        minute = minute % 1440
        for seg in self.segments:
            if seg.start_min <= minute < seg.end_min:
                return seg
        if self.segments and minute == 0:
            return self.segments[0]
        raise ValueError("therapy schedule does not cover the full 24h surface")

    def weighted_mean(self, field_name: str) -> float:
        total = 0.0
        covered = 0
        for seg in self.segments:
            width = seg.end_min - seg.start_min
            total += width * float(getattr(seg, field_name))
            covered += width
        return total / max(covered, 1)

    def apply_level1_action(self, segment_deltas: Iterable[SegmentDelta | dict]) -> "TherapySchedule":
        coerced = [_coerce_delta(delta) for delta in segment_deltas]
        deltas = {delta.segment_id: delta for delta in coerced}
        new_segments: list[TherapySegment] = []
        for seg in self.segments:
            delta = deltas.get(seg.segment_id)
            if delta is None:
                new_segments.append(seg)
                continue
            new_segments.append(
                TherapySegment(
                    segment_id=seg.segment_id,
                    start_min=seg.start_min,
                    end_min=seg.end_min,
                    isf=max(1e-6, seg.isf * (1.0 + delta.isf_delta)),
                    cr=max(1e-6, seg.cr * (1.0 + delta.cr_delta)),
                    basal=max(1e-6, seg.basal * (1.0 + delta.basal_delta)),
                )
            )
        return TherapySchedule(segments=new_segments, tz_identifier=self.tz_identifier)

    def apply_structural_proposal(self, proposal: StructureEdit | dict) -> "TherapySchedule":
        edit = _coerce_edit(proposal)
        idx = next((i for i, seg in enumerate(self.segments) if seg.segment_id == edit.target_segment_id), None)
        if idx is None:
            raise ValueError(f"unknown target segment `{edit.target_segment_id}`")

        segs = list(self.segments)
        target = segs[idx]

        if edit.edit_type in {"split", "add"}:
            split_at = edit.split_at_minute
            if split_at is None or not (target.start_min < split_at < target.end_min):
                raise ValueError("split edit requires split_at_minute inside target segment")
            left = TherapySegment(
                segment_id=f"{target.segment_id}_a",
                start_min=target.start_min,
                end_min=split_at,
                isf=target.isf,
                cr=target.cr,
                basal=target.basal,
            )
            right = TherapySegment(
                segment_id=f"{target.segment_id}_b",
                start_min=split_at,
                end_min=target.end_min,
                isf=target.isf,
                cr=target.cr,
                basal=target.basal,
            )
            segs = segs[:idx] + [left, right] + segs[idx + 1 :]
        elif edit.edit_type == "merge":
            neighbor_id = edit.neighbor_segment_id
            if neighbor_id is None:
                if idx + 1 >= len(segs):
                    raise ValueError("merge edit requires neighbor_segment_id or adjacent segment")
                neighbor_idx = idx + 1
            else:
                neighbor_idx = next((i for i, seg in enumerate(segs) if seg.segment_id == neighbor_id), None)
                if neighbor_idx is None:
                    raise ValueError(f"unknown neighbor segment `{neighbor_id}`")
            a = segs[idx]
            b = segs[neighbor_idx]
            first, second = (a, b) if a.start_min <= b.start_min else (b, a)
            if first.end_min != second.start_min:
                raise ValueError("merge edit requires adjacent segments")
            merged = TherapySegment(
                segment_id=f"{first.segment_id}__{second.segment_id}",
                start_min=first.start_min,
                end_min=second.end_min,
                isf=(first.isf + second.isf) / 2.0,
                cr=(first.cr + second.cr) / 2.0,
                basal=(first.basal + second.basal) / 2.0,
            )
            keep = [seg for seg in segs if seg.segment_id not in {first.segment_id, second.segment_id}]
            segs = sorted(keep + [merged], key=lambda seg: seg.start_min)
        elif edit.edit_type == "remove":
            segs = [seg for seg in segs if seg.segment_id != edit.target_segment_id]
        else:
            raise ValueError(f"unsupported structure edit `{edit.edit_type}`")

        return TherapySchedule(segments=segs, tz_identifier=self.tz_identifier)

    def is_valid(self, min_duration_min: int = 120) -> bool:
        if not self.segments:
            return False
        if self.segments[0].start_min != 0:
            return False
        if self.segments[-1].end_min != 1440:
            return False

        total = 0
        prev_end = 0
        seen_ids: set[str] = set()
        for seg in self.segments:
            if seg.segment_id in seen_ids:
                return False
            seen_ids.add(seg.segment_id)
            if seg.start_min != prev_end:
                return False
            width = seg.end_min - seg.start_min
            if width < min_duration_min:
                return False
            if seg.isf <= 0 or seg.cr <= 0 or seg.basal <= 0:
                return False
            total += width
            prev_end = seg.end_min
        return total == 1440


def make_default_schedule(config: "PatientConfig") -> TherapySchedule:
    """Build a simple 4-block schedule from patient-level therapy multipliers."""

    base_isf = 45.0 / max(config.isf_multiplier, 1e-6)
    base_cr = 12.0 * config.cr_multiplier
    base_basal = 0.85 * config.basal_multiplier

    segments = [
        TherapySegment("overnight", 0, 360, base_isf * 1.04, base_cr * 1.03, base_basal * 0.94),
        TherapySegment("morning", 360, 720, base_isf * 0.93, base_cr * 0.92, base_basal * 1.08),
        TherapySegment("afternoon", 720, 1080, base_isf, base_cr, base_basal),
        TherapySegment("evening", 1080, 1440, base_isf * 0.98, base_cr * 1.02, base_basal * 0.96),
    ]
    return TherapySchedule(segments=segments, tz_identifier="UTC")
