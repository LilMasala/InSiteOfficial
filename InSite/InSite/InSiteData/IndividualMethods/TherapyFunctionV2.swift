//
//  TherapyFunctionV2.swift
//  InSite
//
//  Created by Anand Parikh on 10/6/25.
//
import Foundation


struct TherapyFunctionV2: Codable, Equatable {
    var version: Int = 2
    var tzIdentifier: String            // e.g., "America/Detroit"
    var resolutionMin: Int              // e.g., 30
    var knots: [Knot]                   // sorted; first.offsetMin == 0

    struct Knot: Codable, Equatable {
        var offsetMin: Int              // 0…1439 (minutes since local midnight)
        var basalRate: Double
        var insulinSensitivity: Double
        var carbRatio: Double
    }
}

extension TherapyFunctionV2 {
    func value(at date: Date) -> (basal: Double, isf: Double, cr: Double) {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: tzIdentifier) ?? .current
        let sOD = cal.startOfDay(for: date)
        let m = cal.dateComponents([.minute], from: sOD, to: date).minute ?? 0
        // binary search last knot with offset ≤ m
        let i = max(0, (knots.lastIndex(where: { $0.offsetMin <= m }) ?? 0))
        let k = knots[i]
        return (k.basalRate, k.insulinSensitivity, k.carbRatio)
    }

    /// Returns the active segment bounds (as concrete Dates) and its knot.
    func activeSegment(at date: Date) -> (start: Date, end: Date, knot: Knot) {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: tzIdentifier) ?? .current
        let sod = cal.startOfDay(for: date)
        let m = cal.dateComponents([.minute], from: sod, to: date).minute ?? 0

        let idx = max(0, (knots.lastIndex(where: { $0.offsetMin <= m }) ?? 0))
        let k = knots[idx]
        let next = (idx + 1 < knots.count) ? knots[idx + 1].offsetMin : 1440

        let start = cal.date(byAdding: .minute, value: k.offsetMin, to: sod)!
        let end   = cal.date(byAdding: .minute, value: next,       to: sod)!
        return (start, end, k)
    }
}


// Build V2 from legacy hours (fallback path)
/// Build a step schedule from legacy hour ranges (inclusive endHour, supports wrap).
func makeV2(from hours: [HourRange], tz: TimeZone) -> TherapyFunctionV2 {
    guard !hours.isEmpty else {
        // Single all-day default so value(at:) never zeroes out
        let k = TherapyFunctionV2.Knot(offsetMin: 0, basalRate: 0.8, insulinSensitivity: 45, carbRatio: 10)
        return TherapyFunctionV2(version: 2, tzIdentifier: tz.identifier, resolutionMin: 30, knots: [k])
    }

    // Break ranges (which are hour-based, inclusive end) into [startMin, endMin) half-open blocks.
    // If a range wraps (e.g., 22–3), split into [22:00, 24:00) and [00:00, 04:00).
    struct Block { let start: Int; let end: Int; let b: Double; let isf: Double; let cr: Double }

    var blocks: [Block] = []
    for r in hours {
        let s = max(0, min(23, r.startHour)) * 60
        let eInc = max(0, min(23, r.endHour)) * 60 + 59  // inclusive to last minute of that hour
        let e = min(1440, (eInc + 1))                    // convert to half-open end minute

        if s < e {
            blocks.append(.init(start: s, end: e, b: r.basalRate, isf: r.insulinSensitivity, cr: r.carbRatio))
        } else if s > e { // wrap
            blocks.append(.init(start: s, end: 1440, b: r.basalRate, isf: r.insulinSensitivity, cr: r.carbRatio))
            blocks.append(.init(start: 0, end: e, b: r.basalRate, isf: r.insulinSensitivity, cr: r.carbRatio))
        } else {
            // s == e means full-day; cover everything
            blocks.append(.init(start: 0, end: 1440, b: r.basalRate, isf: r.insulinSensitivity, cr: r.carbRatio))
        }
    }

    // Build knots at every distinct block.start, keep last-writer-wins if overlaps
    let starts = Set(blocks.map { $0.start }).union([0])  // ensure 0 exists
    var knots: [TherapyFunctionV2.Knot] = []
    for kStart in starts.sorted() {
        // choose the block that actually covers kStart (or the latest that begins at kStart)
        if let blk = blocks.last(where: { $0.start == kStart }) ?? blocks.last(where: { $0.start <= kStart && kStart < $0.end }) {
            knots.append(.init(offsetMin: kStart,
                               basalRate: blk.b,
                               insulinSensitivity: blk.isf,
                               carbRatio: blk.cr))
        }
    }

    // Compact consecutive duplicates
    var compact: [TherapyFunctionV2.Knot] = []
    for k in knots.sorted(by: { $0.offsetMin < $1.offsetMin }) {
        if let last = compact.last,
           last.basalRate == k.basalRate,
           last.insulinSensitivity == k.insulinSensitivity,
           last.carbRatio == k.carbRatio { continue }
        compact.append(k)
    }
    if compact.isEmpty {
        compact = [TherapyFunctionV2.Knot(offsetMin: 0, basalRate: 0.8, insulinSensitivity: 45, carbRatio: 10)]
    }

    return TherapyFunctionV2(version: 2, tzIdentifier: tz.identifier, resolutionMin: 30, knots: compact)
}


// Optional: coarsen V2 to HourRange[] (for old UIs that still need it)
func coarsenToHours(_ v2: TherapyFunctionV2) -> [HourRange] {
    var out: [HourRange] = []
    // sample each hour start and merge adjacent equal settings
    var cur: HourRange?
    for h in 0..<24 {
        let minute = h * 60
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: v2.tzIdentifier) ?? .current
        let now = Date()
        let sod = cal.startOfDay(for: now)
        let dt = cal.date(byAdding: .minute, value: minute, to: sod)!
        let (b, isf, cr) = v2.value(at: dt)
        let r = HourRange(startHour: h, endHour: h, carbRatio: cr, basalRate: b, insulinSensitivity: isf)
        if var c = cur {
            if c.basalRate == r.basalRate && c.carbRatio == r.carbRatio && c.insulinSensitivity == r.insulinSensitivity {
                c.endHour = r.endHour
                cur = c
            } else {
                out.append(c)
                cur = r
            }
        } else {
            cur = r
        }
    }
    if let c = cur { out.append(c) }
    return out
}
