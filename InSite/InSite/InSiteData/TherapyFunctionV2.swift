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

extension TherapyFunc tionV2 {
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
