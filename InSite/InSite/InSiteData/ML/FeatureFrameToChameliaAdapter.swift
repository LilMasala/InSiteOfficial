import Foundation

public enum ChameliaSignalValue: Sendable, Equatable {
    case double(Double)
    case int(Int)
    case string(String)

    var objectValue: Any {
        switch self {
        case .double(let value):
            return value
        case .int(let value):
            return value
        case .string(let value):
            return value
        }
    }
}

public struct ChameliaSignalBlob: Sendable, Equatable {
    public let hourStartUtc: Date
    public let signals: [String: ChameliaSignalValue]

    public var numericSignals: [String: Double] {
        signals.reduce(into: [String: Double]()) { partialResult, entry in
            guard case .double(let value) = entry.value else { return }
            partialResult[entry.key] = value
        }
    }

    public var signalObject: [String: Any] {
        signals.reduce(into: [String: Any]()) { partialResult, entry in
            partialResult[entry.key] = entry.value.objectValue
        }
    }

    public var payloadObject: [String: Any] {
        [
            "timestamp": hourStartUtc.timeIntervalSince1970,
            "signals": signalObject
        ]
    }
}

public enum FeatureFrameToChameliaAdapter {
    public static func makeSignalBlob(from frame: FeatureFrameHourly) -> ChameliaSignalBlob {
        var signals: [String: ChameliaSignalValue] = [:]
        put("day_of_week", Double(dayOfWeek(for: frame.hourStartUtc)), into: &signals)

        put("bg_avg", frame.bg_avg, into: &signals)
        putPercent("tir_7d", frame.bg_tir, into: &signals)
        putPercent("pct_low_7d", frame.bg_percentLow, into: &signals)
        putPercent("pct_high_7d", frame.bg_percentHigh, into: &signals)
        put("uroc", frame.bg_uRoc, into: &signals)
        put("bg_delta_7h", frame.bg_deltaAvg7h, into: &signals)
        put("bg_z_7h", frame.bg_zAvg7h, into: &signals)

        put("heart_rate", frame.hr_mean, into: &signals)
        put("hr_delta_7h", frame.hr_delta7h, into: &signals)
        put("hr_z_7h", frame.hr_z7h, into: &signals)
        put("resting_hr", frame.rhr_daily, into: &signals)

        put("active_kcal", frame.kcal_active, into: &signals)
        put("kcal_last3h", frame.kcal_active_last3h, into: &signals)
        put("kcal_last6h", frame.kcal_active_last6h, into: &signals)
        put("active_kcal_delta7h", frame.kcal_active_delta7h, into: &signals)
        put("active_kcal_z7h", frame.kcal_active_z7h, into: &signals)

        put("sleep_total_min", frame.sleep_prev_total_min, into: &signals)
        put("sleep_debt_7d", frame.sleep_debt_7d_min, into: &signals)
        put("mins_since_wake", frame.minutes_since_wake, into: &signals)

        put("move_mins", frame.ex_move_min, into: &signals)
        put("exercise_mins", frame.ex_exercise_min, into: &signals)
        put("exercise_last3h", frame.ex_min_last3h, into: &signals)
        put("hours_since_exercise", frame.ex_hours_since, into: &signals)

        put("cycle_day", frame.days_since_period_start, into: &signals)
        put("cycle_phase_follicular", frame.cycle_follicular, into: &signals)
        put("cycle_phase_luteal", frame.cycle_luteal, into: &signals)
        put("cycle_phase_ovulation", frame.cycle_ovulation, into: &signals)
        put("cycle_phase_menstrual", menstrualPhaseFlag(for: frame.days_since_period_start), into: &signals)

        put("days_since_change", frame.days_since_site_change, into: &signals)
        put("site_location", frame.site_loc_current, into: &signals)
        put("site_repeat", frame.site_loc_same_as_last, into: &signals)

        put("valence", frame.mood_valence, into: &signals)
        put("arousal", frame.mood_arousal, into: &signals)
        put("quad_pos_pos", frame.mood_quad_posPos, into: &signals)
        put("quad_pos_neg", frame.mood_quad_posNeg, into: &signals)
        put("quad_neg_pos", frame.mood_quad_negPos, into: &signals)
        put("quad_neg_neg", frame.mood_quad_negNeg, into: &signals)
        put("hours_since_mood", frame.mood_hours_since, into: &signals)
        put("stress_acute", stressAcute(valence: frame.mood_valence, arousal: frame.mood_arousal), into: &signals)

        put("iob", frame.insulin_iob, into: &signals)
        put("cob", frame.insulin_cob, into: &signals)
        put("recent_bolus_count", frame.insulin_recent_bolus_count, into: &signals)
        put("recent_carb_count", frame.insulin_recent_carb_count, into: &signals)
        put("recent_temp_basal_count", frame.insulin_recent_temp_basal_count, into: &signals)

        return ChameliaSignalBlob(hourStartUtc: frame.hourStartUtc, signals: signals)
    }

    private static func put(_ key: String, _ value: Double?, into signals: inout [String: ChameliaSignalValue]) {
        guard let value else { return }
        signals[key] = .double(value)
    }

    private static func put(_ key: String, _ value: Int?, into signals: inout [String: ChameliaSignalValue]) {
        guard let value else { return }
        signals[key] = .int(value)
    }

    private static func put(_ key: String, _ value: String?, into signals: inout [String: ChameliaSignalValue]) {
        guard let value, !value.isEmpty else { return }
        signals[key] = .string(value)
    }

    private static func putPercent(_ key: String, _ value: Double?, into signals: inout [String: ChameliaSignalValue]) {
        guard let value else { return }
        let normalized = max(0.0, min(1.0, value / 100.0))
        signals[key] = .double(normalized)
    }

    private static func menstrualPhaseFlag(for cycleDay: Int?) -> Int? {
        guard let cycleDay, cycleDay >= 0 else { return nil }
        // The frame schema tracks day-since-period plus non-menstrual one-hots only.
        // Treat the first five cycle days as the menstrual window until the upstream
        // feature frame carries an explicit menstrual-phase flag.
        return (0...4).contains(cycleDay) ? 1 : 0
    }

    private static func stressAcute(valence: Double?, arousal: Double?) -> Double? {
        guard let arousal else { return nil }
        guard arousal > 0.6 else { return 0.0 }
        if let valence, valence < 0 {
            return 1.0
        }
        return 0.5
    }

    private static func dayOfWeek(for date: Date) -> Int {
        Calendar.current.component(.weekday, from: date) - 1
    }
}
