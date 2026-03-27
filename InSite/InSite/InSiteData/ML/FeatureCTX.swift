//
//  FeatureCTX.swift
//  InSite
//
//  Created by Anand Parikh on 10/8/25.
//

import Foundation
import SwiftUI

public struct SiteCTX: Sendable {
    public let hourStartUtc: Date
    public let daysSinceSiteChange: Int
    public let currentSiteLocation: String
    public let lastSiteLocation: String?
    public let isSiteRepeatLocation: Int
}

public struct MenstrualCTX: Sendable {
    public let hourStartUtc: Date
    public let daysSincePeriodStart: Int        // -1 if unknown
    public let cyclePhaseOneHot: (follicular: Int, ovulation: Int, luteal: Int)
}


// MARK: - HR + BG CTX

public struct HRCTX: Sendable {
    public let hourStartUtc: Date
    public let hrMean: Double?
    public let hrRobustMean: Double?
    public let deltaFrom7dSameHour: Double?
    public let zscore7dSameHour: Double?
    public let restingHrDaily: Double?
}

public struct BGCTX: Sendable {
    public let hourStartUtc: Date
    public let startBg: Double?
    public let endBg: Double?
    public let avgBg: Double?
    public let robustAvgBg: Double?
    public let percentLow: Double?     // 0–100
    public let percentHigh: Double?    // 0–100
    public let tir: Double?            // 0–100
    public let uRoc: Double?           // mg/dL/s (unexpected ROC)
    public let expectedEndBg: Double?
    public let deltaAvgFrom7dSameHour: Double?
    public let zscoreAvg7dSameHour: Double?
}

public struct EnergyCTX: Sendable {
    public let hourStartUtc: Date

    // raw
    public let basalKcal: Double?
    public let activeKcal: Double?
    public var totalKcal: Double? {
        if let b = basalKcal, let a = activeKcal { return b + a }
        if let b = basalKcal { return b }
        if let a = activeKcal { return a }
        return nil
    }

    // short windows
    public let activeKcalLast3h: Double?   // sum of previous 3 *completed* hours
    public let activeKcalLast6h: Double?
    public let totalKcalLast6h: Double?

    // circadian baseline vs same-hour history
    public let deltaActiveFrom7dSameHour: Double?
    public let zscoreActive7dSameHour: Double?
}



public struct SleepCTX: Sendable {
    public let hourStartUtc: Date

    // previous night's totals (minutes)
    public let prevTotalMin: Double?
    public let prevRemMin: Double?
    public let prevDeepMin: Double?
    public let prevCoreMin: Double?
    public let prevAwakeMin: Double?

    // debt/quality
    public let sleepDebt7dMin: Double?     // (target*7 - sum last 7 nights), clamp at 0
    public let sleepScore: Double?         // if you add a score later; nil for now

    // timing for current hour
    public let minutesSinceWake: Int?      // needs episode boundaries; nil until you adopt them
    public let isAsleepThisHour: Int?      // 1/0 if hour intersects an asleep state; nil until available
}


// Exercise context for ML
public struct ExerciseCTX: Sendable {
    public let hourStartUtc: Date

    // raw hourly (from HealthKit stats)
    public let moveMinutes: Double?          // Apple “Move Time” minutes this hour
    public let exerciseMinutes: Double?      // Apple “Exercise Time” minutes this hour
    public var totalMinutes: Double? {       // convenience
        guard let m = moveMinutes, let e = exerciseMinutes else { return nil }
        return m + e
    }

    // rolling baselines (same-hour-of-day over past 7 occurrences)
    public let deltaMoveFrom7dSameHour: Double?
    public let zMove7dSameHour: Double?
    public let deltaExerciseFrom7dSameHour: Double?
    public let zExercise7dSameHour: Double?

    // short-horizon activity features (derived)
    public let minutesInLast3h: Double?      // total move+exercise in last 3h
    public let vigorousInLast3h: Int?        // if you later classify vigorous bouts, else leave nil
    public let hoursSinceExercise: Double?   // since last (any) recorded >0 min hour
}

// Body mass context for ML
public struct BodyMassCTX: Sendable {
    public let hourStartUtc: Date
    public let weightKg: Double?             // hourly discrete average (kg)
    public let deltaFrom7dSameHour: Double?
    public let zscore7dSameHour: Double?
}


public struct MoodCTX: Sendable {
    public let hourStartUtc: Date
    public let valence: Double?          // last observed at/ before hour
    public let arousal: Double?
    public let quad_posPos: Int?         // (+valence, +arousal)
    public let quad_posNeg: Int?         // (+valence, -arousal)
    public let quad_negPos: Int?         // (-valence, +arousal)
    public let quad_negNeg: Int?         // (-valence, -arousal)
    public let hoursSinceMood: Double?   // since last event (h)
}






