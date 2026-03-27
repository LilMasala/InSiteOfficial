//
//  FeatureJoiner.swift
//  InSite
//
//  Created by Anand Parikh on 10/8/25.
//

import Foundation
// UTC hour iterator
private func eachHourUTC(from start: Date, to end: Date) -> [Date] {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(secondsFromGMT: 0)!
    func floorHour(_ d: Date) -> Date {
        let c = cal.dateComponents([.year,.month,.day,.hour], from: d)
        return cal.date(from: c)!
    }
    var out: [Date] = []
    var cur = floorHour(start)
    let stop = floorHour(end)
    while cur <= stop {
        out.append(cur)
        cur = cal.date(byAdding: .hour, value: 1, to: cur)!
    }
    return out
}

func makeFeatureFramesHourly(
    span: ClosedRange<Date>,
    bg: [Date: BGCTX],
    hr: [Date: HRCTX],
    energy: [Date: EnergyCTX],
    sleep: [Date: SleepCTX],
    exercise: [Date: ExerciseCTX],
    menstrual: [Date: MenstrualCTX] = [:],
    site: [Date: SiteCTX] = [:],
    mood: [Date: MoodCTX] = [:]            // ← NEW

) -> [FeatureFrameHourly] {

    // canonical key set (avoid heterogeneous array-of-dictionaries)
    var keySet = Set<Date>()
    keySet.formUnion(eachHourUTC(from: span.lowerBound, to: span.upperBound))
    keySet.formUnion(bg.keys)
    keySet.formUnion(hr.keys)
    keySet.formUnion(energy.keys)
    keySet.formUnion(sleep.keys)
    keySet.formUnion(exercise.keys)
    keySet.formUnion(menstrual.keys)
    keySet.formUnion(site.keys)
    keySet.formUnion(site.keys); keySet.formUnion(mood.keys)   // ← NEW


    let hours = keySet.sorted()

    var frames: [FeatureFrameHourly] = []
    frames.reserveCapacity(hours.count)

    for t in hours {
        let b  = bg[t]
        let h  = hr[t]
        let en = energy[t]
        let sl = sleep[t]
        let ex = exercise[t]
        let m  = menstrual[t]
        let si = site[t]
        let mo = mood[t]

        frames.append(FeatureFrameHourly(
            hourStartUtc: t,

            // BG
            bg_avg:                   b?.avgBg,
            bg_tir:                   b?.tir,
            bg_percentLow:            b?.percentLow,
            bg_percentHigh:           b?.percentHigh,
            bg_uRoc:                  b?.uRoc,
            bg_deltaAvg7h:            b?.deltaAvgFrom7dSameHour,
            bg_zAvg7h:                b?.zscoreAvg7dSameHour,

            // HR
            hr_mean:                  h?.hrMean,
            hr_delta7h:               h?.deltaFrom7dSameHour,
            hr_z7h:                   h?.zscore7dSameHour,
            rhr_daily:                h?.restingHrDaily,

            // Energy
            kcal_active:              en?.activeKcal,
            kcal_active_last3h:       en?.activeKcalLast3h,
            kcal_active_last6h:       en?.activeKcalLast6h,
            kcal_active_delta7h:      en?.deltaActiveFrom7dSameHour,
            kcal_active_z7h:          en?.zscoreActive7dSameHour,

            // Sleep
            sleep_prev_total_min:     sl?.prevTotalMin,
            sleep_debt_7d_min:        sl?.sleepDebt7dMin,
            minutes_since_wake:       sl?.minutesSinceWake,

            // Exercise
            ex_move_min:              ex?.moveMinutes,
            ex_exercise_min:          ex?.exerciseMinutes,
            ex_min_last3h:            ex?.minutesInLast3h,
            ex_hours_since:           ex?.hoursSinceExercise,

            // Menstrual
            days_since_period_start:  m?.daysSincePeriodStart,
            cycle_follicular:         m.map { $0.cyclePhaseOneHot.follicular },
            cycle_ovulation:          m.map { $0.cyclePhaseOneHot.ovulation },
            cycle_luteal:             m.map { $0.cyclePhaseOneHot.luteal },

            // Site
            days_since_site_change:   si?.daysSinceSiteChange,
            site_loc_current:         si?.currentSiteLocation,
            site_loc_same_as_last:    si?.isSiteRepeatLocation,
            
            mood_valence: mo?.valence,
            mood_arousal: mo?.arousal,
            mood_quad_posPos: mo?.quad_posPos,
            mood_quad_posNeg: mo?.quad_posNeg,
            mood_quad_negPos: mo?.quad_negPos,
            mood_quad_negNeg: mo?.quad_negNeg,
            mood_hours_since: mo?.hoursSinceMood
        ))
    }

    return frames
}
