//
//  CTXBuilder.swift
//  InSite
//

import Foundation

// MARK: - Shared UTC helpers

private let utcCal: Calendar = {
    var c = Calendar(identifier: .gregorian)
    c.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
    return c
}()

private extension TimeZone {
    static var gmt: TimeZone { TimeZone(secondsFromGMT: 0)! }
}

private func floorToUtcHour(_ d: Date) -> Date {
    let comps = utcCal.dateComponents([.year, .month, .day, .hour], from: d)
    return utcCal.date(from: comps) ?? d
}

private func sameHourIndex(_ d: Date) -> Int { utcCal.component(.hour, from: d) }

/// Rolling "same-hour-of-day" stats (for each timestamp, uses up to `lookbackCount` *prior* observations at the same UTC hour)
private func rollingSameHourStats(
    values: [Date: Double],
    lookbackCount: Int = 7
) -> [Date: (mean: Double, std: Double)] {
    var buckets: [[(Date, Double)]] = Array(repeating: [], count: 24)
    for (k, v) in values { buckets[sameHourIndex(k)].append((k, v)) }
    for i in 0..<24 { buckets[i].sort { $0.0 < $1.0 } }

    var out: [Date: (mean: Double, std: Double)] = [:]

    for bucket in buckets where !bucket.isEmpty {
        for i in 0..<bucket.count {
            let (t, _) = bucket[i]
            let start = max(0, i - lookbackCount)
            guard start < i else { continue } // need at least one prior value

            let window = bucket[start..<i].map { $0.1 }
            guard !window.isEmpty else { continue }

            let mean = window.reduce(0, +) / Double(window.count)
            let varSum = window.reduce(0) { $0 + pow($1 - mean, 2) }
            let std = window.count > 1 ? sqrt(varSum / Double(window.count - 1)) : 0.0
            out[t] = (mean, std)
        }
    }
    return out
}

private func trimmedMean(_ xs: [Double], trim: Double = 0.1) -> Double? {
    guard !xs.isEmpty else { return nil }
    let n = xs.count
    let k = Int((trim * Double(n)).rounded(.down))
    let lo = k
    let hi = max(lo, n - k)
    guard lo < hi else { return nil }
    let slice = xs[lo..<hi]
    guard !slice.isEmpty else { return nil }
    return slice.reduce(0, +) / Double(slice.count)
}

private func clamp(_ x: Double, _ lo: Double, _ hi: Double) -> Double { max(lo, min(hi, x)) }

// MARK: - Menstrual

func buildMenstrualCtxByHour(
    daily: [Date: DailyMenstrualData],
    startUtc: Date,
    endUtc: Date,
    assumedCycleLengthDays: Int = 28
) -> [Date: MenstrualCTX] {

    func phaseOneHot(daysSince: Int) -> (Int, Int, Int) {
        guard daysSince >= 0 else { return (1, 0, 0) } // unknown → default follicular
        let d = ((daysSince % assumedCycleLengthDays) + assumedCycleLengthDays) % assumedCycleLengthDays
        if (14...16).contains(d) { return (0, 1, 0) }   // ovulation (coarse)
        if d >= 17 { return (0, 0, 1) }                 // luteal
        return (1, 0, 0)                                // follicular
    }

    var byDay: [Date: Int] = [:]
    for (k, v) in daily {
        byDay[utcCal.startOfDay(for: k)] = v.daysSincePeriodStart
    }

    var out: [Date: MenstrualCTX] = [:]
    var cur = floorToUtcHour(startUtc)
    let stop = floorToUtcHour(endUtc)

    while cur <= stop && cur < endUtc {
        let ds = byDay[utcCal.startOfDay(for: cur)] ?? -1
        out[cur] = MenstrualCTX(
            hourStartUtc: cur,
            daysSincePeriodStart: ds,
            cyclePhaseOneHot: phaseOneHot(daysSince: ds)
        )
        guard let next = utcCal.date(byAdding: .hour, value: 1, to: cur) else { break }
        cur = next
    }
    return out
}

// MARK: - HR

func buildHRCTXByHour(
    hourlyHR: [Date: HourlyHeartRateData],
    restingDaily: [DailyRestingHeartRateData]
) -> [Date: HRCTX] {

    var hrByHour: [Date: Double] = [:]
    for (_, e) in hourlyHR {
        hrByHour[floorToUtcHour(e.hour)] = e.heartRate
    }

    let stats = rollingSameHourStats(values: hrByHour, lookbackCount: 7)

    var rhrByDay: [Date: Double] = [:]
    for e in restingDaily {
        rhrByDay[utcCal.startOfDay(for: e.date)] = e.restingHeartRate
    }

    var out: [Date: HRCTX] = [:]
    let hours = hrByHour.keys.sorted()

    for t in hours {
        guard let mean = hrByHour[t] else { continue }

        let base = stats[t]
        let delta: Double? = base.map { mean - $0.mean }
        let z: Double? = base.flatMap { b in b.std > 0 ? (mean - b.mean) / b.std : 0.0 }

        let rhr = rhrByDay[utcCal.startOfDay(for: t)]

        out[t] = HRCTX(
            hourStartUtc: t,
            hrMean: mean,
            hrRobustMean: mean,  // swap with robust estimator if/when you add raw-sample processing
            deltaFrom7dSameHour: delta,
            zscore7dSameHour: z,
            restingHrDaily: rhr
        )
    }
    return out
}

// MARK: - BG

func buildBGCTXByHour(
    hourly: [HourlyBgData],
    avg: [HourlyAvgBgData],
    pct: [HourlyBgPercentages],
    uroc: [HourlyBgURoc],
    hourValues: [HourlyBgValues]? = nil
) -> [Date: BGCTX] {

    var startEndByHour: [Date: (Double?, Double?)] = [:]
    for e in hourly {
        startEndByHour[floorToUtcHour(e.startDate)] = (e.startBg, e.endBg)
    }

    var avgByHour: [Date: Double] = [:]
    for e in avg {
        if let v = e.averageBg {
            avgByHour[floorToUtcHour(e.startDate)] = v
        }
    }

    var pctByHour: [Date: (Double, Double)] = [:]
    for e in pct {
        pctByHour[floorToUtcHour(e.startDate)] = (e.percentLow, e.percentHigh)
    }

    var urocByHour: [Date: (Double?, Double?)] = [:] // (uROC, expectedEndBg)
    for e in uroc {
        urocByHour[floorToUtcHour(e.startDate)] = (e.uRoc, e.expectedEndBg)
    }

    var robustAvgByHour: [Date: Double] = [:]
    if let hourValues {
        for e in hourValues {
            let k = floorToUtcHour(e.startDate)
            guard !e.values.isEmpty else { continue }
            if let tm = trimmedMean(e.values.sorted(), trim: 0.1) { robustAvgByHour[k] = tm }
        }
    }

    let avgStats = rollingSameHourStats(values: avgByHour, lookbackCount: 7)

    var keySet = Set<Date>()
    keySet.formUnion(startEndByHour.keys)
    keySet.formUnion(avgByHour.keys)
    keySet.formUnion(pctByHour.keys)
    keySet.formUnion(urocByHour.keys)
    keySet.formUnion(robustAvgByHour.keys)
    let keys = keySet.sorted()

    var out: [Date: BGCTX] = [:]

    for t in keys {
        let (startBg, endBg) = startEndByHour[t] ?? (nil, nil)
        let avgVal = avgByHour[t]
        let robust = robustAvgByHour[t]

        let percents = pctByHour[t] ?? (Double.nan, Double.nan)
        let lowOpt  = percents.0.isNaN ? nil : percents.0
        let highOpt = percents.1.isNaN ? nil : percents.1
        let tirOpt: Double? = {
            guard let low = lowOpt, let high = highOpt else { return nil }
            return clamp(100.0 - low - high, 0.0, 100.0)
        }()

        let (u, expEnd) = urocByHour[t] ?? (nil, nil)

        let base = avgStats[t]
        let deltaAvg = base.flatMap { b in avgVal.map { $0 - b.mean } }
        let zAvg: Double? = {
            guard let b = base, b.std > 0, let a = avgVal else { return nil }
            return (a - b.mean) / b.std
        }()

        out[t] = BGCTX(
            hourStartUtc: t,
            startBg: startBg,
            endBg: endBg,
            avgBg: avgVal,
            robustAvgBg: robust,
            percentLow: lowOpt,
            percentHigh: highOpt,
            tir: tirOpt,
            uRoc: u,
            expectedEndBg: expEnd,
            deltaAvgFrom7dSameHour: deltaAvg,
            zscoreAvg7dSameHour: zAvg
        )
    }
    return out
}

// MARK: - Energy

func buildEnergyCTXByHour(
    hourly: [Date: HourlyEnergyData],
    lookbackSameHourCount: Int = 7
) -> [Date: EnergyCTX] {

    var hourlyActive: [Date: Double] = [:]
    var hourlyBasal:  [Date: Double] = [:]
    for e in hourly.values {
        let k = floorToUtcHour(e.hour)
        hourlyActive[k] = e.activeEnergy
        hourlyBasal[k]  = e.basalEnergy
    }

    let sameHourStats = rollingSameHourStats(values: hourlyActive, lookbackCount: lookbackSameHourCount)

    let keys = Array(Set(hourlyActive.keys).union(hourlyBasal.keys)).sorted()
    var out: [Date: EnergyCTX] = [:]

    for (i, t) in keys.enumerated() {
        let a = hourlyActive[t]
        let b = hourlyBasal[t]

        func sumActive(lastHours: Int) -> Double? {
            guard lastHours > 0 else { return 0 }
            let lo = max(0, i - lastHours)
            let slice = keys[lo..<i].compactMap { hourlyActive[$0] }
            return slice.isEmpty ? nil : slice.reduce(0, +)
        }
        func sumTotal(lastHours: Int) -> Double? {
            guard lastHours > 0 else { return 0 }
            let lo = max(0, i - lastHours)
            let slice = keys[lo..<i].compactMap {
                let aa = hourlyActive[$0]; let bb = hourlyBasal[$0]
                switch (aa, bb) {
                case let (x?, y?): return x + y
                case let (x?, nil): return x
                case let (nil, y?): return y
                default: return nil
                }
            }
            return slice.isEmpty ? nil : slice.reduce(0, +)
        }

        let base = sameHourStats[t]
        let delta: Double? = {
            guard let a = a, let b0 = base?.mean else { return nil }
            return a - b0
        }()
        let z: Double? = {
            guard let a = a, let mean = base?.mean, let std = base?.std, std > 0 else { return nil }
            return (a - mean) / std
        }()

        out[t] = EnergyCTX(
            hourStartUtc: t,
            basalKcal: b,
            activeKcal: a,
            activeKcalLast3h: sumActive(lastHours: 3),
            activeKcalLast6h: sumActive(lastHours: 6),
            totalKcalLast6h:  sumTotal(lastHours: 6),
            deltaActiveFrom7dSameHour: delta,
            zscoreActive7dSameHour: z
        )
    }
    return out
}

// MARK: - Sleep

func buildSleepCTXByHour(
    hourlySpan: ClosedRange<Date>,                  // [startUtc ... endUtc]
    daily: [Date: DailySleepDurations],             // from fetchSleepDurations
    mainWindows: [Date: MainSleepWindow]? = nil,    // optional (when available)
    targetSleepMinPerNight: Double = 7.5 * 60.0
) -> [Date: SleepCTX] {

    var dailyMap: [Date: DailySleepDurations] = [:]
    for (k, v) in daily {
        dailyMap[utcCal.startOfDay(for: k)] = v
    }

    let days = dailyMap.keys.sorted()
    var rolling7dTotal: [Date: Double] = [:]
    for i in 0..<days.count {
        let d = days[i]
        let lo = max(0, i - 6)
        // total per day is the sum of all asleep buckets (your struct already has them)
        let sum = days[lo...i].compactMap { dailyMap[$0] }.map {
            $0.asleepCore + $0.asleepDeep + $0.asleepREM + $0.asleepUnspecified
        }.reduce(0, +)
        rolling7dTotal[d] = sum
    }

    var out: [Date: SleepCTX] = [:]
    var t = floorToUtcHour(hourlySpan.lowerBound)
    let stop = floorToUtcHour(hourlySpan.upperBound)

    while t <= stop {
        let day = utcCal.startOfDay(for: t)
        let prevDay = utcCal.date(byAdding: .day, value: -1, to: day)

        let prev = prevDay.flatMap { dailyMap[$0] }
        let got7d = prevDay.flatMap { rolling7dTotal[$0] }
        let need7d = 7.0 * targetSleepMinPerNight
        let debt = got7d.map { max(0, need7d - $0) }

        var minutesSinceWake: Int? = nil
        if let w = mainWindows?[day], t >= w.end {
            minutesSinceWake = Int(t.timeIntervalSince(w.end) / 60.0)
        }

        out[t] = SleepCTX(
            hourStartUtc: t,
            prevTotalMin: prev.map { $0.asleepCore + $0.asleepDeep + $0.asleepREM + $0.asleepUnspecified },
            prevRemMin: prev?.asleepREM,
            prevDeepMin: prev?.asleepDeep,
            prevCoreMin: prev?.asleepCore,
            prevAwakeMin: prev?.awake,
            sleepDebt7dMin: debt,
            sleepScore: nil,
            minutesSinceWake: minutesSinceWake,
            isAsleepThisHour: nil
        )

        guard let next = utcCal.date(byAdding: .hour, value: 1, to: t) else { break }
        t = next
    }

    return out
}

// MARK: - Exercise

func buildExerciseCTXByHour(
    hourly: [Date: HourlyExerciseData],
    lookbackSameHourCount: Int = 7
) -> [Date: ExerciseCTX] {

    var moveByHour: [Date: Double] = [:]
    var exerciseByHour: [Date: Double] = [:]
    for e in hourly.values {
        let k = floorToUtcHour(e.hour)
        if e.moveMinutes > 0 { moveByHour[k] = (moveByHour[k] ?? 0) + e.moveMinutes }
        if e.exerciseMinutes > 0 { exerciseByHour[k] = (exerciseByHour[k] ?? 0) + e.exerciseMinutes }
    }

    let moveStats     = rollingSameHourStats(values: moveByHour,     lookbackCount: lookbackSameHourCount)
    let exerciseStats = rollingSameHourStats(values: exerciseByHour, lookbackCount: lookbackSameHourCount)

    let hours = Array(Set(moveByHour.keys).union(exerciseByHour.keys)).sorted()
    var out: [Date: ExerciseCTX] = [:]

    var rolling3h: [(t: Date, totalMin: Double)] = []
    var lastExerciseHour: Date? = nil

    for t in hours {
        let move = moveByHour[t]
        let exer = exerciseByHour[t]
        let total = (move ?? 0) + (exer ?? 0)

        if total > 0 { lastExerciseHour = t }

        // maintain ≤3h window (keys are hourly)
        if let threeHoursAgo = utcCal.date(byAdding: .hour, value: -2, to: t) {
            rolling3h.append((t, total))
            rolling3h.removeAll { $0.t < threeHoursAgo }
        }
        let minutesInLast3h = rolling3h.reduce(0.0) { $0 + $1.totalMin }

        let dMove: Double? = {
            guard let m = move, let b = moveStats[t] else { return nil }
            return m - b.mean
        }()
        let zMove: Double? = {
            guard let m = move, let b = moveStats[t], b.std > 0 else { return nil }
            return (m - b.mean) / b.std
        }()

        let dExer: Double? = {
            guard let x = exer, let b = exerciseStats[t] else { return nil }
            return x - b.mean
        }()
        let zExer: Double? = {
            guard let x = exer, let b = exerciseStats[t], b.std > 0 else { return nil }
            return (x - b.mean) / b.std
        }()

        let hoursSince: Double? = {
            guard let last = lastExerciseHour else { return nil }
            let dt = t.timeIntervalSince(last) / 3600.0
            return max(0, dt)
        }()

        out[t] = ExerciseCTX(
            hourStartUtc: t,
            moveMinutes: move,
            exerciseMinutes: exer,
            deltaMoveFrom7dSameHour: dMove,
            zMove7dSameHour: zMove,
            deltaExerciseFrom7dSameHour: dExer,
            zExercise7dSameHour: zExer,
            minutesInLast3h: minutesInLast3h,
            vigorousInLast3h: nil,
            hoursSinceExercise: hoursSince
        )
    }

    return out
}

// MARK: - Body Mass

func buildBodyMassCTXByHour(
    hourly: [HourlyBodyMassData],
    lookbackSameHourCount: Int = 7
) -> [Date: BodyMassCTX] {

    var kgByHour: [Date: Double] = [:]
    for e in hourly {
        kgByHour[floorToUtcHour(e.hour)] = e.weight
    }

    let stats = rollingSameHourStats(values: kgByHour, lookbackCount: lookbackSameHourCount)

    var out: [Date: BodyMassCTX] = [:]
    for t in kgByHour.keys.sorted() {
        guard let w = kgByHour[t] else { continue }

        let delta: Double? = stats[t].map { w - $0.mean }
        let z: Double? = stats[t].flatMap { b in b.std > 0 ? (w - b.mean) / b.std : 0.0 }

        out[t] = BodyMassCTX(
            hourStartUtc: t,
            weightKg: w,
            deltaFrom7dSameHour: delta,
            zscore7dSameHour: z
        )
    }
    return out
}
