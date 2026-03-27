import Foundation
import HealthKit

struct HourlyBgPercentages {
    let startDate: Date
    let endDate: Date
    let percentLow: Double
    let percentHigh: Double
}

struct HourlyBgURoc {
    let startDate: Date
    let endDate: Date
    let uRoc: Double?            // (realROC - expectedROC) in mg/dL per second
    let expectedEndBg: Double?   // model-predicted BG at end of hour (optional but useful)
}

enum BgAnalytics {
    // MARK: - private helpers
    private static func bgModel(startBG: Double, targetBG: Double)
    -> (f: (Double) -> Double, df: (Double) -> Double) {
        if startBG > targetBG {
            let A = 385.96, k = 0.017
            let f: (Double) -> Double = { t in A * exp(-k * t) + targetBG }
            let df: (Double) -> Double = { t in -k * A * exp(-k * t) }
            return (f, df)
        } else if startBG < targetBG {
            let f: (Double) -> Double = { t in targetBG / (1 + exp(-0.2 * pow(t - 43.5, 0.8))) }
            let df: (Double) -> Double = { t in
                let a = pow(t - 43.5, 0.8)
                let e = exp(-0.2 * a)
                let denom2 = pow(1 + e, 2)
                let d_a = 0.8 * pow(max(t - 43.5, 1e-9), -0.2)
                return -targetBG * (e * -0.2 * d_a) / denom2
            }
            return (f, df)
        } else {
            return ({ _ in targetBG }, { _ in 0 })
        }
    }

    private static func expectedBg(startBG: Double, targetBG: Double, deltaMinutes: Double) -> Double? {
        let (f, df) = bgModel(startBG: startBG, targetBG: targetBG)
        let g: NewtonRaphson.Function = { x in f(x) - startBG }
        let solver = NewtonRaphson(initialValue: 0.0, error: 1e-6, function: g, differential: df)
        guard let t0 = try? solver.solve().x else { return nil }
        let t1 = (t0 ?? 0.0) + deltaMinutes
        return f(t1)
    }

    // MARK: - public API
    static func computeHourlyURoc(
        hourlyBgData: [HourlyBgData],
        targetBG: Double = 110.0
    ) -> [HourlyBgURoc] {
        var out: [HourlyBgURoc] = []
        out.reserveCapacity(hourlyBgData.count)

        for h in hourlyBgData {
            guard let start = h.startBg, let end = h.endBg else {
                out.append(HourlyBgURoc(startDate: h.startDate, endDate: h.endDate, uRoc: nil, expectedEndBg: nil))
                continue
            }
            let dt = h.endDate.timeIntervalSince(h.startDate) // seconds
            guard dt > 0 else {
                out.append(HourlyBgURoc(startDate: h.startDate, endDate: h.endDate, uRoc: nil, expectedEndBg: nil))
                continue
            }

            let realROC = (end - start) / dt // mg/dL per sec
            let expectedEnd = expectedBg(startBG: start, targetBG: targetBG, deltaMinutes: dt / 60.0)
            let expectedROC = expectedEnd.map { ($0 - start) / dt }
            let uroc = expectedROC.map { realROC - $0 } ?? 0.0

            out.append(HourlyBgURoc(startDate: h.startDate, endDate: h.endDate, uRoc: uroc, expectedEndBg: expectedEnd))
        }
        return out
    }
}

struct HourlyBgData {
    let startDate: Date
    let endDate: Date
    let startBg: Double?
    let endBg: Double?
}

struct HourlyBgValues {
    let startDate: Date
    let endDate: Date
    let values: [Double]
}

struct HourlyAvgBgData {
    let startDate: Date
    let endDate: Date
    let averageBg: Double?
}

extension HealthStore {

    public func fetchBgRawValuesHourly(
        start: Date,
        end: Date,
        completion: @escaping (Result<[HourlyBgValues], Error>) -> Void
    ) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        fetchAllBgPerHour(
            start: start,
            end: end,
            healthStore: healthStore,
            bloodGlucoseType: bloodGlucoseType,
            dispatchGroup: DispatchGroup(),
            completion: completion
        )
    }

    /// Returns (1) first/last BG per hour, (2) average BG per hour, (3) % low/high per hour.
    /// NOTE: Uses the caller's dispatchGroup only to coordinate these three top-level tasks.
    public func fetchAllBgData(
        start: Date,
        end: Date,
        dispatchGroup: DispatchGroup,
        completion: @escaping (Result<([HourlyBgData], [HourlyAvgBgData], [HourlyBgPercentages]), Error>) -> Void
    ) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        var hourlyBgData: [HourlyBgData] = []
        var avgBgData: [HourlyAvgBgData] = []
        var hourlyPercentages: [HourlyBgPercentages] = []
        var firstError: Error?

        // 1) First/Last per hour
        dispatchGroup.enter()
        fetchBgAtStartandEnd(start: start, end: end, healthStore: healthStore, bloodGlucoseType: bloodGlucoseType) { result in
            //print("BG: fetchBgAtStartandEnd returned \(result)")
            switch result {
            case .success(let data): hourlyBgData = data
            case .failure(let error):
                if firstError == nil { firstError = error }
                print("Error fetching BG start/end: \(error)")
            }
            dispatchGroup.leave()
        }

        // 2) Average per hour
        dispatchGroup.enter()
        fetchAvgBg(start: start, end: end, healthStore: healthStore, bloodGlucoseType: bloodGlucoseType) { result in
            //print("BG: fetchAvgBg returned \(result)")
            switch result {
            case .success(let data): avgBgData = data
            case .failure(let error):
                if firstError == nil { firstError = error }
                print("Error fetching BG average: \(error)")
            }
            dispatchGroup.leave()
        }

        // 3) Percent low/high per hour
        dispatchGroup.enter()
        calculatePercentLowAndHigh(start: start, end: end, healthStore: healthStore, dispatchGroup: dispatchGroup, bloodGlucoseType: bloodGlucoseType) { result in
//            print("BG: calculatePercentLowAndHigh returned \(result)")

            switch result {
            case .success(let data): hourlyPercentages = data
            case .failure(let error):
                if firstError == nil { firstError = error }
                print("Error calculating BG percentages: \(error)")
            }
            dispatchGroup.leave()
        }

        dispatchGroup.notify(queue: .main) {
            print("BG: top-level notify firing")

            if let error = firstError {
                completion(.failure(error))
            } else {
                completion(.success((hourlyBgData, avgBgData, hourlyPercentages)))
            }
        }
    }

    /// True first and last BG *sample values* in each hour window.
    private func fetchBgAtStartandEnd(
        start: Date,
        end: Date,
        healthStore: HKHealthStore,
        bloodGlucoseType: HKQuantityType,
        completion: @escaping (Result<[HourlyBgData], Error>) -> Void
    ) {
        let calendar = Calendar.current
        let unit = HKUnit(from: "mg/dL")

        // Build hour bins first
        var bins: [(start: Date, end: Date)] = []
        var cursor = start
        while cursor < end {
            let next = calendar.date(byAdding: .hour, value: 1, to: cursor)!
            bins.append((start: cursor, end: next))
            cursor = next
        }

        if bins.isEmpty {
            completion(.success([]))
            return
        }

        // Results arrays aligned to bins
        var startVals = Array<Double?>(repeating: nil, count: bins.count)
        var endVals   = Array<Double?>(repeating: nil, count: bins.count)

        let group = DispatchGroup()

        func makeQuery(from: Date, to: Date, sortAscending: Bool, limit: Int, handler: @escaping ([HKQuantitySample]) -> Void) -> HKSampleQuery {
            let predicate = HKQuery.predicateForSamples(withStart: from, end: to, options: .strictStartDate)
            let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: sortAscending)
            return HKSampleQuery(sampleType: bloodGlucoseType, predicate: predicate, limit: limit, sortDescriptors: [sort]) { _, samples, _ in
                handler((samples as? [HKQuantitySample]) ?? [])
            }
        }

        for (i, bin) in bins.enumerated() {
            // first sample in bin
            group.enter()
            let q1 = makeQuery(from: bin.start, to: bin.end, sortAscending: true, limit: 1) { samples in
                startVals[i] = samples.first?.quantity.doubleValue(for: unit)
                group.leave()
            }
            healthStore.execute(q1)

            // last sample in bin
            group.enter()
            let q2 = makeQuery(from: bin.start, to: bin.end, sortAscending: false, limit: 1) { samples in
                endVals[i] = samples.first?.quantity.doubleValue(for: unit)
                group.leave()
            }
            healthStore.execute(q2)
        }

        group.notify(queue: .main) {
            let out: [HourlyBgData] = bins.enumerated().map { (i, bin) in
                HourlyBgData(startDate: bin.start, endDate: bin.end, startBg: startVals[i], endBg: endVals[i])
            }
            completion(.success(out))
        }
    }

    /// Returns *all* BG sample values for each hour window. Uses its own DispatchGroup internally.
    private func fetchAllBgPerHour(
        start: Date,
        end: Date,
        healthStore: HKHealthStore,
        bloodGlucoseType: HKQuantityType,
        dispatchGroup: DispatchGroup, // kept for API compatibility; not reused internally
        completion: @escaping (Result<[HourlyBgValues], Error>) -> Void
    ) {
        let calendar = Calendar.current
        let unit = HKUnit(from: "mg/dL")

        // Build hour bins first
        var bins: [(start: Date, end: Date)] = []
        var cursor = start
        while cursor < end {
            let next = calendar.date(byAdding: .hour, value: 1, to: cursor)!
            bins.append((start: cursor, end: next))
            cursor = next
        }

        if bins.isEmpty {
            completion(.success([]))
            return
        }

        // Collect values per index; then render in order to avoid race conditions
        var valuesByIndex: [[Double]?] = Array(repeating: nil, count: bins.count)
        let group = DispatchGroup()

        for (i, bin) in bins.enumerated() {
            group.enter()
            let predicate = HKQuery.predicateForSamples(withStart: bin.start, end: bin.end, options: .strictStartDate)
            // Sort ensures deterministic order (oldest → newest)
            let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

            let q = HKSampleQuery(sampleType: bloodGlucoseType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, samples, error in
                defer { group.leave() }
                if let error = error {
                    completion(.failure(error))
                    return
                }
                guard let samples = samples as? [HKQuantitySample] else {
                    completion(.failure(HealthStoreError.dataUnavailable("bg-hour-values")))
                    return
                }
                let vals = samples.map { $0.quantity.doubleValue(for: unit) }
                valuesByIndex[i] = vals
            }
            healthStore.execute(q)
        }

        group.notify(queue: .main) {
            // If any index is still nil (shouldn't happen), treat as empty
            let out: [HourlyBgValues] = bins.enumerated().map { (i, bin) in
                HourlyBgValues(startDate: bin.start, endDate: bin.end, values: valuesByIndex[i] ?? [])
            }
            completion(.success(out))
        }
    }

    /// Hourly average via statistics collection (fast and correct).
    private func fetchAvgBg(
        start: Date,
        end: Date,
        healthStore: HKHealthStore,
        bloodGlucoseType: HKQuantityType,
        completion: @escaping (Result<[HourlyAvgBgData], Error>) -> Void
    ) {
        var dateComponents = DateComponents()
        dateComponents.hour = 1

        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)

        let query = HKStatisticsCollectionQuery(
            quantityType: bloodGlucoseType,
            quantitySamplePredicate: predicate,
            options: [.discreteAverage],
            anchorDate: start,
            intervalComponents: dateComponents
        )

        query.initialResultsHandler = { _, result, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            guard let result = result else {
                completion(.failure(HealthStoreError.dataUnavailable("bg-avg")))
                return
            }

            var out: [HourlyAvgBgData] = []
            result.enumerateStatistics(from: start, to: end) { statistic, _ in
                let avg = statistic.averageQuantity()?.doubleValue(for: HKUnit(from: "mg/dL"))
                out.append(HourlyAvgBgData(startDate: statistic.startDate, endDate: statistic.endDate, averageBg: avg))
            }
            completion(.success(out))
        }

        healthStore.execute(query)
    }

    /// Computes % low (<80 mg/dL) and % high (>180 mg/dL) per hour using *all* samples in that hour.
    /// NOTE: Accepts a dispatchGroup param for API compatibility but manages its own group internally.
    private func calculatePercentLowAndHigh(
        start: Date,
        end: Date,
        healthStore: HKHealthStore,
        dispatchGroup: DispatchGroup, // not reused internally
        bloodGlucoseType: HKQuantityType,
        completion: @escaping (Result<[HourlyBgPercentages], Error>) -> Void
    ) {
        let lowBg = 80.0
        let highBg = 180.0

        fetchAllBgPerHour(start: start, end: end, healthStore: healthStore, bloodGlucoseType: bloodGlucoseType, dispatchGroup: dispatchGroup) { result in
            switch result {
            case .failure(let error):
                completion(.failure(error))
            case .success(let perHour):
                var out: [HourlyBgPercentages] = []
                out.reserveCapacity(perHour.count)

                for h in perHour {
                    let n = h.values.count
                    if n == 0 {
                        out.append(HourlyBgPercentages(startDate: h.startDate, endDate: h.endDate, percentLow: 0, percentHigh: 0))
                        continue
                    }
                    let lowCount = h.values.lazy.filter { $0 < lowBg }.count
                    let highCount = h.values.lazy.filter { $0 > highBg }.count
                    let percentLow = (Double(lowCount) / Double(n)) * 100.0
                    let percentHigh = (Double(highCount) / Double(n)) * 100.0
                    out.append(HourlyBgPercentages(startDate: h.startDate, endDate: h.endDate, percentLow: percentLow, percentHigh: percentHigh))
                }
                completion(.success(out))
            }
        }
    }
}
