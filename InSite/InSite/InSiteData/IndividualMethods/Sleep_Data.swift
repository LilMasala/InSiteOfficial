//
//  Sleep_Data.swift
//  InSite
//
//  Created by Anand Parikh on 2/1/24.
//

import Foundation
import HealthKit

struct DailySleepDurations {
    let date: Date
    var awake: TimeInterval = 0
    var asleepCore: TimeInterval = 0
    var asleepDeep: TimeInterval = 0
    var asleepREM: TimeInterval = 0
    var asleepUnspecified: TimeInterval = 0

    mutating func addSleepState(state: HKCategoryValueSleepAnalysis, duration: TimeInterval) {
        switch state {
        case .awake:
            awake += duration
        case .asleepCore:
            asleepCore += duration
        case .asleepDeep:
            asleepDeep += duration
        case .asleepREM:
            asleepREM += duration
        case .asleepUnspecified:
            asleepUnspecified += duration
        default:
            break
        }
    }
}
extension HealthStore {
    func fetchSleepDurations(startDate: Date, endDate: Date, completion: @escaping (Result<[Date: DailySleepDurations], Error>) -> Void) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sortDescriptor]) { (query, samples, error) in
            guard let samples = samples as? [HKCategorySample], error == nil else {
                completion(.failure(error ?? HealthStoreError.dataUnavailable("sleep")))
                return
            }

            var sleepDurations = [Date: DailySleepDurations]()
            let calendar = Calendar.current

            for sample in samples {
                let stateValue = HKCategoryValueSleepAnalysis(rawValue: sample.value)!
                let startOfDay = calendar.startOfDay(for: sample.startDate)
                let endOfDay = calendar.startOfDay(for: sample.endDate)

                var day = startOfDay
                while day <= endOfDay, day <= endDate {
                    let dayEnd = calendar.date(byAdding: .day, value: 1, to: day)!
                    let overlapStart = max(sample.startDate, day)
                    let overlapEnd = min(sample.endDate, dayEnd)
                    let duration = overlapEnd.timeIntervalSince(overlapStart) / 60.0 // Duration in minutes

                    if duration > 0 {
                        sleepDurations[day, default: DailySleepDurations(date: day)].addSleepState(state: stateValue, duration: duration)
                    }

                    day = dayEnd
                }
            }

            // Ensure every day in the range has an entry
            var day = startDate
            while day <= endDate {
                sleepDurations[day, default: DailySleepDurations(date: day)]
                day = calendar.date(byAdding: .day, value: 1, to: day)!
            }

            completion(.success(sleepDurations))
        }

        healthStore.execute(query)
    }
}


/// Optional helper: returns the main overnight sleep interval per day (start,end).
/// "Main sleep" ≈ the longest contiguous 'asleep*' span crossing local midnight.
public struct MainSleepWindow {
    public let day: Date          // UTC midnight of the *wake day*
    public let start: Date        // exact timestamp
    public let end: Date          // exact timestamp
}

extension HealthStore {
    public func fetchMainSleepWindows(
        startDate: Date,
        endDate: Date,
        completion: @escaping (Result<[Date: MainSleepWindow], Error>) -> Void
    ) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable)); return
        }
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

        let q = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, samples, err in
            guard let samples = samples as? [HKCategorySample], err == nil else {
                completion(.failure(err ?? HealthStoreError.dataUnavailable("sleep-episodes"))); return
            }
            // Heuristic: group contiguous asleep* states into episodes; pick the longest per wake day.
            let cal = Calendar(identifier: .gregorian)
            var episodesByWakeDay: [Date: (Date, Date)] = [:]

            var curStart: Date? = nil
            var curEnd: Date? = nil
            func closeEpisode() {
                guard let s = curStart, let e = curEnd, e > s else { return }
                // wake day = UTC midnight for end time
                let wakeDay = cal.startOfDay(for: e)
                if let prev = episodesByWakeDay[wakeDay] {
                    if (e.timeIntervalSince(s) > prev.1.timeIntervalSince(prev.0)) {
                        episodesByWakeDay[wakeDay] = (s,e)
                    }
                } else {
                    episodesByWakeDay[wakeDay] = (s,e)
                }
            }

            for s in samples {
                let v = HKCategoryValueSleepAnalysis(rawValue: s.value) ?? .asleepUnspecified
                let isAsleep = (v == .asleepCore || v == .asleepDeep || v == .asleepREM || v == .asleepUnspecified)
                if isAsleep {
                    if curStart == nil { curStart = s.startDate }
                    curEnd = max(curEnd ?? s.endDate, s.endDate)
                } else {
                    closeEpisode()
                    curStart = nil; curEnd = nil
                }
            }
            closeEpisode()

            let out = episodesByWakeDay.mapValues { MainSleepWindow(day: cal.startOfDay(for: $0.1), start: $0.0, end: $0.1) }
            completion(.success(out))
        }
        healthStore.execute(q)
    }
}
