//
//  Menstrual_Data.swift
//  InSite
//
//  Created by Anand Parikh on 12/19/23.
//

import Foundation
import HealthKit


struct DailyMenstrualData {
    let date: Date
    let daysSincePeriodStart: Int
}
extension HealthStore {
    func fetchMenstrualData(
            startDate: Date,
            endDate: Date,
            lastKnownPeriodStart: Date? = nil,           // ← NEW
            completion: @escaping (Result<[Date: DailyMenstrualData], Error>) -> Void
        ) {
            guard let healthStore = self.healthStore else {
                completion(.failure(HealthStoreError.notAvailable)); return
            }

            // Optionally extend the query window back to find a baseline start
            let calendar = Calendar.current
            let lookbackStart = calendar.date(byAdding: .day, value: -40, to: startDate) ?? startDate
            let queryStart = (lastKnownPeriodStart != nil) ? startDate : lookbackStart

            let predicate = HKQuery.predicateForSamples(withStart: queryStart, end: endDate, options: .strictStartDate)
            let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

            let q = HKSampleQuery(sampleType: menstrualType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sort]) { _, samples, error in
                guard let samples = samples as? [HKCategorySample], error == nil else {
                    completion(.failure(error ?? HealthStoreError.dataUnavailable("menstrual"))); return
                }

                var lastPeriodStart: Date? = lastKnownPeriodStart
                // If no external baseline, try to find the last start ≤ startDate from fetched samples
                if lastPeriodStart == nil {
                    lastPeriodStart = samples
                        .map { $0.startDate }
                        .filter { $0 <= startDate }
                        .max()
                }

                var menstrualData: [Date: DailyMenstrualData] = [:]
                var currentDate = startDate
                let dayStarts = Set(samples.map { calendar.startOfDay(for: $0.startDate) })

                while currentDate <= endDate {
                    let dayStart = calendar.startOfDay(for: currentDate)
                    if dayStarts.contains(dayStart) {
                        lastPeriodStart = dayStart
                    }

                    let days = lastPeriodStart.map {
                        calendar.dateComponents([.day], from: calendar.startOfDay(for: $0), to: dayStart).day ?? -1
                    } ?? -1

                    menstrualData[dayStart] = DailyMenstrualData(date: dayStart, daysSincePeriodStart: days)
                    currentDate = calendar.date(byAdding: .day, value: 1, to: currentDate)!
                }

                completion(.success(menstrualData))
            }

            healthStore.execute(q)
        }
}
