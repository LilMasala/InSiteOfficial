import Foundation
import HealthKit

struct DailyRestingHeartRateData {
    let date: Date
    let restingHeartRate: Double
}

extension HealthStore {
    func fetchDailyRestingHeartRate(startDate: Date, endDate: Date, completion: @escaping (Result<[DailyRestingHeartRateData], Error>) -> Void) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        let dateComponents = DateComponents(day: 1)  // Daily statistics

        let query = HKStatisticsCollectionQuery(quantityType: restingHeartRateType,
                                                quantitySamplePredicate: predicate,
                                                options: [.discreteAverage],
                                                anchorDate: startDate,
                                                intervalComponents: dateComponents)

        query.initialResultsHandler = { _, result, error in
            guard let result = result else {
                completion(.failure(error ?? HealthStoreError.dataUnavailable("resting-heart-rate")))
                return
            }

            var restingRates: [DailyRestingHeartRateData] = []

            result.enumerateStatistics(from: startDate, to: endDate) { statistic, _ in
                let date = statistic.startDate
                if let average = statistic.averageQuantity() {
                    let averageRate = average.doubleValue(for: HKUnit(from: "count/min"))
                    let dataPoint = DailyRestingHeartRateData(date: date, restingHeartRate: averageRate)
                    restingRates.append(dataPoint)
                }
            }

            completion(.success(restingRates))
        }

        healthStore.execute(query)
    }
}
