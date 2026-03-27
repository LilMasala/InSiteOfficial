import Foundation
import HealthKit

struct HourlyHeartRateData {
    let hour: Date
    var heartRate: Double

    init(hour: Date, heartRate: Double = 0) {
        self.hour = hour
        self.heartRate = heartRate
    }
}

struct DailyAverageHeartRateData {
    let date: Date // Represents the end date of the 7-day period
    let averageHeartRate: Double
}

extension HealthStore {
    private func fetchHourlyHeartRate(start: Date, end: Date, healthStore: HKHealthStore, completion: @escaping (Result<[HourlyHeartRateData], Error>) -> Void) {

        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        var dateComponents = DateComponents()
        dateComponents.hour = 1  // Set the interval for each hour.

        let query = HKStatisticsCollectionQuery(quantityType: heartRateType,
                                                quantitySamplePredicate: predicate,
                                                options: [.discreteAverage],
                                                anchorDate: start,
                                                intervalComponents: dateComponents)

        query.initialResultsHandler = { _, result, error in
            guard let result = result else {
                completion(.failure(error ?? HealthStoreError.dataUnavailable("heart-hourly")))
                return
            }

            var heartRateData: [HourlyHeartRateData] = []

            result.enumerateStatistics(from: start, to: end) { statistic, _ in
                let date = statistic.startDate
                if let average = statistic.averageQuantity() {
                    let averageHeartRate = average.doubleValue(for: HKUnit(from: "count/min"))
                    heartRateData.append(HourlyHeartRateData(hour: date, heartRate: averageHeartRate))
                }
            }

            completion(.success(heartRateData))
        }

        healthStore.execute(query)
    }
}


extension HealthStore {

    // This method fetches hourly heart rate data.
    public func fetchAndCombineHourlyHeartRateData(start: Date, end: Date, dispatchGroup: DispatchGroup, completion: @escaping (Result<([Date: HourlyHeartRateData], [DailyAverageHeartRateData]), Error>) -> Void) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        var hourlyHeartRateData = [Date: HourlyHeartRateData]()
        var averageHeartRateData = [DailyAverageHeartRateData]()

        dispatchGroup.enter()
        fetchHourlyHeartRate(start: start, end: end, healthStore: healthStore) { result in
            switch result {
            case .success(let results):
                for data in results {
                    hourlyHeartRateData[data.hour] = data
                }
            case .failure(let error):
                print("Error fetching hourly heart rate: \(error)")
            }
            dispatchGroup.leave()
        }

        dispatchGroup.enter() // Enter before calling the method to ensure synchronization
        fetchDailyAverageHeartRate(startDate: start, endDate: end) { result in
            switch result {
            case .success(let dailyAverageHeartRateDict):
            // Convert each (Date, Double) pair into a DailyAverageHeartRateData object
            averageHeartRateData = dailyAverageHeartRateDict.map { date, averageRate in
                DailyAverageHeartRateData(date: date, averageHeartRate: averageRate)
            }.sorted(by: { $0.date < $1.date }) // Optionally, sort the array by date
            case .failure(let error):
                print("Error fetching daily average heart rate: \(error)")
            }
            // IMPORTANT: Leave the dispatch group after fetching and processing the daily average heart rate data
            dispatchGroup.leave()
        }
        
        // Once both queries are complete, process the combined data
        dispatchGroup.notify(queue: .main) {
            completion(.success((hourlyHeartRateData, averageHeartRateData)))
        }
    }

    func fetchDailyAverageHeartRate(startDate: Date, endDate: Date, completion: @escaping (Result<[Date: Double], Error>) -> Void) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        var allAverages = [Date: Double]()
        let calendar = Calendar.current
        let daysBetween = calendar.dateComponents([.day], from: startDate, to: endDate).day ?? 0
        let dispatchGroup = DispatchGroup()

        for dayOffset in 0...daysBetween {
            guard let currentDate = calendar.date(byAdding: .day, value: dayOffset, to: startDate) else { continue }
            let periodStart = calendar.date(byAdding: .day, value: -7, to: currentDate)!

            dispatchGroup.enter()
            fetchWeeklyAverageHeartRate(start: periodStart, end: currentDate, healthStore: healthStore) { average in
                if let average = average {
                    allAverages[currentDate] = average
                }
                dispatchGroup.leave()
            }
        }

        dispatchGroup.notify(queue: .main) {
            completion(.success(allAverages))
        }
    }

    private func fetchWeeklyAverageHeartRate(start: Date, end: Date, healthStore: HKHealthStore, completion: @escaping (Double?) -> Void) {
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        let dateComponents = DateComponents(day: 1)  // Daily statistics

        let query = HKStatisticsCollectionQuery(quantityType: heartRateType,
                                                quantitySamplePredicate: predicate,
                                                options: [.discreteAverage],
                                                anchorDate: start,
                                                intervalComponents: dateComponents)

        query.initialResultsHandler = { _, result, _ in
            guard let result = result else {
                completion(nil)
                return
            }

            var totalSum: Double = 0
            var daysWithData: Int = 0

            result.enumerateStatistics(from: start, to: end) { statistic, _ in
                if let average = statistic.averageQuantity() {
                    totalSum += average.doubleValue(for: HKUnit(from: "count/min"))
                    daysWithData += 1
                }
            }

            let averageDailyRate = daysWithData > 0 ? totalSum / Double(daysWithData) : 0
            completion(averageDailyRate)
        }

        healthStore.execute(query)
    }
}
