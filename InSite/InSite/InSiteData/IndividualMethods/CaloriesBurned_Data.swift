//
//  CaloriesBurned_Data.swift
//  InSite
//
//  Created by Anand Parikh on 2/1/24.
//

import Foundation
import HealthKit

struct HourlyEnergyData {
    let hour: Date
    var basalEnergy: Double
    var activeEnergy: Double
    var totalEnergy: Double { basalEnergy + activeEnergy }

    init(hour: Date, basalEnergy: Double = 0, activeEnergy: Double = 0) {
        self.hour = hour
        self.basalEnergy = basalEnergy
        self.activeEnergy = activeEnergy
    }
}

struct DailyAverageEnergyData {
    let date: Date // Represents the end date of the 7-day period
    let averageActiveEnergy: Double
}



extension HealthStore {

    // This method now fetches both basal and active energy data,
    // then combines them into total energy burned per hour.
    public func fetchAndCombineHourlyEnergyData(start: Date, end: Date, dispatchGroup: DispatchGroup,completion: @escaping (Result<([Date: HourlyEnergyData], [DailyAverageEnergyData]), Error>) -> Void) {
        guard let healthStore = self.healthStore else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }

        // Using a dictionary to map each hour to its energy data
        var combinedEnergyData = [Date: HourlyEnergyData]()
        var averageEnergyData = [DailyAverageEnergyData]()

        // Basal Energy
        dispatchGroup.enter()
        fetchHourlyEnergyData(for: basalEnergyType, start: start, end: end, healthStore: healthStore) { result in
            switch result {
            case .success(let results):
                for data in results {
                    if combinedEnergyData[data.hour] == nil {
                        combinedEnergyData[data.hour] = data
                    } else {
                        combinedEnergyData[data.hour]?.basalEnergy = data.basalEnergy
                    }
                }
            case .failure(let error):
                print("Error fetching basal energy: \(error)")
            }
            dispatchGroup.leave()
        }

        // Active Energy
        dispatchGroup.enter()
        fetchHourlyEnergyData(for: activeEnergyType, start: start, end: end, healthStore: healthStore) { result in
            switch result {
            case .success(let results):
                for data in results {
                    if combinedEnergyData[data.hour] == nil {
                        combinedEnergyData[data.hour] = data
                    } else {
                        combinedEnergyData[data.hour]?.activeEnergy = data.activeEnergy
                    }
                }
            case .failure(let error):
                print("Error fetching active energy: \(error)")
            }
            dispatchGroup.leave()
        }

        dispatchGroup.enter() // Enter before calling the method to ensure synchronization
        fetchDailyAverageActiveEnergy(startDate: start, endDate: end) { result in
            switch result {
            case .success(let dailyAverageEnergyDict):
            // Convert each (Date, Double) pair into a DailyAverageEnergyData object
            averageEnergyData = dailyAverageEnergyDict.map { date, averageEnergy in
                DailyAverageEnergyData(date: date, averageActiveEnergy: averageEnergy)
            }.sorted(by: { $0.date < $1.date }) // Optionally, sort the array by date
            case .failure(let error):
                print("Error fetching average active energy: \(error)")
            }
            // IMPORTANT: Leave the dispatch group after fetching and processing the daily average energy data
            dispatchGroup.leave()
        }
        // Once both queries are complete, process the combined data
        dispatchGroup.notify(queue: .main) {
            self.processCombinedEnergyData(combinedEnergyData)
            completion(.success((combinedEnergyData, averageEnergyData)))
        }
    }
    func fetchDailyAverageActiveEnergy(startDate: Date, endDate: Date, completion: @escaping (Result<[Date: Double], Error>) -> Void) {
            guard let healthStore = self.healthStore else {
                completion(.failure(HealthStoreError.notAvailable))
                return
            }
            var allAverages: [Date: Double] = [:]  // Use a dictionary to map each date to its average
            let calendar = Calendar.current
            let daysBetween = calendar.dateComponents([.day], from: startDate, to: endDate).day ?? 0

            // Dispatch group to synchronize the asynchronous fetching of weekly averages
            let fetchGroup = DispatchGroup()

            for dayOffset in 0...daysBetween {
                guard let currentDate = calendar.date(byAdding: .day, value: dayOffset, to: startDate) else { continue }
                let periodStart = calendar.date(byAdding: .day, value: -7, to: currentDate)!

                fetchGroup.enter()  // Mark the beginning of an asynchronous task
            fetchWeeklyAverageActiveCalories(start: periodStart, end: currentDate, healthStore: healthStore, quantityType: activeEnergyType) { average in
                    if let average = average {
                        allAverages[currentDate] = average
                    }
                    fetchGroup.leave()  // Mark the completion of the asynchronous task
                }
            }

            // Once all the asynchronous tasks are completed, execute the completion block
            fetchGroup.notify(queue: .main) {
                completion(.success(allAverages))
            }
        }


    private func fetchWeeklyAverageActiveCalories(start: Date, end: Date, healthStore: HKHealthStore, quantityType: HKQuantityType, completion: @escaping (Double?) -> Void) {
        guard let healthStore = self.healthStore else { return }
        // Constructing the predicate to filter the data
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)

        // Setting the interval for daily statistics
        var dateComponents = DateComponents()
        dateComponents.day = 1  // Aggregating data by day

        // Creating the query with cumulativeSum to get the total calories for each day
        let query = HKStatisticsCollectionQuery(quantityType: quantityType,
                                                quantitySamplePredicate: predicate,
                                                options: [.cumulativeSum],
                                                anchorDate: start,
                                                intervalComponents: dateComponents)

        // Processing the results to compute the weekly average
        query.initialResultsHandler = { _, result, _ in
            guard let result = result else {
                completion(nil)  // Call completion with nil in case of failure
                return
            }

            var totalSum: Double = 0
            var daysWithData: Int = 0

            // Summing the total calories for days with data
            result.enumerateStatistics(from: start, to: end) { statistic, _ in
                if let sum = statistic.sumQuantity() {
                    totalSum += sum.doubleValue(for: HKUnit.kilocalorie())
                    daysWithData += 1
                }
            }

            // Computing the average; divide by 7 to get the weekly average
            let averageDailyCalories = daysWithData > 0 ? totalSum / Double(7) : 0
            completion(averageDailyCalories)  // Returning the average
        }

        // Executing the query
        healthStore.execute(query)
    }
    // Placeholder for processing combined data
    private func processCombinedEnergyData(_ data: [Date: HourlyEnergyData]) {
        for (hour, energyData) in data {
            print("Hour: \(hour), Total Energy: \(energyData.totalEnergy)")
        }
    }

    private func fetchHourlyEnergyData(for quantityType: HKQuantityType, start: Date, end: Date, healthStore: HKHealthStore, completion: @escaping (Result<[HourlyEnergyData], Error>) -> Void) {
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        var dateComponents = DateComponents()
        dateComponents.hour = 1  // Hourly intervals.

        let query = HKStatisticsCollectionQuery(quantityType: quantityType,
                                                quantitySamplePredicate: predicate,
                                                options: [.cumulativeSum],
                                                anchorDate: start,
                                                intervalComponents: dateComponents)

        query.initialResultsHandler = { _, result, error in
            guard let result = result else {
                completion(.failure(error ?? HealthStoreError.dataUnavailable("energy")))
                return
            }

            var energyData: [HourlyEnergyData] = []

            result.enumerateStatistics(from: start, to: end) { statistic, _ in
                let date = statistic.startDate
                let totalEnergy = statistic.sumQuantity()?.doubleValue(for: HKUnit.kilocalorie()) ?? 0

                // Determine whether the fetched data is for basal or active energy
                if quantityType == self.basalEnergyType {
                    energyData.append(HourlyEnergyData(hour: date, basalEnergy: totalEnergy, activeEnergy: 0))
                } else if quantityType == self.activeEnergyType {
                    energyData.append(HourlyEnergyData(hour: date, basalEnergy: 0, activeEnergy: totalEnergy))
                }
            }

            completion(.success(energyData))
        }

        healthStore.execute(query)
    }
}

