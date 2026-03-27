import Foundation
import HealthKit

enum DataType {
    case move
    case exercise
}

struct HourlyExerciseData {
    let hour: Date
    var moveMinutes: Double
    var exerciseMinutes: Double
    var totalMinutes: Double {
        moveMinutes + exerciseMinutes
    }
}

struct DailyAverageExerciseData {
    let date: Date
    var averageMoveMinutes: Double
    var averageExerciseMinutes: Double
    var averageTotalMinutes: Double {
        (averageMoveMinutes + averageExerciseMinutes) / 2
    }
}

struct LastExerciseData {
    var hoursSinceLightExercise: Double?
    var hoursSinceIntenseExercise: Double?
}

extension HealthStore {
    
    func fetchAndCombineExerciseData(
            start: Date,
            end: Date,
            dispatchGroup: DispatchGroup,
            completion: @escaping (Result<([Date: HourlyExerciseData], [Date: DailyAverageExerciseData]), Error>) -> Void
        ) {
            var hourlyExerciseData = [Date: HourlyExerciseData]()
            var dailyAverageExerciseData = [Date: DailyAverageExerciseData]()
            let syncQ = DispatchQueue(label: "com.yourapp.hourlyExerciseDataQueue")

            
            guard let healthStore = self.healthStore else {
                completion(.failure(HealthStoreError.notAvailable))
                return
            }
            
            // 1) MOVE MINUTES (enter -> guaranteed leave)
            dispatchGroup.enter()
            fetchHourlyExerciseData(for: appleMoveTimeType, dataType: .move, start: start, end: end) { result in
                defer { dispatchGroup.leave() }
                switch result {
                case .success(let results):
                    for (hour, newData) in results {
                        // `hour` is already a Date
                        let hourDate = hour
                        syncQ.sync {
                            if var existing = hourlyExerciseData[hourDate] {
                                existing.moveMinutes += newData.totalMinutes
                                hourlyExerciseData[hourDate] = existing
                            } else {
                                var entry = HourlyExerciseData(hour: hourDate, moveMinutes: 0, exerciseMinutes: 0)
                                entry.moveMinutes += newData.totalMinutes
                                hourlyExerciseData[hourDate] = entry
                            }
                        }
                    }
                case .failure(let error):
                    print("Error fetching move data: \(error)")
                }
            }

            // 2) EXERCISE MINUTES (enter -> guaranteed leave)
            dispatchGroup.enter()
            fetchHourlyExerciseData(for: appleExerciseTimeType, dataType: .exercise, start: start, end: end) { result in
                defer { dispatchGroup.leave() }
                switch result {
                case .success(let results):
                    for (hour, newData) in results {
                        let hourDate = hour
                        syncQ.sync {
                            if var existing = hourlyExerciseData[hourDate] {
                                existing.exerciseMinutes += newData.totalMinutes
                                hourlyExerciseData[hourDate] = existing
                            } else {
                                var entry = HourlyExerciseData(hour: hourDate, moveMinutes: 0, exerciseMinutes: 0)
                                entry.exerciseMinutes += newData.totalMinutes
                                hourlyExerciseData[hourDate] = entry
                            }
                        }
                    }
                case .failure(let error):
                    print("Error fetching exercise minutes: \(error)")
                }
            }

            // 3) Notify ONCE after both hourly queries finish
            dispatchGroup.notify(queue: .main) {
                // Build per-day totals, then (optionally) convert to averages
                var perDayTotals: [Date: (move: Double, exercise: Double, hoursCount: Int)] = [:]
                for (hour, data) in hourlyExerciseData {
                    let dayStart = Calendar.current.startOfDay(for: hour)
                    var agg = perDayTotals[dayStart] ?? (0, 0, 0)
                    agg.move += data.moveMinutes
                    agg.exercise += data.exerciseMinutes
                    agg.hoursCount += 1
                    perDayTotals[dayStart] = agg
                }

                for (day, agg) in perDayTotals {
                    dailyAverageExerciseData[day] = DailyAverageExerciseData(
                        date: day,
                        averageMoveMinutes: agg.move,        // actually totals now
                        averageExerciseMinutes: agg.exercise // totals too
                    )
                }

                completion(.success((hourlyExerciseData, dailyAverageExerciseData)))
            }

            // ---- helpers (unchanged except for small correctness nits) ----

            func fetchHourlyExerciseData(
                for quantityType: HKQuantityType,
                dataType: DataType,
                start: Date,
                end: Date,
                completion: @escaping (Result<[Date: HourlyExerciseData], Error>) -> Void
            ) {
                let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
                var interval = DateComponents(); interval.hour = 1

                // ✅ Align to the hour, not midnight
                let anchorDate = Calendar.current.dateInterval(of: .hour, for: start)?.start ?? start

                let query = HKStatisticsCollectionQuery(
                    quantityType: quantityType,
                    quantitySamplePredicate: predicate,
                    options: [.cumulativeSum],
                    anchorDate: anchorDate,
                    intervalComponents: interval
                )

                query.initialResultsHandler = { _, results, error in
                    guard let results = results else {
                        completion(.failure(error ?? HealthStoreError.dataUnavailable("exercise")))
                        return
                    }

                    var data: [Date: HourlyExerciseData] = [:]
                    results.enumerateStatistics(from: start, to: end) { statistic, _ in
                        let hour = statistic.startDate
                        let minutes = statistic.sumQuantity()?.doubleValue(for: .minute()) ?? 0
                        var entry = data[hour] ?? HourlyExerciseData(hour: hour, moveMinutes: 0, exerciseMinutes: 0)
                        switch dataType {
                        case .move:     entry.moveMinutes += minutes
                        case .exercise: entry.exerciseMinutes += minutes
                        }
                        data[hour] = entry
                    }
                    completion(.success(data))
                }

                // Uses the captured instance from the outer guard
                healthStore.execute(query)
            }
    
        
        func fetchDailyAverageExerciseData(start: Date, end: Date, healthStore: HKHealthStore, completion: @escaping (Result<[Date: DailyAverageExerciseData], Error>) -> Void) {
            var dailyAverageExerciseData = [Date: DailyAverageExerciseData]()
            
            let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
            var interval = DateComponents()
            interval.day = 1
            
            let query = HKStatisticsCollectionQuery(quantityType: appleMoveTimeType, quantitySamplePredicate: predicate, options: [.cumulativeSum], anchorDate: start, intervalComponents: interval)
            query.initialResultsHandler = { _, results, error in
                guard let results = results else {
                    completion(.failure(error ?? HealthStoreError.dataUnavailable("exercise-move")))
                    return
                }
                
                var moveData: [Date: Double] = [:]
                results.enumerateStatistics(from: start, to: end) { statistic, _ in
                    let date = Calendar.current.startOfDay(for: statistic.startDate)
                    let moveMinutes = statistic.sumQuantity()?.doubleValue(for: .minute()) ?? 0
                    moveData[date] = moveMinutes
                }
                
                let exerciseQuery = HKStatisticsCollectionQuery(quantityType: self.appleExerciseTimeType, quantitySamplePredicate: predicate, options: [.cumulativeSum], anchorDate: start, intervalComponents: interval)
                exerciseQuery.initialResultsHandler = { _, exerciseResults, error in
                    guard let exerciseResults = exerciseResults else {
                        completion(.failure(error ?? HealthStoreError.dataUnavailable("exercise")))
                        return
                    }
                    
                    var exerciseData: [Date: Double] = [:]
                    exerciseResults.enumerateStatistics(from: start, to: end) { statistic, _ in
                        let date = Calendar.current.startOfDay(for: statistic.startDate)
                        let exerciseMinutes = statistic.sumQuantity()?.doubleValue(for: .minute()) ?? 0
                        exerciseData[date] = exerciseMinutes
                    }
                    
                    let allDates = Set(moveData.keys).union(exerciseData.keys)
                    for date in allDates {
                        let moveMinutes = moveData[date, default: 0]
                        let exerciseMinutes = exerciseData[date, default: 0]
                        dailyAverageExerciseData[date] = DailyAverageExerciseData(date: date, averageMoveMinutes: moveMinutes, averageExerciseMinutes: exerciseMinutes)
                    }
                    
                    completion(.success(dailyAverageExerciseData))
                }
                
                healthStore.execute(exerciseQuery)
            }
            
            healthStore.execute(query)
        }
        
        func determineLastExerciseData(healthStore: HKHealthStore, completion: @escaping (LastExerciseData) -> Void) {
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
            let query = HKSampleQuery(sampleType: HKObjectType.workoutType(), predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { query, results, error in
                guard let workouts = results as? [HKWorkout], let lastWorkout = workouts.first else {
                    if let error = error {
                        print("Error fetching the last workout: \(error.localizedDescription)")
                    }
                    completion(LastExerciseData(hoursSinceLightExercise: nil, hoursSinceIntenseExercise: nil))
                    return
                }
                
                let now = Date()
                let hoursSinceLastExercise = now.timeIntervalSince(lastWorkout.endDate) / 3600 // Convert seconds to hours
                
                let intensity = lastWorkout.totalEnergyBurned?.doubleValue(for: .kilocalorie()) ?? 0
                let isIntense = intensity > 500 // Example threshold for intense exercise
                
                let lastExerciseData = LastExerciseData(
                    hoursSinceLightExercise: isIntense ? nil : hoursSinceLastExercise,
                    hoursSinceIntenseExercise: isIntense ? hoursSinceLastExercise : nil
                )
                
                completion(lastExerciseData)
            }
            
            healthStore.execute(query)
        }
    }
}
