import Foundation

class HealthDataFetcher {
    private var healthStore: HealthStore?
    init() {
        healthStore = HealthStore()
    }

    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        healthStore?.requestAuthorization(completion: completion)
    }

    func fetchAllBgData(start: Date, end: Date, group: DispatchGroup, completion: @escaping (Result<([HourlyBgData], [HourlyAvgBgData], [HourlyBgPercentages]), Error>) -> Void) {
        healthStore?.fetchAllBgData(start: start, end: end, dispatchGroup: group, completion: completion)
    }

    func fetchHeartRateData(start: Date, end: Date, group: DispatchGroup, completion: @escaping (Result<([Date: HourlyHeartRateData], [DailyAverageHeartRateData]), Error>) -> Void) {
        healthStore?.fetchAndCombineHourlyHeartRateData(start: start, end: end, dispatchGroup: group, completion: completion)
    }

    func fetchExerciseData(start: Date, end: Date, group: DispatchGroup, completion: @escaping (Result<([Date: HourlyExerciseData], [Date: DailyAverageExerciseData]), Error>) -> Void) {
        healthStore?.fetchAndCombineExerciseData(start: start, end: end, dispatchGroup: group, completion: completion)
    }

    func fetchMenstrualData(start: Date, end: Date, completion: @escaping (Result<[Date: DailyMenstrualData], Error>) -> Void) {
        healthStore?.fetchMenstrualData(startDate: start, endDate: end, completion: completion)
    }

    func fetchBodyMassData(start: Date, end: Date, group: DispatchGroup, completion: @escaping (Result<[HourlyBodyMassData], Error>) -> Void) {
        healthStore?.fetchHourlyMassData(start: start, end: end, dispatchGroup: group, completion: completion)
    }

    func fetchRestingHeartRate(start: Date, end: Date, completion: @escaping (Result<[DailyRestingHeartRateData], Error>) -> Void) {
        healthStore?.fetchDailyRestingHeartRate(startDate: start, endDate: end, completion: completion)
    }

    func fetchSleepDurations(start: Date, end: Date, completion: @escaping (Result<[Date: DailySleepDurations], Error>) -> Void) {
        healthStore?.fetchSleepDurations(startDate: start, endDate: end, completion: completion)
    }

    func fetchEnergyData(start: Date, end: Date, group: DispatchGroup, completion: @escaping (Result<([Date: HourlyEnergyData], [DailyAverageEnergyData]), Error>) -> Void) {
        healthStore?.fetchAndCombineHourlyEnergyData(start: start, end: end, dispatchGroup: group, completion: completion)
    }
    
    func fetchBgRawValuesHourly(
            start: Date,
            end: Date,
            completion: @escaping (Result<[HourlyBgValues], Error>) -> Void
        ) {
            healthStore?.fetchBgRawValuesHourly(start: start, end: end, completion: completion)
        }
}
