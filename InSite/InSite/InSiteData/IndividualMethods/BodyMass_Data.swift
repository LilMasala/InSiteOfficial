import Foundation
import HealthKit

struct HourlyBodyMassData {
    let hour: Date
    let weight: Double
}

extension HealthStore {
    
    func fetchHourlyMassData(start: Date, end: Date, dispatchGroup: DispatchGroup, completion: @escaping (Result<[HourlyBodyMassData], Error>) -> Void) {
        guard let healthStore = self.healthStore, let bodyMassType = HKObjectType.quantityType(forIdentifier: .bodyMass) else {
            completion(.failure(HealthStoreError.notAvailable))
            return
        }
        
        var bodyMassData = [HourlyBodyMassData]()
        
        dispatchGroup.enter()
        fetchHourlyMassDataQuery(start: start, end: end, healthStore: healthStore, bodyMassType: bodyMassType) { result in
            switch result {
            case .success(let results):
                for result in results {
                    let date = result.startDate
                    if let average = result.averageQuantity() {
                        let bodyMassValue = average.doubleValue(for: HKUnit.gramUnit(with: .kilo))
                        let dataPoint = HourlyBodyMassData(hour: date, weight: bodyMassValue)
                        bodyMassData.append(dataPoint)
                    }
                }
                completion(.success(bodyMassData))
            case .failure(let error):
                completion(.failure(error))
            }
            dispatchGroup.leave()
        }
    }

    
    private func fetchHourlyMassDataQuery(start: Date, end: Date, healthStore: HKHealthStore, bodyMassType: HKQuantityType, completion: @escaping (Result<[HKStatistics], Error>) -> Void) {
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
        var dateComponents = DateComponents()
        dateComponents.hour = 1
        
        let query = HKStatisticsCollectionQuery(quantityType: bodyMassType, quantitySamplePredicate: predicate, options: [.discreteAverage], anchorDate: start, intervalComponents: dateComponents)
        
        query.initialResultsHandler = { query, results, error in
            guard let statsCollection = results else {
                completion(.failure(error ?? HealthStoreError.dataUnavailable("body-mass")))
                return
            }
            var statistics = [HKStatistics]()
            statsCollection.enumerateStatistics(from: start, to: end) { statistic, stop in
                statistics.append(statistic)
            }
            completion(.success(statistics))
        }
        
        healthStore.execute(query)
    }
}
