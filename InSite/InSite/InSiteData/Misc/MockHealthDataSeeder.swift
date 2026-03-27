#if DEBUG
import Foundation
import HealthKit

class HealthKitSeeder {
    static let healthStore = HKHealthStore()

    static func seedAll() {
        requestAuthorization {
            let calendar = Calendar.current
            let now = Date()
            guard let start = calendar.date(byAdding: .day, value: -30, to: now) else { return }

            for dayOffset in 0..<30 {
                guard let day = calendar.date(byAdding: .day, value: dayOffset, to: start) else { continue }
                let dayStart = calendar.startOfDay(for: day)

                for hour in 0..<24 {
                    guard let hourStart = calendar.date(byAdding: .hour, value: hour, to: dayStart),
                          let hourEnd = calendar.date(byAdding: .hour, value: 1, to: hourStart) else { continue }

                    // Heart Rate
                    let hr = Double.random(in: 60...100)
                    saveQuantity(type: .heartRate, unit: HKUnit(from: "count/min"), value: hr, start: hourStart, end: hourEnd)

                    // Steps
                    let steps = Double.random(in: 0...1000)
                    saveQuantity(type: .stepCount, unit: .count(), value: steps, start: hourStart, end: hourEnd)

                    // Active Energy
                    let activeEnergy = Double.random(in: 10...100)
                    saveQuantity(type: .activeEnergyBurned, unit: .kilocalorie(), value: activeEnergy, start: hourStart, end: hourEnd)

                    // Basal Energy
                    let basalEnergy = Double.random(in: 40...80)
                    saveQuantity(type: .basalEnergyBurned, unit: .kilocalorie(), value: basalEnergy, start: hourStart, end: hourEnd)

                    // Exercise Time
                    let exercise = Double.random(in: 0...30)
                    saveQuantity(type: .appleExerciseTime, unit: .minute(), value: exercise, start: hourStart, end: hourEnd)

                    // Body Mass
                    let weight = Double.random(in: 60...100)
                    saveQuantity(type: .bodyMass, unit: .gramUnit(with: .kilo), value: weight, start: hourStart, end: hourEnd)
                }

                // Resting Heart Rate (daily)
                let restingHR = Double.random(in: 55...75)
                saveQuantity(type: .restingHeartRate, unit: HKUnit(from: "count/min"), value: restingHR, start: dayStart, end: dayStart)

                // Menstrual Flow (category sample)
                let flowType = Int.random(in: 1...3) // 1: light, 2: medium, 3: heavy
                saveCategory(type: .menstrualFlow, value: flowType, start: dayStart, end: dayStart)

                // Sleep Analysis
                let sleepStart = calendar.date(byAdding: .hour, value: 23, to: dayStart)!
                let sleepEnd = calendar.date(byAdding: .hour, value: 7, to: sleepStart)!
                saveCategory(type: .sleepAnalysis, value: HKCategoryValueSleepAnalysis.asleep.rawValue, start: sleepStart, end: sleepEnd)
            }

            print("✅ Finished seeding HealthKit data.")
        }
    }

    static func requestAuthorization(completion: @escaping () -> Void) {
        let typesToShare: Set = [
            HKQuantityType.quantityType(forIdentifier: .heartRate)!,
            HKQuantityType.quantityType(forIdentifier: .stepCount)!,
            HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!,
            HKQuantityType.quantityType(forIdentifier: .basalEnergyBurned)!,
            HKQuantityType.quantityType(forIdentifier: .appleExerciseTime)!,
            HKQuantityType.quantityType(forIdentifier: .bodyMass)!,
            HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!,
            HKCategoryType.categoryType(forIdentifier: .menstrualFlow)!,
            HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
        ]

        healthStore.requestAuthorization(toShare: typesToShare, read: []) { success, error in
            if success {
                DispatchQueue.main.async {
                    completion()
                }
            } else {
                print("❌ HealthKit auth failed: \(error?.localizedDescription ?? "unknown error")")
            }
        }
    }

    static func saveQuantity(type: HKQuantityTypeIdentifier, unit: HKUnit, value: Double, start: Date, end: Date) {
        guard let quantityType = HKQuantityType.quantityType(forIdentifier: type) else { return }
        let quantity = HKQuantity(unit: unit, doubleValue: value)
        let sample = HKQuantitySample(type: quantityType, quantity: quantity, start: start, end: end)
        healthStore.save(sample) { success, error in
            if !success {
                print("❌ Failed to save \(type.rawValue): \(error?.localizedDescription ?? "unknown error")")
            }
        }
    }

    static func saveCategory(type: HKCategoryTypeIdentifier, value: Int, start: Date, end: Date) {
        guard let categoryType = HKCategoryType.categoryType(forIdentifier: type) else { return }
        let sample = HKCategorySample(type: categoryType, value: value, start: start, end: end)
        healthStore.save(sample) { success, error in
            if !success {
                print("❌ Failed to save \(type.rawValue): \(error?.localizedDescription ?? "unknown error")")
            }
        }
    }
}
#endif
