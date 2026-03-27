////
////  BloodGlucose_Data.swift
////  InSite
////
////  Created by Anand Parikh on 6/24/24.
//
import Foundation
import HealthKit
//
public struct NewtonRaphson {

    /// function type, y = f(x)
    public typealias Function = (Double) -> Double
    /// solution type. x == nil means  f'(x) = 0 caused a division by zero
    /// steps is an optional output to indicate how many steps used by this solution
    public typealias Solution = (x: Double?, steps: Int)
    let initialValue: Double
    let error: Double
    let function: Function
    let differential: Function


    /// initialize a Newton Raphson equation solver
    /// - parameter initialValue: a preset (random) value of the equation
    /// - parameter error: error control
    /// - parameter function: the function of the equation - f(x)
    /// - parameter differential: the differential function of the equaltion f'(x)
    init(initialValue: Double = 0,
         error: Double = 1e-16,
         function: @escaping Function,
         differential: @escaping Function) {
        self.initialValue = initialValue
        self.error = error
        self.function = function
        self.differential = differential
    }
    // swiftlint:disable identifier_name
    /// solve the equation
    /// - returns: a solution
    public func solve() throws -> Solution {
        var x = initialValue
        var e = Double.infinity
        var steps = 0
        repeat {
            let delta = differential(x)
            guard abs(delta) > error else {
                return (nil, steps)
            }
            let y = x - function(x) / delta
            e = abs(x - y)
            x = y
            steps += 1
        } while e > error
        return (x: x, steps: steps)

    }
}

//
//import Foundation
//import HealthKit
//
//struct HourlyBloodGlucoseData {
//    let hour: Date
//    var bloodGlucose: Double
//
//    init(hour: Date, bloodGlucose: Double = 0) {
//        self.hour = hour
//        self.bloodGlucose = bloodGlucose
//    }
//}
//
//class BloodGlucoseManager {
//
//    private var totalGlucose: Double = 0.0
//    private var count: Int = 0
//    private var countBelowLowerBound: Int = 0
//    private var countAboveUpperBound: Int = 0
//    private var lowerBound: Double
//    private var upperBound: Double
//
//    private var startBG: Double = 0.0
//    private var startTime: Date?
//    private var targetBG: Int = 110
//    private var mostRecentBG: Double = 0.00
//    private var mostRecentBGTime: Date?
//    private var currentFunction: ((Double) -> Double)!
//    private var currentDifferential: ((Double) -> Double)!
//    private var startingX: Double?
//    private var newX: Double?
//
//    static let shared = BloodGlucoseManager(lowerBound: 80, upperBound: 170)
//    var healthStore = HealthStore()
//
//    private init(lowerBound: Double, upperBound: Double) {
//        self.lowerBound = lowerBound
//        self.upperBound = upperBound
//    }
//
//    public func updateGlucoseData() {
//        let start = Date().addingTimeInterval(-7 * 24 * 60 * 60) // 7 days ago
//        let end = Date()
//        let dispatchGroup = DispatchGroup()
//
//        healthStore.fetchAndCombineHourlyBloodGlucoseData(start: start, end: end, dispatchGroup: dispatchGroup) { hourlyData in
//            for (_, data) in hourlyData {
//                let glucoseValue = data.bloodGlucose
//                self.totalGlucose += glucoseValue
//                self.count += 1
//                if glucoseValue < self.lowerBound {
//                    self.countBelowLowerBound += 1
//                } else if glucoseValue > self.upperBound {
//                    self.countAboveUpperBound += 1
//                }
//
//                self.mostRecentBG = glucoseValue
//                self.mostRecentBGTime = data.hour
//
//                // Add difference in time to newX
//                if let startTime = self.startTime {
//                    let differenceInTime = Date().timeIntervalSince(startTime) / 60 // convert to minutes
//                    self.newX = self.startingX! + differenceInTime
//                }
//
//                self.setFunctionsAndDerivatives(startBG: self.startBG, targetBG: Double(self.targetBG))
//            }
//        }
//    }
//
//    private func setFunctionsAndDerivatives(startBG: Double, targetBG: Double) {
//        if startBG > targetBG {
//            currentFunction = { t in 385.96 * exp(-0.017 * t) + targetBG }
//            currentDifferential = { t in -0.017 * 385.96 * exp(-0.017 * t) }
//        } else if startBG < targetBG {
//            currentFunction = { t in targetBG / (1 + exp(-0.2 * pow((t - 43.5), 0.8))) }
//            currentDifferential = { t in
//                let part1 = pow((t - 43.5), 0.8)
//                let part2 = exp(-0.2 * part1)
//                let part3 = pow((1 + part2), -2)
//                let part4 = exp(-0.2 * part1)
//                let part5 = pow((t - 43.5), -0.2)
//                return -targetBG * part3 * part4 * -0.2 * 0.8 * part5
//            }
//        } else if startBG == targetBG {
//            currentFunction = { t in targetBG }
//            currentDifferential = { t in 0 }
//        } else {  // startBG < 55
//            currentFunction = { t in 4 * t }
//            currentDifferential = { t in 4 }
//        }
//    }
//
//    public func initiateStartTime() {
//        guard let currentFunction = currentFunction,
//              let currentDifferential = currentDifferential else {
//            return
//        }
//        self.startBG = self.mostRecentBG
//        let y_target = startBG
//
//        let g: NewtonRaphson.Function = { x in currentFunction(x) - y_target }
//        let g_differential = currentDifferential
//
//        let initialGuess = 0.0
//
//        let solver = NewtonRaphson(
//            initialValue: initialGuess,
//            error: 1e-6,
//            function: g,
//            differential: g_differential
//        )
//
//        do {
//            let result = try solver.solve()
//            // Store the x value that corresponds to the startBG
//            self.startingX = result.x
//            self.newX = result.x
//        } catch {
//            print("NewtonRaphson solver failed with error: \(error)")
//        }
//    }
//
//    var averageRealROC: Double {
//        guard let mostRecentBGTime = mostRecentBGTime, let startTime = startTime else {
//            print("mostRecentBGTime or startTime is nil")
//            return 0.0
//        }
//
//        let timeInterval = mostRecentBGTime.timeIntervalSince(startTime)
//        return (mostRecentBG - startBG) / timeInterval
//    }
//
//    public var expectedBG: Double? {
//        guard let currentFunction = currentFunction, let newX = newX else {
//            return nil
//        }
//
//        return currentFunction(newX)
//    }
//
//    var averageExpectedROC: Double? {
//        guard let expectedBG = expectedBG, let newX = newX, let startingX = startingX else {
//            print("Expected BG, New X, or Starting X is nil")
//            return nil
//        }
//
//        return (expectedBG - startBG) / (newX - startingX)
//    }
//
//    public var uROC: Double? {
//        return averageRealROC - (averageExpectedROC ?? averageRealROC)
//    }
//
//    public var avgBG: Double {
//        return totalGlucose / Double(count)
//    }
//
//    public var percentLow: Double {
//        return (Double(countBelowLowerBound) / Double(count)) * 100
//    }
//
//    public var percentHigh: Double {
//        return (Double(countAboveUpperBound) / Double(count)) * 100
//    }
//
//    public func resetAvgBG() {
//        totalGlucose = 0.0
//        count = 0
//    }
//
//    public func resetPercentLow() {
//        countBelowLowerBound = 0
//    }
//
//    public func resetPercentHigh() {
//        countAboveUpperBound = 0
//    }
//
//    public func reset() {
//        print(self.totalGlucose)
//        print(self.countAboveUpperBound)
//        print(self.countBelowLowerBound)
//        print(self.uROC ?? 0.0)
//        self.totalGlucose = 0.0
//        self.count = 0
//        self.countBelowLowerBound = 0
//        self.countAboveUpperBound = 0
//
//        // Reset the BG and time related variables
//        self.startBG = 0.0
//        self.startTime = nil
//        self.mostRecentBG = 0.0
//        self.mostRecentBGTime = nil
//        self.currentFunction = nil
//        self.currentDifferential = nil
//        self.startingX = nil
//        self.newX = nil
//    }
//
//    public func startNewCycle() {
//        reset()
//        initiateStartTime()
//    }
//}
