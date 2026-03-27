//
//  InSiteTests.swift
//  InSiteTests
//
//  Created by Anand Parikh on 12/13/23.
//

import XCTest
@testable import InSite

final class InSiteTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        // Any test you write for XCTest can be annotated as throws and async.
        // Mark your test throws to produce an unexpected failure when your test encounters an uncaught error.
        // Mark your test async to allow awaiting for asynchronous code to complete. Check the results with assertions afterwards.
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

    func testChameliaConfigValues() throws {
        XCTAssertEqual(
            ChameliaConfig.baseURL.absoluteString,
            "https://chamelia-136217612465.us-central1.run.app"
        )
        XCTAssertEqual(ChameliaConfig.timeoutSeconds, 30, accuracy: 0.001)
    }

    func testFeatureFrameToChameliaAdapterMapsAllSignals() throws {
        let frame = FeatureFrameHourly(
            hourStartUtc: Date(timeIntervalSince1970: 1_700_000_000),
            bg_avg: 112,
            bg_tir: 76,
            bg_percentLow: 2,
            bg_percentHigh: 18,
            bg_uRoc: 1.4,
            bg_deltaAvg7h: -6,
            bg_zAvg7h: 0.7,
            hr_mean: 71,
            hr_delta7h: 4,
            hr_z7h: 1.2,
            rhr_daily: 58,
            kcal_active: 120,
            kcal_active_last3h: 240,
            kcal_active_last6h: 380,
            kcal_active_delta7h: 25,
            kcal_active_z7h: 1.5,
            sleep_prev_total_min: 435,
            sleep_debt_7d_min: 60,
            minutes_since_wake: 95,
            ex_move_min: 32,
            ex_exercise_min: 28,
            ex_min_last3h: 40,
            ex_hours_since: 6,
            days_since_period_start: 3,
            cycle_follicular: 1,
            cycle_ovulation: 0,
            cycle_luteal: 0,
            days_since_site_change: 2,
            site_loc_current: "abdomen_left",
            site_loc_same_as_last: 0,
            mood_valence: 0.4,
            mood_arousal: 0.8,
            mood_quad_posPos: 0,
            mood_quad_posNeg: 1,
            mood_quad_negPos: 0,
            mood_quad_negNeg: 0,
            mood_hours_since: 4
        )

        let blob = FeatureFrameToChameliaAdapter.makeSignalBlob(from: frame)

        XCTAssertEqual(blob.signals.count, 39)
        XCTAssertEqual(blob.signals["tir_7d"], .double(0.76))
        XCTAssertEqual(blob.signals["pct_low_7d"], .double(0.02))
        XCTAssertEqual(blob.signals["pct_high_7d"], .double(0.18))
        XCTAssertEqual(blob.signals["cycle_phase_menstrual"], .int(1))
        XCTAssertEqual(blob.signals["site_location"], .string("abdomen_left"))
        XCTAssertEqual(blob.signals["move_mins"], .double(32))
        XCTAssertEqual(blob.signals["stress_acute"], .double(0.5))
    }

    func testFeatureFrameToChameliaAdapterSkipsNilFields() throws {
        let frame = FeatureFrameHourly(
            hourStartUtc: Date(timeIntervalSince1970: 1_700_000_000),
            bg_avg: nil,
            bg_tir: 80,
            bg_percentLow: nil,
            bg_percentHigh: nil,
            bg_uRoc: nil,
            bg_deltaAvg7h: nil,
            bg_zAvg7h: nil,
            hr_mean: nil,
            hr_delta7h: nil,
            hr_z7h: nil,
            rhr_daily: nil,
            kcal_active: nil,
            kcal_active_last3h: nil,
            kcal_active_last6h: nil,
            kcal_active_delta7h: nil,
            kcal_active_z7h: nil,
            sleep_prev_total_min: nil,
            sleep_debt_7d_min: nil,
            minutes_since_wake: nil,
            ex_move_min: nil,
            ex_exercise_min: nil,
            ex_min_last3h: nil,
            ex_hours_since: nil,
            days_since_period_start: nil,
            cycle_follicular: nil,
            cycle_ovulation: nil,
            cycle_luteal: nil,
            days_since_site_change: nil,
            site_loc_current: "",
            site_loc_same_as_last: nil,
            mood_valence: nil,
            mood_arousal: nil,
            mood_quad_posPos: nil,
            mood_quad_posNeg: nil,
            mood_quad_negPos: nil,
            mood_quad_negNeg: nil,
            mood_hours_since: nil
        )

        let blob = FeatureFrameToChameliaAdapter.makeSignalBlob(from: frame)

        XCTAssertEqual(blob.signals.count, 1)
        XCTAssertEqual(blob.signals["tir_7d"], .double(0.8))
        XCTAssertNil(blob.signals["bg_avg"])
        XCTAssertNil(blob.signals["cycle_phase_menstrual"])
        XCTAssertNil(blob.signals["site_location"])
        XCTAssertNil(blob.signals["stress_acute"])
    }

    func testFeatureFrameToChameliaAdapterStressAcuteProxy() throws {
        let negativeValence = FeatureFrameHourly(
            hourStartUtc: Date(timeIntervalSince1970: 1_700_000_000),
            bg_avg: nil,
            bg_tir: nil,
            bg_percentLow: nil,
            bg_percentHigh: nil,
            bg_uRoc: nil,
            bg_deltaAvg7h: nil,
            bg_zAvg7h: nil,
            hr_mean: nil,
            hr_delta7h: nil,
            hr_z7h: nil,
            rhr_daily: nil,
            kcal_active: nil,
            kcal_active_last3h: nil,
            kcal_active_last6h: nil,
            kcal_active_delta7h: nil,
            kcal_active_z7h: nil,
            sleep_prev_total_min: nil,
            sleep_debt_7d_min: nil,
            minutes_since_wake: nil,
            ex_move_min: nil,
            ex_exercise_min: nil,
            ex_min_last3h: nil,
            ex_hours_since: nil,
            days_since_period_start: nil,
            cycle_follicular: nil,
            cycle_ovulation: nil,
            cycle_luteal: nil,
            days_since_site_change: nil,
            site_loc_current: nil,
            site_loc_same_as_last: nil,
            mood_valence: -0.2,
            mood_arousal: 0.9,
            mood_quad_posPos: nil,
            mood_quad_posNeg: nil,
            mood_quad_negPos: nil,
            mood_quad_negNeg: nil,
            mood_hours_since: nil
        )

        let neutralValence = FeatureFrameHourly(
            hourStartUtc: negativeValence.hourStartUtc,
            bg_avg: nil,
            bg_tir: nil,
            bg_percentLow: nil,
            bg_percentHigh: nil,
            bg_uRoc: nil,
            bg_deltaAvg7h: nil,
            bg_zAvg7h: nil,
            hr_mean: nil,
            hr_delta7h: nil,
            hr_z7h: nil,
            rhr_daily: nil,
            kcal_active: nil,
            kcal_active_last3h: nil,
            kcal_active_last6h: nil,
            kcal_active_delta7h: nil,
            kcal_active_z7h: nil,
            sleep_prev_total_min: nil,
            sleep_debt_7d_min: nil,
            minutes_since_wake: nil,
            ex_move_min: nil,
            ex_exercise_min: nil,
            ex_min_last3h: nil,
            ex_hours_since: nil,
            days_since_period_start: nil,
            cycle_follicular: nil,
            cycle_ovulation: nil,
            cycle_luteal: nil,
            days_since_site_change: nil,
            site_loc_current: nil,
            site_loc_same_as_last: nil,
            mood_valence: 0.2,
            mood_arousal: 0.9,
            mood_quad_posPos: nil,
            mood_quad_posNeg: nil,
            mood_quad_negPos: nil,
            mood_quad_negNeg: nil,
            mood_hours_since: nil
        )

        let lowArousal = FeatureFrameHourly(
            hourStartUtc: negativeValence.hourStartUtc,
            bg_avg: nil,
            bg_tir: nil,
            bg_percentLow: nil,
            bg_percentHigh: nil,
            bg_uRoc: nil,
            bg_deltaAvg7h: nil,
            bg_zAvg7h: nil,
            hr_mean: nil,
            hr_delta7h: nil,
            hr_z7h: nil,
            rhr_daily: nil,
            kcal_active: nil,
            kcal_active_last3h: nil,
            kcal_active_last6h: nil,
            kcal_active_delta7h: nil,
            kcal_active_z7h: nil,
            sleep_prev_total_min: nil,
            sleep_debt_7d_min: nil,
            minutes_since_wake: nil,
            ex_move_min: nil,
            ex_exercise_min: nil,
            ex_min_last3h: nil,
            ex_hours_since: nil,
            days_since_period_start: nil,
            cycle_follicular: nil,
            cycle_ovulation: nil,
            cycle_luteal: nil,
            days_since_site_change: nil,
            site_loc_current: nil,
            site_loc_same_as_last: nil,
            mood_valence: -0.2,
            mood_arousal: 0.2,
            mood_quad_posPos: nil,
            mood_quad_posNeg: nil,
            mood_quad_negPos: nil,
            mood_quad_negNeg: nil,
            mood_hours_since: nil
        )

        XCTAssertEqual(
            FeatureFrameToChameliaAdapter.makeSignalBlob(from: negativeValence).signals["stress_acute"],
            .double(1.0)
        )
        XCTAssertEqual(
            FeatureFrameToChameliaAdapter.makeSignalBlob(from: neutralValence).signals["stress_acute"],
            .double(0.5)
        )
        XCTAssertEqual(
            FeatureFrameToChameliaAdapter.makeSignalBlob(from: lowArousal).signals["stress_acute"],
            .double(0.0)
        )
    }

    func testFeatureFrameToChameliaAdapterNumericSignalsOmitNonDoubles() throws {
        let frame = FeatureFrameHourly(
            hourStartUtc: Date(timeIntervalSince1970: 1_700_000_000),
            bg_avg: 111,
            bg_tir: 82,
            bg_percentLow: nil,
            bg_percentHigh: nil,
            bg_uRoc: nil,
            bg_deltaAvg7h: nil,
            bg_zAvg7h: nil,
            hr_mean: nil,
            hr_delta7h: nil,
            hr_z7h: nil,
            rhr_daily: nil,
            kcal_active: nil,
            kcal_active_last3h: nil,
            kcal_active_last6h: nil,
            kcal_active_delta7h: nil,
            kcal_active_z7h: nil,
            sleep_prev_total_min: nil,
            sleep_debt_7d_min: nil,
            minutes_since_wake: 120,
            ex_move_min: nil,
            ex_exercise_min: nil,
            ex_min_last3h: nil,
            ex_hours_since: nil,
            days_since_period_start: 2,
            cycle_follicular: 1,
            cycle_ovulation: 0,
            cycle_luteal: 0,
            days_since_site_change: 5,
            site_loc_current: "arm_right",
            site_loc_same_as_last: 1,
            mood_valence: 0.1,
            mood_arousal: 0.8,
            mood_quad_posPos: 1,
            mood_quad_posNeg: 0,
            mood_quad_negPos: 0,
            mood_quad_negNeg: 0,
            mood_hours_since: 1
        )

        let numericSignals = FeatureFrameToChameliaAdapter.makeSignalBlob(from: frame).numericSignals

        XCTAssertEqual(numericSignals["bg_avg"], 111)
        XCTAssertEqual(numericSignals["tir_7d"], 0.82)
        XCTAssertEqual(numericSignals["stress_acute"], 0.5)
        XCTAssertNil(numericSignals["site_location"])
        XCTAssertNil(numericSignals["site_repeat"])
        XCTAssertNil(numericSignals["mins_since_wake"])
        XCTAssertNil(numericSignals["cycle_phase_follicular"])
    }

    func testQuestionnaireToPriorsAllNilReturnsPopulationDefaults() throws {
        let result = QuestionnaireToPriors.compute(QuestionnaireAnswers())
        let isf = try XCTUnwrap(result.physicalPriors["isf_multiplier"])
        let sleepOffset = try XCTUnwrap(result.physicalPriors["sleep_schedule_offset_h"])
        let sleepRegularity = try XCTUnwrap(result.physicalPriors["sleep_regularity"])

        XCTAssertEqual(isf.mean, 1.00, accuracy: 0.0001)
        XCTAssertEqual(sleepOffset.mean, 0.0, accuracy: 0.0001)
        XCTAssertEqual(sleepRegularity.std, 0.14, accuracy: 0.0001)
        XCTAssertEqual(result.aggressiveness, 0.5, accuracy: 0.0001)
    }

    func testQuestionnaireToPriorsAthleteBoostsISF() throws {
        let answers = QuestionnaireAnswers(
            exerciseFreq: .daily,
            exerciseType: .cardio,
            exerciseIntensity: .hard,
            fitnessLevel: .veryFit
        )

        let result = QuestionnaireToPriors.compute(answers)
        XCTAssertGreaterThan(result.physicalPriors["isf_multiplier"]?.mean ?? 0, 1.10)
    }

    func testQuestionnaireToPriorsNightOwlShiftsSleepOffset() throws {
        let answers = QuestionnaireAnswers(bedtimeCategory: .veryLate)
        let result = QuestionnaireToPriors.compute(answers)

        XCTAssertGreaterThan(result.physicalPriors["sleep_schedule_offset_h"]?.mean ?? 0, 2.5)
        XCTAssertGreaterThan(result.physicalPriors["sleep_schedule_offset_h"]?.std ?? 0, 0)
    }

}
