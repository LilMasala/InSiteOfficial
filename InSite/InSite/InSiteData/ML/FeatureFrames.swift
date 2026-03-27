//
//  FeatureFrames.swift
//  InSite
//
//  Created by Anand Parikh on 10/8/25.
//
import Foundation

public struct FeatureFrameHourly: Sendable {
    public let hourStartUtc: Date

    // Core outcomes/targets you might predict later (keep nullable)
    public let bg_avg: Double?
    public let bg_tir: Double?
    public let bg_percentLow: Double?
    public let bg_percentHigh: Double?
    public let bg_uRoc: Double?

    // BG CTX
    public let bg_deltaAvg7h: Double?
    public let bg_zAvg7h: Double?

    // HR CTX
    public let hr_mean: Double?
    public let hr_delta7h: Double?
    public let hr_z7h: Double?
    public let rhr_daily: Double?

    // Energy CTX
    public let kcal_active: Double?
    public let kcal_active_last3h: Double?
    public let kcal_active_last6h: Double?
    public let kcal_active_delta7h: Double?
    public let kcal_active_z7h: Double?

    // Sleep CTX
    public let sleep_prev_total_min: Double?
    public let sleep_debt_7d_min: Double?
    public let minutes_since_wake: Int?

    // Exercise CTX
    public let ex_move_min: Double?
    public let ex_exercise_min: Double?
    public let ex_min_last3h: Double?
    public let ex_hours_since: Double?

    // Menstrual + Site CTX (optional for many hours)
    public let days_since_period_start: Int?
    public let cycle_follicular: Int?
    public let cycle_ovulation: Int?
    public let cycle_luteal: Int?

    public let days_since_site_change: Int?
    public let site_loc_current: String?
    public let site_loc_same_as_last: Int?   // 1/0
    
    
    public let mood_valence: Double?
    public let mood_arousal: Double?
    public let mood_quad_posPos: Int?
    public let mood_quad_posNeg: Int?
    public let mood_quad_negPos: Int?
    public let mood_quad_negNeg: Int?
    public let mood_hours_since: Double?
}
