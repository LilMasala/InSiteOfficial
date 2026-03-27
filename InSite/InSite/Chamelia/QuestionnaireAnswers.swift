import Foundation

enum ExerciseFreq: String, CaseIterable, Codable {
    case never = "never"
    case lightWeek = "1_2x_week"
    case modWeek = "3_5x_week"
    case daily = "daily"
}

enum ExerciseType: String, CaseIterable, Codable {
    case cardio = "cardio"
    case strength = "strength"
    case mixed = "mixed"
    case light = "light"
    case none = "none"
}

enum ExerciseIntensity: String, CaseIterable, Codable {
    case casual = "casual"
    case moderate = "moderate"
    case hard = "hard"
    case intense = "intense"
}

enum FitnessLevel: String, CaseIterable, Codable {
    case low = "low"
    case average = "average"
    case fit = "fit"
    case veryFit = "very_fit"
}

enum BedtimeCategory: String, CaseIterable, Codable {
    case early = "before_10pm"
    case normal = "10pm_midnight"
    case late = "midnight_2am"
    case veryLate = "after_2am"
}

enum SleepHours: String, CaseIterable, Codable {
    case under6 = "under_6"
    case sixSeven = "6_7"
    case sevenEight = "7_8"
    case over8 = "over_8"
}

enum SleepConsistency: String, CaseIterable, Codable {
    case very = "very_consistent"
    case fairly = "fairly_consistent"
    case variable = "pretty_variable"
    case irregular = "completely_irregular"
}

enum RestedFeeling: String, CaseIterable, Codable {
    case usually = "usually"
    case sometimes = "sometimes"
    case rarely = "rarely"
}

enum FirstMealTime: String, CaseIterable, Codable {
    case early = "before_7am"
    case normal = "7am_9am"
    case late = "9am_11am"
    case skip = "after_11am_or_skip"
}

enum BreakfastSkip: String, CaseIterable, Codable {
    case never = "almost_never"
    case sometimes = "sometimes"
    case often = "often"
    case always = "almost_always"
}

enum LunchSkip: String, CaseIterable, Codable {
    case never = "almost_never"
    case sometimes = "sometimes"
    case often = "often"
    case always = "almost_always"
}

enum MealFrequency: String, CaseIterable, Codable {
    case oneTwo = "1_2"
    case three = "3"
    case fourFive = "4_5"
    case sixPlus = "6_plus"
}

enum MealConsistency: String, CaseIterable, Codable {
    case very = "very_consistent"
    case fairly = "fairly_consistent"
    case variable = "pretty_variable"
    case chaotic = "very_irregular"
}

enum PortionSize: String, CaseIterable, Codable {
    case small = "small"
    case average = "average"
    case generous = "generous"
    case large = "large"
}

enum DietType: String, CaseIterable, Codable {
    case lowCarb = "low_carb"
    case moderate = "moderate"
    case highCarb = "high_carb"
    case veryVariable = "very_variable"
}

enum LastMealTime: String, CaseIterable, Codable {
    case early = "before_7pm"
    case normal = "7pm_9pm"
    case late = "after_9pm"
    case veryLate = "midnight_snacks"
}

enum StressLevel: String, CaseIterable, Codable {
    case rarely = "rarely"
    case sometimes = "sometimes"
    case often = "often"
    case always = "almost_always"
}

enum StressBgEffect: String, CaseIterable, Codable {
    case noticeably = "yes_noticeably"
    case little = "a_little"
    case notReally = "not_really"
    case unsure = "not_sure"
}

enum MoodVariability: String, CaseIterable, Codable {
    case stable = "very_stable"
    case some = "some_variation"
    case variable = "quite_variable"
    case very = "very_variable"
}

enum ScheduleType: String, CaseIterable, Codable {
    case regular = "regular_9_5"
    case shift = "shift_work"
    case variable = "very_variable"
}

enum CyclePresence: String, CaseIterable, Codable {
    case regular = "yes_regular"
    case irregular = "yes_irregular"
    case no = "no"
}

enum CycleBgEffect: String, CaseIterable, Codable {
    case noticeably = "yes_noticeably"
    case little = "a_little"
    case notReally = "not_really"
    case unsure = "not_sure"
}

enum CycleHunger: String, CaseIterable, Codable {
    case noticeably = "yes_noticeably"
    case little = "a_little"
    case notReally = "not_really"
}

enum CycleMood: String, CaseIterable, Codable {
    case noticeably = "yes_noticeably"
    case little = "a_little"
    case notReally = "not_really"
}

enum InsulinSensitivity: String, CaseIterable, Codable {
    case high = "high"
    case normal = "normal"
    case low = "low"
    case unsure = "not_sure"
}

enum CarbSpike: String, CaseIterable, Codable {
    case lot = "yes_a_lot"
    case average = "average"
    case notMuch = "not_much"
    case unsure = "not_sure"
}

enum Aggressiveness: String, CaseIterable, Codable {
    case veryCautious = "very_cautious"
    case moderate = "moderate"
    case willing = "willing"
    case veryWilling = "very_willing"
}

enum ComplianceLevel: String, CaseIterable, Codable {
    case exact = "exactly"
    case close = "pretty_closely"
    case rough = "roughly"
    case forget = "sometimes_forget"
}

enum CheckFrequency: String, CaseIterable, Codable {
    case everyDay = "every_day"
    case mostDays = "most_days"
    case sometimes = "whenever_i_remember"
    case rarely = "probably_rarely"
}

enum TrustLevel: String, CaseIterable, Codable {
    case skeptical = "very_skeptical"
    case cautious = "cautiously_open"
    case trusting = "fairly_trusting"
    case veryTrusting = "very_trusting"
}

enum HypoglycemiaFearLevel: String, CaseIterable, Codable {
    case very = "very"
    case somewhat = "somewhat"
    case notMuch = "not_much"
}

enum RecommendationCadence: String, CaseIterable, Codable {
    case daily = "daily"
    case weekly = "weekly"
    case significant = "only_significant_changes"
}

struct QuestionnaireAnswers: Codable, Equatable {
    var bedtimeCategory: BedtimeCategory? = nil
    var sleepHours: SleepHours? = nil
    var sleepConsistency: SleepConsistency? = nil
    var restedFeeling: RestedFeeling? = nil

    var exerciseFreq: ExerciseFreq? = nil
    var exerciseType: ExerciseType? = nil
    var exerciseIntensity: ExerciseIntensity? = nil
    var fitnessLevel: FitnessLevel? = nil

    var firstMealTime: FirstMealTime? = nil
    var breakfastSkip: BreakfastSkip? = nil
    var lunchSkip: LunchSkip? = nil
    var mealFrequency: MealFrequency? = nil
    var mealConsistency: MealConsistency? = nil
    var portionSize: PortionSize? = nil
    var dietType: DietType? = nil
    var lastMealTime: LastMealTime? = nil

    var stressLevel: StressLevel? = nil
    var stressBgEffect: StressBgEffect? = nil
    var moodVariability: MoodVariability? = nil
    var scheduleType: ScheduleType? = nil

    var cyclePresence: CyclePresence? = nil
    var cycleBgEffect: CycleBgEffect? = nil
    var cycleHunger: CycleHunger? = nil
    var cycleMood: CycleMood? = nil

    var insulinSensitivity: InsulinSensitivity? = nil
    var carbSpike: CarbSpike? = nil
    var aggressiveness: Aggressiveness? = nil
    var complianceLevel: ComplianceLevel? = nil
    var checkFrequency: CheckFrequency? = nil
    var trustLevel: TrustLevel? = nil

    var hasAnyAnswer: Bool {
        bedtimeCategory != nil || sleepHours != nil || sleepConsistency != nil || restedFeeling != nil ||
        exerciseFreq != nil || exerciseType != nil || exerciseIntensity != nil || fitnessLevel != nil ||
        firstMealTime != nil || breakfastSkip != nil || lunchSkip != nil || mealFrequency != nil ||
        mealConsistency != nil || portionSize != nil || dietType != nil || lastMealTime != nil ||
        stressLevel != nil || stressBgEffect != nil || moodVariability != nil || scheduleType != nil ||
        cyclePresence != nil || cycleBgEffect != nil || cycleHunger != nil || cycleMood != nil ||
        insulinSensitivity != nil || carbSpike != nil || aggressiveness != nil ||
        complianceLevel != nil || checkFrequency != nil || trustLevel != nil
    }
}

struct QuestionnairePreferenceDraft: Codable, Equatable {
    var hypoglycemiaFear: HypoglycemiaFearLevel? = nil
    var recommendationCadence: RecommendationCadence? = nil
}
