import Foundation

struct QuestionnaireToPriors {
    struct PriorResult {
        var physicalPriors: [String: (mean: Double, std: Double)]
        var aggressiveness: Double
        var complianceNoise: Double
        var engagementDecay: Double
        var initialTrust: Double
    }

    static func compute(_ answers: QuestionnaireAnswers) -> PriorResult {
        var priors = populationDefaults()

        if let bedtime = answers.bedtimeCategory {
            priors["sleep_schedule_offset_h"] = bedtimeToPrior(bedtime)
        }
        if let hours = answers.sleepHours {
            let values = sleepHoursToPriors(hours)
            priors["sleep_total_min_mean"] = values.minutes
            priors["sleep_efficiency"] = values.efficiency
        }
        if let consistency = answers.sleepConsistency {
            priors["sleep_regularity"] = sleepConsistencyToPrior(consistency)
        }
        if let rested = answers.restedFeeling {
            priors["sleep_efficiency"] = adjust(priors["sleep_efficiency"], meanDelta: restedFeelingDelta(rested), stdOverride: nil)
        }

        if let freq = answers.exerciseFreq {
            priors["activity_propensity"] = exerciseFreqToPrior(freq)
        }
        if let type = answers.exerciseType {
            let mapping = exerciseTypeToPriors(type, activityMean: priors["activity_propensity"]?.mean ?? 0.45)
            priors["exercise_intensity_mean"] = mapping.intensity
            priors["fitness_level"] = mapping.fitness
            priors["base_rhr"] = mapping.baseRhr
            priors["isf_multiplier"] = mapping.isf
        }
        if let intensity = answers.exerciseIntensity {
            let update = exerciseIntensityAdjustment(intensity)
            priors["exercise_intensity_mean"] = adjust(priors["exercise_intensity_mean"], meanDelta: update.intensityDelta, stdOverride: nil)
            if let baseRhrDelta = update.baseRhrDelta {
                priors["base_rhr"] = adjust(priors["base_rhr"], meanDelta: baseRhrDelta, stdOverride: nil)
            }
        }
        if let fitness = answers.fitnessLevel {
            let update = fitnessAdjustment(fitness)
            priors["fitness_level"] = adjust(priors["fitness_level"], meanDelta: update.fitnessDelta, stdOverride: nil)
            if let baseRhrDelta = update.baseRhrDelta {
                priors["base_rhr"] = adjust(priors["base_rhr"], meanDelta: baseRhrDelta, stdOverride: nil)
            }
        }

        if let firstMeal = answers.firstMealTime {
            let mapping = firstMealToPriors(firstMeal)
            priors["meal_schedule_offset_h"] = mapping.offset
            if let breakfast = mapping.breakfastSkip {
                priors["skips_breakfast_p"] = breakfast
            }
        }
        if let breakfastSkip = answers.breakfastSkip {
            priors["skips_breakfast_p"] = breakfastSkipToPrior(breakfastSkip)
        }
        if let lunchSkip = answers.lunchSkip {
            priors["skips_lunch_p"] = lunchSkipToPrior(lunchSkip)
        }
        if let mealFrequency = answers.mealFrequency {
            priors["n_meals_per_day_mean"] = mealFrequencyToPrior(mealFrequency)
        }
        if let mealConsistency = answers.mealConsistency {
            priors["meal_regularity"] = mealConsistencyToPrior(mealConsistency)
        }
        if let portion = answers.portionSize {
            priors["meal_size_multiplier"] = portionSizeToPrior(portion)
        }
        if let diet = answers.dietType {
            let mapping = dietTypeToPriors(diet)
            priors["cr_multiplier"] = mapping.cr
            if let mealSizeDelta = mapping.mealSizeMeanDelta {
                priors["meal_size_multiplier"] = adjust(priors["meal_size_multiplier"], meanDelta: mealSizeDelta, stdOverride: nil)
            }
        }
        if let lastMeal = answers.lastMealTime {
            let update = lastMealAdjustment(lastMeal)
            if let meanDelta = update.offsetMeanDelta {
                priors["meal_schedule_offset_h"] = adjust(priors["meal_schedule_offset_h"], meanDelta: meanDelta, stdOverride: nil)
            }
            priors["meal_schedule_offset_h"] = adjust(priors["meal_schedule_offset_h"], meanDelta: 0, stdOverride: max(0.02, (priors["meal_schedule_offset_h"]?.std ?? 0.25) + update.offsetStdDelta))
        }

        if let stress = answers.stressLevel {
            let mapping = stressLevelToPriors(stress)
            priors["stress_baseline"] = mapping.baseline
            priors["stress_reactivity"] = mapping.reactivity
        }
        if let effect = answers.stressBgEffect {
            priors["stress_reactivity"] = adjust(priors["stress_reactivity"], meanDelta: stressBgEffectDelta(effect), stdOverride: nil)
        }
        if let mood = answers.moodVariability {
            priors["mood_stability"] = moodVariabilityToPrior(mood)
        }
        if let schedule = answers.scheduleType {
            let update = scheduleTypeAdjustment(schedule)
            if let sleepStdDelta = update.sleepOffsetStdDelta {
                priors["sleep_schedule_offset_h"] = adjust(priors["sleep_schedule_offset_h"], meanDelta: 0, stdOverride: max(0.02, (priors["sleep_schedule_offset_h"]?.std ?? 0.35) + sleepStdDelta))
            }
            if let stressDelta = update.stressBaselineDelta {
                priors["stress_baseline"] = adjust(priors["stress_baseline"], meanDelta: stressDelta, stdOverride: nil)
            }
            if let sleepRegularityDelta = update.sleepRegularityMeanDelta {
                priors["sleep_regularity"] = adjust(priors["sleep_regularity"], meanDelta: sleepRegularityDelta, stdOverride: nil)
            }
            if let sleepRegularityStdDelta = update.sleepRegularityStdDelta {
                priors["sleep_regularity"] = adjust(priors["sleep_regularity"], meanDelta: 0, stdOverride: max(0.02, (priors["sleep_regularity"]?.std ?? 0.14) + sleepRegularityStdDelta))
            }
            if let mealRegularityStdDelta = update.mealRegularityStdDelta {
                priors["meal_regularity"] = adjust(priors["meal_regularity"], meanDelta: 0, stdOverride: max(0.02, (priors["meal_regularity"]?.std ?? 0.14) + mealRegularityStdDelta))
            }
        }

        if let cycle = answers.cyclePresence {
            let mapping = cyclePresenceToPrior(cycle)
            priors["cycle_sensitivity"] = mapping
        }
        if let effect = answers.cycleBgEffect,
           answers.cyclePresence == .regular || answers.cyclePresence == .irregular {
            let current = priors["cycle_sensitivity"] ?? (0.0, 0.0)
            priors["cycle_sensitivity"] = cycleBgEffectAdjustment(effect, current: current)
        }
        if let hunger = answers.cycleHunger,
           answers.cyclePresence == .regular || answers.cyclePresence == .irregular {
            priors["luteal_meal_size_boost"] = cycleHungerToPrior(hunger)
        }
        if let mood = answers.cycleMood,
           answers.cyclePresence == .regular || answers.cyclePresence == .irregular {
            priors["luteal_mood_drop"] = cycleMoodToPrior(mood)
        }

        if let sensitivity = answers.insulinSensitivity {
            priors["isf_multiplier"] = adjust(priors["isf_multiplier"], meanDelta: insulinSensitivityDelta(sensitivity), stdOverride: nil)
        }
        if let carbSpike = answers.carbSpike {
            priors["cr_multiplier"] = adjust(priors["cr_multiplier"], meanDelta: carbSpikeDelta(carbSpike), stdOverride: nil)
        }

        applyCrossInteractions(&priors, answers: answers)

        return PriorResult(
            physicalPriors: priors,
            aggressiveness: aggressivenessValue(answers.aggressiveness),
            complianceNoise: complianceValue(answers.complianceLevel),
            engagementDecay: engagementValue(answers.checkFrequency),
            initialTrust: trustValue(answers.trustLevel)
        )
    }

    static func hypoglycemiaFearValue(_ fear: HypoglycemiaFearLevel?) -> Double {
        switch fear {
        case .very: return 0.9
        case .somewhat: return 0.6
        case .notMuch: return 0.3
        case nil: return 0.5
        }
    }

    static func burdenSensitivityValue(_ cadence: RecommendationCadence?) -> Double {
        switch cadence {
        case .daily: return 0.3
        case .weekly: return 0.6
        case .significant: return 0.9
        case nil: return 0.5
        }
    }

    private static func populationDefaults() -> [String: (mean: Double, std: Double)] {
        [
            "isf_multiplier": (1.00, 0.12),
            "cr_multiplier": (1.00, 0.12),
            "basal_multiplier": (1.00, 0.10),
            "base_rhr": (63.0, 7.0),
            "activity_propensity": (0.45, 0.15),
            "sleep_regularity": (0.62, 0.14),
            "sleep_total_min_mean": (400.0, 55.0),
            "sleep_efficiency": (0.82, 0.09),
            "sleep_schedule_offset_h": (0.0, 0.35),
            "stress_reactivity": (0.50, 0.14),
            "stress_baseline": (0.22, 0.12),
            "cycle_sensitivity": (0.0, 0.0),
            "meal_regularity": (0.58, 0.14),
            "exercise_intensity_mean": (0.50, 0.14),
            "fitness_level": (0.55, 0.14),
            "mood_stability": (0.65, 0.12),
            "skips_breakfast_p": (0.15, 0.10),
            "skips_lunch_p": (0.08, 0.07),
            "n_meals_per_day_mean": (3.0, 0.5),
            "meal_size_multiplier": (1.00, 0.12),
            "meal_schedule_offset_h": (0.0, 0.25),
            "luteal_meal_size_boost": (0.08, 0.03),
            "luteal_mood_drop": (0.15, 0.05)
        ]
    }

    private static func applyCrossInteractions(
        _ priors: inout [String: (mean: Double, std: Double)],
        answers: QuestionnaireAnswers
    ) {
        if (priors["activity_propensity"]?.mean ?? 0) > 0.60,
           (priors["sleep_regularity"]?.mean ?? 1) < 0.50 {
            priors["stress_reactivity"] = adjust(priors["stress_reactivity"], meanDelta: 0.10, stdOverride: nil)
        }

        if (priors["sleep_schedule_offset_h"]?.mean ?? 0) > 1.5,
           answers.scheduleType == .shift {
            priors["sleep_regularity"] = (
                (priors["sleep_regularity"]?.mean ?? 0.62) - 0.08,
                (priors["sleep_regularity"]?.std ?? 0.14) + 0.06
            )
        }

        if answers.dietType == .veryVariable {
            let current = priors["cr_multiplier"] ?? (1.0, 0.12)
            priors["cr_multiplier"] = (current.mean, max(current.std, 0.16))
        }

        if (priors["stress_baseline"]?.mean ?? 0) > 0.45,
           (priors["mood_stability"]?.mean ?? 1) < 0.40 {
            priors["initial_trust_proxy"] = (max(0.10, 0.35 - 0.08), 0.08)
        }

        if (priors["activity_propensity"]?.mean ?? 0) > 0.70,
           answers.exerciseType == .cardio,
           (priors["sleep_regularity"]?.mean ?? 0) > 0.70 {
            priors["isf_multiplier"] = adjust(priors["isf_multiplier"], meanDelta: 0.05, stdOverride: nil)
        }

        if answers.dietType == .highCarb,
           (priors["skips_breakfast_p"]?.mean ?? 0) > 0.40 || (priors["skips_lunch_p"]?.mean ?? 0) > 0.40 {
            let current = priors["meal_size_multiplier"] ?? (1.0, 0.12)
            priors["meal_size_multiplier"] = (current.mean, current.std + 0.06)
        }
    }

    private static func bedtimeToPrior(_ bedtime: BedtimeCategory) -> (Double, Double) {
        switch bedtime {
        case .early: return (-1.5, 0.25)
        case .normal: return (0.0, 0.25)
        case .late: return (1.5, 0.35)
        case .veryLate: return (3.0, 0.45)
        }
    }

    private static func sleepHoursToPriors(_ value: SleepHours) -> (minutes: (Double, Double), efficiency: (Double, Double)) {
        switch value {
        case .under6: return ((330, 25), (0.76, 0.08))
        case .sixSeven: return ((390, 25), (0.81, 0.07))
        case .sevenEight: return ((450, 25), (0.85, 0.06))
        case .over8: return ((510, 30), (0.87, 0.06))
        }
    }

    private static func sleepConsistencyToPrior(_ value: SleepConsistency) -> (Double, Double) {
        switch value {
        case .very: return (0.90, 0.05)
        case .fairly: return (0.70, 0.08)
        case .variable: return (0.45, 0.10)
        case .irregular: return (0.18, 0.10)
        }
    }

    private static func restedFeelingDelta(_ value: RestedFeeling) -> Double {
        switch value {
        case .usually: return 0.04
        case .sometimes: return 0.0
        case .rarely: return -0.05
        }
    }

    private static func exerciseFreqToPrior(_ value: ExerciseFreq) -> (Double, Double) {
        switch value {
        case .never: return (0.08, 0.06)
        case .lightWeek: return (0.32, 0.10)
        case .modWeek: return (0.65, 0.10)
        case .daily: return (0.88, 0.07)
        }
    }

    private static func exerciseTypeToPriors(_ value: ExerciseType, activityMean: Double) -> (intensity: (Double, Double), fitness: (Double, Double), baseRhr: (Double, Double), isf: (Double, Double)) {
        switch value {
        case .cardio:
            return ((0.72, 0.10), (activityMean * 0.95, 0.10), (51, 6), (1.16, 0.10))
        case .strength:
            return ((0.68, 0.10), (activityMean * 0.88, 0.10), (57, 7), (1.08, 0.10))
        case .mixed:
            return ((0.64, 0.10), (activityMean * 0.90, 0.10), (58, 7), (1.11, 0.10))
        case .light:
            return ((0.32, 0.10), (activityMean * 0.70, 0.10), (64, 8), (1.02, 0.10))
        case .none:
            return ((0.18, 0.10), (activityMean * 0.55, 0.10), (66, 8), (1.00, 0.10))
        }
    }

    private static func exerciseIntensityAdjustment(_ value: ExerciseIntensity) -> (intensityDelta: Double, baseRhrDelta: Double?) {
        switch value {
        case .casual: return (-0.10, nil)
        case .moderate: return (0.0, nil)
        case .hard: return (0.08, nil)
        case .intense: return (0.15, -3)
        }
    }

    private static func fitnessAdjustment(_ value: FitnessLevel) -> (fitnessDelta: Double, baseRhrDelta: Double?) {
        switch value {
        case .low: return (-0.10, nil)
        case .average: return (0.0, nil)
        case .fit: return (0.08, nil)
        case .veryFit: return (0.15, -4)
        }
    }

    private static func firstMealToPriors(_ value: FirstMealTime) -> (offset: (Double, Double), breakfastSkip: (Double, Double)?) {
        switch value {
        case .early: return ((-1.5, 0.25), nil)
        case .normal: return ((0.0, 0.25), nil)
        case .late: return ((1.5, 0.30), nil)
        case .skip: return ((2.5, 0.40), (0.75, 0.15))
        }
    }

    private static func breakfastSkipToPrior(_ value: BreakfastSkip) -> (Double, Double) {
        switch value {
        case .never: return (0.05, 0.04)
        case .sometimes: return (0.25, 0.10)
        case .often: return (0.55, 0.12)
        case .always: return (0.85, 0.08)
        }
    }

    private static func lunchSkipToPrior(_ value: LunchSkip) -> (Double, Double) {
        switch value {
        case .never: return (0.04, 0.03)
        case .sometimes: return (0.20, 0.10)
        case .often: return (0.45, 0.12)
        case .always: return (0.75, 0.10)
        }
    }

    private static func mealFrequencyToPrior(_ value: MealFrequency) -> (Double, Double) {
        switch value {
        case .oneTwo: return (1.8, 0.4)
        case .three: return (3.0, 0.4)
        case .fourFive: return (4.5, 0.4)
        case .sixPlus: return (6.5, 0.6)
        }
    }

    private static func mealConsistencyToPrior(_ value: MealConsistency) -> (Double, Double) {
        switch value {
        case .very: return (0.90, 0.05)
        case .fairly: return (0.65, 0.09)
        case .variable: return (0.38, 0.10)
        case .chaotic: return (0.15, 0.08)
        }
    }

    private static func portionSizeToPrior(_ value: PortionSize) -> (Double, Double) {
        switch value {
        case .small: return (0.78, 0.10)
        case .average: return (1.00, 0.10)
        case .generous: return (1.18, 0.10)
        case .large: return (1.35, 0.12)
        }
    }

    private static func dietTypeToPriors(_ value: DietType) -> (cr: (Double, Double), mealSizeMeanDelta: Double?) {
        switch value {
        case .lowCarb: return ((0.86, 0.09), -0.08)
        case .moderate: return ((1.00, 0.10), nil)
        case .highCarb: return ((1.14, 0.10), 0.08)
        case .veryVariable: return ((1.00, 0.18), nil)
        }
    }

    private static func lastMealAdjustment(_ value: LastMealTime) -> (offsetMeanDelta: Double?, offsetStdDelta: Double) {
        switch value {
        case .early: return (nil, -0.1)
        case .normal: return (nil, 0.0)
        case .late: return (nil, 0.2)
        case .veryLate: return (0.5, 0.3)
        }
    }

    private static func stressLevelToPriors(_ value: StressLevel) -> (baseline: (Double, Double), reactivity: (Double, Double)) {
        switch value {
        case .rarely: return ((0.07, 0.05), (0.28, 0.08))
        case .sometimes: return ((0.22, 0.08), (0.48, 0.10))
        case .often: return ((0.48, 0.10), (0.68, 0.10))
        case .always: return ((0.72, 0.09), (0.84, 0.08))
        }
    }

    private static func stressBgEffectDelta(_ value: StressBgEffect) -> Double {
        switch value {
        case .noticeably: return 0.15
        case .little: return 0.05
        case .notReally: return -0.10
        case .unsure: return 0.0
        }
    }

    private static func moodVariabilityToPrior(_ value: MoodVariability) -> (Double, Double) {
        switch value {
        case .stable: return (0.88, 0.06)
        case .some: return (0.68, 0.09)
        case .variable: return (0.42, 0.10)
        case .very: return (0.22, 0.10)
        }
    }

    private static func scheduleTypeAdjustment(_ value: ScheduleType) -> (
        sleepOffsetStdDelta: Double?,
        stressBaselineDelta: Double?,
        sleepRegularityMeanDelta: Double?,
        sleepRegularityStdDelta: Double?,
        mealRegularityStdDelta: Double?
    ) {
        switch value {
        case .regular:
            return (nil, nil, nil, nil, nil)
        case .shift:
            return (0.5, 0.08, -0.12, nil, nil)
        case .variable:
            return (nil, nil, nil, 0.08, 0.08)
        }
    }

    private static func cyclePresenceToPrior(_ value: CyclePresence) -> (Double, Double) {
        switch value {
        case .regular: return (0.50, 0.14)
        case .irregular: return (0.40, 0.18)
        case .no: return (0.0, 0.0)
        }
    }

    private static func cycleBgEffectAdjustment(_ value: CycleBgEffect, current: (Double, Double)) -> (Double, Double) {
        switch value {
        case .noticeably: return (current.0 + 0.28, 0.09)
        case .little: return (current.0 + 0.08, current.1)
        case .notReally: return (current.0 - 0.18, 0.09)
        case .unsure: return current
        }
    }

    private static func cycleHungerToPrior(_ value: CycleHunger) -> (Double, Double) {
        switch value {
        case .noticeably: return (0.14, 0.04)
        case .little: return (0.07, 0.03)
        case .notReally: return (0.02, 0.02)
        }
    }

    private static func cycleMoodToPrior(_ value: CycleMood) -> (Double, Double) {
        switch value {
        case .noticeably: return (0.22, 0.05)
        case .little: return (0.10, 0.04)
        case .notReally: return (0.03, 0.02)
        }
    }

    private static func insulinSensitivityDelta(_ value: InsulinSensitivity) -> Double {
        switch value {
        case .high: return 0.12
        case .normal, .unsure: return 0.0
        case .low: return -0.12
        }
    }

    private static func carbSpikeDelta(_ value: CarbSpike) -> Double {
        switch value {
        case .lot: return 0.10
        case .average, .unsure: return 0.0
        case .notMuch: return -0.10
        }
    }

    private static func aggressivenessValue(_ value: Aggressiveness?) -> Double {
        switch value {
        case .veryCautious: return 0.2
        case .moderate, nil: return 0.5
        case .willing: return 0.7
        case .veryWilling: return 0.85
        }
    }

    private static func complianceValue(_ value: ComplianceLevel?) -> Double {
        switch value {
        case .exact: return 0.05
        case .close: return 0.15
        case .rough: return 0.28
        case .forget: return 0.42
        case nil: return 0.15
        }
    }

    private static func engagementValue(_ value: CheckFrequency?) -> Double {
        switch value {
        case .everyDay: return 0.02
        case .mostDays: return 0.06
        case .sometimes: return 0.14
        case .rarely: return 0.25
        case nil: return 0.06
        }
    }

    private static func trustValue(_ value: TrustLevel?) -> Double {
        switch value {
        case .skeptical: return 0.15
        case .cautious: return 0.35
        case .trusting: return 0.60
        case .veryTrusting: return 0.80
        case nil: return 0.35
        }
    }

    private static func adjust(
        _ prior: (mean: Double, std: Double)?,
        meanDelta: Double,
        stdOverride: Double?
    ) -> (Double, Double) {
        let base = prior ?? (0.0, 0.1)
        return (base.mean + meanDelta, stdOverride ?? base.std)
    }
}
