import SwiftUI
import FirebaseAuth
import FirebaseFirestore

struct ChameliaInsightsSnapshot {
    struct CountBreakdown: Equatable {
        let accepted: Int
        let partial: Int
        let rejected: Int
    }

    struct TrendSeries: Equatable {
        let dates: [String]
        let tirRolling14d: [Double]
        let bgAvgRolling14d: [Double]
        let pctLowRolling14d: [Double]
        let pctHighRolling14d: [Double]
        let moodValenceDaily: [Double?]
        let moodArousalDaily: [Double?]
        let stressAcuteDaily: [Double?]
    }

    struct RecommendationHistoryItem: Identifiable, Equatable {
        let day: Int
        let dateText: String
        let recommendationDate: Date?
        let actionKind: String
        let actionLevel: Int?
        let actionFamily: String?
        let response: String?
        let scheduleChanged: Bool
        let realizedCost: Double?
        let predictedImprovement: Double?
        let predictedOutcomes: PredictedTradeoff?
        let outcomeSummary: OutcomeSummary?
        let storedSegmentSummaries: [String]
        let storedStructureSummaries: [String]

        var id: String { "\(day)-\(dateText)-\(actionKind)" }
    }

    struct PredictedTradeoff: Equatable {
        let deltaTIR: Double
        let deltaPercentLow: Double
        let deltaPercentHigh: Double
        let deltaAverageBG: Double
        let deltaCostMean: Double
        let deltaCVaR: Double?
    }

    struct OutcomeSummary: Equatable {
        let tirDelta: Double
        let percentLowDelta: Double
        let percentHighDelta: Double
        let averageBGDelta: Double
        let costDelta: Double
        let positive: Bool
    }

    struct CalibrationSummary: Equatable {
        let pairedCount: Int
        let tirMAE: Double?
        let percentLowMAE: Double?
        let percentHighMAE: Double?
        let averageBGMAE: Double?
        let costMAE: Double?
        let tirDirectionMatchRate: Double?
        let percentLowDirectionMatchRate: Double?
        let percentHighDirectionMatchRate: Double?
    }

    struct HealthSnapshot: Equatable {
        struct TrendPoint: Identifiable, Equatable {
            let dateLabel: String
            let value: Double

            var id: String { dateLabel }
        }

        let averageBg: Double?
        let tir: Double?
        let percentLow: Double?
        let percentHigh: Double?
        let averageHeartRate: Double?
        let restingHeartRate: Double?
        let exerciseMinutes: Double?
        let activeEnergy: Double?
        let sleepHours: Double?
        let daysSinceSiteChange: Int?
        let bgTrend: [TrendPoint]
        let tirTrend: [TrendPoint]
        let exerciseTrend: [TrendPoint]
        let sleepTrend: [TrendPoint]
        let bgDelta: Double?
        let tirDelta: Double?
        let exerciseDelta: Double?
        let sleepDelta: Double?
    }

    let status: GraduationStatus?
    let recommendationCount: Int
    let graduatedDay: Int?
    let acceptOrPartialRate: Double?
    let realizedPositiveOutcomeRate: Double?
    let counts: CountBreakdown
    let tirMean: Double?
    let tirBaseline14dMean: Double?
    let tirFinal14dMean: Double?
    let tirDeltaBaselineVsFinal14d: Double?
    let pctLowMean: Double?
    let pctHighMean: Double?
    let postGraduationSurfaceDays: Int?
    let postGraduationNoSurfaceDays: Int?
    let topBlockReasons: [(reason: String, count: Int)]
    let trendSeries: TrendSeries?
    let history: [RecommendationHistoryItem]
    let latestProfileName: String?
    let latestTherapyUpdate: Date?
    let lastDecisionReason: String?
    let jepaStatus: String?
    let jepaActiveDays: Int?
    let configuratorModeSummary: String?
    let calibrationSummary: CalibrationSummary?
    let health: HealthSnapshot?

    var isGraduated: Bool {
        status?.graduated == true || graduatedDay != nil
    }

    var recommendationOutcomeSeries: [Double] {
        history.compactMap { $0.outcomeSummary?.tirDelta }
    }

    var recommendationCostSeries: [Double] {
        history.compactMap { item in
            item.outcomeSummary?.costDelta ?? item.realizedCost
        }
    }
}

@MainActor
final class ChameliaInsightsStore: ObservableObject {
    @Published private(set) var snapshot: ChameliaInsightsSnapshot?
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?

    private let db = Firestore.firestore()

    func refresh(userId: String?, fallbackStatus: GraduationStatus? = nil) async {
        guard let userId, !userId.isEmpty else {
            snapshot = nil
            errorMessage = nil
            isLoading = false
            return
        }

        isLoading = true
        defer { isLoading = false }

        do {
            async let reportDocument = db.collection("users")
                .document(userId)
                .collection("sim_reports")
                .document("latest")
                .getDocument()

            async let simLogDocuments = db.collection("users")
                .document(userId)
                .collection("sim_log")
                .document("entries")
                .collection("items")
                .order(by: "day", descending: true)
                .limit(to: 1)
                .getDocuments()

            async let bgAverageDocs = db.collection("users")
                .document(userId)
                .collection("blood_glucose")
                .document("average")
                .collection("items")
                .order(by: "startUtc", descending: true)
                .limit(to: 24 * 30)
                .getDocuments()

            async let bgPercentDocs = db.collection("users")
                .document(userId)
                .collection("blood_glucose")
                .document("percent")
                .collection("items")
                .order(by: "startUtc", descending: true)
                .limit(to: 24 * 30)
                .getDocuments()

            async let hrDailyDocs = db.collection("users")
                .document(userId)
                .collection("heart_rate")
                .document("daily_average")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            async let restingHrDocs = db.collection("users")
                .document(userId)
                .collection("resting_heart_rate")
                .document("daily")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            async let exerciseDailyDocs = db.collection("users")
                .document(userId)
                .collection("exercise")
                .document("daily_average")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            async let energyDailyDocs = db.collection("users")
                .document(userId)
                .collection("energy")
                .document("daily_average")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            async let sleepDailyDocs = db.collection("users")
                .document(userId)
                .collection("sleep")
                .document("daily")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            async let siteChangeDailyDocs = db.collection("users")
                .document(userId)
                .collection("site_changes")
                .document("daily")
                .collection("items")
                .order(by: "dateUtc", descending: true)
                .limit(to: 30)
                .getDocuments()

            let latestTherapySnapshot = try? await TherapySettingsLogManager.shared.getLatestValidTherapySnapshot(limit: 12)
            let (
                reportSnapshot,
                logSnapshot,
                bgAverageSnapshot,
                bgPercentSnapshot,
                hrDailySnapshot,
                restingHrSnapshot,
                exerciseDailySnapshot,
                energyDailySnapshot,
                sleepDailySnapshot,
                siteChangeDailySnapshot
            ) = try await (
                reportDocument,
                simLogDocuments,
                bgAverageDocs,
                bgPercentDocs,
                hrDailyDocs,
                restingHrDocs,
                exerciseDailyDocs,
                energyDailyDocs,
                sleepDailyDocs,
                siteChangeDailyDocs
            )

            let parsed = try parseSnapshot(
                reportData: reportSnapshot.data(),
                latestLogData: logSnapshot.documents.first?.data(),
                fallbackStatus: fallbackStatus,
                latestTherapySnapshot: latestTherapySnapshot,
                healthSnapshot: parseHealthSnapshot(
                    bgAverageDocs: bgAverageSnapshot.documents,
                    bgPercentDocs: bgPercentSnapshot.documents,
                    hrDailyDocs: hrDailySnapshot.documents,
                    restingHrDocs: restingHrSnapshot.documents,
                    exerciseDailyDocs: exerciseDailySnapshot.documents,
                    energyDailyDocs: energyDailySnapshot.documents,
                    sleepDailyDocs: sleepDailySnapshot.documents,
                    siteChangeDailyDocs: siteChangeDailySnapshot.documents
                )
            )

            snapshot = parsed
            errorMessage = nil
        } catch {
            snapshot = nil
            errorMessage = readableMessage(for: error)
            print("[ChameliaInsights] failed to load insights for \(userId): \(error)")
        }
    }

    private func parseSnapshot(
        reportData: [String: Any]?,
        latestLogData: [String: Any]?,
        fallbackStatus: GraduationStatus?,
        latestTherapySnapshot: TherapySnapshot?,
        healthSnapshot: ChameliaInsightsSnapshot.HealthSnapshot?
    ) throws -> ChameliaInsightsSnapshot {
        guard let reportData else {
            if let fallbackStatus {
                return ChameliaInsightsSnapshot(
                    status: fallbackStatus,
                    recommendationCount: 0,
                    graduatedDay: nil,
                    acceptOrPartialRate: nil,
                    realizedPositiveOutcomeRate: nil,
                    counts: .init(accepted: 0, partial: 0, rejected: 0),
                    tirMean: nil,
                    tirBaseline14dMean: nil,
                    tirFinal14dMean: nil,
                    tirDeltaBaselineVsFinal14d: nil,
                    pctLowMean: nil,
                    pctHighMean: nil,
                    postGraduationSurfaceDays: nil,
                    postGraduationNoSurfaceDays: nil,
                    topBlockReasons: [],
                    trendSeries: nil,
                    history: [],
                    latestProfileName: latestTherapySnapshot?.profileName,
                    latestTherapyUpdate: latestTherapySnapshot?.timestamp,
                    lastDecisionReason: latestLogData.flatMap { decisionReason(from: $0) },
                    jepaStatus: fallbackStatus.beliefMode,
                    jepaActiveDays: nil,
                    configuratorModeSummary: fallbackStatus.configuratorMode,
                    calibrationSummary: nil,
                    health: healthSnapshot
                )
            }
            throw NSError(domain: "ChameliaInsights", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "No mature-account Chamelia report exists yet for this user."
            ])
        }

        let finalStatus = graduationStatus(from: reportData["final_status"] as? [String: Any]) ?? fallbackStatus
        let trendSeries = trendSeries(from: reportData["trend_series"] as? [String: Any])
        let outcomesByDay = outcomeTimeline(from: reportData["realized_outcome_timeline"] as? [[String: Any]]).mapToDictionary()
        let history = recommendationHistory(
            from: reportData["recommendation_timeline"] as? [[String: Any]],
            outcomesByDay: outcomesByDay
        )

        return ChameliaInsightsSnapshot(
            status: finalStatus,
            recommendationCount: intValue(reportData["recommendation_count"]) ?? history.count,
            graduatedDay: intValue(reportData["graduated_day"]),
            acceptOrPartialRate: doubleValue(reportData["accept_or_partial_rate"]),
            realizedPositiveOutcomeRate: doubleValue(reportData["realized_positive_outcome_rate"]),
            counts: .init(
                accepted: intValue(reportData["accepted_count"]) ?? 0,
                partial: intValue(reportData["partial_count"]) ?? 0,
                rejected: intValue(reportData["rejected_count"]) ?? 0
            ),
            tirMean: doubleValue(reportData["tir_mean"]),
            tirBaseline14dMean: doubleValue(reportData["tir_baseline_14d_mean"]),
            tirFinal14dMean: doubleValue(reportData["tir_final_14d_mean"]),
            tirDeltaBaselineVsFinal14d: doubleValue(reportData["tir_delta_baseline_vs_final_14d"]),
            pctLowMean: doubleValue(reportData["pct_low_mean"]),
            pctHighMean: doubleValue(reportData["pct_high_mean"]),
            postGraduationSurfaceDays: intValue(reportData["post_graduation_surface_days"]),
            postGraduationNoSurfaceDays: intValue(reportData["post_graduation_no_surface_days"]),
            topBlockReasons: sortedCounts(from: reportData["block_reasons"]).prefix(3).map { $0 },
            trendSeries: trendSeries,
            history: history,
            latestProfileName: latestTherapySnapshot?.profileName,
            latestTherapyUpdate: latestTherapySnapshot?.timestamp,
            lastDecisionReason: latestLogData.flatMap { decisionReason(from: $0) },
            jepaStatus: stringValue(reportData["jepa_status"]) ?? finalStatus?.beliefMode,
            jepaActiveDays: intValue(reportData["jepa_active_days"]),
            configuratorModeSummary: configuratorModeSummary(from: reportData["configurator_mode_counts"]) ?? finalStatus?.configuratorMode,
            calibrationSummary: calibrationSummary(from: reportData["calibration_summary"] as? [String: Any]),
            health: healthSnapshot
        )
    }

    private func recommendationHistory(
        from raw: [[String: Any]]?,
        outcomesByDay: [Int: ChameliaInsightsSnapshot.OutcomeSummary]
    ) -> [ChameliaInsightsSnapshot.RecommendationHistoryItem] {
        (raw ?? []).compactMap { item in
            guard let day = intValue(item["day"]) else { return nil }
            return .init(
                day: day,
                dateText: stringValue(item["date"]) ?? "Day \(day)",
                recommendationDate: parseRecommendationDate(from: item["date"]),
                actionKind: stringValue(item["action_kind"]) ?? "Recommendation",
                actionLevel: intValue(item["action_level"]),
                actionFamily: stringValue(item["action_family"]),
                response: stringValue(item["patient_response"]),
                scheduleChanged: boolValue(item["schedule_changed"]) ?? false,
                realizedCost: doubleValue(item["realized_cost"]),
                predictedImprovement: doubleValue(item["predicted_improvement"]),
                predictedOutcomes: predictedTradeoff(from: item["predicted_outcomes"] as? [String: Any]),
                outcomeSummary: outcomesByDay[day],
                storedSegmentSummaries: stringArray(item["segment_summaries"]),
                storedStructureSummaries: stringArray(item["structure_summaries"])
            )
        }
        .sorted { $0.day > $1.day }
    }

    private func outcomeTimeline(from raw: [[String: Any]]?) -> [ChameliaInsightsSnapshot.OutcomeSummaryWithDay] {
        (raw ?? []).compactMap { item in
            guard let day = intValue(item["day"]) else { return nil }
            return .init(
                day: day,
                tirDelta: doubleValue(item["tir_delta"]) ?? 0,
                percentLowDelta: doubleValue(item["pct_low_delta"]) ?? 0,
                percentHighDelta: doubleValue(item["pct_high_delta"]) ?? 0,
                averageBGDelta: doubleValue(item["bg_avg_delta"]) ?? 0,
                costDelta: doubleValue(item["cost_delta"]) ?? 0,
                positive: boolValue(item["positive"]) ?? false
            )
        }
    }

    private func predictedTradeoff(from raw: [String: Any]?) -> ChameliaInsightsSnapshot.PredictedTradeoff? {
        guard let raw else { return nil }
        return .init(
            deltaTIR: doubleValue(raw["delta_tir"]) ?? 0,
            deltaPercentLow: doubleValue(raw["delta_pct_low"]) ?? 0,
            deltaPercentHigh: doubleValue(raw["delta_pct_high"]) ?? 0,
            deltaAverageBG: doubleValue(raw["delta_bg_avg"]) ?? 0,
            deltaCostMean: doubleValue(raw["delta_cost_mean"]) ?? 0,
            deltaCVaR: doubleValue(raw["delta_cvar"])
        )
    }

    private func trendSeries(from raw: [String: Any]?) -> ChameliaInsightsSnapshot.TrendSeries? {
        guard let raw else { return nil }
        return .init(
            dates: stringArray(raw["dates"]),
            tirRolling14d: doubleArray(raw["tir_rolling_14d"]),
            bgAvgRolling14d: doubleArray(raw["bg_avg_rolling_14d"]),
            pctLowRolling14d: doubleArray(raw["pct_low_rolling_14d"]),
            pctHighRolling14d: doubleArray(raw["pct_high_rolling_14d"]),
            moodValenceDaily: optionalDoubleArray(raw["mood_valence_daily"]),
            moodArousalDaily: optionalDoubleArray(raw["mood_arousal_daily"]),
            stressAcuteDaily: optionalDoubleArray(raw["stress_acute_daily"])
        )
    }

    private func calibrationSummary(from raw: [String: Any]?) -> ChameliaInsightsSnapshot.CalibrationSummary? {
        guard let raw else { return nil }
        return .init(
            pairedCount: intValue(raw["paired_count"]) ?? 0,
            tirMAE: doubleValue(raw["tir_mae"]),
            percentLowMAE: doubleValue(raw["pct_low_mae"]),
            percentHighMAE: doubleValue(raw["pct_high_mae"]),
            averageBGMAE: doubleValue(raw["bg_avg_mae"]),
            costMAE: doubleValue(raw["cost_mae"]),
            tirDirectionMatchRate: doubleValue(raw["tir_direction_match_rate"]),
            percentLowDirectionMatchRate: doubleValue(raw["pct_low_direction_match_rate"]),
            percentHighDirectionMatchRate: doubleValue(raw["pct_high_direction_match_rate"])
        )
    }

    private func parseHealthSnapshot(
        bgAverageDocs: [QueryDocumentSnapshot],
        bgPercentDocs: [QueryDocumentSnapshot],
        hrDailyDocs: [QueryDocumentSnapshot],
        restingHrDocs: [QueryDocumentSnapshot],
        exerciseDailyDocs: [QueryDocumentSnapshot],
        energyDailyDocs: [QueryDocumentSnapshot],
        sleepDailyDocs: [QueryDocumentSnapshot],
        siteChangeDailyDocs: [QueryDocumentSnapshot]
    ) -> ChameliaInsightsSnapshot.HealthSnapshot? {
        let bgAverageByDay = aggregateDailySeries(from: bgAverageDocs, key: "startUtc") { data in
            doubleValue(data["averageBg"])
        }
        let bgPercentByDay = aggregateDailyPairSeries(from: bgPercentDocs, key: "startUtc") { data in
            (
                doubleValue(data["percentLow"]),
                doubleValue(data["percentHigh"])
            )
        }
        let hrByDay = scalarDailySeries(from: hrDailyDocs, dateKey: "dateUtc", valueKey: "averageHeartRate")
        let restingByDay = scalarDailySeries(from: restingHrDocs, dateKey: "dateUtc", valueKey: "restingHeartRate")
        let exerciseByDay = scalarDailySeries(from: exerciseDailyDocs, dateKey: "dateUtc", valueKey: "averageExerciseMinutes")
        let energyByDay = scalarDailySeries(from: energyDailyDocs, dateKey: "dateUtc", valueKey: "averageActiveEnergy")
        let sleepByDay = scalarDailySeries(from: sleepDailyDocs, dateKey: "dateUtc", extractor: sleepHours(from:))

        let lowByDay = bgPercentByDay.mapValues(\.low)
        let highByDay = bgPercentByDay.mapValues(\.high)
        let tirByDay = Dictionary(uniqueKeysWithValues: bgPercentByDay.map { key, value in
            (key, max(0, 1 - value.low - value.high))
        })
        let latestSiteAge = siteChangeDailyDocs
            .compactMap { intValue($0.data()["daysSinceChange"]) }
            .first

        if bgAverageByDay.isEmpty,
           tirByDay.isEmpty,
           hrByDay.isEmpty,
           restingByDay.isEmpty,
           exerciseByDay.isEmpty,
           energyByDay.isEmpty,
           sleepByDay.isEmpty,
           latestSiteAge == nil {
            return nil
        }

        return .init(
            averageBg: averageOfRecent(bgAverageByDay, days: 7),
            tir: averageOfRecent(tirByDay, days: 7),
            percentLow: averageOfRecent(lowByDay, days: 7),
            percentHigh: averageOfRecent(highByDay, days: 7),
            averageHeartRate: averageOfRecent(hrByDay, days: 7),
            restingHeartRate: averageOfRecent(restingByDay, days: 7),
            exerciseMinutes: averageOfRecent(exerciseByDay, days: 7),
            activeEnergy: averageOfRecent(energyByDay, days: 7),
            sleepHours: averageOfRecent(sleepByDay, days: 7),
            daysSinceSiteChange: latestSiteAge,
            bgTrend: trendPoints(from: bgAverageByDay),
            tirTrend: trendPoints(from: tirByDay),
            exerciseTrend: trendPoints(from: exerciseByDay),
            sleepTrend: trendPoints(from: sleepByDay),
            bgDelta: compareRecentWindow(bgAverageByDay, days: 7),
            tirDelta: compareRecentWindow(tirByDay, days: 7),
            exerciseDelta: compareRecentWindow(exerciseByDay, days: 7),
            sleepDelta: compareRecentWindow(sleepByDay, days: 7)
        )
    }

    private func graduationStatus(from raw: [String: Any]?) -> GraduationStatus? {
        guard let raw else { return nil }
        return GraduationStatus(
            graduated: boolValue(raw["graduated"]) ?? false,
            nDays: intValue(raw["n_days"]) ?? 0,
            winRate: doubleValue(raw["win_rate"]) ?? 0,
            safetyViolations: intValue(raw["safety_violations"]) ?? 0,
            consecutiveDays: intValue(raw["consecutive_days"]) ?? 0,
            beliefEntropy: doubleValue(raw["belief_entropy"]),
            familiarity: doubleValue(raw["familiarity"]),
            concordance: doubleValue(raw["concordance"]),
            calibration: doubleValue(raw["calibration"]),
            trustLevel: doubleValue(raw["trust_level"]),
            burnoutLevel: doubleValue(raw["burnout_level"]),
            noSurfaceStreak: intValue(raw["no_surface_streak"]),
            beliefMode: stringValue(raw["belief_mode"]),
            jepaActive: boolValue(raw["jepa_active"]),
            configuratorMode: stringValue(raw["configurator_mode"]),
            lastDecisionReason: stringValue(raw["last_decision_reason"])
        )
    }

    private func scalarDailySeries(
        from docs: [QueryDocumentSnapshot],
        dateKey: String,
        valueKey: String
    ) -> [String: Double] {
        scalarDailySeries(from: docs, dateKey: dateKey) { data in
            doubleValue(data[valueKey])
        }
    }

    private func sleepHours(from data: [String: Any]) -> Double? {
        let core = doubleValue(data["asleepCore"]) ?? 0
        let deep = doubleValue(data["asleepDeep"]) ?? 0
        let rem = doubleValue(data["asleepREM"]) ?? 0
        let unspecified = doubleValue(data["asleepUnspecified"]) ?? 0
        let totalMinutes = core + deep + rem + unspecified
        return totalMinutes / 60.0
    }

    private func scalarDailySeries(
        from docs: [QueryDocumentSnapshot],
        dateKey: String,
        extractor: ([String: Any]) -> Double?
    ) -> [String: Double] {
        var values: [String: Double] = [:]
        for doc in docs {
            let data = doc.data()
            guard let dayId = dayId(from: data[dateKey] as Any?) else { continue }
            guard let value = extractor(data) else { continue }
            values[dayId] = value
        }
        return values
    }

    private func aggregateDailySeries(
        from docs: [QueryDocumentSnapshot],
        key: String,
        extractor: ([String: Any]) -> Double?
    ) -> [String: Double] {
        var buckets: [String: [Double]] = [:]
        for doc in docs {
            let data = doc.data()
            guard let dayId = dayId(from: data[key] as Any?) else { continue }
            guard let value = extractor(data) else { continue }
            buckets[dayId, default: []].append(value)
        }
        return Dictionary(uniqueKeysWithValues: buckets.map { key, values in
            (key, values.reduce(0, +) / Double(values.count))
        })
    }

    private func aggregateDailyPairSeries(
        from docs: [QueryDocumentSnapshot],
        key: String,
        extractor: ([String: Any]) -> (Double?, Double?)
    ) -> [String: (low: Double, high: Double)] {
        var lowBuckets: [String: [Double]] = [:]
        var highBuckets: [String: [Double]] = [:]
        for doc in docs {
            let data = doc.data()
            guard let dayId = dayId(from: data[key] as Any?) else { continue }
            let pair = extractor(data)
            if let low = pair.0 {
                lowBuckets[dayId, default: []].append(low)
            }
            if let high = pair.1 {
                highBuckets[dayId, default: []].append(high)
            }
        }
        let allKeys = Set(lowBuckets.keys).union(highBuckets.keys)
        return Dictionary(uniqueKeysWithValues: allKeys.map { key in
            let low = (lowBuckets[key].map { $0.reduce(0, +) / Double($0.count) } ?? 0) / 100.0
            let high = (highBuckets[key].map { $0.reduce(0, +) / Double($0.count) } ?? 0) / 100.0
            return (key, (low, high))
        })
    }

    private func dayId(from raw: Any?) -> String? {
        if let string = raw as? String {
            if string.count >= 10 {
                return String(string.prefix(10))
            }
            return string
        }
        return nil
    }

    private func sortedSeries(_ values: [String: Double]) -> [(String, Double)] {
        values.sorted { $0.key < $1.key }
    }

    private func trendPoints(from values: [String: Double], limit: Int = 21) -> [ChameliaInsightsSnapshot.HealthSnapshot.TrendPoint] {
        sortedSeries(values).suffix(limit).map { dayId, value in
            .init(dateLabel: dayId, value: value)
        }
    }

    private func averageOfRecent(_ values: [String: Double], days: Int) -> Double? {
        let recent = Array(sortedSeries(values).suffix(days)).map(\.1)
        guard !recent.isEmpty else { return nil }
        return recent.reduce(0, +) / Double(recent.count)
    }

    private func compareRecentWindow(_ values: [String: Double], days: Int) -> Double? {
        let ordered = sortedSeries(values).map(\.1)
        guard ordered.count >= days + 1 else { return nil }
        let recent = Array(ordered.suffix(days))
        let prior = Array(ordered.dropLast(days).suffix(days))
        guard !recent.isEmpty, !prior.isEmpty else { return nil }
        let recentMean = recent.reduce(0, +) / Double(recent.count)
        let priorMean = prior.reduce(0, +) / Double(prior.count)
        return recentMean - priorMean
    }

    private func decisionReason(from raw: [String: Any]) -> String? {
        stringValue(raw["last_decision_reason"])
            ?? stringValue(raw["decision_block_reason"])
            ?? {
                if raw["recommendation"] != nil { return "Recommendation surfaced" }
                return nil
            }()
    }

    private func configuratorModeSummary(from raw: Any?) -> String? {
        guard let rawCounts = raw as? [String: Any] else { return nil }
        let sorted = rawCounts.compactMap { key, value -> (String, Int)? in
            guard let count = intValue(value), count > 0 else { return nil }
            return (key, count)
        }
        .sorted { $0.1 > $1.1 }

        guard let first = sorted.first else { return nil }
        return "\(first.0.replacingOccurrences(of: "_", with: " ").capitalized) · \(first.1)d"
    }

    private func sortedCounts(from raw: Any?) -> [(reason: String, count: Int)] {
        guard let rawCounts = raw as? [String: Any] else { return [] }
        return rawCounts.compactMap { key, value -> (String, Int)? in
            guard let count = intValue(value), count > 0 else { return nil }
            return (key.replacingOccurrences(of: "_", with: " ").capitalized, count)
        }
        .sorted { $0.1 > $1.1 }
    }

    private func stringArray(_ raw: Any?) -> [String] {
        (raw as? [Any])?.compactMap(stringValue) ?? []
    }

    private func parseRecommendationDate(from raw: Any?) -> Date? {
        guard let text = stringValue(raw) else { return nil }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = iso.date(from: text) {
            return date
        }

        let fallbackIso = ISO8601DateFormatter()
        if let date = fallbackIso.date(from: text) {
            return date
        }

        let formats = [
            "yyyy-MM-dd",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm:ss",
            "MMM d, yyyy",
            "M/d/yyyy"
        ]
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        for format in formats {
            formatter.dateFormat = format
            if let date = formatter.date(from: text) {
                return date
            }
        }
        return nil
    }

    private func doubleArray(_ raw: Any?) -> [Double] {
        (raw as? [Any])?.compactMap(doubleValue) ?? []
    }

    private func optionalDoubleArray(_ raw: Any?) -> [Double?] {
        (raw as? [Any])?.map { value in
            if value is NSNull { return nil }
            return doubleValue(value)
        } ?? []
    }

    private func intValue(_ raw: Any?) -> Int? {
        if let int = raw as? Int { return int }
        if let number = raw as? NSNumber { return number.intValue }
        if let string = raw as? String { return Int(string) }
        return nil
    }

    private func doubleValue(_ raw: Any?) -> Double? {
        if let double = raw as? Double { return double }
        if let number = raw as? NSNumber { return number.doubleValue }
        if let string = raw as? String { return Double(string) }
        return nil
    }

    private func boolValue(_ raw: Any?) -> Bool? {
        if let bool = raw as? Bool { return bool }
        if let number = raw as? NSNumber { return number.boolValue }
        if let string = raw as? String { return Bool(string) }
        return nil
    }

    private func stringValue(_ raw: Any?) -> String? {
        if let string = raw as? String, !string.isEmpty { return string }
        if let number = raw as? NSNumber { return number.stringValue }
        return nil
    }

    private func readableMessage(for error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}

private extension ChameliaInsightsSnapshot {
    struct OutcomeSummaryWithDay {
        let day: Int
        let tirDelta: Double
        let percentLowDelta: Double
        let percentHighDelta: Double
        let averageBGDelta: Double
        let costDelta: Double
        let positive: Bool

        var asOutcome: OutcomeSummary {
            .init(
                tirDelta: tirDelta,
                percentLowDelta: percentLowDelta,
                percentHighDelta: percentHighDelta,
                averageBGDelta: averageBGDelta,
                costDelta: costDelta,
                positive: positive
            )
        }
    }
}

private extension Array where Element == ChameliaInsightsSnapshot.OutcomeSummaryWithDay {
    func mapToDictionary() -> [Int: ChameliaInsightsSnapshot.OutcomeSummary] {
        Dictionary(uniqueKeysWithValues: map { ($0.day, $0.asOutcome) })
    }
}

struct ChameliaInsightsEntryCard: View {
    let snapshot: ChameliaInsightsSnapshot?
    let fallbackStatus: GraduationStatus?
    let accent: Color
    let isLoading: Bool
    let errorMessage: String?

    private var status: GraduationStatus? {
        snapshot?.status ?? fallbackStatus
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(status?.graduated == true ? "Chamelia is live" : "Chamelia progress")
                        .font(.title3.weight(.bold))
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                statusBadge
            }

            if let snapshot {
                HStack(spacing: 10) {
                    insightPill(title: "Recs", value: "\(snapshot.recommendationCount)")
                    insightPill(title: "Accept+", value: snapshot.acceptOrPartialRate.percentString)
                    insightPill(title: "TIR Δ", value: snapshot.tirDeltaBaselineVsFinal14d.signedPercentString)
                }
            } else if let status {
                HStack(spacing: 10) {
                    insightPill(title: "Days", value: "\(status.nDays)")
                    insightPill(title: "Win", value: (status.winRate).percentString)
                    insightPill(title: "Streak", value: "\(status.consecutiveDays)")
                }
            }

            if let message = errorMessage, !message.isEmpty {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .stroke(accent.opacity(0.12), lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if isLoading {
                ProgressView()
                    .padding(16)
                    .tint(accent)
            }
        }
    }

    private var subtitle: String {
        if let snapshot, snapshot.isGraduated {
            return "Recommendations, outcomes, and health trends are synced to this account."
        }
        if let status, status.graduated {
            return "This account has already graduated out of pure shadow mode."
        }
        return "See shadow progress, recommendation history, and recent health trends."
    }

    private var statusBadge: some View {
        Text((status?.graduated ?? false) ? "Live" : "Shadow")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(((status?.graduated ?? false) ? Color.green : accent).opacity(0.16), in: Capsule())
            .foregroundStyle((status?.graduated ?? false) ? Color.green : accent)
    }

    private func insightPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct ChameliaStatusSummaryCard: View {
    let status: GraduationStatus
    let accent: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(status.graduated ? "Chamelia live" : "Shadow progress", systemImage: status.graduated ? "bolt.heart.fill" : "hourglass.bottomhalf.filled")
                    .font(.headline)
                    .foregroundStyle(status.graduated ? Color.green : accent)
                Spacer()
                Text(status.graduated ? "Live" : "Shadow")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background((status.graduated ? Color.green : accent).opacity(0.16), in: Capsule())
                    .foregroundStyle(status.graduated ? Color.green : accent)
            }

            if status.graduated {
                Text("Chamelia is actively evaluating recommendations on top of your current schedule.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                Text("Chamelia is still building evidence before surfacing recommendations.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                metricChip(title: "Observed days", value: "\(status.nDays)")
                metricChip(title: "Win rate", value: status.winRate.percentString)
                metricChip(title: "Streak", value: "\(status.consecutiveDays)")
                metricChip(title: "Safety", value: "\(status.safetyViolations)")
            }
        }
        .insightCardStyle()
    }

    private func metricChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct ChameliaInsightsView: View {
    @ObservedObject var store: ChameliaInsightsStore
    @EnvironmentObject private var themeManager: ThemeManager
    @ObservedObject private var siteChange = SiteChangeData.shared
    @State private var selectedChart: ChartDetailPayload?
    @State private var selectedRecommendation: ChameliaInsightsSnapshot.RecommendationHistoryItem?

    private var accent: Color { themeManager.theme.accent }

    var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme)
                .ignoresSafeArea()

            if store.isLoading && store.snapshot == nil {
                ProgressView("Loading Chamelia insights…")
                    .tint(accent)
            } else if let errorMessage = store.errorMessage, store.snapshot == nil {
                ScrollView {
                    VStack(spacing: 16) {
                        errorCard(message: errorMessage)
                    }
                    .padding(16)
                }
            } else if let snapshot = store.snapshot {
                ScrollView {
                    VStack(spacing: 18) {
                        headerCard(snapshot: snapshot)
                        healthCard(snapshot: snapshot)
                        operationalCard(snapshot: snapshot)
                        recommendationHistoryCard(snapshot: snapshot)
                    }
                    .padding(16)
                }
                .refreshable {
                    await store.refresh(userId: Auth.auth().currentUser?.uid, fallbackStatus: ChameliaDashboardStore.shared.state.status)
                }
            } else {
                ScrollView {
                    VStack(spacing: 16) {
                        emptyCard
                    }
                    .padding(16)
                }
            }
        }
        .navigationTitle("Chamelia & Health")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedChart) { payload in
            ChartDetailView(payload: payload)
        }
        .sheet(item: $selectedRecommendation) { item in
            RecommendationDetailSheet(item: item, accent: accent)
        }
    }

    private func headerCard(snapshot: ChameliaInsightsSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(snapshot.isGraduated ? "Chamelia is live" : "Chamelia is learning")
                        .font(.system(.title2, design: .rounded).weight(.bold))
                    Text(snapshot.isGraduated
                         ? "This account has already graduated out of shadow mode. The recommendation engine is active."
                         : "This account is still accumulating shadow evidence before surfacing recommendations.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(snapshot.isGraduated ? "Live" : "Shadow")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background((snapshot.isGraduated ? Color.green : accent).opacity(0.16), in: Capsule())
                    .foregroundStyle(snapshot.isGraduated ? Color.green : accent)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                dashboardMetric(title: "Recommendations", value: "\(snapshot.recommendationCount)", tint: accent)
                dashboardMetric(title: "Accept + partial", value: snapshot.acceptOrPartialRate.percentString, tint: .green)
                dashboardMetric(title: "Positive outcomes", value: snapshot.realizedPositiveOutcomeRate.percentString, tint: .green)
                dashboardMetric(title: "Graduated day", value: snapshot.graduatedDay.map(String.init) ?? "—", tint: accent)
                dashboardMetric(title: "Observed days", value: snapshot.status.map { "\($0.nDays)" } ?? "—", tint: accent)
                dashboardMetric(title: "Win rate", value: snapshot.status?.winRate.percentString ?? "—", tint: snapshot.status?.winRate ?? 0 >= 0.6 ? .green : .orange)
                dashboardMetric(title: "Safety violations", value: snapshot.status.map { "\($0.safetyViolations)" } ?? "—", tint: (snapshot.status?.safetyViolations ?? 0) == 0 ? .green : .red)
                dashboardMetric(title: "Consecutive good days", value: snapshot.status.map { "\($0.consecutiveDays)" } ?? "—", tint: accent)
            }

            if let winRate = snapshot.status?.winRate {
                explanatoryCallout(
                    title: "What win rate means",
                    text: "Win rate is Chamelia's shadow scorecard hit rate, not direct clinical success. It measures how often Chamelia's counterfactual judgment counted as a win while learning.",
                    emphasis: winRate >= 0.6 ? .good : .neutral
                )
            }

            if let calibrationSummary = snapshot.calibrationSummary, calibrationSummary.pairedCount > 0 {
                explanatoryCallout(
                    title: "Prediction calibration",
                    text: "Across \(calibrationSummary.pairedCount) followed recommendations, TIR direction matched realized outcomes \(calibrationSummary.tirDirectionMatchRate.percentString) of the time and % low direction matched \(calibrationSummary.percentLowDirectionMatchRate.percentString).",
                    emphasis: (calibrationSummary.tirDirectionMatchRate ?? 0) >= 0.6 ? .good : .neutral
                )
            }

            HStack(spacing: 10) {
                infoTag(title: "Belief", value: snapshot.jepaStatus ?? snapshot.status?.beliefMode ?? "Unknown")
                if let configuratorModeSummary = snapshot.configuratorModeSummary {
                    infoTag(title: "Mode", value: configuratorModeSummary)
                }
                if let jepaActiveDays = snapshot.jepaActiveDays, jepaActiveDays > 0 {
                    infoTag(title: "JEPA", value: "\(jepaActiveDays)d active")
                }
            }

            if let status = snapshot.status {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    if let familiarity = status.familiarity {
                        infoTag(title: "Familiarity", value: familiarity.percentString)
                    }
                    if let concordance = status.concordance {
                        infoTag(title: "Concordance", value: concordance.percentString)
                    }
                    if let calibration = status.calibration {
                        infoTag(title: "Calibration", value: calibration.percentString)
                    }
                    if let trustLevel = status.trustLevel {
                        infoTag(title: "Trust", value: trustLevel.percentString)
                    }
                    if let burnoutLevel = status.burnoutLevel {
                        infoTag(title: "Burnout", value: burnoutLevel.percentString)
                    }
                    if let noSurfaceStreak = status.noSurfaceStreak, noSurfaceStreak > 0 {
                        infoTag(title: "No-surface", value: "\(noSurfaceStreak)d")
                    }
                }
            }

            if let calibrationSummary = snapshot.calibrationSummary, calibrationSummary.pairedCount > 0 {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    if let tirMAE = calibrationSummary.tirMAE {
                        infoTag(title: "TIR MAE", value: tirMAE.percentString)
                    }
                    if let percentLowMAE = calibrationSummary.percentLowMAE {
                        infoTag(title: "% Low MAE", value: percentLowMAE.percentString)
                    }
                    if let percentHighMAE = calibrationSummary.percentHighMAE {
                        infoTag(title: "% High MAE", value: percentHighMAE.percentString)
                    }
                }
            }

            if let lastDecisionReason = snapshot.lastDecisionReason, !lastDecisionReason.isEmpty {
                Label(lastDecisionReason, systemImage: "text.bubble.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .insightCardStyle()
    }

    private func healthCard(snapshot: ChameliaInsightsSnapshot) -> some View {
        NavigationLink {
            HealthStatsDetailView(snapshot: snapshot, accent: accent)
        } label: {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Health trends")
                        .font(.headline)
                    Spacer()
                    Text("Firebase")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(accent.opacity(0.14), in: Capsule())
                        .foregroundStyle(accent)
                }

                if let health = snapshot.health {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        trendMetricCard(
                            title: "TIR",
                            current: health.tir,
                            delta: health.tirDelta,
                            data: health.tirTrend.map(\.value),
                            tint: .green,
                            format: .percent
                        )
                        trendMetricCard(
                            title: "Average BG",
                            current: health.averageBg,
                            delta: health.bgDelta,
                            data: health.bgTrend.map(\.value),
                            tint: accent,
                            format: .number
                        )
                    }

                    HStack(spacing: 10) {
                        infoTag(title: "Sleep", value: health.sleepHours.map { "\($0.formatted(.number.precision(.fractionLength(1))))h" } ?? "—")
                        infoTag(title: "Exercise", value: health.exerciseMinutes.map { "\($0.formatted(.number.precision(.fractionLength(0))))m" } ?? "—")
                        infoTag(title: "Site age", value: health.daysSinceSiteChange.map { "\($0)d" } ?? "—")
                    }
                } else {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        trendMetricCard(
                            title: "TIR",
                            current: snapshot.trendSeries?.tirRolling14d.last ?? snapshot.tirFinal14dMean,
                            delta: rollingDelta(snapshot.trendSeries?.tirRolling14d),
                            data: snapshot.trendSeries?.tirRolling14d,
                            tint: .green,
                            format: .percent
                        )
                        trendMetricCard(
                            title: "Average BG",
                            current: snapshot.trendSeries?.bgAvgRolling14d.last,
                            delta: rollingDelta(snapshot.trendSeries?.bgAvgRolling14d),
                            data: snapshot.trendSeries?.bgAvgRolling14d,
                            tint: accent,
                            format: .number
                        )
                    }
                }

                Text("Open the full health screen for Firebase-backed glucose, activity, sleep, and site-change metrics.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .insightCardStyle()
        }
        .buttonStyle(.plain)
    }

    private func operationalCard(snapshot: ChameliaInsightsSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Operational context")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                dashboardMetric(title: "Current profile", value: snapshot.latestProfileName ?? "—", tint: accent)
                dashboardMetric(title: "Last therapy update", value: snapshot.latestTherapyUpdate.map(relativeDateString) ?? "—", tint: accent)
                dashboardMetric(title: "Site age", value: "\(siteChange.daysSinceSiteChange) days", tint: accent)
                dashboardMetric(title: "Post-grad surfaced", value: snapshot.postGraduationSurfaceDays.map(String.init) ?? "—", tint: .green)
                dashboardMetric(title: "Post-grad withheld", value: snapshot.postGraduationNoSurfaceDays.map(String.init) ?? "—", tint: .secondary)
                dashboardMetric(title: "Recent block reason", value: snapshot.topBlockReasons.first?.reason ?? "None", tint: .orange)
            }
        }
        .insightCardStyle()
    }

    private func recommendationHistoryCard(snapshot: ChameliaInsightsSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Recommendation history")
                .font(.headline)

            explanatoryCallout(
                title: "How to read realized cost",
                text: "Realized cost is Chamelia's penalty score after the recommendation window. Lower is better. A negative cost delta means the outcome improved versus the period before the recommendation.",
                emphasis: .neutral
            )

            if recommendationOutcomeSectionAvailable(snapshot: snapshot) {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    if let payload = tirHistoryPayload(snapshot: snapshot) {
                        expandableInsightChartCard(
                            payload: payload,
                            current: payload.values.last,
                            delta: payload.values.count >= 2 ? payload.values.last.map { $0 - payload.values.first! } : nil,
                            tint: .green
                        )
                    }

                    if let payload = recommendationOutcomePayload(snapshot: snapshot) {
                        expandableInsightChartCard(
                            payload: payload,
                            current: payload.values.last,
                            delta: payload.values.count >= 2 ? payload.values.last.map { $0 - payload.values.first! } : nil,
                            tint: accent
                        )
                    }
                }
            }

            if snapshot.history.isEmpty {
                Text("No surfaced recommendations are stored for this account yet.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(snapshot.history.prefix(10)) { item in
                    Button {
                        selectedRecommendation = item
                    } label: {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.actionKind.replacingOccurrences(of: "_", with: " ").capitalized)
                                        .font(.subheadline.weight(.semibold))
                                    Text(item.dateText)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(item.response?.capitalized ?? "Pending")
                                    .font(.caption.weight(.semibold))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(responseTint(for: item).opacity(0.14), in: Capsule())
                                    .foregroundStyle(responseTint(for: item))
                            }

                            HStack(spacing: 8) {
                                infoTag(title: "Level", value: item.actionLevel.map(String.init) ?? "—")
                                if let family = item.actionFamily {
                                    infoTag(title: "Family", value: family.replacingOccurrences(of: "_", with: " ").capitalized)
                                }
                                infoTag(title: "Changed", value: item.scheduleChanged ? "Yes" : "No")
                            }

                            if let outcome = item.outcomeSummary {
                                Text(outcomeText(for: outcome))
                                    .font(.caption)
                                    .foregroundStyle(outcome.positive ? Color.green : .secondary)
                            } else if let predicted = item.predictedOutcomes {
                                Text(predictedTradeoffText(for: predicted))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else if let realizedCost = item.realizedCost {
                                Text("Realized cost \(realizedCost.formatted(.number.precision(.fractionLength(2))))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            HStack {
                                Spacer()
                                Label("Open details", systemImage: "chevron.right")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(accent)
                            }
                        }
                        .padding(14)
                        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .insightCardStyle()
    }

    private func recommendationOutcomeSectionAvailable(snapshot: ChameliaInsightsSnapshot) -> Bool {
        tirHistoryPayload(snapshot: snapshot) != nil || recommendationOutcomePayload(snapshot: snapshot) != nil
    }

    private func tirHistoryPayload(snapshot: ChameliaInsightsSnapshot) -> ChartDetailPayload? {
        if let tirTrend = snapshot.health?.tirTrend.map(\.value), tirTrend.count >= 2 {
            return .init(
                title: "Long-term TIR",
                subtitle: "Firebase-backed time-in-range trend across the recent synced window.",
                values: tirTrend,
                tint: .green,
                format: .percent
            )
        }

        if let rollingTir = snapshot.trendSeries?.tirRolling14d, rollingTir.count >= 2 {
            return .init(
                title: "Long-term TIR",
                subtitle: "Seeded report rolling 14-day TIR trend across the mature-account history window.",
                values: rollingTir,
                tint: .green,
                format: .percent
            )
        }

        return nil
    }

    private func recommendationOutcomePayload(snapshot: ChameliaInsightsSnapshot) -> ChartDetailPayload? {
        let values = snapshot.recommendationOutcomeSeries
        guard values.count >= 2 else { return nil }
        return .init(
            title: "Recommendation outcomes",
            subtitle: "TIR change after each surfaced recommendation. Positive values mean the follow-up window improved.",
            values: values,
            tint: accent,
            format: .percent
        )
    }

    private func dashboardMetric(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(tint)
                .lineLimit(2)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func explanatoryCallout(title: String, text: String, emphasis: CalloutEmphasis) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(emphasis.color)
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(emphasis.color.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func trendMetricCard(
        title: String,
        current: Double?,
        delta: Double?,
        data: [Double]?,
        tint: Color,
        format: InsightNumberFormat
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if let delta {
                    Text(formatDelta(delta, format: format))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(delta >= 0 ? tint : .orange)
                }
            }

            Text(formatValue(current, format: format))
                .font(.title3.weight(.bold))
                .foregroundStyle(.primary)

            TrendSparkline(values: data ?? [], tint: tint)
                .frame(height: 34)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func expandableInsightChartCard(
        payload: ChartDetailPayload,
        current: Double?,
        delta: Double?,
        tint: Color
    ) -> some View {
        Button {
            selectedChart = payload
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text(payload.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let delta {
                        Text(formatDelta(delta, format: payload.format))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(delta >= 0 ? tint : .orange)
                    }
                }

                Text(formatValue(current, format: payload.format))
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.primary)

                TrendSparkline(values: payload.values, tint: tint)
                    .frame(height: 34)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func infoTag(title: String, value: String) -> some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.primary.opacity(0.06), in: Capsule())
    }

    private func errorCard(message: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Unable to load Chamelia insights", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .insightCardStyle()
    }

    private var emptyCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("No Chamelia insight data yet")
                .font(.headline)
            Text("Sync the account or wait for seeded report artifacts before this screen can show trends and recommendation history.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .insightCardStyle()
    }

    private func responseTint(for item: ChameliaInsightsSnapshot.RecommendationHistoryItem) -> Color {
        switch item.response?.lowercased() {
        case "accept":
            return .green
        case "partial":
            return accent
        case "reject":
            return .orange
        default:
            return .secondary
        }
    }

    private func outcomeText(for outcome: ChameliaInsightsSnapshot.OutcomeSummary) -> String {
        let tirText = outcome.tirDelta.signedPercentString
        let costText = outcome.costDelta.formatted(.number.precision(.fractionLength(2)))
        if outcome.positive {
            return "Positive follow-up outcome · TIR \(tirText) · % low \(outcome.percentLowDelta.signedPercentString) · cost \(costText)"
        }
        return "Follow-up mixed/neutral · TIR \(tirText) · % low \(outcome.percentLowDelta.signedPercentString) · cost \(costText)"
    }

    private func predictedTradeoffText(for predicted: ChameliaInsightsSnapshot.PredictedTradeoff) -> String {
        "Predicted TIR \(predicted.deltaTIR.signedPercentString) · % low \(predicted.deltaPercentLow.signedPercentString) · % high \(predicted.deltaPercentHigh.signedPercentString)"
    }

    private func rollingDelta(_ values: [Double]?) -> Double? {
        guard let values, values.count >= 2 else { return nil }
        let current = values[values.count - 1]
        let priorIndex = max(0, values.count - 15)
        return current - values[priorIndex]
    }

    private func formatValue(_ value: Double?, format: InsightNumberFormat) -> String {
        guard let value else { return "—" }
        switch format {
        case .percent:
            return value.percentString
        case .number:
            return value.formatted(.number.precision(.fractionLength(0)))
        }
    }

    private func formatDelta(_ value: Double, format: InsightNumberFormat) -> String {
        switch format {
        case .percent:
            return value.signedPercentString
        case .number:
            let rounded = Int(value.rounded())
            return rounded > 0 ? "+\(rounded)" : "\(rounded)"
        }
    }

    private func relativeDateString(_ date: Date) -> String {
        RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
    }
}

private enum InsightNumberFormat {
    case percent
    case number
}

private enum CalloutEmphasis {
    case good
    case neutral

    var color: Color {
        switch self {
        case .good:
            return .green
        case .neutral:
            return .orange
        }
    }
}

private struct ChartDetailPayload: Identifiable {
    let title: String
    let subtitle: String
    let values: [Double]
    let tint: Color
    let format: InsightNumberFormat

    var id: String { title }
}

private struct HealthStatsDetailView: View {
    let snapshot: ChameliaInsightsSnapshot
    let accent: Color
    @State private var selectedChart: ChartDetailPayload?

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let health = snapshot.health {
                    firebaseHealthOverview(health)
                    firebaseHealthCharts(health)
                    firebaseRecoveryCards(health)
                } else {
                    reportFallbackCard
                }
            }
            .padding(16)
        }
        .background(Color.clear)
        .navigationTitle("Health Stats")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedChart) { payload in
            ChartDetailView(payload: payload)
        }
    }

    private func firebaseHealthOverview(_ health: ChameliaInsightsSnapshot.HealthSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Firebase health snapshot")
                    .font(.headline)
                Spacer()
                Text("Live data")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(accent.opacity(0.14), in: Capsule())
                    .foregroundStyle(accent)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                metricCard(title: "Average BG", value: formatValue(health.averageBg, format: .number), tint: accent)
                metricCard(title: "Time in range", value: formatValue(health.tir, format: .percent), tint: .green)
                metricCard(title: "% Low", value: formatValue(health.percentLow, format: .percent), tint: .orange)
                metricCard(title: "% High", value: formatValue(health.percentHigh, format: .percent), tint: .pink)
                metricCard(title: "Avg HR", value: health.averageHeartRate.map { "\($0.formatted(.number.precision(.fractionLength(0)))) bpm" } ?? "—", tint: accent)
                metricCard(title: "Resting HR", value: health.restingHeartRate.map { "\($0.formatted(.number.precision(.fractionLength(0)))) bpm" } ?? "—", tint: accent)
                metricCard(title: "Exercise", value: health.exerciseMinutes.map { "\($0.formatted(.number.precision(.fractionLength(0)))) min" } ?? "—", tint: accent)
                metricCard(title: "Sleep", value: health.sleepHours.map { "\($0.formatted(.number.precision(.fractionLength(1)))) h" } ?? "—", tint: accent)
                metricCard(title: "Active energy", value: health.activeEnergy.map { "\($0.formatted(.number.precision(.fractionLength(0)))) cal" } ?? "—", tint: accent)
                metricCard(title: "Days since site change", value: health.daysSinceSiteChange.map(String.init) ?? "—", tint: accent)
            }
        }
        .insightCardStyle()
    }

    private func firebaseHealthCharts(_ health: ChameliaInsightsSnapshot.HealthSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Recent trends")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                detailTrendCard(title: "TIR", current: health.tir, delta: health.tirDelta, data: health.tirTrend.map(\.value), tint: .green, format: .percent, subtitle: "Recent Firebase time-in-range trend")
                detailTrendCard(title: "Average BG", current: health.averageBg, delta: health.bgDelta, data: health.bgTrend.map(\.value), tint: accent, format: .number, subtitle: "Recent Firebase glucose trend")
                detailTrendCard(title: "Exercise", current: health.exerciseMinutes, delta: health.exerciseDelta, data: health.exerciseTrend.map(\.value), tint: accent, format: .number, subtitle: "Recent exercise-minutes trend")
                detailTrendCard(title: "Sleep", current: health.sleepHours, delta: health.sleepDelta, data: health.sleepTrend.map(\.value), tint: .blue, format: .number, subtitle: "Recent sleep-hours trend")
            }
        }
        .insightCardStyle()
    }

    private func firebaseRecoveryCards(_ health: ChameliaInsightsSnapshot.HealthSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Context")
                .font(.headline)

            HStack(spacing: 10) {
                infoTag(title: "Source", value: "Firestore uploads")
                if let siteAge = health.daysSinceSiteChange {
                    infoTag(title: "Site", value: "\(siteAge)d old")
                }
                if let exercise = health.exerciseMinutes {
                    infoTag(title: "Exercise", value: "\(exercise.formatted(.number.precision(.fractionLength(0))))m")
                }
            }
        }
        .insightCardStyle()
    }

    private var reportFallbackCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Health stats are still syncing")
                .font(.headline)
            Text("Firebase health collections are empty for this account, so the app is falling back to seeded report trends only.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                infoTag(title: "Baseline 14d", value: snapshot.tirBaseline14dMean.percentString)
                infoTag(title: "Final 14d", value: snapshot.tirFinal14dMean.percentString)
                infoTag(title: "Δ TIR", value: snapshot.tirDeltaBaselineVsFinal14d.signedPercentString)
            }
        }
        .insightCardStyle()
    }

    private func metricCard(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func detailTrendCard(
        title: String,
        current: Double?,
        delta: Double?,
        data: [Double],
        tint: Color,
        format: InsightNumberFormat,
        subtitle: String
    ) -> some View {
        Button {
            selectedChart = .init(
                title: title,
                subtitle: subtitle,
                values: data,
                tint: tint,
                format: format
            )
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let delta {
                        Text(formatDelta(delta, format: format))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(delta >= 0 ? tint : .orange)
                    }
                }

                Text(formatValue(current, format: format))
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.primary)

                TrendSparkline(values: data, tint: tint)
                    .frame(height: 34)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func infoTag(title: String, value: String) -> some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.primary.opacity(0.06), in: Capsule())
    }

    private func formatValue(_ value: Double?, format: InsightNumberFormat) -> String {
        guard let value else { return "—" }
        switch format {
        case .percent:
            return value.formatted(.percent.precision(.fractionLength(0)))
        case .number:
            return value.formatted(.number.precision(.fractionLength(1)))
        }
    }

    private func formatDelta(_ value: Double, format: InsightNumberFormat) -> String {
        switch format {
        case .percent:
            let rounded = Int((value * 100).rounded())
            return rounded > 0 ? "+\(rounded)%" : "\(rounded)%"
        case .number:
            let rounded = value.formatted(.number.precision(.fractionLength(1)))
            return value > 0 ? "+\(rounded)" : rounded
        }
    }
}

private struct ChartDetailView: View {
    let payload: ChartDetailPayload
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(payload.title)
                        .font(.system(.title2, design: .rounded).weight(.bold))
                    Text(payload.subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                TrendSparkline(values: payload.values, tint: payload.tint)
                    .frame(height: 220)
                    .padding(.vertical, 10)

                if let latest = payload.values.last {
                    Text("Latest: \(formatted(latest))")
                        .font(.subheadline.weight(.semibold))
                }

                if let first = payload.values.first, let last = payload.values.last {
                    Text("Window change: \(formattedDelta(last - first))")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding(20)
            .navigationTitle(payload.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func formatted(_ value: Double) -> String {
        switch payload.format {
        case .percent:
            return value.formatted(.percent.precision(.fractionLength(0)))
        case .number:
            return value.formatted(.number.precision(.fractionLength(1)))
        }
    }

    private func formattedDelta(_ value: Double) -> String {
        switch payload.format {
        case .percent:
            let percent = Int((value * 100).rounded())
            return percent > 0 ? "+\(percent)%" : "\(percent)%"
        case .number:
            let text = value.formatted(.number.precision(.fractionLength(1)))
            return value > 0 ? "+\(text)" : text
        }
    }
}

private struct RecommendationDetailSheet: View {
    let item: ChameliaInsightsSnapshot.RecommendationHistoryItem
    let accent: Color
    @Environment(\.dismiss) private var dismiss
    @State private var profileChangeState: RecommendationProfileChangeState = .idle

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(item.actionKind.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.system(.title2, design: .rounded).weight(.bold))
                        Text(item.dateText)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 10) {
                        detailPill(title: "Response", value: item.response?.capitalized ?? "Pending", tint: responseTint)
                        detailPill(title: "Level", value: item.actionLevel.map(String.init) ?? "—", tint: accent)
                        detailPill(title: "Changed", value: item.scheduleChanged ? "Yes" : "No", tint: accent)
                    }

                    if let family = item.actionFamily {
                        detailBlock(title: "Action family") {
                            Text(family.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.subheadline)
                        }
                    }

                    if let predicted = item.predictedOutcomes {
                        detailBlock(title: "Predicted tradeoff") {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("These estimates came from Chamelia's rollout model at recommendation time.")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                comparisonRow(title: "TIR", value: predicted.deltaTIR.signedPercentString, favorable: predicted.deltaTIR >= 0)
                                comparisonRow(title: "% Low", value: predicted.deltaPercentLow.signedPercentString, favorable: predicted.deltaPercentLow <= 0)
                                comparisonRow(title: "% High", value: predicted.deltaPercentHigh.signedPercentString, favorable: predicted.deltaPercentHigh <= 0)
                                comparisonRow(title: "Avg BG", value: signedNumber(predicted.deltaAverageBG), favorable: predicted.deltaAverageBG <= 0)
                                comparisonRow(title: "Cost", value: signedDecimal(predicted.deltaCostMean), favorable: predicted.deltaCostMean <= 0)
                            }
                        }
                    } else if let predictedImprovement = item.predictedImprovement {
                        detailBlock(title: "Predicted tradeoff") {
                            Text("This history artifact only preserved Chamelia's aggregate predicted cost improvement: \(signedDecimal(predictedImprovement)).")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }

                    therapyChangeBlock

                    detailBlock(title: "What happened") {
                        if let outcome = item.outcomeSummary {
                            VStack(alignment: .leading, spacing: 8) {
                                Text(outcome.positive ? "This recommendation had a positive follow-up window." : "This recommendation had a mixed or neutral follow-up window.")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(outcome.positive ? Color.green : .primary)
                                Text("TIR delta: \(outcome.tirDelta.signedPercentString)")
                                    .font(.subheadline)
                                Text("% low delta: \(outcome.percentLowDelta.signedPercentString)")
                                    .font(.subheadline)
                                Text("% high delta: \(outcome.percentHighDelta.signedPercentString)")
                                    .font(.subheadline)
                                Text("Average BG delta: \(signedNumber(outcome.averageBGDelta))")
                                    .font(.subheadline)
                                Text("Cost delta: \(outcome.costDelta.formatted(.number.precision(.fractionLength(2))))")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                        } else if let realizedCost = item.realizedCost {
                            Text("No explicit outcome window was stored, but the realized cost for this recommendation was \(realizedCost.formatted(.number.precision(.fractionLength(2)))).")
                                .font(.subheadline)
                        } else {
                            Text("No follow-up outcome summary has been recorded for this recommendation yet.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }

                    detailBlock(title: "Interpretation") {
                        Text("Accepted or partially accepted recommendations are worth reading alongside TIR delta and realized cost. Lower realized cost is better, and positive TIR delta means the follow-up window improved.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }

                    Spacer(minLength: 0)
                }
                .padding(20)
            }
            .navigationTitle("Recommendation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .task(id: item.id) {
            await loadProfileChangesIfNeeded()
        }
    }

    private var responseTint: Color {
        switch item.response?.lowercased() {
        case "accept":
            return .green
        case "partial":
            return accent
        case "reject":
            return .orange
        default:
            return .secondary
        }
    }

    private func detailPill(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tint)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(tint.opacity(0.1), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func detailBlock<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    @ViewBuilder
    private var therapyChangeBlock: some View {
        detailBlock(title: "Profile changes") {
            switch profileChangeState {
            case .idle, .loading:
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Reconstructing therapy changes from saved history and therapy snapshots…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

            case .resolved(let resolved):
                VStack(alignment: .leading, spacing: 12) {
                    if !resolved.blockChanges.isEmpty {
                        ForEach(resolved.blockChanges) { change in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text(change.label)
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    if change.isStructureAffected {
                                        Text("Structure changed")
                                            .font(.caption.weight(.semibold))
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(Color.orange.opacity(0.14), in: Capsule())
                                            .foregroundStyle(.orange)
                                    }
                                }

                                valueChangeRow(title: "ISF", oldValue: change.oldISF, newValue: change.newISF)
                                valueChangeRow(title: "CR", oldValue: change.oldCR, newValue: change.newCR)
                                valueChangeRow(title: "Basal", oldValue: change.oldBasal, newValue: change.newBasal)
                            }
                            .padding(12)
                            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                    }

                    if !resolved.structureSummaries.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Structure summary")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            ForEach(resolved.structureSummaries, id: \.self) { summary in
                                Text("• \(summary)")
                                    .font(.subheadline)
                            }
                        }
                    }

                    if !resolved.fallbackSummaries.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Stored summary")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            ForEach(resolved.fallbackSummaries, id: \.self) { summary in
                                Text("• \(summary)")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Text(resolved.provenanceText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            case .unavailable(let message):
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func valueChangeRow(title: String, oldValue: Double?, newValue: Double?) -> some View {
        HStack {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .leading)
            Text(formattedValue(oldValue))
                .font(.subheadline.monospacedDigit())
            Image(systemName: "arrow.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(formattedValue(newValue))
                .font(.subheadline.monospacedDigit().weight(.semibold))
        }
    }

    private func formattedValue(_ value: Double?) -> String {
        guard let value else { return "—" }
        return value.formatted(.number.precision(.fractionLength(2)))
    }

    private func comparisonRow(title: String, value: String, favorable: Bool) -> some View {
        HStack {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
            Text(value)
                .font(.subheadline.monospacedDigit().weight(.semibold))
                .foregroundStyle(favorable ? Color.green : .orange)
        }
    }

    private func signedNumber(_ value: Double) -> String {
        let rounded = Int(value.rounded())
        if rounded > 0 { return "+\(rounded)" }
        return "\(rounded)"
    }

    private func signedDecimal(_ value: Double) -> String {
        let formatted = value.formatted(.number.precision(.fractionLength(2)))
        return value > 0 ? "+\(formatted)" : formatted
    }

    private func loadProfileChangesIfNeeded() async {
        profileChangeState = .loading

        if let resolved = await reconstructProfileChanges() {
            profileChangeState = .resolved(resolved)
            return
        }

        let fallbackSummaries = item.storedSegmentSummaries + item.storedStructureSummaries
        if !fallbackSummaries.isEmpty {
            profileChangeState = .resolved(
                .init(
                    blockChanges: [],
                    structureSummaries: item.storedStructureSummaries,
                    fallbackSummaries: item.storedSegmentSummaries,
                    provenanceText: "These details come directly from the stored recommendation artifact."
                )
            )
            return
        }

        profileChangeState = .unavailable(
            "This history artifact does not include exact before/after therapy values, and nearby therapy snapshots were not sufficient to reconstruct them honestly."
        )
    }

    private func reconstructProfileChanges() async -> ResolvedRecommendationChanges? {
        guard let targetDate = item.recommendationDate else { return nil }

        do {
            let snapshots = try await TherapySettingsLogManager.shared.loadSnapshots(
                since: Calendar.current.date(byAdding: .day, value: -45, to: targetDate) ?? targetDate,
                until: Calendar.current.date(byAdding: .day, value: 45, to: targetDate) ?? targetDate
            )
            let ordered = snapshots.sorted { $0.timestamp < $1.timestamp }
            guard let before = ordered.last(where: { $0.timestamp <= targetDate }),
                  let after = ordered.first(where: { $0.timestamp > targetDate }) else {
                return nil
            }

            let blockChanges = RecommendationProfileDiff.compute(before: before.hourRanges, after: after.hourRanges)
            if blockChanges.isEmpty, item.storedSegmentSummaries.isEmpty, item.storedStructureSummaries.isEmpty {
                return nil
            }

            let structureChanged = RecommendationProfileDiff.structureChanged(before: before.hourRanges, after: after.hourRanges)
            let structureSummaries = !item.storedStructureSummaries.isEmpty
                ? item.storedStructureSummaries
                : (structureChanged ? ["Profile structure changed between the pre- and post-recommendation therapy snapshots."] : [])

            return .init(
                blockChanges: blockChanges,
                structureSummaries: structureSummaries,
                fallbackSummaries: item.storedSegmentSummaries,
                provenanceText: "These changes were reconstructed from therapy snapshots saved around this recommendation date."
            )
        } catch {
            print("[ChameliaInsights] failed to reconstruct profile changes for item=\(item.id) error=\(error)")
            return nil
        }
    }
}

private enum RecommendationProfileChangeState {
    case idle
    case loading
    case resolved(ResolvedRecommendationChanges)
    case unavailable(String)
}

private struct ResolvedRecommendationChanges {
    let blockChanges: [ProfileBlockChange]
    let structureSummaries: [String]
    let fallbackSummaries: [String]
    let provenanceText: String
}

private struct ProfileBlockChange: Identifiable {
    let label: String
    let oldISF: Double?
    let newISF: Double?
    let oldCR: Double?
    let newCR: Double?
    let oldBasal: Double?
    let newBasal: Double?
    let isStructureAffected: Bool

    var id: String { label }
}

private enum RecommendationProfileDiff {
    static func compute(before: [HourRange], after: [HourRange]) -> [ProfileBlockChange] {
        let oldByLabel = Dictionary(uniqueKeysWithValues: before.map { ($0.timeLabel, $0) })
        let newByLabel = Dictionary(uniqueKeysWithValues: after.map { ($0.timeLabel, $0) })
        let sharedLabels = Array(Set(oldByLabel.keys).intersection(newByLabel.keys)).sorted()

        var changes = sharedLabels.compactMap { label -> ProfileBlockChange? in
            guard let old = oldByLabel[label], let new = newByLabel[label] else { return nil }
            let changed =
                abs(old.insulinSensitivity - new.insulinSensitivity) > 0.0001 ||
                abs(old.carbRatio - new.carbRatio) > 0.0001 ||
                abs(old.basalRate - new.basalRate) > 0.0001
            guard changed else { return nil }
            return ProfileBlockChange(
                label: label,
                oldISF: old.insulinSensitivity,
                newISF: new.insulinSensitivity,
                oldCR: old.carbRatio,
                newCR: new.carbRatio,
                oldBasal: old.basalRate,
                newBasal: new.basalRate,
                isStructureAffected: false
            )
        }

        if changes.isEmpty {
            let overlapChanges = overlapBasedChanges(before: before, after: after)
            changes.append(contentsOf: overlapChanges)
        }

        return changes
    }

    static func structureChanged(before: [HourRange], after: [HourRange]) -> Bool {
        guard before.count == after.count else { return true }
        let oldRanges = before.map { ($0.startMinute, $0.endMinute) }
        let newRanges = after.map { ($0.startMinute, $0.endMinute) }
        return !oldRanges.elementsEqual(newRanges, by: { lhs, rhs in
            lhs.0 == rhs.0 && lhs.1 == rhs.1
        })
    }

    private static func overlapBasedChanges(before: [HourRange], after: [HourRange]) -> [ProfileBlockChange] {
        after.compactMap { newRange in
            guard let oldRange = before.max(by: { overlapMinutes($0, newRange) < overlapMinutes($1, newRange) }) else {
                return nil
            }
            let overlap = overlapMinutes(oldRange, newRange)
            guard overlap > 0 else { return nil }
            let valuesChanged =
                abs(oldRange.insulinSensitivity - newRange.insulinSensitivity) > 0.0001 ||
                abs(oldRange.carbRatio - newRange.carbRatio) > 0.0001 ||
                abs(oldRange.basalRate - newRange.basalRate) > 0.0001
            guard valuesChanged else { return nil }
            return ProfileBlockChange(
                label: newRange.timeLabel,
                oldISF: oldRange.insulinSensitivity,
                newISF: newRange.insulinSensitivity,
                oldCR: oldRange.carbRatio,
                newCR: newRange.carbRatio,
                oldBasal: oldRange.basalRate,
                newBasal: newRange.basalRate,
                isStructureAffected: true
            )
        }
    }

    private static func overlapMinutes(_ lhs: HourRange, _ rhs: HourRange) -> Int {
        max(0, min(lhs.endMinute, rhs.endMinute) - max(lhs.startMinute, rhs.startMinute))
    }
}

private struct TrendSparkline: View {
    let values: [Double]
    let tint: Color

    var body: some View {
        GeometryReader { proxy in
            if values.count < 2 {
                Capsule()
                    .fill(tint.opacity(0.14))
                    .frame(height: 6)
                    .frame(maxHeight: .infinity, alignment: .center)
            } else {
                let trimmed = Array(values.suffix(21))
                let minValue = trimmed.min() ?? 0
                let maxValue = trimmed.max() ?? 1
                let span = max(maxValue - minValue, 0.0001)
                Path { path in
                    for (index, value) in trimmed.enumerated() {
                        let x = proxy.size.width * CGFloat(index) / CGFloat(max(trimmed.count - 1, 1))
                        let y = proxy.size.height * (1 - CGFloat((value - minValue) / span))
                        if index == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(tint, style: StrokeStyle(lineWidth: 2.4, lineCap: .round, lineJoin: .round))
            }
        }
    }
}

private extension View {
    func insightCardStyle() -> some View {
        self
            .padding(18)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
    }
}

private extension Optional where Wrapped == Double {
    var percentString: String {
        guard let value = self else { return "—" }
        return value.formatted(.percent.precision(.fractionLength(0)))
    }

    var signedPercentString: String {
        guard let value = self else { return "—" }
        let percent = Int((value * 100).rounded())
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }
}

private extension Double {
    var percentString: String {
        self.formatted(.percent.precision(.fractionLength(0)))
    }

    var signedPercentString: String {
        let percent = Int((self * 100).rounded())
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }
}
