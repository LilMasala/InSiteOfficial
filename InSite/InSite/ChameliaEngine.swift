import Foundation

enum ChameliaError: Error, LocalizedError {
    case networkError(Error)
    case serverError(Int, String)
    case notFound
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case let .networkError(error as URLError):
            switch error.code {
            case .timedOut:
                return "Chamelia took too long to respond. Try again in a moment."
            case .notConnectedToInternet:
                return "You're offline. Reconnect to the internet and try again."
            default:
                return "Chamelia couldn't be reached right now."
            }
        case let .networkError(error):
            return "Chamelia couldn't be reached right now. \(error.localizedDescription)"
        case let .serverError(status, message):
            if message.isEmpty {
                return "Chamelia returned a server error (\(status))."
            }
            return "Chamelia server error (\(status)): \(message)"
        case .notFound:
            return "No Chamelia state exists yet for this account."
        case let .decodingError(error):
            return "Chamelia returned data in an unexpected format. \(error.localizedDescription)"
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .networkError:
            return "Try syncing again after your connection stabilizes."
        case .serverError:
            return "If this keeps happening, wait a minute and try again."
        case .notFound:
            return "This is normal for a first-time account."
        case .decodingError:
            return "Try again after the next app sync."
        }
    }
}

struct GraduationStatus: Codable, Equatable {
    let graduated: Bool
    let nDays: Int
    let winRate: Double
    let safetyViolations: Int
    let consecutiveDays: Int
    let beliefEntropy: Double?
    let familiarity: Double?
    let concordance: Double?
    let calibration: Double?
    let trustLevel: Double?
    let burnoutLevel: Double?
    let noSurfaceStreak: Int?
    let beliefMode: String?
    let jepaActive: Bool?
    let configuratorMode: String?
    let lastDecisionReason: String?

    enum CodingKeys: String, CodingKey {
        case graduated
        case nDays = "n_days"
        case winRate = "win_rate"
        case safetyViolations = "safety_violations"
        case consecutiveDays = "consecutive_days"
        case beliefEntropy = "belief_entropy"
        case familiarity
        case concordance
        case calibration
        case trustLevel = "trust_level"
        case burnoutLevel = "burnout_level"
        case noSurfaceStreak = "no_surface_streak"
        case beliefMode = "belief_mode"
        case jepaActive = "jepa_active"
        case configuratorMode = "configurator_mode"
        case lastDecisionReason = "last_decision_reason"
    }

    init(
        graduated: Bool,
        nDays: Int,
        winRate: Double,
        safetyViolations: Int,
        consecutiveDays: Int,
        beliefEntropy: Double? = nil,
        familiarity: Double? = nil,
        concordance: Double? = nil,
        calibration: Double? = nil,
        trustLevel: Double? = nil,
        burnoutLevel: Double? = nil,
        noSurfaceStreak: Int? = nil,
        beliefMode: String? = nil,
        jepaActive: Bool? = nil,
        configuratorMode: String? = nil,
        lastDecisionReason: String? = nil
    ) {
        self.graduated = graduated
        self.nDays = nDays
        self.winRate = winRate
        self.safetyViolations = safetyViolations
        self.consecutiveDays = consecutiveDays
        self.beliefEntropy = beliefEntropy
        self.familiarity = familiarity
        self.concordance = concordance
        self.calibration = calibration
        self.trustLevel = trustLevel
        self.burnoutLevel = burnoutLevel
        self.noSurfaceStreak = noSurfaceStreak
        self.beliefMode = beliefMode
        self.jepaActive = jepaActive
        self.configuratorMode = configuratorMode
        self.lastDecisionReason = lastDecisionReason
    }
}

enum ChameliaActionFamily: String, Codable, Equatable {
    case parameterAdjustment = "parameter_adjustment"
    case structureEdit = "structure_edit"
    case continuousSchedule = "continuous_schedule"
}

struct ConnectedAppCapabilities: Codable, Equatable {
    let appId: String
    let supportsScalarSchedule: Bool
    let supportsPiecewiseSchedule: Bool
    let supportsContinuousSchedule: Bool
    let maxSegments: Int
    let minSegmentDurationMin: Int
    let maxSegmentsAddable: Int
    let level1Enabled: Bool
    let level2Enabled: Bool
    let level3Enabled: Bool
    let structuralChangeRequiresConsent: Bool

    enum CodingKeys: String, CodingKey {
        case appId = "app_id"
        case supportsScalarSchedule = "supports_scalar_schedule"
        case supportsPiecewiseSchedule = "supports_piecewise_schedule"
        case supportsContinuousSchedule = "supports_continuous_schedule"
        case maxSegments = "max_segments"
        case minSegmentDurationMin = "min_segment_duration_min"
        case maxSegmentsAddable = "max_segments_addable"
        case level1Enabled = "level_1_enabled"
        case level2Enabled = "level_2_enabled"
        case level3Enabled = "level_3_enabled"
        case structuralChangeRequiresConsent = "structural_change_requires_consent"
    }

    static func insiteDefaults(level2Enabled: Bool) -> ConnectedAppCapabilities {
        ConnectedAppCapabilities(
            appId: "insite",
            supportsScalarSchedule: true,
            supportsPiecewiseSchedule: true,
            supportsContinuousSchedule: false,
            maxSegments: 8,
            minSegmentDurationMin: 120,
            maxSegmentsAddable: 2,
            level1Enabled: true,
            level2Enabled: level2Enabled,
            level3Enabled: false,
            structuralChangeRequiresConsent: true
        )
    }
}

struct TherapySegmentConfig: Codable, Equatable, Identifiable {
    let segmentId: String
    let startMin: Int
    let endMin: Int
    let isf: Double
    let cr: Double
    let basal: Double

    var id: String { segmentId }

    enum CodingKeys: String, CodingKey {
        case segmentId = "segment_id"
        case startMin = "start_min"
        case endMin = "end_min"
        case isf
        case cr
        case basal
    }
}

struct ProfileSummary: Codable, Equatable {
    let id: String
    let name: String
    let segmentCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case segmentCount = "segment_count"
    }
}

struct ConnectedAppState: Codable, Equatable {
    let scheduleVersion: String
    let currentSegments: [TherapySegmentConfig]
    let allowStructuralRecommendations: Bool
    let allowContinuousSchedule: Bool
    let activeProfileId: String?
    let availableProfiles: [ProfileSummary]

    enum CodingKeys: String, CodingKey {
        case scheduleVersion = "schedule_version"
        case currentSegments = "current_segments"
        case allowStructuralRecommendations = "allow_structural_recommendations"
        case allowContinuousSchedule = "allow_continuous_schedule"
        case activeProfileId = "active_profile_id"
        case availableProfiles = "available_profiles"
    }
}

struct SegmentDeltaPayload: Codable, Equatable, Identifiable {
    let segmentId: String
    let isfDelta: Double
    let crDelta: Double
    let basalDelta: Double

    var id: String { segmentId }

    enum CodingKeys: String, CodingKey {
        case segmentId = "segment_id"
        case isfDelta = "isf_delta"
        case crDelta = "cr_delta"
        case basalDelta = "basal_delta"
    }
}

struct StructureEditPayload: Codable, Equatable, Identifiable {
    let editType: String
    let targetSegmentId: String
    let splitAtMinute: Int?
    let neighborSegmentId: String?

    var id: String {
        [
            editType,
            targetSegmentId,
            splitAtMinute.map(String.init),
            neighborSegmentId
        ]
        .compactMap { $0 }
        .joined(separator: "|")
    }

    enum CodingKeys: String, CodingKey {
        case editType = "edit_type"
        case targetSegmentId = "target_segment_id"
        case splitAtMinute = "split_at_minute"
        case neighborSegmentId = "neighbor_segment_id"
    }
}

struct ScheduledActionPayload: Codable, Equatable {
    let kind: String
    let level: Int?
    let family: ChameliaActionFamily?
    let segmentDeltas: [SegmentDeltaPayload]
    let structuralEdits: [StructureEditPayload]

    enum CodingKeys: String, CodingKey {
        case kind
        case level
        case family
        case segmentDeltas = "segment_deltas"
        case structuralEdits = "structural_edits"
    }
}

struct RecommendationSegmentSummary: Codable, Equatable, Identifiable {
    let segmentId: String
    let label: String
    let isf: String
    let cr: String
    let basal: String

    var id: String { segmentId }

    enum CodingKeys: String, CodingKey {
        case segmentId = "segment_id"
        case label
        case isf
        case cr
        case basal
    }
}

struct TherapyAction: Codable, Equatable {
    let kind: String
    let deltas: [String: Double]
    let level: Int?
    let family: ChameliaActionFamily?
    let segmentDeltas: [SegmentDeltaPayload]
    let structuralEdits: [StructureEditPayload]

    enum CodingKeys: String, CodingKey {
        case kind
        case deltas
        case level
        case family
        case segmentDeltas = "segment_deltas"
        case structuralEdits = "structural_edits"
    }

    init(
        kind: String,
        deltas: [String: Double],
        level: Int? = nil,
        family: ChameliaActionFamily? = nil,
        segmentDeltas: [SegmentDeltaPayload] = [],
        structuralEdits: [StructureEditPayload] = []
    ) {
        self.kind = kind
        self.deltas = deltas
        self.level = level
        self.family = family
        self.segmentDeltas = segmentDeltas
        self.structuralEdits = structuralEdits
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(String.self, forKey: .kind)
        deltas = try container.decodeIfPresent([String: Double].self, forKey: .deltas) ?? [:]
        level = try container.decodeIfPresent(Int.self, forKey: .level)
        family = try container.decodeIfPresent(ChameliaActionFamily.self, forKey: .family)
        segmentDeltas = try container.decodeIfPresent([SegmentDeltaPayload].self, forKey: .segmentDeltas) ?? []
        structuralEdits = try container.decodeIfPresent([StructureEditPayload].self, forKey: .structuralEdits) ?? []
    }
}

struct BurnoutAttribution: Codable, Equatable {
    let deltaHat: Double
    let pTreated: Double
    let pBaseline: Double
    let upperCI: Double
    let horizon: Int

    enum CodingKeys: String, CodingKey {
        case deltaHat = "delta_hat"
        case pTreated = "p_treated"
        case pBaseline = "p_baseline"
        case upperCI = "upper_ci"
        case horizon
    }
}

struct PredictedOutcomeSummary: Codable, Equatable {
    let baselineTIR: Double
    let treatedTIR: Double
    let deltaTIR: Double
    let baselinePercentLow: Double
    let treatedPercentLow: Double
    let deltaPercentLow: Double
    let baselinePercentHigh: Double
    let treatedPercentHigh: Double
    let deltaPercentHigh: Double
    let baselineAverageBG: Double
    let treatedAverageBG: Double
    let deltaAverageBG: Double
    let baselineCostMean: Double
    let treatedCostMean: Double
    let deltaCostMean: Double
    let baselineCVaR: Double
    let treatedCVaR: Double
    let deltaCVaR: Double

    enum CodingKeys: String, CodingKey {
        case baselineTIR = "baseline_tir"
        case treatedTIR = "treated_tir"
        case deltaTIR = "delta_tir"
        case baselinePercentLow = "baseline_pct_low"
        case treatedPercentLow = "treated_pct_low"
        case deltaPercentLow = "delta_pct_low"
        case baselinePercentHigh = "baseline_pct_high"
        case treatedPercentHigh = "treated_pct_high"
        case deltaPercentHigh = "delta_pct_high"
        case baselineAverageBG = "baseline_bg_avg"
        case treatedAverageBG = "treated_bg_avg"
        case deltaAverageBG = "delta_bg_avg"
        case baselineCostMean = "baseline_cost_mean"
        case treatedCostMean = "treated_cost_mean"
        case deltaCostMean = "delta_cost_mean"
        case baselineCVaR = "baseline_cvar"
        case treatedCVaR = "treated_cvar"
        case deltaCVaR = "delta_cvar"
    }
}

struct ConfidenceBreakdown: Codable, Equatable {
    let familiarity: Double
    let concordance: Double
    let calibration: Double
    let effectSupport: Double
    let selectionPenalty: Double
    let finalConfidence: Double

    enum CodingKeys: String, CodingKey {
        case familiarity
        case concordance
        case calibration
        case effectSupport = "effect_support"
        case selectionPenalty = "selection_penalty"
        case finalConfidence = "final_confidence"
    }
}

struct PredictedUncertaintySummary: Codable, Equatable {
    let tirStd: Double
    let percentLowStd: Double
    let percentHighStd: Double
    let averageBGStd: Double
    let costStd: Double

    enum CodingKeys: String, CodingKey {
        case tirStd = "tir_std"
        case percentLowStd = "pct_low_std"
        case percentHighStd = "pct_high_std"
        case averageBGStd = "bg_avg_std"
        case costStd = "cost_std"
    }
}

struct RecommendationPackage: Codable, Equatable {
    let action: TherapyAction
    let predictedImprovement: Double
    let confidence: Double
    let confidenceBreakdown: ConfidenceBreakdown?
    let effectSize: Double
    let cvarValue: Double
    let burnoutAttribution: BurnoutAttribution?
    let predictedOutcomes: PredictedOutcomeSummary?
    let predictedUncertainty: PredictedUncertaintySummary?
    let actionLevel: Int
    let actionFamily: ChameliaActionFamily?
    let recommendationScope: String
    let targetProfileId: String?
    let detectedRegime: String?
    let segmentSummaries: [RecommendationSegmentSummary]
    let structureSummaries: [String]

    enum CodingKeys: String, CodingKey {
        case action
        case predictedImprovement = "predicted_improvement"
        case confidence
        case confidenceBreakdown = "confidence_breakdown"
        case effectSize = "effect_size"
        case cvarValue = "cvar_value"
        case burnoutAttribution = "burnout_attribution"
        case predictedOutcomes = "predicted_outcomes"
        case predictedUncertainty = "predicted_uncertainty"
        case actionLevel = "action_level"
        case actionFamily = "action_family"
        case recommendationScope = "recommendation_scope"
        case targetProfileId = "target_profile_id"
        case detectedRegime = "detected_regime"
        case segmentSummaries = "segment_summaries"
        case structureSummaries = "structure_summaries"
    }

    init(
        action: TherapyAction,
        predictedImprovement: Double,
        confidence: Double,
        confidenceBreakdown: ConfidenceBreakdown? = nil,
        effectSize: Double,
        cvarValue: Double,
        burnoutAttribution: BurnoutAttribution?,
        predictedOutcomes: PredictedOutcomeSummary? = nil,
        predictedUncertainty: PredictedUncertaintySummary? = nil,
        actionLevel: Int = 1,
        actionFamily: ChameliaActionFamily? = nil,
        recommendationScope: String = "patch_current",
        targetProfileId: String? = nil,
        detectedRegime: String? = nil,
        segmentSummaries: [RecommendationSegmentSummary] = [],
        structureSummaries: [String] = []
    ) {
        self.action = action
        self.predictedImprovement = predictedImprovement
        self.confidence = confidence
        self.confidenceBreakdown = confidenceBreakdown
        self.effectSize = effectSize
        self.cvarValue = cvarValue
        self.burnoutAttribution = burnoutAttribution
        self.predictedOutcomes = predictedOutcomes
        self.predictedUncertainty = predictedUncertainty
        self.actionLevel = actionLevel
        self.actionFamily = actionFamily
        self.recommendationScope = recommendationScope
        self.targetProfileId = targetProfileId
        self.detectedRegime = detectedRegime
        self.segmentSummaries = segmentSummaries
        self.structureSummaries = structureSummaries
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        action = try container.decode(TherapyAction.self, forKey: .action)
        predictedImprovement = try container.decode(Double.self, forKey: .predictedImprovement)
        confidence = try container.decode(Double.self, forKey: .confidence)
        confidenceBreakdown = try container.decodeIfPresent(ConfidenceBreakdown.self, forKey: .confidenceBreakdown)
        effectSize = try container.decode(Double.self, forKey: .effectSize)
        cvarValue = try container.decode(Double.self, forKey: .cvarValue)
        burnoutAttribution = try container.decodeIfPresent(BurnoutAttribution.self, forKey: .burnoutAttribution)
        predictedOutcomes = try container.decodeIfPresent(PredictedOutcomeSummary.self, forKey: .predictedOutcomes)
        predictedUncertainty = try container.decodeIfPresent(PredictedUncertaintySummary.self, forKey: .predictedUncertainty)
        actionLevel = try container.decodeIfPresent(Int.self, forKey: .actionLevel) ?? action.level ?? 1
        actionFamily = try container.decodeIfPresent(ChameliaActionFamily.self, forKey: .actionFamily) ?? action.family
        recommendationScope = try container.decodeIfPresent(String.self, forKey: .recommendationScope) ?? "patch_current"
        targetProfileId = try container.decodeIfPresent(String.self, forKey: .targetProfileId)
        detectedRegime = try container.decodeIfPresent(String.self, forKey: .detectedRegime)
        segmentSummaries = try container.decodeIfPresent([RecommendationSegmentSummary].self, forKey: .segmentSummaries) ?? []
        structureSummaries = try container.decodeIfPresent([String].self, forKey: .structureSummaries) ?? []
    }
}

struct ChameliaPreferences: Codable, Equatable {
    let aggressiveness: Double
    let hypoglycemiaFear: Double
    let burdenSensitivity: Double
    let persona: String
    let physicalPriors: [String: [Double]]

    enum CodingKeys: String, CodingKey {
        case aggressiveness
        case hypoglycemiaFear = "hypoglycemia_fear"
        case burdenSensitivity = "burden_sensitivity"
        case persona
        case physicalPriors = "physical_priors"
    }

    init(
        aggressiveness: Double,
        hypoglycemiaFear: Double,
        burdenSensitivity: Double,
        persona: String,
        physicalPriors: [String: [Double]] = [:]
    ) {
        self.aggressiveness = aggressiveness
        self.hypoglycemiaFear = hypoglycemiaFear
        self.burdenSensitivity = burdenSensitivity
        self.persona = persona
        self.physicalPriors = physicalPriors
    }
}

struct ChameliaResponse: Codable, Equatable {
    let ok: Bool
    let patientId: String
    let status: GraduationStatus?
    let recId: Int64?
    let recommendation: RecommendationPackage?

    enum CodingKeys: String, CodingKey {
        case ok
        case patientId = "patient_id"
        case status
        case recId = "rec_id"
        case recommendation
    }
}

actor ChameliaEngine {
    static let shared = ChameliaEngine()

    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private let loggingEnabled = true

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    func initialize(patientId: String, preferences: ChameliaPreferences) async throws {
        let request = InitializeRequest(patientId: patientId, preferences: preferences)
        log("initialize start patient=\(patientId)")
        let _: ChameliaResponse = try await post(path: "/chamelia_initialize_patient", body: request)
        log("initialize success patient=\(patientId)")
    }

    func observe(patientId: String, timestamp: Double, signals: [String: Double]) async throws {
        let request = SignalRequest(
            patientId: patientId,
            timestamp: timestamp,
            signals: signals,
            connectedAppCapabilities: nil,
            connectedAppState: nil
        )
        log("observe start patient=\(patientId) timestamp=\(timestamp) signals=\(signals.count)")
        let _: ChameliaResponse = try await post(path: "/chamelia_observe", body: request)
        log("observe success patient=\(patientId)")
    }

    func step(
        patientId: String,
        timestamp: Double,
        signals: [String: Double],
        connectedAppCapabilities: ConnectedAppCapabilities? = nil,
        connectedAppState: ConnectedAppState? = nil
    ) async throws -> RecommendationPackage? {
        let response = try await stepResult(
            patientId: patientId,
            timestamp: timestamp,
            signals: signals,
            connectedAppCapabilities: connectedAppCapabilities,
            connectedAppState: connectedAppState
        )
        return response.recommendation
    }

    func stepResult(
        patientId: String,
        timestamp: Double,
        signals: [String: Double],
        connectedAppCapabilities: ConnectedAppCapabilities? = nil,
        connectedAppState: ConnectedAppState? = nil
    ) async throws -> ChameliaResponse {
        let request = SignalRequest(
            patientId: patientId,
            timestamp: timestamp,
            signals: signals,
            connectedAppCapabilities: connectedAppCapabilities,
            connectedAppState: connectedAppState
        )
        log("step start patient=\(patientId) timestamp=\(timestamp) signals=\(signals.count)")
        let response: ChameliaResponse = try await post(path: "/chamelia_step", body: request)
        log("step success patient=\(patientId) recommendation=\(response.recommendation != nil)")
        return response
    }

    func recordOutcome(patientId: String, recId: Int, response: String, signals: [String: Double], cost: Double) async throws {
        guard ["reject", "partial", "accept"].contains(response) else {
            throw ChameliaError.serverError(0, "Invalid outcome response: \(response)")
        }

        let request = RecordOutcomeRequest(
            patientId: patientId,
            recId: recId,
            response: response,
            signals: signals,
            cost: cost
        )
        log("recordOutcome start patient=\(patientId) recId=\(recId) response=\(response) signals=\(signals.count) cost=\(cost)")
        let _: ChameliaResponse = try await post(path: "/chamelia_record_outcome", body: request)
        log("recordOutcome success patient=\(patientId) recId=\(recId)")
    }

    func graduationStatus(patientId: String) async throws -> GraduationStatus {
        let request = PatientRequest(patientId: patientId)
        log("graduationStatus start patient=\(patientId)")
        let response: ChameliaResponse = try await post(path: "/chamelia_graduation_status", body: request)
        guard let status = response.status else {
            throw ChameliaError.serverError(200, "Missing graduation status in response")
        }
        log("graduationStatus success patient=\(patientId) nDays=\(status.nDays) graduated=\(status.graduated)")
        return status
    }

    func save(patientId: String) async throws {
        let request = PatientRequest(patientId: patientId)
        log("save start patient=\(patientId)")
        let _: ChameliaResponse = try await post(path: "/chamelia_save_patient", body: request)
        log("save success patient=\(patientId)")
    }

    func load(patientId: String) async throws {
        let request = PatientRequest(patientId: patientId)
        log("load start patient=\(patientId)")
        let _: ChameliaResponse = try await post(path: "/chamelia_load_patient", body: request)
        log("load success patient=\(patientId)")
    }

    private func post<RequestBody: Encodable, ResponseBody: Decodable>(
        path: String,
        body: RequestBody
    ) async throws -> ResponseBody {
        var request = URLRequest(url: ChameliaConfig.baseURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = ChameliaConfig.timeoutSeconds
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try encoder.encode(body)
        } catch {
            log("encode failure path=\(path) error=\(error)")
            throw ChameliaError.networkError(error)
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            log("network failure path=\(path) error=\(error)")
            throw ChameliaError.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            log("invalid response path=\(path)")
            throw ChameliaError.serverError(-1, "Invalid HTTP response")
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let message = decodeErrorMessage(from: data) ?? HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            log("server error path=\(path) status=\(httpResponse.statusCode) message=\(message)")
            if httpResponse.statusCode == 404 {
                throw ChameliaError.notFound
            }
            throw ChameliaError.serverError(httpResponse.statusCode, message)
        }

        do {
            return try decoder.decode(ResponseBody.self, from: data)
        } catch {
            let rawBody = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            log("decode failure path=\(path) error=\(error) body=\(rawBody)")
            throw ChameliaError.decodingError(error)
        }
    }

    private func decodeErrorMessage(from data: Data) -> String? {
        guard !data.isEmpty else { return nil }
        if let payload = try? decoder.decode(ErrorResponse.self, from: data) {
            return payload.error
        }
        return String(data: data, encoding: .utf8)
    }

    private func log(_ message: String) {
        guard loggingEnabled else { return }
        print("[ChameliaEngine] \(message)")
    }
}

private struct PatientRequest: Encodable {
    let patientId: String

    enum CodingKeys: String, CodingKey {
        case patientId = "patient_id"
    }
}

private struct InitializeRequest: Encodable {
    let patientId: String
    let preferences: ChameliaPreferences

    enum CodingKeys: String, CodingKey {
        case patientId = "patient_id"
        case preferences
    }
}

private struct SignalRequest: Encodable {
    let patientId: String
    let timestamp: Double
    let signals: [String: Double]
    let connectedAppCapabilities: ConnectedAppCapabilities?
    let connectedAppState: ConnectedAppState?

    enum CodingKeys: String, CodingKey {
        case patientId = "patient_id"
        case timestamp
        case signals
        case connectedAppCapabilities = "connected_app_capabilities"
        case connectedAppState = "connected_app_state"
    }
}

private struct RecordOutcomeRequest: Encodable {
    let patientId: String
    let recId: Int
    let response: String
    let signals: [String: Double]
    let cost: Double

    enum CodingKeys: String, CodingKey {
        case patientId = "patient_id"
        case recId = "rec_id"
        case response
        case signals
        case cost
    }
}

private struct ErrorResponse: Decodable {
    let ok: Bool?
    let error: String
}
