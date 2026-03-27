import Foundation
import Combine
import FirebaseAuth

struct ChameliaDashboardState: Codable, Equatable {
    var status: GraduationStatus?
    var recId: Int64?
    var recommendation: RecommendationPackage?
    var latestSignals: [String: Double]
    var lastUpdatedAt: Date?

    init(
        status: GraduationStatus? = nil,
        recId: Int64? = nil,
        recommendation: RecommendationPackage? = nil,
        latestSignals: [String: Double] = [:],
        lastUpdatedAt: Date? = nil
    ) {
        self.status = status
        self.recId = recId
        self.recommendation = recommendation
        self.latestSignals = latestSignals
        self.lastUpdatedAt = lastUpdatedAt
    }
}

@MainActor
final class ChameliaDashboardStore: ObservableObject {
    static let shared = ChameliaDashboardStore()

    @Published private(set) var state = ChameliaDashboardState()
    @Published private(set) var activeUserId: String?
    @Published private(set) var isRefreshing = false
    @Published private(set) var latestErrorMessage: String?

    private let engine = ChameliaEngine.shared
    private let defaults = UserDefaults.standard
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private init() {}

    func bootstrapCurrentUser() async {
        guard let user = Auth.auth().currentUser else {
            clearSession()
            return
        }

        isRefreshing = true
        loadCached(userId: user.uid)
        do {
            let status = try await engine.graduationStatus(patientId: user.uid)
            latestErrorMessage = nil
            update(userId: user.uid, status: status)
        } catch {
            latestErrorMessage = readableMessage(for: error)
            print("[ChameliaDashboardStore] graduation status refresh failed: \(error)")
        }
        isRefreshing = false
    }

    func loadCached(userId: String) {
        activeUserId = userId
        state = decodeState(for: userId) ?? ChameliaDashboardState()
    }

    func update(
        userId: String,
        status: GraduationStatus? = nil,
        recId: Int64? = nil,
        recommendation: RecommendationPackage? = nil,
        latestSignals: [String: Double]? = nil,
        clearRecommendation: Bool = false
    ) {
        var next = decodeState(for: userId) ?? ChameliaDashboardState()
        if let status {
            next.status = status
        }
        if let latestSignals {
            next.latestSignals = latestSignals
        }
        if clearRecommendation {
            next.recId = nil
            next.recommendation = nil
        } else {
            if let recId {
                next.recId = recId
            }
            if let recommendation {
                next.recommendation = recommendation
            }
        }
        next.lastUpdatedAt = Date()

        persist(next, for: userId)
        activeUserId = userId
        state = next
    }

    func clearRecommendation(userId: String) {
        update(userId: userId, clearRecommendation: true)
    }

    func clearSession() {
        activeUserId = nil
        state = ChameliaDashboardState()
        latestErrorMessage = nil
        isRefreshing = false
    }

    func clearTransientError() {
        latestErrorMessage = nil
    }

    private func persist(_ state: ChameliaDashboardState, for userId: String) {
        guard let data = try? encoder.encode(state) else { return }
        defaults.set(data, forKey: key(for: userId))
    }

    private func decodeState(for userId: String) -> ChameliaDashboardState? {
        guard let data = defaults.data(forKey: key(for: userId)) else { return nil }
        return try? decoder.decode(ChameliaDashboardState.self, from: data)
    }

    private func key(for userId: String) -> String {
        "ChameliaDashboardState.\(userId)"
    }

    private func readableMessage(for error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}
