import Foundation
import Combine
import CryptoKit
import Firebase
import FirebaseAuth
import HealthKit
import Security

enum HealthBackfillDataType: String, CaseIterable, Codable, Identifiable {
    case bloodGlucose
    case heartRate
    case restingHeartRate
    case energy
    case exercise
    case sleep
    case bodyMass
    case menstrual

    var id: String { rawValue }

    var title: String {
        switch self {
        case .bloodGlucose: return "Blood glucose"
        case .heartRate: return "Heart rate"
        case .restingHeartRate: return "Resting heart rate"
        case .energy: return "Energy"
        case .exercise: return "Exercise"
        case .sleep: return "Sleep"
        case .bodyMass: return "Body mass"
        case .menstrual: return "Menstrual"
        }
    }

    var detail: String {
        switch self {
        case .bloodGlucose: return "CGM readings, hourly averages, and low/high trends"
        case .heartRate: return "Hourly heart rate and daily averages"
        case .restingHeartRate: return "Daily resting heart rate"
        case .energy: return "Active and basal energy burn"
        case .exercise: return "Move minutes and exercise minutes"
        case .sleep: return "Daily sleep totals"
        case .bodyMass: return "Weight samples"
        case .menstrual: return "Cycle-related context when available"
        }
    }
}

struct HealthBackfillConfiguration: Codable, Equatable {
    static let defaultDays = 30

    var bloodGlucoseDays: Int = Self.defaultDays
    var heartRateDays: Int = Self.defaultDays
    var restingHeartRateDays: Int = Self.defaultDays
    var energyDays: Int = Self.defaultDays
    var exerciseDays: Int = Self.defaultDays
    var sleepDays: Int = Self.defaultDays
    var bodyMassDays: Int = Self.defaultDays
    var menstrualDays: Int = Self.defaultDays

    static func defaults() -> HealthBackfillConfiguration {
        HealthBackfillConfiguration()
    }

    func days(for type: HealthBackfillDataType) -> Int {
        switch type {
        case .bloodGlucose: return bloodGlucoseDays
        case .heartRate: return heartRateDays
        case .restingHeartRate: return restingHeartRateDays
        case .energy: return energyDays
        case .exercise: return exerciseDays
        case .sleep: return sleepDays
        case .bodyMass: return bodyMassDays
        case .menstrual: return menstrualDays
        }
    }

    mutating func setDays(_ days: Int, for type: HealthBackfillDataType) {
        let clamped = max(0, days)
        switch type {
        case .bloodGlucose: bloodGlucoseDays = clamped
        case .heartRate: heartRateDays = clamped
        case .restingHeartRate: restingHeartRateDays = clamped
        case .energy: energyDays = clamped
        case .exercise: exerciseDays = clamped
        case .sleep: sleepDays = clamped
        case .bodyMass: bodyMassDays = clamped
        case .menstrual: menstrualDays = clamped
        }
    }

    func startDate(for type: HealthBackfillDataType, reference: Date, calendar: Calendar = .current) -> Date? {
        let lookbackDays = days(for: type)
        guard lookbackDays > 0 else { return nil }
        return calendar.date(byAdding: .day, value: -lookbackDays, to: reference)
    }

    var maximumDays: Int {
        HealthBackfillDataType.allCases.map(days(for:)).max() ?? 0
    }

    private static func key(for uid: String) -> String {
        "HealthBackfillConfiguration.\(uid)"
    }

    static func load(for uid: String?) -> HealthBackfillConfiguration {
        guard let uid, !uid.isEmpty,
              let data = UserDefaults.standard.data(forKey: key(for: uid)),
              let decoded = try? JSONDecoder().decode(HealthBackfillConfiguration.self, from: data) else {
            return defaults()
        }
        return decoded
    }

    func save(for uid: String?) {
        guard let uid, !uid.isEmpty,
              let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.key(for: uid))
    }
}

enum NightscoutAuthMode: String, Codable, CaseIterable, Identifiable {
    case accessToken
    case apiSecret

    var id: String { rawValue }

    var title: String {
        switch self {
        case .accessToken: return "Access Token"
        case .apiSecret: return "API Secret"
        }
    }
}

struct NightscoutCalibrationSummary: Codable, Equatable {
    var windowDays: Int
    var sampleCount: Int
    var recentTIR: Double
    var recentPercentLow: Double
    var recentPercentHigh: Double

    var calibrationTargets: [String: Double] {
        [
            "recent_tir": recentTIR,
            "recent_pct_low": recentPercentLow,
            "recent_pct_high": recentPercentHigh,
        ]
    }
}

struct NightscoutTherapySegment: Codable, Equatable {
    var startMinute: Int
    var endMinute: Int
    var basalRate: Double
    var insulinSensitivity: Double
    var carbRatio: Double
}

struct NightscoutTreatmentEvent: Codable, Equatable {
    var eventId: String
    var sourceEventId: String?
    var eventType: String
    var timestamp: Date
    var insulin: Double?
    var carbs: Double?
    var rate: Double?
    var durationMinutes: Int?
    var enteredBy: String?
    var notes: String?
    var source: String
}

struct NightscoutBootstrapSummary: Codable, Equatable {
    var lastValidatedAt: Date
    var units: String?
    var enabledPlugins: [String]
    var latestProfileName: String?
    var latestProfileStartDate: Date?
    var therapySegments: [NightscoutTherapySegment]
    var latestIOB: Double?
    var latestCOB: Double?
    var recentBolusCount: Int
    var recentCarbEntryCount: Int
    var recentTempBasalCount: Int
    var recentTreatmentEvents: [NightscoutTreatmentEvent]
    var calibration: NightscoutCalibrationSummary?
}

struct NightscoutConnectionState: Codable, Equatable {
    var baseURLString: String = ""
    var authMode: NightscoutAuthMode = .accessToken
    var lastValidatedAt: Date?
    var summary: NightscoutBootstrapSummary?
    var lastErrorMessage: String?

    var isConnected: Bool { !baseURLString.isEmpty }
    var hasValidatedSummary: Bool { summary != nil }

    var normalizedBaseURLString: String {
        baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }
}

enum NightscoutConnectionError: LocalizedError {
    case invalidURL
    case missingCredential
    case invalidResponse
    case serverError(Int)
    case malformedPayload(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Enter a valid Nightscout site URL."
        case .missingCredential:
            return "Add a Nightscout access token or API secret."
        case .invalidResponse:
            return "Nightscout returned an unexpected response."
        case let .serverError(code):
            return "Nightscout returned a server error (\(code))."
        case let .malformedPayload(detail):
            return "Nightscout data was missing expected fields. \(detail)"
        }
    }
}

private enum NightscoutKeychain {
    static func service(for uid: String) -> String { "InSite.Nightscout.\(uid)" }
    private static let account = "credential"

    static func save(_ credential: String, for uid: String) {
        let service = service(for: uid)
        let encoded = Data(credential.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        var add = query
        add[kSecValueData as String] = encoded
        SecItemAdd(add as CFDictionary, nil)
    }

    static func load(for uid: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service(for: uid),
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return nil
        }
        return value
    }

    static func delete(for uid: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service(for: uid),
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

private actor NightscoutClient {
    func validateAndBootstrap(
        baseURLString: String,
        authMode: NightscoutAuthMode,
        credential: String,
        windowDays: Int = 14
    ) async throws -> NightscoutBootstrapSummary {
        let trimmed = baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var baseURL = URL(string: trimmed), !trimmed.isEmpty else {
            throw NightscoutConnectionError.invalidURL
        }
        while baseURL.absoluteString.hasSuffix("/") {
            baseURL.deleteLastPathComponent()
        }

        let statusJSON = try await requestJSON(
            baseURL: baseURL,
            path: "/api/v1/status.json",
            authMode: authMode,
            credential: credential
        )
        guard let status = statusJSON as? [String: Any] else {
            throw NightscoutConnectionError.invalidResponse
        }

        let settings = status["settings"] as? [String: Any]
        let units = settings?["units"] as? String
        let enabled = settings?["enable"] as? [String] ?? settings?["ENABLE"] as? [String] ?? []

        let profile = try await fetchLatestProfile(baseURL: baseURL, authMode: authMode, credential: credential)
        let iobCob = try await fetchLatestDeviceStatus(baseURL: baseURL, authMode: authMode, credential: credential)
        let treatments = try await fetchRecentTreatments(baseURL: baseURL, authMode: authMode, credential: credential, windowDays: windowDays)
        let calibration = try await fetchCalibrationSummary(baseURL: baseURL, authMode: authMode, credential: credential, windowDays: windowDays, units: units)

        return NightscoutBootstrapSummary(
            lastValidatedAt: Date(),
            units: units,
            enabledPlugins: enabled,
            latestProfileName: profile.name,
            latestProfileStartDate: profile.startDate,
            therapySegments: profile.segments,
            latestIOB: iobCob.iob,
            latestCOB: iobCob.cob,
            recentBolusCount: treatments.bolusCount,
            recentCarbEntryCount: treatments.carbCount,
            recentTempBasalCount: treatments.tempBasalCount,
            recentTreatmentEvents: treatments.events,
            calibration: calibration
        )
    }

    private func requestJSON(
        baseURL: URL,
        path: String,
        queryItems: [URLQueryItem] = [],
        authMode: NightscoutAuthMode,
        credential: String
    ) async throws -> Any {
        guard !credential.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw NightscoutConnectionError.missingCredential
        }
        guard var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false) else {
            throw NightscoutConnectionError.invalidURL
        }
        var items = queryItems
        if authMode == .accessToken {
            items.append(URLQueryItem(name: "token", value: credential))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw NightscoutConnectionError.invalidURL
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        if authMode == .apiSecret {
            request.setValue(sha1Hex(credential), forHTTPHeaderField: "api-secret")
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NightscoutConnectionError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw NightscoutConnectionError.serverError(http.statusCode)
        }
        return try JSONSerialization.jsonObject(with: data)
    }

    private func fetchLatestProfile(baseURL: URL, authMode: NightscoutAuthMode, credential: String) async throws -> (name: String?, startDate: Date?, segments: [NightscoutTherapySegment]) {
        let json = try await requestJSON(
            baseURL: baseURL,
            path: "/api/v1/profile.json",
            queryItems: [URLQueryItem(name: "count", value: "1")],
            authMode: authMode,
            credential: credential
        )
        let docs = json as? [[String: Any]] ?? []
        guard let first = docs.first else { return (nil, nil, []) }
        let name = first["defaultProfile"] as? String ?? first["name"] as? String
        let startDate = parseNightscoutDate(first["startDate"]) ?? parseNightscoutDate(first["created_at"])
        let profilePayload: [String: Any]? = {
            if let store = first["store"] as? [String: Any], let name, let payload = store[name] as? [String: Any] {
                return payload
            }
            return first
        }()
        let segments = buildTherapySegments(from: profilePayload)
        return (name, startDate, segments)
    }

    private func fetchLatestDeviceStatus(baseURL: URL, authMode: NightscoutAuthMode, credential: String) async throws -> (iob: Double?, cob: Double?) {
        let json = try await requestJSON(
            baseURL: baseURL,
            path: "/api/v1/devicestatus.json",
            queryItems: [URLQueryItem(name: "count", value: "1")],
            authMode: authMode,
            credential: credential
        )
        guard let first = (json as? [[String: Any]])?.first else { return (nil, nil) }
        let openAPS = first["openaps"] as? [String: Any]
        let iobDict = openAPS?["iob"] as? [String: Any]
        let suggested = openAPS?["suggested"] as? [String: Any]
        let iob = doubleValue(iobDict?["iob"])
        let cob = doubleValue(suggested?["COB"]) ?? doubleValue(suggested?["cob"])
        return (iob, cob)
    }

    private func fetchRecentTreatments(baseURL: URL, authMode: NightscoutAuthMode, credential: String, windowDays: Int) async throws -> (bolusCount: Int, carbCount: Int, tempBasalCount: Int, events: [NightscoutTreatmentEvent]) {
        let since = ISO8601DateFormatter().string(from: Calendar.current.date(byAdding: .day, value: -windowDays, to: Date()) ?? Date())
        let json = try await requestJSON(
            baseURL: baseURL,
            path: "/api/v1/treatments.json",
            queryItems: [
                URLQueryItem(name: "count", value: "500"),
                URLQueryItem(name: "find[created_at][$gte]", value: since),
            ],
            authMode: authMode,
            credential: credential
        )
        let docs = json as? [[String: Any]] ?? []
        var bolusCount = 0
        var carbCount = 0
        var tempBasalCount = 0
        var events: [NightscoutTreatmentEvent] = []
        for doc in docs {
            let eventType = (doc["eventType"] as? String ?? "").lowercased()
            let insulin = doubleValue(doc["insulin"]) ?? 0
            let carbs = doubleValue(doc["carbs"]) ?? 0
            if insulin > 0 || eventType.contains("bolus") { bolusCount += 1 }
            if carbs > 0 || eventType.contains("carb") { carbCount += 1 }
            if eventType.contains("temp basal") { tempBasalCount += 1 }

            guard let timestamp = parseNightscoutDate(doc["created_at"]) ?? parseNightscoutDate(doc["timestamp"]) else {
                continue
            }
            let sourceId = stringValue(doc["_id"]) ?? stringValue(doc["identifier"]) ?? stringValue(doc["syncIdentifier"])
            let derivedId = sourceId ?? "\(Int(timestamp.timeIntervalSince1970))-\(eventType)-\(insulin)-\(carbs)"
            events.append(
                NightscoutTreatmentEvent(
                    eventId: "nightscout-\(derivedId)",
                    sourceEventId: sourceId,
                    eventType: (doc["eventType"] as? String) ?? "unknown",
                    timestamp: timestamp,
                    insulin: doubleValue(doc["insulin"]),
                    carbs: doubleValue(doc["carbs"]),
                    rate: doubleValue(doc["absolute"]) ?? doubleValue(doc["rate"]),
                    durationMinutes: intValue(doc["duration"]),
                    enteredBy: doc["enteredBy"] as? String,
                    notes: doc["notes"] as? String,
                    source: "nightscout"
                )
            )
        }
        events.sort { $0.timestamp < $1.timestamp }
        return (bolusCount, carbCount, tempBasalCount, events)
    }

    private func fetchCalibrationSummary(baseURL: URL, authMode: NightscoutAuthMode, credential: String, windowDays: Int, units: String?) async throws -> NightscoutCalibrationSummary? {
        let sinceDate = Calendar.current.date(byAdding: .day, value: -windowDays, to: Date()) ?? Date()
        let since = ISO8601DateFormatter().string(from: sinceDate)
        var upperBound: Date? = nil
        var values: [Double] = []

        for _ in 0..<6 {
            var queryItems = [
                URLQueryItem(name: "count", value: "1000"),
                URLQueryItem(name: "find[dateString][$gte]", value: since),
            ]
            if let upperBound {
                queryItems.append(URLQueryItem(name: "find[dateString][$lt]", value: ISO8601DateFormatter().string(from: upperBound)))
            }

            let json = try await requestJSON(
                baseURL: baseURL,
                path: "/api/v1/entries.json",
                queryItems: queryItems,
                authMode: authMode,
                credential: credential
            )
            let docs = json as? [[String: Any]] ?? []
            guard !docs.isEmpty else { break }

            var oldest: Date?
            for doc in docs {
                if let sgv = doubleValue(doc["sgv"]) {
                    values.append(sgv)
                }
                if let date = parseNightscoutDate(doc["dateString"]) ?? parseNightscoutDate(doc["date"]) {
                    if oldest == nil || date < oldest! {
                        oldest = date
                    }
                }
            }

            guard docs.count >= 1000, let oldest, oldest > sinceDate else { break }
            upperBound = oldest.addingTimeInterval(-1)
        }

        guard !values.isEmpty else { return nil }
        let thresholds: (low: Double, high: Double) = {
            if (units ?? "").lowercased().contains("mmol") {
                return (3.9, 10.0)
            }
            return (70, 180)
        }()
        let low = values.filter { $0 < thresholds.low }.count
        let inRange = values.filter { $0 >= thresholds.low && $0 <= thresholds.high }.count
        let high = values.filter { $0 > thresholds.high }.count
        let total = Double(values.count)
        return NightscoutCalibrationSummary(
            windowDays: windowDays,
            sampleCount: values.count,
            recentTIR: Double(inRange) / total,
            recentPercentLow: Double(low) / total,
            recentPercentHigh: Double(high) / total
        )
    }

    private func buildTherapySegments(from profile: [String: Any]?) -> [NightscoutTherapySegment] {
        guard let profile else { return [] }
        let basal = parseSchedule(profile["basal"])
        let sens = parseSchedule(profile["sens"])
        let carb = parseSchedule(profile["carbratio"])
        let boundaries = Set([0] + basal.map(\.minute) + sens.map(\.minute) + carb.map(\.minute)).sorted()
        guard !boundaries.isEmpty else { return [] }

        func value(at minute: Int, from schedule: [(minute: Int, value: Double)]) -> Double? {
            schedule.last(where: { $0.minute <= minute })?.value ?? schedule.last?.value
        }

        var segments: [NightscoutTherapySegment] = []
        for (index, startMinute) in boundaries.enumerated() {
            let endMinute = index + 1 < boundaries.count ? boundaries[index + 1] : 1440
            guard endMinute > startMinute,
                  let basalRate = value(at: startMinute, from: basal),
                  let insulinSensitivity = value(at: startMinute, from: sens),
                  let carbRatio = value(at: startMinute, from: carb) else {
                continue
            }
            segments.append(
                NightscoutTherapySegment(
                    startMinute: startMinute,
                    endMinute: endMinute,
                    basalRate: basalRate,
                    insulinSensitivity: insulinSensitivity,
                    carbRatio: carbRatio
                )
            )
        }
        return segments
    }

    private func parseSchedule(_ raw: Any?) -> [(minute: Int, value: Double)] {
        guard let payloads = raw as? [[String: Any]] else { return [] }
        return payloads.compactMap { item in
            guard let time = item["time"] as? String,
                  let minute = minuteOfDay(from: time),
                  let value = doubleValue(item["value"]) ?? doubleValue(item["rate"]) else {
                return nil
            }
            return (minute: minute, value: value)
        }
        .sorted { $0.minute < $1.minute }
    }

    private func minuteOfDay(from time: String) -> Int? {
        let parts = time.split(separator: ":")
        guard parts.count >= 2,
              let hour = Int(parts[0]),
              let minute = Int(parts[1]) else {
            return nil
        }
        return max(0, min(23, hour)) * 60 + max(0, min(59, minute))
    }

    private func parseNightscoutDate(_ raw: Any?) -> Date? {
        if let string = raw as? String {
            return ISO8601DateFormatter().date(from: string)
        }
        if let millis = doubleValue(raw) {
            return Date(timeIntervalSince1970: millis > 10_000_000_000 ? millis / 1000.0 : millis)
        }
        return nil
    }

    private func doubleValue(_ raw: Any?) -> Double? {
        switch raw {
        case let value as Double:
            return value
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    private func sha1Hex(_ secret: String) -> String {
        let encoded = secret.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? secret
        let digest = Insecure.SHA1.hash(data: Data(encoded.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

@MainActor
final class NightscoutConnectionStore: ObservableObject {
    static let shared = NightscoutConnectionStore()

    @Published private(set) var state = NightscoutConnectionState()

    private let client = NightscoutClient()
    private let defaults = UserDefaults.standard

    private init() {
        refresh()
    }

    func refresh(for uid: String? = Auth.auth().currentUser?.uid) {
        state = loadState(for: uid)
    }

    func connect(baseURLString: String, authMode: NightscoutAuthMode, credential: String, uid: String? = Auth.auth().currentUser?.uid) async throws {
        guard let uid, !uid.isEmpty else { return }
        var nextState = state
        nextState.baseURLString = baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        nextState.authMode = authMode
        nextState.lastErrorMessage = nil
        saveState(nextState, for: uid)
        NightscoutKeychain.save(credential.trimmingCharacters(in: .whitespacesAndNewlines), for: uid)
        state = nextState
        try await validateConnection(for: uid)
    }

    func validateConnection(for uid: String? = Auth.auth().currentUser?.uid) async throws {
        guard let uid, !uid.isEmpty else { return }
        var nextState = loadState(for: uid)
        guard let credential = NightscoutKeychain.load(for: uid), !credential.isEmpty else {
            throw NightscoutConnectionError.missingCredential
        }

        do {
            let summary = try await client.validateAndBootstrap(
                baseURLString: nextState.baseURLString,
                authMode: nextState.authMode,
                credential: credential
            )
            nextState.lastValidatedAt = summary.lastValidatedAt
            nextState.summary = summary
            nextState.lastErrorMessage = nil
            saveState(nextState, for: uid)
            state = nextState
        } catch {
            nextState.lastErrorMessage = error.localizedDescription
            saveState(nextState, for: uid)
            state = nextState
            throw error
        }
    }

    func disconnect(for uid: String? = Auth.auth().currentUser?.uid) {
        guard let uid, !uid.isEmpty else { return }
        NightscoutKeychain.delete(for: uid)
        defaults.removeObject(forKey: key(for: uid))
        state = NightscoutConnectionState()
    }

    func calibrationTargets(for uid: String? = Auth.auth().currentUser?.uid) -> [String: Double] {
        loadState(for: uid).summary?.calibration?.calibrationTargets ?? [:]
    }

    private func key(for uid: String) -> String {
        "NightscoutConnectionState.\(uid)"
    }

    private func loadState(for uid: String?) -> NightscoutConnectionState {
        guard let uid, !uid.isEmpty,
              let data = defaults.data(forKey: key(for: uid)),
              let decoded = try? JSONDecoder().decode(NightscoutConnectionState.self, from: data) else {
            return NightscoutConnectionState()
        }
        return decoded
    }

    private func saveState(_ state: NightscoutConnectionState, for uid: String) {
        guard let data = try? JSONEncoder().encode(state) else { return }
        defaults.set(data, forKey: key(for: uid))
    }
}

// MARK: - ActiveProfileResolver (in-memory; no network per row)
private struct ActiveProfileResolver {
    struct Interval { let start: Date; let end: Date; let snap: TherapySnapshot }
    private let intervals: [Interval]

    init(snapshots: [TherapySnapshot], windowStart: Date, windowEnd: Date, baseline: TherapySnapshot?) {
        var snaps = snapshots.sorted { $0.timestamp < $1.timestamp }
        if let b = baseline, snaps.first?.timestamp != b.timestamp {
            snaps.insert(b, at: 0)
        }
        var ivs: [Interval] = []
        for i in 0..<snaps.count {
            let s = max(windowStart, snaps[i].timestamp)
            let e = (i + 1 < snaps.count) ? min(windowEnd, snaps[i+1].timestamp) : windowEnd
            if s < e { ivs.append(.init(start: s, end: e, snap: snaps[i])) }
        }
        self.intervals = ivs
    }

    func profileId(at date: Date) -> String? {
        intervals.last { date >= $0.start && date < $0.end }?.snap.profileId
    }
}

// MARK: - DataManager
final class DataManager {
    static let shared = DataManager()

    private let fetcher = HealthDataFetcher()
    private let uploader = HealthDataUploader()
    private let chameliaEngine = ChameliaEngine.shared
    private let chameliaStateManager = ChameliaStateManager.shared
    private var authListener: AuthStateDidChangeListenerHandle?

    private init() {
        uploader.refresh(for: Auth.auth().currentUser?.uid)
        authListener = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            guard let self = self else { return }
            if let uid = user?.uid {
                self.uploader.refresh(for: uid)
            } else {
                self.uploader.clear()
            }
        }
    }

    deinit {
        if let handle = authListener {
            Auth.auth().removeStateDidChangeListener(handle)
        }
    }

    func handleLogout(for uid: String?) { uploader.clear() }

    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        fetcher.requestAuthorization(completion: completion)
    }

    func initialBackfillConfiguration(for uid: String? = Auth.auth().currentUser?.uid) -> HealthBackfillConfiguration {
        HealthBackfillConfiguration.load(for: uid)
    }

    func shouldPromptForInitialBackfill(userId: String? = Auth.auth().currentUser?.uid) -> Bool {
        guard let userId, !userId.isEmpty else { return false }
        return !UserDefaults.standard.bool(forKey: "HasDoneInitialSync.\(userId)")
    }

    // MARK: - Sync entrypoint (stops spinner after fetches + writes)
    func syncHealthData(initialBackfill: HealthBackfillConfiguration? = nil, completion: @escaping () -> Void) {
        guard let uid = Auth.auth().currentUser?.uid else {
            print("[DataManager] No authenticated user; skipping health data sync.")
            DispatchQueue.main.async { completion() }
            return
        }
        uploader.refresh(for: uid)

        let tz = TimeZone(identifier: "America/Detroit") ?? .current
        let lastSyncKey = "LastSyncDate.\(uid)"
        let firstRunKey = "HasDoneInitialSync.\(uid)"

        let now = Date()
        let hasDoneInitial = UserDefaults.standard.bool(forKey: firstRunKey)
        let incrementalStartDate: Date = (UserDefaults.standard.object(forKey: lastSyncKey) as? Date)
            ?? Calendar.current.date(byAdding: .day, value: hasDoneInitial ? -3 : -30, to: now)!
        let initialConfiguration = hasDoneInitial ? nil : (initialBackfill ?? initialBackfillConfiguration(for: uid))
        let overallStartDate: Date = {
            if let initialConfiguration,
               initialConfiguration.maximumDays > 0,
               let configuredStart = Calendar.current.date(byAdding: .day, value: -initialConfiguration.maximumDays, to: now) {
                return configuredStart
            }
            return incrementalStartDate
        }()
        let endDate = now

        let startDatesByType: [HealthBackfillDataType: Date] = {
            guard let initialConfiguration else {
                return Dictionary(uniqueKeysWithValues: HealthBackfillDataType.allCases.map { ($0, incrementalStartDate) })
            }
            return Dictionary(uniqueKeysWithValues: HealthBackfillDataType.allCases.compactMap { type in
                guard let startDate = initialConfiguration.startDate(for: type, reference: now) else { return nil }
                return (type, startDate)
            })
        }()
        
        var bg_hourly: [HourlyBgData] = []
        var bg_avg: [HourlyAvgBgData] = []
        var bg_pct: [HourlyBgPercentages] = []
        var bg_uroc: [HourlyBgURoc] = []

        var hr_hourly: [Date: HourlyHeartRateData] = [:]
        var hr_restingDaily: [DailyRestingHeartRateData] = []

        var ex_hourly: [Date: HourlyExerciseData] = [:]

        var sleep_daily: [Date: DailySleepDurations] = [:]      // you already upload this; reuse for CTX
        var energy_hourly: [Date: HourlyEnergyData] = [:]

        var menstrual_daily: [Date: DailyMenstrualData] = [:]

        
        // Two-phase coordination
        let fetches = DispatchGroup()
        let writes  = DispatchGroup()
        

        Task {
            // Build one in-memory resolver (no per-row awaits)
            let snapshots = (try? await TherapySettingsLogManager.shared
                .loadSnapshots(since: overallStartDate, until: endDate)) ?? []
            let baseline = try? await TherapySettingsLogManager.shared
                .getActiveTherapyProfile(at: overallStartDate.addingTimeInterval(-1))
            let resolver = ActiveProfileResolver(
                snapshots: snapshots,
                windowStart: overallStartDate,
                windowEnd: endDate,
                baseline: baseline
            )

            // Fire backfills in parallel (do not hold UI spinner on these)
            self.backfillTherapySettingsByHour(from: overallStartDate, to: endDate, tz: tz)
            self.backfillSiteChangeDaily(from: overallStartDate, to: endDate, tz: tz)

            // ---- Blood Glucose ----
            if let bgStartDate = startDatesByType[.bloodGlucose] {
                fetches.enter()
                let bgInner = DispatchGroup()
                fetcher.fetchAllBgData(start: bgStartDate, end: endDate, group: bgInner) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let (hourly, avg, pct)):
                        bg_hourly = hourly
                        bg_avg    = avg
                        bg_pct    = pct

                        let hourlyEnriched = hourly.map { ($0, resolver.profileId(at: $0.startDate)) }
                        writes.enter()
                        self.uploader.uploadHourlyBgData(hourlyEnriched) { writes.leave() }

                        let avgEnriched = avg.map { ($0, resolver.profileId(at: $0.startDate)) }
                        writes.enter()
                        self.uploader.uploadAverageBgData(avgEnriched) { writes.leave() }

                        let pctEnriched = pct.map { ($0, resolver.profileId(at: $0.startDate)) }
                        writes.enter()
                        self.uploader.uploadHourlyBgPercentages(pctEnriched) { writes.leave() }

                        let uroc = BgAnalytics.computeHourlyURoc(hourlyBgData: hourly, targetBG: 110)

                        bg_uroc = uroc
                        let urocEnriched = uroc.map { ($0, resolver.profileId(at: $0.startDate)) }
                        writes.enter()
                        self.uploader.uploadHourlyBgURoc(urocEnriched) { writes.leave() }

                    case .failure(let err):
                        print("[sync] BG error:", err.localizedDescription)
                    }
                }
            }

            // ---- Heart Rate ----
            if let heartRateStartDate = startDatesByType[.heartRate] {
                fetches.enter()
                let hrInner = DispatchGroup()
                fetcher.fetchHeartRateData(start: heartRateStartDate, end: endDate, group: hrInner) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let (hourly, dailyAvg)):
                        hr_hourly = hourly

                        var enriched: [Date: (HourlyHeartRateData, String?)] = [:]
                        for (d, e) in hourly { enriched[d] = (e, resolver.profileId(at: d)) }
                        writes.enter()
                        self.uploader.uploadHourlyHeartRateData(enriched) { writes.leave() }

                        writes.enter()
                        self.uploader.uploadDailyAverageHeartRateData(dailyAvg) { writes.leave() }

                    case .failure(let err):
                        print("[sync] HR error:", err.localizedDescription)
                    }
                }
            }

            // ---- Exercise ----
            if let exerciseStartDate = startDatesByType[.exercise] {
                fetches.enter()
                let exInner = DispatchGroup()
                fetcher.fetchExerciseData(start: exerciseStartDate, end: endDate, group: exInner) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let (hourly, dailyAvg)):
                        ex_hourly = hourly
                        var enriched: [Date: (HourlyExerciseData, String?)] = [:]
                        for (d, e) in hourly { enriched[d] = (e, resolver.profileId(at: d)) }
                        writes.enter()
                        self.uploader.uploadHourlyExerciseData(enriched) { writes.leave() }

                        writes.enter()
                        self.uploader.uploadDailyAverageExerciseData(dailyAvg) { writes.leave() }

                    case .failure(let err):
                        print("[sync] Exercise error:", err.localizedDescription)
                    }
                }
            }

            // ---- Menstrual ----
            if let menstrualStartDate = startDatesByType[.menstrual] {
                fetches.enter()
                self.fetcher.fetchMenstrualData(start: menstrualStartDate, end: endDate) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let data):
                        menstrual_daily = data
                        writes.enter()
                        self.uploader.uploadMenstrualData(data) { writes.leave() }
                    case .failure(let err):
                        print("[sync] Menstrual error:", err.localizedDescription)
                    }
                }
            }

            // ---- Body Mass ----
            if let bodyMassStartDate = startDatesByType[.bodyMass] {
                fetches.enter()
                let bmInner = DispatchGroup()
                self.fetcher.fetchBodyMassData(start: bodyMassStartDate, end: endDate, group: bmInner) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let data):
                        let enriched = data.map { ($0, resolver.profileId(at: $0.hour)) }
                        writes.enter()
                        self.uploader.uploadBodyMassData(enriched) { writes.leave() }
                    case .failure(let err):
                        print("[sync] BodyMass error:", err.localizedDescription)
                    }
                }
            }

            // ---- Resting HR ----
            if let restingHeartRateStartDate = startDatesByType[.restingHeartRate] {
                fetches.enter()
                self.fetcher.fetchRestingHeartRate(start: restingHeartRateStartDate, end: endDate) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let data):
                        hr_restingDaily = data
                        writes.enter()

                        self.uploader.uploadRestingHeartRateData(data) { writes.leave() }
                    case .failure(let err):
                        print("[sync] RestingHR error:", err.localizedDescription)
                    }
                }
            }

            // ---- Sleep ----
            if let sleepStartDate = startDatesByType[.sleep] {
                fetches.enter()
                self.fetcher.fetchSleepDurations(start: sleepStartDate, end: endDate) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let data):
                        sleep_daily = data
                        writes.enter()
                        self.uploader.uploadSleepDurations(data) { writes.leave() }
                    case .failure(let err):
                        print("[sync] Sleep error:", err.localizedDescription)
                    }
                }
            }

            // ---- Energy ----
            if let energyStartDate = startDatesByType[.energy] {
                fetches.enter()
                let enInner = DispatchGroup()
                self.fetcher.fetchEnergyData(start: energyStartDate, end: endDate, group: enInner) { result in
                    defer { fetches.leave() }
                    switch result {
                    case .success(let (hourly, dailyAvg)):
                        energy_hourly = hourly
                        var enriched: [Date: (HourlyEnergyData, String?)] = [:]
                        for (d, e) in hourly { enriched[d] = (e, resolver.profileId(at: d)) }
                        writes.enter()
                        self.uploader.uploadHourlyEnergyData(enriched) { writes.leave() }

                        writes.enter()
                        self.uploader.uploadDailyAverageEnergyData(dailyAvg) { writes.leave() }

                    case .failure(let err):
                        print("[sync] Energy error:", err.localizedDescription)
                    }
                }
            }

            // Phase 2: after all fetches complete, wait for all writes; then finish.
            fetches.notify(queue: .global()) {
                Task {
                    // ---- Build CTXs safely on a background queue ----
                    func floorToHourUTC(_ d: Date) -> Date {
                        var cal = Calendar(identifier: .gregorian)
                        cal.timeZone = TimeZone(secondsFromGMT: 0)!
                        let comps = cal.dateComponents([.year,.month,.day,.hour], from: d)
                        return cal.date(from: comps)!
                    }
                    let span: ClosedRange<Date> = floorToHourUTC(overallStartDate)...floorToHourUTC(endDate)
                    let nightscoutSummary = await self.validatedNightscoutSummary(for: uid)

                    let bgCtx: [Date: BGCTX] = buildBGCTXByHour(
                        hourly: bg_hourly,
                        avg: bg_avg,
                        pct: bg_pct,
                        uroc: bg_uroc,
                        hourValues: nil
                    )
                    let hrCtx: [Date: HRCTX] = buildHRCTXByHour(
                        hourlyHR: hr_hourly,
                        restingDaily: hr_restingDaily
                    )
                    let energyCtx: [Date: EnergyCTX] = buildEnergyCTXByHour(
                        hourly: energy_hourly
                    )
                    let sleepCtx: [Date: SleepCTX] = buildSleepCTXByHour(
                        hourlySpan: span,
                        daily: sleep_daily,
                        mainWindows: nil,
                        targetSleepMinPerNight: 7.5 * 60.0
                    )
                    let exerciseCtx: [Date: ExerciseCTX] = buildExerciseCTXByHour(
                        hourly: ex_hourly
                    )
                    let menstrualCtx: [Date: MenstrualCTX] = buildMenstrualCtxByHour(
                        daily: menstrual_daily,
                        startUtc: span.lowerBound,
                        endUtc: span.upperBound
                    )
                    let moodEvents = MoodCache.shared.load()
                    let moodCtx: [Date: MoodCTX] = buildMoodCTXByHour(span: span, events: moodEvents, maxCarryHours: 24)
                    let insulinCtx: [Date: InsulinCTX] = buildInsulinCTXByHour(summary: nightscoutSummary, span: span)

                    var latestFrame: FeatureFrameHourly?

                    if let nightscoutSummary {
                        writes.enter()
                        await self.applyNightscoutSummary(nightscoutSummary, userId: uid)
                        writes.leave()
                    }

                    writes.enter()
                    SiteChangeData.shared.buildSiteCtxByHour(startUtc: span.lowerBound, endUtc: span.upperBound) { siteCtx in
                        let frames = makeFeatureFramesHourly(
                            span: span,
                            bg: bgCtx,
                            hr: hrCtx,
                            energy: energyCtx,
                            sleep: sleepCtx,
                            exercise: exerciseCtx,
                            menstrual: menstrualCtx,
                            site: siteCtx,
                            mood: moodCtx,
                            insulin: insulinCtx
                        )
                        latestFrame = frames.max(by: { $0.hourStartUtc < $1.hourStartUtc })

                        if !frames.isEmpty {
                            self.uploader.uploadFeatureFramesHourly(frames) {
                                writes.leave()
                            }
                        } else {
                            writes.leave()
                        }
                    }

                    writes.notify(queue: .main) {
                        Task {
                            if
                                let currentUser = Auth.auth().currentUser,
                                currentUser.uid == uid,
                                !currentUser.isAnonymous,
                                let latestFrame
                            {
                                await self.syncChameliaAfterHealthSync(
                                    userId: currentUser.uid,
                                    frame: latestFrame,
                                    syncDate: endDate
                                )
                            }

                            await MainActor.run {
                                UserDefaults.standard.set(endDate, forKey: lastSyncKey)
                                if !hasDoneInitial {
                                    initialConfiguration?.save(for: uid)
                                    UserDefaults.standard.set(true, forKey: firstRunKey)
                                }
                                completion()
                            }
                        }
                    }
                }
            }

        }
    }
}

private extension DataManager {
    func validatedNightscoutSummary(for userId: String) async -> NightscoutBootstrapSummary? {
        do {
            try await NightscoutConnectionStore.shared.validateConnection(for: userId)
            let state = await MainActor.run { NightscoutConnectionStore.shared.state }
            return state.summary
        } catch {
            print("[DataManager] Nightscout sync skipped/failed: \(error.localizedDescription)")
            return nil
        }
    }

    func applyNightscoutSummary(_ summary: NightscoutBootstrapSummary, userId: String) async {
        if !summary.therapySegments.isEmpty {
            let snapshot = TherapySnapshot(
                timestamp: summary.latestProfileStartDate ?? summary.lastValidatedAt,
                profileId: "nightscout_active",
                profileName: summary.latestProfileName ?? "Nightscout Imported Profile",
                hourRanges: summary.therapySegments.map {
                    HourRange(
                        startMinute: $0.startMinute,
                        endMinute: $0.endMinute,
                        carbRatio: $0.carbRatio,
                        basalRate: $0.basalRate,
                        insulinSensitivity: $0.insulinSensitivity
                    )
                },
                therapyFunctionV2: nil
            )

            do {
                let existing = try await TherapySettingsLogManager.shared.getLatestValidTherapySnapshot(limit: 6)
                let isDuplicate = existing?.profileName == snapshot.profileName &&
                    existing?.timestamp == snapshot.timestamp &&
                    existing?.hourRanges == snapshot.hourRanges
                if !isDuplicate {
                    try await TherapySettingsLogManager.shared.logImportedSnapshot(snapshot, uid: userId)
                }
            } catch {
                print("[DataManager] Nightscout therapy import failed: \(error.localizedDescription)")
            }
        }

        await withCheckedContinuation { continuation in
            self.uploader.uploadNightscoutInsulinContext(summary) {
                continuation.resume()
            }
        }
    }

    func syncNightscoutAfterHealthSync(userId: String) async {
        guard let summary = await validatedNightscoutSummary(for: userId) else { return }
        await applyNightscoutSummary(summary, userId: userId)
    }

    func syncChameliaAfterHealthSync(userId: String, frame: FeatureFrameHourly, syncDate: Date) async {
        let signalBlob = FeatureFrameToChameliaAdapter.makeSignalBlob(from: frame)
        let numericSignals = signalBlob.numericSignals
        guard !numericSignals.isEmpty else {
            print("[DataManager] Skipping Chamelia sync: no numeric signals for latest frame.")
            return
        }

        var latestStatus: GraduationStatus?
        var latestRecommendation: RecommendationPackage?

        print(
            "[DataManager] Chamelia sync start user=\(userId) frame=\(signalBlob.hourStartUtc) numericSignals=\(numericSignals.count)"
        )

        do {
            try await chameliaEngine.observe(
                patientId: userId,
                timestamp: signalBlob.hourStartUtc.timeIntervalSince1970,
                signals: numericSignals
            )
        } catch ChameliaError.notFound {
            print("[DataManager] Chamelia patient not initialized; skipping observe/step.")
            return
        } catch {
            print("[DataManager] Chamelia observe failed: \(error)")
            return
        }

        do {
            latestStatus = try await chameliaEngine.graduationStatus(patientId: userId)
        } catch ChameliaError.notFound {
            print("[DataManager] Chamelia patient not initialized while loading status.")
        } catch {
            print("[DataManager] Chamelia graduation status failed: \(error)")
        }

        if shouldRunDailyChameliaStep(userId: userId, on: syncDate) {
            do {
                print("[DataManager] Running daily Chamelia step for user=\(userId)")
                let connectedAppCapabilities = buildConnectedAppCapabilities()
                let connectedAppState = buildConnectedAppState()
                let stepResponse = try await chameliaEngine.stepResult(
                    patientId: userId,
                    timestamp: signalBlob.hourStartUtc.timeIntervalSince1970,
                    signals: numericSignals,
                    connectedAppCapabilities: connectedAppCapabilities,
                    connectedAppState: connectedAppState
                )
                latestRecommendation = stepResponse.recommendation
                markDailyChameliaStepRan(userId: userId, on: syncDate)
                print("[DataManager] Daily Chamelia step completed for user=\(userId)")
                let statusSnapshot = latestStatus
                let recommendationSnapshot = latestRecommendation
                await MainActor.run {
                    ChameliaDashboardStore.shared.update(
                        userId: userId,
                        status: statusSnapshot,
                        recId: stepResponse.recId,
                        recommendation: recommendationSnapshot,
                        latestSignals: numericSignals
                    )
                }
            } catch ChameliaError.notFound {
                print("[DataManager] Chamelia patient not initialized; skipping daily step.")
            } catch {
                print("[DataManager] Chamelia step failed: \(error)")
            }
        } else {
            print("[DataManager] Daily Chamelia step already ran for user=\(userId)")
        }

        let statusSnapshot = latestStatus
        let recommendationSnapshot = latestRecommendation
        await MainActor.run {
            ChameliaDashboardStore.shared.update(
                userId: userId,
                status: statusSnapshot,
                recommendation: recommendationSnapshot,
                latestSignals: numericSignals,
                clearRecommendation: false
            )
        }

        if shouldRunDailyChameliaSave(userId: userId, on: syncDate) {
            do {
                print("[DataManager] Running daily Chamelia save for user=\(userId)")
                _ = try await chameliaStateManager.saveToFirebase(userId: userId)
                markDailyChameliaSaveRan(userId: userId, on: syncDate)
                print("[DataManager] Daily Chamelia save completed for user=\(userId)")
            } catch {
                print("[DataManager] Chamelia save failed: \(error)")
            }
        } else {
            print("[DataManager] Daily Chamelia save already ran for user=\(userId)")
        }
    }

    func shouldRunDailyChameliaStep(userId: String, on date: Date) -> Bool {
        let key = "LastChameliaStepDate.\(userId)"
        return !Calendar.current.isDate(UserDefaults.standard.object(forKey: key) as? Date ?? .distantPast, inSameDayAs: date)
    }

    func markDailyChameliaStepRan(userId: String, on date: Date) {
        UserDefaults.standard.set(date, forKey: "LastChameliaStepDate.\(userId)")
    }

    func shouldRunDailyChameliaSave(userId: String, on date: Date) -> Bool {
        let key = "LastChameliaSyncSaveDate.\(userId)"
        return !Calendar.current.isDate(UserDefaults.standard.object(forKey: key) as? Date ?? .distantPast, inSameDayAs: date)
    }

    func markDailyChameliaSaveRan(userId: String, on date: Date) {
        UserDefaults.standard.set(date, forKey: "LastChameliaSyncSaveDate.\(userId)")
    }

    func buildConnectedAppCapabilities() -> ConnectedAppCapabilities {
        let level2Enabled = ChameliaSettingsStore.level2Enabled()
        return .insiteDefaults(level2Enabled: level2Enabled)
    }

    func buildConnectedAppState() -> ConnectedAppState {
        let level2Enabled = ChameliaSettingsStore.level2Enabled()
        let store = ProfileDataStore()
        let profiles = store.loadProfiles()
        let activeProfile = activeProfile(in: profiles, store: store)
        let segments = activeProfile.map(makeTherapySegments(from:)) ?? []
        let summaries = profiles.map { profile in
            ProfileSummary(
                id: profile.id,
                name: profile.name,
                segmentCount: profile.hourRanges.count
            )
        }

        return ConnectedAppState(
            scheduleVersion: activeProfile?.id ?? "unspecified",
            currentSegments: segments,
            allowStructuralRecommendations: level2Enabled,
            allowContinuousSchedule: false,
            activeProfileId: activeProfile?.id,
            availableProfiles: summaries
        )
    }

    func activeProfile(in profiles: [DiabeticProfile], store: ProfileDataStore) -> DiabeticProfile? {
        if let activeId = store.loadActiveProfileID(),
           let profile = profiles.first(where: { $0.id == activeId }) {
            return profile
        }
        return profiles.first
    }

    func makeTherapySegments(from profile: DiabeticProfile) -> [TherapySegmentConfig] {
        profile.hourRanges.map { range in
            return TherapySegmentConfig(
                segmentId: stableSegmentId(forStartMin: range.startMinute, endMin: range.endMinute),
                startMin: range.startMinute,
                endMin: range.endMinute,
                isf: range.insulinSensitivity,
                cr: range.carbRatio,
                basal: range.basalRate
            )
        }
    }

    func stableSegmentId(forStartMin startMin: Int, endMin: Int) -> String {
        "\(startMin)-\(endMin)"
    }
}

// MARK: - Therapy hourly backfill (unchanged logic)
extension DataManager {
    func backfillTherapySettingsByHour(from startDate: Date, to endDate: Date, tz: TimeZone = .current) {
            Task {
                let key = "LastTherapyHourBackfill"
                let last = (UserDefaults.standard.object(forKey: key) as? Date)
                let windowStart = max(last ?? startDate, startDate)
                let windowEnd   = endDate

                var snaps = (try? await TherapySettingsLogManager.shared.loadSnapshots(since: windowStart, until: windowEnd)) ?? []

                if let baseline = try? await TherapySettingsLogManager.shared
                    .getActiveTherapyProfile(at: windowStart.addingTimeInterval(-1)),
                   snaps.first?.timestamp != baseline.timestamp {
                    snaps.insert(baseline, at: 0)
                }

                guard !snaps.isEmpty else {
                    print("No therapy snapshots; skipping therapy hourly backfill")
                    UserDefaults.standard.set(windowEnd, forKey: key)
                    return
                }
                snaps.sort { $0.timestamp < $1.timestamp }

                struct Interval { let start: Date; let end: Date; let snap: TherapySnapshot }
                var intervals: [Interval] = []
                for i in 0..<snaps.count {
                    let s = max(windowStart, snaps[i].timestamp)
                    let e = (i + 1 < snaps.count) ? min(windowEnd, snaps[i+1].timestamp) : windowEnd
                    if s < e { intervals.append(.init(start: s, end: e, snap: snaps[i])) }
                }
                guard !intervals.isEmpty else {
                    print("No intervals within window; skipping")
                    UserDefaults.standard.set(windowEnd, forKey: key)
                    return
                }

                var hours: [TherapyHour] = []
                for hourStart in eachHourUTC(from: intervals.first!.start, to: intervals.last!.end) {
                    guard let iv = intervals.last(where: { hourStart >= $0.start && hourStart < $0.end }) else { continue }

                    // ---- NEW: get a V2 schedule for this snapshot ----
                    let scheduleTZ = TimeZone(identifier: iv.snap.therapyFunctionV2?.tzIdentifier ?? tz.identifier) ?? tz
                    let v2: TherapyFunctionV2 = {
                        if let s = iv.snap.therapyFunctionV2 { return s }
                        return makeV2(from: iv.snap.hourRanges, tz: scheduleTZ)
                    }()

                    // Localize the hourStart to the schedule's TZ
                    var cal = Calendar(identifier: .gregorian); cal.timeZone = scheduleTZ
                    let localHourStart = hourStart // same instant, different calendar interpretation handled by value(at:)

                    // Evaluate exact settings at the *start of the local hour*
                    let (basal, isf, cr) = v2.value(at: localHourStart)
                    let lh = cal.component(.hour, from: localHourStart)

                    hours.append(.init(
                        hourStartUtc: hourStart,
                        profileId: iv.snap.profileId,
                        profileName: iv.snap.profileName,
                        snapshotTimestamp: iv.snap.timestamp,
                        carbRatio: cr,
                        basalRate: basal,
                        insulinSensitivity: isf,
                        localTz: scheduleTZ,
                        localHour: lh
                    ))
                }

                print("Therapy hourly to upload: \(hours.count) rows [\(intervals.first!.start) – \(intervals.last!.end)]")
                guard !hours.isEmpty else {
                    UserDefaults.standard.set(windowEnd, forKey: key)
                    return
                }

                uploader.uploadTherapySettingsByHour(hours)
                UserDefaults.standard.set(windowEnd, forKey: key)
            }
        }

    // --- helpers ---
    private func eachHourUTC(from start: Date, to end: Date) -> [Date] {
        var out: [Date] = []
        let cal = Calendar(identifier: .gregorian)
        var cur = floorToHourUTC(start)
        let stop = floorToHourUTC(end)
        while cur <= stop {
            out.append(cur)
            cur = cal.date(byAdding: .hour, value: 1, to: cur)!
        }
        return out
    }

    private func floorToHourUTC(_ d: Date) -> Date {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        let comps = cal.dateComponents([.year,.month,.day,.hour], from: d)
        return cal.date(from: comps)!
    }

    private func localMinute(for utcHourStart: Date, tz: TimeZone) -> Int {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = tz
        let comps = cal.dateComponents([.hour, .minute], from: utcHourStart)
        return (comps.hour ?? 0) * 60 + (comps.minute ?? 0)
    }

    private func rangeFor(localMinute: Int, in ranges: [HourRange]) -> HourRange? {
        return ranges
            .filter { $0.contains(minuteOfDay: localMinute) }
            .sorted { span($0) < span($1) }
            .first
    }

    private func span(_ r: HourRange) -> Int {
        r.durationMinutes
    }
}

// MARK: - Site-change daily backfill
extension DataManager {
    func backfillSiteChangeDaily(from startDate: Date, to endDate: Date, tz: TimeZone = .current) {
        Task { [weak self] in
            guard let self = self else { return }
            guard let uid = Auth.auth().currentUser?.uid else {
                print("[DataManager] No authenticated user; skipping site-change daily backfill.")
                return
            }
            self.uploader.refresh(for: uid)

            let db = Firestore.firestore()
            let eventsRef = db.collection("users").document(uid)
                .collection("site_changes").document("events")
                .collection("items")
            let dailyRef = db.collection("users").document(uid)
                .collection("site_changes").document("daily")
                .collection("items")

            var cal = Calendar(identifier: .gregorian)
            cal.timeZone = tz

            // --- Clamp “daily” seed to strictly before endDate's start-of-day (ignore "today") ---
            let today = cal.startOfDay(for: endDate)

            let isoDay = ISO8601DateFormatter()
            isoDay.timeZone = TimeZone(secondsFromGMT: 0)
            isoDay.formatOptions = [.withFullDate]

            // Use string compare on ISO full-date (lexicographic-safe)
            let todayStr = isoDay.string(from: today)

            let latestDailySnap = try? await dailyRef
                .whereField("dateUtc", isLessThan: todayStr)  // < today only
                .order(by: "dateUtc", descending: true)
                .limit(to: 1)
                .getDocuments()

            let lastDailyDateStr = latestDailySnap?.documents.first?.data()["dateUtc"] as? String
            let lastDailyDate: Date? = lastDailyDateStr.flatMap { isoDay.date(from: $0) }
            let dayAfterLastDaily = lastDailyDate.map { cal.date(byAdding: .day, value: 1, to: $0)! }

            // --- Derive writeStartAnchor AFTER clamp; never later than startOfDay(endDate) ---
            let unclampedAnchor = [dayAfterLastDaily, startDate].compactMap { $0 }.max() ?? startDate
            let writeStartAnchor = min(unclampedAnchor, today)

            // --- Fetch baseline + window events bounded by writeStartAnchor .. endDate ---
            let baselineSnap = try? await eventsRef
                .order(by: "timestamp", descending: true)
                .whereField("timestamp", isLessThan: Timestamp(date: writeStartAnchor))
                .limit(to: 1)
                .getDocuments()

            let windowSnap = try? await eventsRef
                .order(by: "timestamp", descending: false)
                .whereField("timestamp", isGreaterThanOrEqualTo: Timestamp(date: writeStartAnchor))
                .whereField("timestamp", isLessThanOrEqualTo: Timestamp(date: endDate))
                .getDocuments()

            struct Ev { let date: Date; let location: String }
            var events: [Ev] = []
            if let b = baselineSnap?.documents.first {
                let d = b.data()
                if let ts = (d["timestamp"] as? Timestamp)?.dateValue()
                    ?? (d["createdAt"] as? Timestamp)?.dateValue(),
                   let loc = d["location"] as? String {
                    events.append(Ev(date: ts, location: loc))
                }
            }
            if let w = windowSnap?.documents {
                for doc in w {
                    let d = doc.data()
                    if let ts = (d["timestamp"] as? Timestamp)?.dateValue()
                        ?? (d["createdAt"] as? Timestamp)?.dateValue(),
                       let loc = d["location"] as? String {
                        events.append(Ev(date: ts, location: loc))
                    }
                }
            }

            guard !events.isEmpty else {
                print("No site-change events found; skipping daily backfill.")
                return
            }
            events.sort { $0.date < $1.date }

            struct Seg { let start: Date; let end: Date; let origin: Date; let location: String }
            var segs: [Seg] = []
            for i in 0..<events.count {
                let e = events[i]
                let segStart = max(cal.startOfDay(for: writeStartAnchor), cal.startOfDay(for: e.date))
                let nextStart: Date = {
                    if i + 1 < events.count {
                        return cal.date(byAdding: .day, value: -1, to: cal.startOfDay(for: events[i+1].date))!
                    } else {
                        return today
                    }
                }()
                let segEnd = min(today, nextStart)
                if segStart <= segEnd {
                    segs.append(.init(start: segStart, end: segEnd,
                                      origin: cal.startOfDay(for: e.date),
                                      location: e.location))
                }
            }

            var rows: [(Date, Int, String)] = []
            for s in segs {
                var cur = s.start
                while cur <= s.end {
                    let days = cal.dateComponents([.day], from: s.origin, to: cur).day ?? 0
                    rows.append((cur, max(0, days), s.location))
                    cur = cal.date(byAdding: .day, value: 1, to: cur)!
                }
            }

            guard !rows.isEmpty else {
                print("No daily rows to upsert.")
                return
            }
            self.uploader.upsertDailySiteStatus(rows.map { (date: $0.0, daysSince: $0.1, location: $0.2) })
        }
    }
}


extension DataManager {
    func recordMood(_ point: MoodPoint, completion: (() -> Void)? = nil) {
        uploader.uploadMoodEvents([point], onDone: completion)
    }
}
