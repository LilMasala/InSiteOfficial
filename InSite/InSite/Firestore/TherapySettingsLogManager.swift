import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift
import FirebaseAuth

enum TherapySettingsLogManagerError: LocalizedError {
    case missingAuthenticatedUser
    case invalidSnapshotDocument(String)
    case noValidRemoteSnapshot(String)

    var errorDescription: String? {
        switch self {
        case .missingAuthenticatedUser:
            return "No authenticated user is available for therapy snapshot hydration."
        case .invalidSnapshotDocument(let reason):
            return reason
        case .noValidRemoteSnapshot(let reason):
            return reason
        }
    }
}

struct TherapySnapshot: Codable, Identifiable {
    @DocumentID var id: String?
    var timestamp: Date
    var profileId: String
    var profileName: String
    var hourRanges: [HourRange]
    var therapyFunctionV2: TherapyFunctionV2?
}

final class TherapySettingsLogManager {
    static let shared = TherapySettingsLogManager()
    private init() {}

    private func logCollection(for uid: String) -> CollectionReference {
        Firestore.firestore().collection("users").document(uid).collection("therapy_settings_log")
    }

    private var cache: [TherapySnapshot] = []

    private func decodeSnapshot(from document: QueryDocumentSnapshot) throws -> TherapySnapshot {
        let raw = document.data()

        guard let timestamp = timestamp(from: raw["timestamp"]) else {
            throw TherapySettingsLogManagerError.invalidSnapshotDocument(
                "Document \(document.documentID) is missing a valid timestamp."
            )
        }

        let profileId = (raw["profileId"] as? String)
            ?? (raw["profile_id"] as? String)
            ?? document.documentID
        let profileName = (raw["profileName"] as? String)
            ?? (raw["profile_name"] as? String)
            ?? "Imported Profile"

        let therapyFunctionV2 = try decodeTherapyFunctionV2(from: raw["therapyFunctionV2"] ?? raw["therapy_function_v2"])
        let decodedHourRanges = try decodeHourRanges(from: raw["hourRanges"] ?? raw["hour_ranges"])
        let hourRanges = !decodedHourRanges.isEmpty
            ? decodedHourRanges
            : (therapyFunctionV2.map(coarsenToHours) ?? [])

        return TherapySnapshot(
            id: document.documentID,
            timestamp: timestamp,
            profileId: profileId,
            profileName: profileName,
            hourRanges: hourRanges,
            therapyFunctionV2: therapyFunctionV2
        )
    }

    private func decodeHourRanges(from raw: Any?) throws -> [HourRange] {
        guard let raw else { return [] }
        guard let rangePayloads = raw as? [[String: Any]] else {
            throw TherapySettingsLogManagerError.invalidSnapshotDocument("hourRanges is not an array of dictionaries.")
        }

        return try rangePayloads.enumerated().map { index, payload in
            let idString = payload["id"] as? String
            let id = idString.flatMap(UUID.init(uuidString:)) ?? UUID()
            let carbRatio = (payload["carbRatio"] as? NSNumber)?.doubleValue
                ?? (payload["carb_ratio"] as? NSNumber)?.doubleValue
            let basalRate = (payload["basalRate"] as? NSNumber)?.doubleValue
                ?? (payload["basal_rate"] as? NSNumber)?.doubleValue
            let insulinSensitivity = (payload["insulinSensitivity"] as? NSNumber)?.doubleValue
                ?? (payload["insulin_sensitivity"] as? NSNumber)?.doubleValue

            guard let carbRatio, let basalRate, let insulinSensitivity else {
                throw TherapySettingsLogManagerError.invalidSnapshotDocument(
                    "hourRanges[\(index)] is missing carb ratio, basal rate, or insulin sensitivity."
                )
            }

            if let startMinute = (payload["startMinute"] as? NSNumber)?.intValue
                ?? (payload["start_minute"] as? NSNumber)?.intValue,
               let endMinute = (payload["endMinute"] as? NSNumber)?.intValue
                ?? (payload["end_minute"] as? NSNumber)?.intValue {
                return HourRange(
                    id: id,
                    startMinute: startMinute,
                    endMinute: endMinute,
                    carbRatio: carbRatio,
                    basalRate: basalRate,
                    insulinSensitivity: insulinSensitivity
                )
            }

            let startHour = (payload["startHour"] as? NSNumber)?.intValue
                ?? (payload["start_hour"] as? NSNumber)?.intValue
            let endHour = (payload["endHour"] as? NSNumber)?.intValue
                ?? (payload["end_hour"] as? NSNumber)?.intValue

            guard let startHour, let endHour else {
                throw TherapySettingsLogManagerError.invalidSnapshotDocument(
                    "hourRanges[\(index)] is missing minute or hour bounds."
                )
            }

            return HourRange(
                id: id,
                startHour: startHour,
                endHour: endHour,
                carbRatio: carbRatio,
                basalRate: basalRate,
                insulinSensitivity: insulinSensitivity
            )
        }
    }

    private func decodeTherapyFunctionV2(from raw: Any?) throws -> TherapyFunctionV2? {
        guard let raw else { return nil }
        guard let payload = raw as? [String: Any] else {
            throw TherapySettingsLogManagerError.invalidSnapshotDocument("therapyFunctionV2 is not a dictionary.")
        }

        let tzIdentifier = (payload["tzIdentifier"] as? String)
            ?? (payload["tz_identifier"] as? String)
            ?? TimeZone.current.identifier
        let version = (payload["version"] as? NSNumber)?.intValue ?? 2
        let resolutionMin = (payload["resolutionMin"] as? NSNumber)?.intValue
            ?? (payload["resolution_min"] as? NSNumber)?.intValue
            ?? 30

        guard let knotPayloads = payload["knots"] as? [[String: Any]] else {
            throw TherapySettingsLogManagerError.invalidSnapshotDocument("therapyFunctionV2.knots is missing or invalid.")
        }

        let knots = try knotPayloads.enumerated().map { index, knot in
            guard
                let offsetMin = (knot["offsetMin"] as? NSNumber)?.intValue ?? (knot["offset_min"] as? NSNumber)?.intValue,
                let basalRate = (knot["basalRate"] as? NSNumber)?.doubleValue ?? (knot["basal_rate"] as? NSNumber)?.doubleValue,
                let insulinSensitivity = (knot["insulinSensitivity"] as? NSNumber)?.doubleValue ?? (knot["insulin_sensitivity"] as? NSNumber)?.doubleValue,
                let carbRatio = (knot["carbRatio"] as? NSNumber)?.doubleValue ?? (knot["carb_ratio"] as? NSNumber)?.doubleValue
            else {
                throw TherapySettingsLogManagerError.invalidSnapshotDocument(
                    "therapyFunctionV2.knots[\(index)] is missing required values."
                )
            }

            return TherapyFunctionV2.Knot(
                offsetMin: offsetMin,
                basalRate: basalRate,
                insulinSensitivity: insulinSensitivity,
                carbRatio: carbRatio
            )
        }

        return TherapyFunctionV2(
            version: version,
            tzIdentifier: tzIdentifier,
            resolutionMin: resolutionMin,
            knots: knots.sorted { $0.offsetMin < $1.offsetMin }
        )
    }

    private func timestamp(from raw: Any?) -> Date? {
        if let timestamp = raw as? Timestamp {
            return timestamp.dateValue()
        }
        if let date = raw as? Date {
            return date
        }
        return nil
    }

    func logTherapySettingsChange(profile: DiabeticProfile, timestamp: Date = Date()) async throws {
        guard let uid = Auth.auth().currentUser?.uid else { return }
        let snapshot = TherapySnapshot(timestamp: timestamp, profileId: profile.id, profileName: profile.name, hourRanges: profile.hourRanges)
        try logCollection(for: uid).addDocument(from: snapshot)
        cache.append(snapshot)
        cache.sort { $0.timestamp < $1.timestamp }
    }

    func getActiveTherapyProfile(at date: Date) async throws -> TherapySnapshot? {
        if let cached = cache.last(where: { $0.timestamp <= date }) {
            return cached
        }
        guard let uid = Auth.auth().currentUser?.uid else {
            throw TherapySettingsLogManagerError.missingAuthenticatedUser
        }
        let query = logCollection(for: uid)
            .order(by: "timestamp", descending: true)
            .whereField("timestamp", isLessThanOrEqualTo: date)
            .limit(to: 1)
        let snapshot = try await query.getDocuments().documents.first
        if let snapshot = snapshot {
            let snap = try decodeSnapshot(from: snapshot)
            cache.append(snap)
            cache.sort { $0.timestamp < $1.timestamp }
            return snap
        }
        return nil
    }

    func loadSnapshots(since startDate: Date, until endDate: Date) async throws -> [TherapySnapshot] {
        guard let uid = Auth.auth().currentUser?.uid else {
            throw TherapySettingsLogManagerError.missingAuthenticatedUser
        }

        let query = logCollection(for: uid)
            .whereField("timestamp", isGreaterThanOrEqualTo: startDate)
            .whereField("timestamp", isLessThan: endDate)
            .order(by: "timestamp", descending: false)

        let documents = try await query.getDocuments().documents
        var snapshots: [TherapySnapshot] = []
        for document in documents {
            do {
                snapshots.append(try decodeSnapshot(from: document))
            } catch {
                print("[TherapySettingsLogManager] loadSnapshots decode failed doc=\(document.documentID) error=\(error.localizedDescription)")
            }
        }

        if !snapshots.isEmpty {
            cache.append(contentsOf: snapshots)
            cache.sort { $0.timestamp < $1.timestamp }

            var deduped: [TherapySnapshot] = []
            var seenIDs = Set<String>()
            for snapshot in cache.reversed() {
                let key = snapshot.id ?? "\(snapshot.profileId)|\(snapshot.timestamp.timeIntervalSince1970)"
                if seenIDs.insert(key).inserted {
                    deduped.append(snapshot)
                }
            }
            cache = deduped.reversed()
        }

        return snapshots
    }

    func getLatestValidTherapySnapshot(limit: Int = 20) async throws -> TherapySnapshot? {
        guard let uid = Auth.auth().currentUser?.uid else {
            throw TherapySettingsLogManagerError.missingAuthenticatedUser
        }

        let query = logCollection(for: uid)
            .order(by: "timestamp", descending: true)
            .limit(to: limit)

        let documents = try await query.getDocuments().documents
        print("[TherapySettingsLogManager] getLatestValidTherapySnapshot fetched_docs=\(documents.count) uid=\(uid)")

        var decodedSnapshots: [TherapySnapshot] = []
        var decodeErrors: [String] = []

        for document in documents {
            let timestampDescription: String
            if let timestamp = timestamp(from: document.data()["timestamp"]) {
                timestampDescription = ISO8601DateFormatter().string(from: timestamp)
            } else {
                timestampDescription = "missing"
            }

            print("[TherapySettingsLogManager] considering doc_id=\(document.documentID) timestamp=\(timestampDescription)")

            do {
                let snapshot = try decodeSnapshot(from: document)
                decodedSnapshots.append(snapshot)
                print(
                    "[TherapySettingsLogManager] decode_success doc_id=\(document.documentID) profile_id=\(snapshot.profileId) hour_ranges=\(snapshot.hourRanges.count)"
                )
            } catch {
                let message = error.localizedDescription
                decodeErrors.append("doc \(document.documentID): \(message)")
                print("[TherapySettingsLogManager] decode_failure doc_id=\(document.documentID) error=\(message)")
            }
        }

        if !decodedSnapshots.isEmpty {
            cache.append(contentsOf: decodedSnapshots)
            cache.sort { $0.timestamp < $1.timestamp }
        }

        if documents.isEmpty {
            print("[TherapySettingsLogManager] no remote therapy snapshots found")
            return nil
        }

        guard let chosen = decodedSnapshots.first(where: { !$0.hourRanges.isEmpty }) ?? decodedSnapshots.first else {
            throw TherapySettingsLogManagerError.noValidRemoteSnapshot(
                "Fetched \(documents.count) remote therapy snapshots, but none could be decoded. \(decodeErrors.joined(separator: " | "))"
            )
        }

        print(
            "[TherapySettingsLogManager] chosen_snapshot timestamp=\(ISO8601DateFormatter().string(from: chosen.timestamp)) profile_id=\(chosen.profileId) hour_ranges=\(chosen.hourRanges.count)"
        )
        return chosen
    }
}
