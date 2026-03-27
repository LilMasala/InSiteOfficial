//
//  HealthDataUploader.swift
//  InSite
//

import Foundation
import FirebaseFirestore
import FirebaseAuth

// MARK: - Canonical cadences & kinds

enum Cadence: String { case hourly, daily, event }

enum DataKind: String {
    case bloodGlucose        = "blood_glucose"
    case heartRate           = "heart_rate"
    case energy              = "energy"
    case exercise            = "exercise"
    case sleep               = "sleep"
    case bodyMass            = "body_mass"
    case restingHeartRate    = "resting_heart_rate"
    case therapySettings     = "therapy_settings"
    case menstrual           = "menstrual"
    case siteChanges         = "site_changes"
    case features    = "features"
    case mood = "mood"

    /// Default subpath per cadence (collections are always .../<subpath>/items)
    func defaultSubpath(for cadence: Cadence) -> String {
        switch (self, cadence) {
        case (.bloodGlucose, .hourly):      return "hourly"
        case (.bloodGlucose, .daily):       return "daily"
        case (.heartRate, .hourly):         return "hourly"
        case (.heartRate, .daily):          return "daily_average"
        case (.energy, .hourly):            return "hourly"
        case (.energy, .daily):             return "daily_average"
        case (.exercise, .hourly):          return "hourly"
        case (.exercise, .daily):           return "daily_average"
        case (.sleep, .daily):              return "daily"
        case (.bodyMass, .hourly):          return "hourly"
        case (.restingHeartRate, .daily):   return "daily"
        case (.therapySettings, .hourly):   return "hourly"
        case (.therapySettings, .event):    return "events"
        case (.menstrual, .daily):          return "daily"
        case (.siteChanges, .daily):        return "daily"
        case (.siteChanges, .event):        return "events"
        case (.features, .hourly):         return "ml_feature_frames"
        case (.mood, .event):              return "events"     // ← NEW
        case (.mood, .hourly):             return "hourly"     // ← NEW
            
        default:                             return cadence.rawValue
        }
    }
}

// MARK: - Time helpers (UTC)

private let isoHour: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.timeZone = TimeZone(secondsFromGMT: 0)
    f.formatOptions = [.withInternetDateTime] // no fractional seconds, stable doc IDs
    return f
}()

private let isoDay: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.timeZone = TimeZone(secondsFromGMT: 0)
    f.formatOptions = [.withFullDate]
    return f
}()

private func floorToHourUTC(_ d: Date) -> Date {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(secondsFromGMT: 0)!
    let c = cal.dateComponents([.year,.month,.day,.hour], from: d)
    return cal.date(from: c)!
}

func isoHourId(_ date: Date) -> String { isoHour.string(from: floorToHourUTC(date)) }
func isoDayId(_ date: Date)  -> String { isoDay.string(from: date) }
func uuidDocId()             -> String { UUID().uuidString }

// MARK: - TherapyHour (unchanged shape)

struct TherapyHour {
    let hourStartUtc: Date
    let profileId: String
    let profileName: String
    let snapshotTimestamp: Date
    let carbRatio: Double
    let basalRate: Double
    let insulinSensitivity: Double
    // optional: local tz info for convenience
    let localTz: TimeZone?
    let localHour: Int?
}

// MARK: - StreamRecord protocol (tiny mappers implement this)

protocol StreamRecord {
    static var kind: DataKind { get }
    static var cadence: Cadence { get }
    /// Override the subpath under the kind (e.g., "percent", "uROC", "average"). Default is kind.defaultSubpath.
    static var subpathOverride: String? { get }

    var documentId: String { get }         // deterministic ID (UTC hour/day) or UUID for events
    var payload: [String: Any] { get }     // merge-safe body (no giant blobs)
}

extension StreamRecord {
    static var subpathOverride: String? { nil }
}

// MARK: - Generic, idempotent uploader

final class FirestoreStreamUploader {
    private let db = Firestore.firestore()
    private let uid: String
    private let batchSize = 450

    init?(uid: String? = Auth.auth().currentUser?.uid) {
        guard let uid = uid else { return nil }
        self.uid = uid
    }

    private func itemsCollection<R: StreamRecord>(_: R.Type) -> CollectionReference {
        let sub = R.subpathOverride ?? R.kind.defaultSubpath(for: R.cadence)
        return db.collection("users")
                 .document(uid)
                 .collection(R.kind.rawValue)
                 .document(sub)
                 .collection("items")
    }

    /// Upsert records (merge) in chunks; replay-safe if documentId is stable.
    func upsert<R: StreamRecord>(
            _ records: [R],
            label: String,
            completion: (() -> Void)? = nil
        ) {
            guard !records.isEmpty else { completion?(); return }
            let col = itemsCollection(R.self)

            let batchSize = 450
            let chunks: [[R]] = stride(from: 0, to: records.count, by: batchSize).map {
                Array(records[$0..<min($0 + batchSize, records.count)])
            }

            let group = DispatchGroup()
            for chunk in chunks {
                group.enter()
                let batch = db.batch()
                for r in chunk {
                    batch.setData(r.payload, forDocument: col.document(r.documentId), merge: true)
                }
                batch.commit { err in
                    if let err = err { print("[\(label)] commit error:", err) }
                    group.leave()
                }
            }
            group.notify(queue: .global()) { completion?() }
        }
}

// MARK: - Mappers (1 small struct per logical stream)

// --- Blood Glucose (hourly raw) ---
struct BGHourlyRecord: StreamRecord {
    static let kind: DataKind = .bloodGlucose
    static let cadence: Cadence = .hourly

    let start: Date
    let end: Date
    let startBg: Double?
    let endBg: Double?
    let therapyProfileId: String?

    var documentId: String { isoHourId(start) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "startUtc": isoHourId(start),
            "endUtc": isoHour.string(from: end)
        ]
        if let v = startBg { d["startBg"] = v }
        if let v = endBg   { d["endBg"]   = v }
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

// --- Blood Glucose (hourly average) ---
struct BGAverageHourlyRecord: StreamRecord {
    static let kind: DataKind = .bloodGlucose
    static let cadence: Cadence = .hourly
    static let subpathOverride: String? = "average"   // goes to blood_glucose/average/items

    let start: Date
    let end: Date
    let averageBg: Double?
    let therapyProfileId: String?

    var documentId: String { isoHourId(start) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "startUtc": isoHourId(start),
            "endUtc": isoHour.string(from: end)
        ]
        if let v = averageBg { d["averageBg"] = v }
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

// --- Blood Glucose (hourly %low/%high) ---
struct BGPercentHourlyRecord: StreamRecord {
    static let kind: DataKind = .bloodGlucose
    static let cadence: Cadence = .hourly
    static let subpathOverride: String? = "percent"

    let start: Date
    let end: Date
    let percentLow: Double
    let percentHigh: Double
    let therapyProfileId: String?

    var documentId: String { isoHourId(start) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "startUtc": isoHourId(start),
            "endUtc": isoHour.string(from: end),
            "percentLow": percentLow,
            "percentHigh": percentHigh
        ]
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

// --- Blood Glucose (hourly uROC) ---
struct BGURocHourlyRecord: StreamRecord {
    static let kind: DataKind = .bloodGlucose
    static let cadence: Cadence = .hourly
    static let subpathOverride: String? = "uROC"

    let start: Date
    let end: Date
    let uRoc: Double?              // mg/dL per sec
    let expectedEndBg: Double?
    let therapyProfileId: String?

    var documentId: String { isoHourId(start) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "startUtc": isoHourId(start),
            "endUtc": isoHour.string(from: end)
        ]
        if let u = uRoc           { d["uRoc"] = u }
        if let e = expectedEndBg  { d["expectedEndBg"] = e }
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

// --- Heart Rate ---
struct HRHourlyRecord: StreamRecord {
    static let kind: DataKind = .heartRate
    static let cadence: Cadence = .hourly

    let hour: Date
    let heartRate: Double
    let therapyProfileId: String?

    var documentId: String { isoHourId(hour) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "hourUtc": isoHourId(hour),
            "heartRate": heartRate
        ]
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

struct HRDailyAverageRecord: StreamRecord {
    static let kind: DataKind = .heartRate
    static let cadence: Cadence = .daily   // goes to heart_rate/daily_average/items

    let date: Date
    let averageHeartRate: Double

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "averageHeartRate": averageHeartRate
        ]
    }
}

// --- Exercise ---
struct ExerciseHourlyRecord: StreamRecord {
    static let kind: DataKind = .exercise
    static let cadence: Cadence = .hourly

    let hour: Date
    let moveMinutes: Double
    let exerciseMinutes: Double
    let totalMinutes: Double
    let therapyProfileId: String?

    var documentId: String { isoHourId(hour) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "hourUtc": isoHourId(hour),
            "moveMinutes": moveMinutes,
            "exerciseMinutes": exerciseMinutes,
            "totalMinutes": totalMinutes
        ]
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

struct ExerciseDailyAverageRecord: StreamRecord {
    static let kind: DataKind = .exercise
    static let cadence: Cadence = .daily  // exercise/daily_average/items

    let date: Date
    let averageMoveMinutes: Double
    let averageExerciseMinutes: Double
    let averageTotalMinutes: Double

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "averageMoveMinutes": averageMoveMinutes,
            "averageExerciseMinutes": averageExerciseMinutes,
            "averageTotalMinutes": averageTotalMinutes
        ]
    }
}

// --- Menstrual ---
struct MenstrualDailyRecord: StreamRecord {
    static let kind: DataKind = .menstrual
    static let cadence: Cadence = .daily

    let date: Date
    let daysSincePeriodStart: Int

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "daysSincePeriodStart": daysSincePeriodStart
        ]
    }
}

// --- Body Mass ---
struct BodyMassHourlyRecord: StreamRecord {
    static let kind: DataKind = .bodyMass
    static let cadence: Cadence = .hourly

    let hour: Date
    let weight: Double
    let therapyProfileId: String?

    var documentId: String { isoHourId(hour) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "hourUtc": isoHourId(hour),
            "weight": weight
        ]
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

// --- Resting HR ---
struct RestingHRDailyRecord: StreamRecord {
    static let kind: DataKind = .restingHeartRate
    static let cadence: Cadence = .daily

    let date: Date
    let restingHeartRate: Double

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "restingHeartRate": restingHeartRate
        ]
    }
}

// --- Sleep ---
struct SleepDailyRecord: StreamRecord {
    static let kind: DataKind = .sleep
    static let cadence: Cadence = .daily

    let date: Date
    let awake: Double
    let asleepCore: Double
    let asleepDeep: Double
    let asleepREM: Double
    let asleepUnspecified: Double

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "awake": awake,
            "asleepCore": asleepCore,
            "asleepDeep": asleepDeep,
            "asleepREM": asleepREM,
            "asleepUnspecified": asleepUnspecified
        ]
    }
}

// --- Energy ---
struct EnergyHourlyRecord: StreamRecord {
    static let kind: DataKind = .energy
    static let cadence: Cadence = .hourly

    let hour: Date
    let basalEnergy: Double
    let activeEnergy: Double
    let totalEnergy: Double
    let therapyProfileId: String?

    var documentId: String { isoHourId(hour) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "hourUtc": isoHourId(hour),
            "basalEnergy": basalEnergy,
            "activeEnergy": activeEnergy,
            "totalEnergy": totalEnergy
        ]
        if let p = therapyProfileId { d["therapyProfileId"] = p }
        return d
    }
}

struct EnergyDailyAverageRecord: StreamRecord {
    static let kind: DataKind = .energy
    static let cadence: Cadence = .daily

    let date: Date
    let averageActiveEnergy: Double

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "averageActiveEnergy": averageActiveEnergy
        ]
    }
}

// --- Therapy Settings (hourly projection of snapshot to UTC hour) ---
struct TherapySettingsHourlyRecord: StreamRecord {
    static let kind: DataKind = .therapySettings
    static let cadence: Cadence = .hourly

    let hourStartUtc: Date
    let profileId: String
    let profileName: String
    let snapshotTimestamp: Date
    let carbRatio: Double
    let basalRate: Double
    let insulinSensitivity: Double
    let localTzId: String?
    let localHour: Int?

    var documentId: String { isoHourId(hourStartUtc) }
    var payload: [String: Any] {
        var d: [String: Any] = [
            "hourStartUtc": isoHourId(hourStartUtc),          // string id you already use
            "hourStartTs": Timestamp(date: hourStartUtc),     // <— add this line
            "profileId": profileId,
            "profileName": profileName,
            "snapshotTimestamp": isoHour.string(from: snapshotTimestamp),
            "carbRatio": carbRatio,
            "basalRate": basalRate,
            "insulinSensitivity": insulinSensitivity
        ]
        if let tz = localTzId { d["localTz"] = tz }
        if let lh = localHour { d["localHour"] = lh }
        return d
    }
}

// --- Site change (event + derived daily) ---
struct SiteChangeEventRecord: StreamRecord {
    static let kind: DataKind = .siteChanges
    static let cadence: Cadence = .event

    let id: String       // use UUID
    let location: String
    let localTzId: String
    let timestamp: Date  // client-side timestamp as hint

    var documentId: String { id }
    var payload: [String: Any] {
        [
            "location": location,
            "localTz": localTzId,
            "clientTimestamp": isoHour.string(from: timestamp),
            "createdAt": FieldValue.serverTimestamp(),
            "timestamp": FieldValue.serverTimestamp() // authoritative
        ]
    }
}

struct SiteChangeDailyRecord: StreamRecord {
    static let kind: DataKind = .siteChanges
    static let cadence: Cadence = .daily

    let date: Date
    let daysSince: Int
    let location: String

    var documentId: String { isoDayId(date) }
    var payload: [String: Any] {
        [
            "dateUtc": isoDayId(date),
            "daysSinceChange": daysSince,
            "location": location,
            "computedAt": FieldValue.serverTimestamp()
        ]
    }
}

// MARK: - Convenience façade (optionally keep this name for call sites)

final class HealthDataUploader {
    private var uploader: FirestoreStreamUploader?
    private var cachedUid: String?
    var skipWrites: Bool = false

    init() {
        refresh(for: Auth.auth().currentUser?.uid)
    }

    func refresh(for uid: String?) {
        guard cachedUid != uid else { return }
        cachedUid = uid
        if let uid = uid, !uid.isEmpty {
            uploader = FirestoreStreamUploader(uid: uid)
        } else {
            uploader = nil
        }
    }

    func clear() {
        cachedUid = nil
        uploader = nil
    }

    private func currentUploader(function: String = #function) -> FirestoreStreamUploader? {
        if uploader == nil {
            refresh(for: Auth.auth().currentUser?.uid)
        }
        guard let uploader else {
            print("[HealthDataUploader] Missing uploader for \(function); ensure user is authenticated before syncing.")
            return nil
        }
        return uploader
    }

    // ---- BG ----
    func uploadHourlyBgData(_ data: [(HourlyBgData, String?)], onDone: (() -> Void)? = nil) {
           guard !skipWrites, let up = currentUploader() else { onDone?(); return }
           let recs = data.map { BGHourlyRecord(start: $0.0.startDate, end: $0.0.endDate,
                                                startBg: $0.0.startBg, endBg: $0.0.endBg,
                                                therapyProfileId: $0.1) }
           up.upsert(recs, label: "bg hourly", completion: onDone)
       }
    

    func uploadAverageBgData(_ data: [(HourlyAvgBgData, String?)], onDone: (() -> Void)? = nil) {
            guard !skipWrites, let up = currentUploader() else { onDone?(); return }
            let recs = data.map { BGAverageHourlyRecord(start: $0.0.startDate, end: $0.0.endDate,
                                                        averageBg: $0.0.averageBg,
                                                        therapyProfileId: $0.1) }
            up.upsert(recs, label: "bg hourly average", completion: onDone)
        }

    func uploadHourlyBgPercentages(_ data: [(HourlyBgPercentages, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [BGPercentHourlyRecord] = data.map { (e, pid) in
            BGPercentHourlyRecord(start: e.startDate, end: e.endDate, percentLow: e.percentLow, percentHigh: e.percentHigh, therapyProfileId: pid)
        }
        up.upsert(recs, label: "bg hourly percent", completion: onDone)
    }

    func uploadHourlyBgURoc(_ data: [(HourlyBgURoc, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [BGURocHourlyRecord] = data.map { (e, pid) in
            BGURocHourlyRecord(start: e.startDate, end: e.endDate, uRoc: e.uRoc, expectedEndBg: e.expectedEndBg, therapyProfileId: pid)
        }
        up.upsert(recs, label: "bg hourly uROC", completion: onDone)
    }

    // ---- HR ----
    func uploadHourlyHeartRateData(_ data: [Date: (HourlyHeartRateData, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [HRHourlyRecord] = data.values.map { (entry, pid) in
            HRHourlyRecord(hour: entry.hour, heartRate: entry.heartRate, therapyProfileId: pid)
        }
        up.upsert(recs, label: "hr hourly", completion: onDone)
    }

    func uploadDailyAverageHeartRateData(_ data: [DailyAverageHeartRateData], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.map { HRDailyAverageRecord(date: $0.date, averageHeartRate: $0.averageHeartRate) }
        up.upsert(recs, label: "hr daily avg", completion: onDone)
    }

    // ---- Exercise ----
    func uploadHourlyExerciseData(_ data: [Date: (HourlyExerciseData, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [ExerciseHourlyRecord] = data.values.map { (e, pid) in
            ExerciseHourlyRecord(hour: e.hour, moveMinutes: e.moveMinutes, exerciseMinutes: e.exerciseMinutes, totalMinutes: e.totalMinutes, therapyProfileId: pid)
        }
        up.upsert(recs, label: "exercise hourly", completion: onDone)
    }

    func uploadDailyAverageExerciseData(_ data: [Date: DailyAverageExerciseData], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.values.map {
            ExerciseDailyAverageRecord(date: $0.date, averageMoveMinutes: $0.averageMoveMinutes, averageExerciseMinutes: $0.averageExerciseMinutes, averageTotalMinutes: $0.averageTotalMinutes)
        }
        up.upsert(recs, label: "exercise daily avg", completion: onDone)
    }

    // ---- Menstrual ----
    func uploadMenstrualData(_ data: [Date: DailyMenstrualData], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.values.map { MenstrualDailyRecord(date: $0.date, daysSincePeriodStart: $0.daysSincePeriodStart) }
        up.upsert(recs, label: "menstrual daily", completion: onDone)
    }

    // ---- Body mass ----
    func uploadBodyMassData(_ data: [(HourlyBodyMassData, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [BodyMassHourlyRecord] = data.map { (e, pid) in
            BodyMassHourlyRecord(hour: e.hour, weight: e.weight, therapyProfileId: pid)
        }
        up.upsert(recs, label: "body mass hourly", completion: onDone)
    }

    // ---- Resting HR ----
    func uploadRestingHeartRateData(_ data: [DailyRestingHeartRateData], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.map { RestingHRDailyRecord(date: $0.date, restingHeartRate: $0.restingHeartRate) }
        up.upsert(recs, label: "resting hr daily", completion: onDone)
    }

    // ---- Sleep ----
    func uploadSleepDurations(_ data: [Date: DailySleepDurations], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.values.map {
            SleepDailyRecord(date: $0.date, awake: $0.awake, asleepCore: $0.asleepCore, asleepDeep: $0.asleepDeep, asleepREM: $0.asleepREM, asleepUnspecified: $0.asleepUnspecified)
        }
        up.upsert(recs, label: "sleep daily", completion: onDone)
    }

    // ---- Energy ----
    func uploadHourlyEnergyData(_ data: [Date: (HourlyEnergyData, String?)], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [EnergyHourlyRecord] = data.values.map { (e, pid) in
            EnergyHourlyRecord(hour: e.hour, basalEnergy: e.basalEnergy, activeEnergy: e.activeEnergy, totalEnergy: e.totalEnergy, therapyProfileId: pid)
        }
        up.upsert(recs, label: "energy hourly", completion: onDone)
    }

    func uploadDailyAverageEnergyData(_ data: [DailyAverageEnergyData], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = data.map { EnergyDailyAverageRecord(date: $0.date, averageActiveEnergy: $0.averageActiveEnergy) }
        up.upsert(recs, label: "energy daily avg", completion: onDone)
    }

    // ---- Therapy settings (hourly) ----
    func uploadTherapySettingsByHour(_ hours: [TherapyHour], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs: [TherapySettingsHourlyRecord] = hours.map { h in
            TherapySettingsHourlyRecord(
                hourStartUtc: h.hourStartUtc,
                profileId: h.profileId,
                profileName: h.profileName,
                snapshotTimestamp: h.snapshotTimestamp,
                carbRatio: h.carbRatio,
                basalRate: h.basalRate,
                insulinSensitivity: h.insulinSensitivity,
                localTzId: h.localTz?.identifier,
                localHour: h.localHour
            )
        }
        up.upsert(recs, label: "therapy settings hourly", completion: onDone)
    }
}

// MARK: - Site change convenience (event write + same-tick daily seed + backfill)

extension HealthDataUploader {
    /// Record a site change (event), seed today's daily=0, then backfill recent days (idempotent).
    func recordSiteChange(location: String,
                          localTz: TimeZone = .current,
                          backfillDays: Int = 14) {
        guard !skipWrites, let up = currentUploader() else { return }

        // 1) Event (UUID doc id; serverTimestamp for authoritative time)
        let ev = SiteChangeEventRecord(
            id: uuidDocId(),
            location: location,
            localTzId: localTz.identifier,
            timestamp: Date()
        )
        up.upsert([ev], label: "site change event")

        // 2) Seed today's derived daily doc for instant UX
        let today = Date()
        let seed = SiteChangeDailyRecord(date: today, daysSince: 0, location: location)
        up.upsert([seed], label: "site change daily seed")

        // 3) Backfill last N days with authoritative event timestamp
        let end = today
        let start = Calendar.current.date(byAdding: .day, value: -backfillDays, to: end) ?? end
        DataManager.shared.backfillSiteChangeDaily(from: start, to: end, tz: localTz)
    }
}

extension HealthDataUploader {
    func upsertDailySiteStatus(_ days: [(date: Date, daysSince: Int, location: String)]) {
        guard !skipWrites, let up = currentUploader() else { return }
        let recs = days.map { SiteChangeDailyRecord(date: $0.date, daysSince: $0.daysSince, location: $0.location) }
        up.upsert(recs, label: "site daily status")
    }
}



extension HealthDataUploader {
    struct FeatureFrameHourlyRecord: StreamRecord {
        static let kind: DataKind = .features        // ← was .bloodGlucose
        static let cadence: Cadence = .hourly
        static let subpathOverride: String? = "ml_feature_frames"

        let f: FeatureFrameHourly
        var documentId: String { isoHourId(f.hourStartUtc) }
        var payload: [String: Any] {
            var d: [String: Any] = ["hourStartUtc": isoHourId(f.hourStartUtc)]
            // write only non-nil fields
            func put(_ k: String, _ v: Any?) { if let v = v { d[k] = v } }
            put("bg_avg", f.bg_avg); put("bg_tir", f.bg_tir)
            put("bg_percentLow", f.bg_percentLow); put("bg_percentHigh", f.bg_percentHigh)
            put("bg_uRoc", f.bg_uRoc); put("bg_deltaAvg7h", f.bg_deltaAvg7h); put("bg_zAvg7h", f.bg_zAvg7h)

            put("hr_mean", f.hr_mean); put("hr_delta7h", f.hr_delta7h)
            put("hr_z7h", f.hr_z7h); put("rhr_daily", f.rhr_daily)

            put("kcal_active", f.kcal_active); put("kcal_active_last3h", f.kcal_active_last3h)
            put("kcal_active_last6h", f.kcal_active_last6h)
            put("kcal_active_delta7h", f.kcal_active_delta7h); put("kcal_active_z7h", f.kcal_active_z7h)

            put("sleep_prev_total_min", f.sleep_prev_total_min)
            put("sleep_debt_7d_min", f.sleep_debt_7d_min)
            put("minutes_since_wake", f.minutes_since_wake)

            put("ex_move_min", f.ex_move_min); put("ex_exercise_min", f.ex_exercise_min)
            put("ex_min_last3h", f.ex_min_last3h); put("ex_hours_since", f.ex_hours_since)

            put("days_since_period_start", f.days_since_period_start)
            put("cycle_follicular", f.cycle_follicular)
            put("cycle_ovulation", f.cycle_ovulation)
            put("cycle_luteal", f.cycle_luteal)

            put("days_since_site_change", f.days_since_site_change)
            put("site_loc_current", f.site_loc_current)
            put("site_loc_same_as_last", f.site_loc_same_as_last)
            
            put("mood_valence", f.mood_valence)
            put("mood_arousal", f.mood_arousal)
            put("mood_quad_posPos", f.mood_quad_posPos)
            put("mood_quad_posNeg", f.mood_quad_posNeg)
            put("mood_quad_negPos", f.mood_quad_negPos)
            put("mood_quad_negNeg", f.mood_quad_negNeg)
            put("mood_hours_since", f.mood_hours_since)
            return d
        }
    }

    func uploadFeatureFramesHourly(_ frames: [FeatureFrameHourly], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        up.upsert(frames.map { FeatureFrameHourlyRecord(f: $0) }, label: "ml feature frames", completion: onDone)
    }
}




extension HealthDataUploader {
    func uploadMoodEvents(_ events: [MoodPoint], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = events.map {
            MoodEventRecord(id: uuidDocId(), timestamp: $0.timestamp, valence: $0.valence, arousal: $0.arousal)
        }
        up.upsert(recs, label: "mood events", completion: onDone)
    }

    func uploadMoodHourlyCtx(_ ctx: [Date: MoodCTX], onDone: (() -> Void)? = nil) {
        guard !skipWrites, let up = currentUploader() else { onDone?(); return }
        let recs = ctx.sorted { $0.key < $1.key }.map { (_, m) in
            MoodHourlyCtxRecord(
                hour: m.hourStartUtc,
                valence: m.valence, arousal: m.arousal,
                quad_posPos: m.quad_posPos, quad_posNeg: m.quad_posNeg,
                quad_negPos: m.quad_negPos, quad_negNeg: m.quad_negNeg,
                hoursSinceMood: m.hoursSinceMood
            )
        }
        up.upsert(recs, label: "mood hourly ctx", completion: onDone)
    }
}
