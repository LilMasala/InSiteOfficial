//
//  SiteChangeData.swift
//  InSite
//
//  Created by Anand Parikh on 12/14/23.
//

import Foundation
import Firebase
import FirebaseAuth


struct SiteChangeEvent: Codable {
    let timestamp: Date        // server timestamp reflected on read
    let location: String       // consider an enum later
    let localTzId: String
}


class SiteChangeData: ObservableObject {
    static let shared = SiteChangeData()

    @Published var daysSinceSiteChange: Int = 0
    @Published var siteChangeLocation: String = "Not selected"

    private enum DefaultsKey {
        static let lastUpdate = "LastSiteChangeUpdateDate"
        static let lastChangeDate = "LastSiteChangeDate"
        static let lastLocation = "LastSiteChangeLocation"
    }

    private var authListener: AuthStateDidChangeListenerHandle?

    private init() {
        refreshState(for: Auth.auth().currentUser?.uid)
        authListener = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            self?.refreshState(for: user?.uid)
        }
    }

    deinit {
        if let handle = authListener {
            Auth.auth().removeStateDidChangeListener(handle)
        }
    }

    private func key(for base: String, uid: String? = Auth.auth().currentUser?.uid) -> String {
        guard let uid = uid, !uid.isEmpty else { return base }
        return "\(base)_\(uid)"
    }

    private func storedDate(for base: String, uid: String? = Auth.auth().currentUser?.uid) -> Date? {
        UserDefaults.standard.object(forKey: key(for: base, uid: uid)) as? Date
    }

    private func setStoredDate(_ date: Date?, for base: String, uid: String? = Auth.auth().currentUser?.uid) {
        let defaults = UserDefaults.standard
        let key = self.key(for: base, uid: uid)
        if let date {
            defaults.set(date, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    private func storedLocation(uid: String? = Auth.auth().currentUser?.uid) -> String? {
        UserDefaults.standard.string(forKey: key(for: DefaultsKey.lastLocation, uid: uid))
    }

    private func setStoredLocation(_ location: String?, uid: String? = Auth.auth().currentUser?.uid) {
        let defaults = UserDefaults.standard
        let key = self.key(for: DefaultsKey.lastLocation, uid: uid)
        if let location, !location.isEmpty {
            defaults.set(location, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    private var lastUpdateDate: Date? {
        get { storedDate(for: DefaultsKey.lastUpdate) }
        set { setStoredDate(newValue, for: DefaultsKey.lastUpdate) }
    }

    private func lastChangeDate(for uid: String? = Auth.auth().currentUser?.uid) -> Date? {
        storedDate(for: DefaultsKey.lastChangeDate, uid: uid)
    }

    private var lastChangeDate: Date? {
        get { lastChangeDate(for: Auth.auth().currentUser?.uid) }
        set { setStoredDate(newValue, for: DefaultsKey.lastChangeDate) }
    }

    var latestSiteChangeDate: Date? {
        lastChangeDate
    }

    func refreshState(for uid: String? = Auth.auth().currentUser?.uid) {
        guard let uid = uid, !uid.isEmpty else {
            // fall back to local cached values if not signed in
            let location = storedLocation() ?? "Not selected"
            let days = Self.daysSinceChange(from: lastChangeDate)
            DispatchQueue.main.async {
                self.siteChangeLocation = location
                self.daysSinceSiteChange = days
            }
            return
        }

        let db = Firestore.firestore()
        let dailyQuery = db
            .collection("users").document(uid)
            .collection("site_changes").document("daily")
            .collection("items")
            .order(by: "dateUtc", descending: true)
            .limit(to: 1)

        let eventsQuery = db
            .collection("users").document(uid)
            .collection("site_changes").document("events")
            .collection("items")
            .order(by: "createdAt", descending: true)
            .limit(to: 1)

        dailyQuery.getDocuments { snap, err in
            if let err = err {
                print("refreshState site_changes/daily error:", err)
                let location = self.storedLocation(uid: uid) ?? "Not selected"
                let days = Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                DispatchQueue.main.async {
                    self.siteChangeLocation = location
                    self.daysSinceSiteChange = days
                }
            } else if let data = snap?.documents.first?.data() {
                let days = data["daysSinceChange"] as? Int ?? 0
                let loc = data["location"] as? String ?? "Not selected"
                self.setStoredLocation(loc, uid: uid)
                DispatchQueue.main.async {
                    self.siteChangeLocation = loc
                    self.daysSinceSiteChange = days
                }
            } else {
                let location = self.storedLocation(uid: uid) ?? "Not selected"
                let days = Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                DispatchQueue.main.async {
                    self.siteChangeLocation = location
                    self.daysSinceSiteChange = days
                }
            }
        }

        eventsQuery.getDocuments { snap, err in
            if let err = err {
                print("refreshState site_changes/events error:", err)
                return
            }

            guard let data = snap?.documents.first?.data() else { return }
            let remoteDate =
                (data["timestamp"] as? Timestamp)?.dateValue() ??
                (data["createdAt"] as? Timestamp)?.dateValue() ??
                Self.parseClientTimestamp(data["clientTimestamp"])

            let remoteLocation = data["location"] as? String

            if let remoteDate {
                self.setStoredDate(remoteDate, for: DefaultsKey.lastChangeDate, uid: uid)
            }
            if let remoteLocation, !remoteLocation.isEmpty {
                self.setStoredLocation(remoteLocation, uid: uid)
            }

            DispatchQueue.main.async {
                if let remoteLocation, !remoteLocation.isEmpty {
                    self.siteChangeLocation = remoteLocation
                }
                if let remoteDate {
                    self.daysSinceSiteChange = Self.daysSinceChange(from: remoteDate)
                }
            }
        }
    }



    func setSiteChange(location: String) {
        siteChangeLocation = location
        daysSinceSiteChange = 0
        lastChangeDate = Date()
        setStoredLocation(location)
    }

    func updateDaysSinceSiteChange() {
        let days = Self.daysSinceChange(from: lastChangeDate)
        DispatchQueue.main.async {
            self.daysSinceSiteChange = days
        }
    }

    func clearData(for uid: String?) {
        let defaults = UserDefaults.standard
        let keys = [
            key(for: DefaultsKey.lastUpdate, uid: uid),
            key(for: DefaultsKey.lastChangeDate, uid: uid),
            key(for: DefaultsKey.lastLocation, uid: uid)
        ]
        keys.forEach { defaults.removeObject(forKey: $0) }

        if uid == Auth.auth().currentUser?.uid || Auth.auth().currentUser == nil {
            DispatchQueue.main.async {
                self.siteChangeLocation = "Not selected"
                self.daysSinceSiteChange = 0
            }
        }
    }

    private static func daysSinceChange(from last: Date?) -> Int {
        guard let last else { return 0 }
        let calendar = Calendar.current
        let startLast = calendar.startOfDay(for: last)
        let startToday = calendar.startOfDay(for: Date())
        let delta = calendar.dateComponents([.day], from: startLast, to: startToday).day ?? 0
        return max(0, delta)
    }

    private static func parseClientTimestamp(_ raw: Any?) -> Date? {
        guard let text = raw as? String, !text.isEmpty else { return nil }

        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: text) {
            return date
        }

        let basic = ISO8601DateFormatter()
        if let date = basic.date(from: text) {
            return date
        }

        return nil
    }
}



extension SiteChangeData {
    /// Build per-hour SiteCTX from site_changes/daily, using [startUtc, endUtc).
    /// - Note: We fetch one extra prior day so we can compute "repeat vs last" reliably.
    func buildSiteCtxByHour(
        startUtc: Date,
        endUtc: Date,
        tz: TimeZone = .current,
        completion: @escaping ([Date: SiteCTX]) -> Void
    ) {
        // If not signed in, synthesize a constant CTX from local cache
        guard let uid = Auth.auth().currentUser?.uid, !uid.isEmpty else {
            let loc = self.storedLocation() ?? "Not selected"
            let days = Self.daysSinceChange(from: self.lastChangeDate)
            let hours = hourlyBins(start: startUtc, end: endUtc)
            var out: [Date: SiteCTX] = [:]
            for h in hours {
                out[h] = SiteCTX(
                    hourStartUtc: h,
                    daysSinceSiteChange: days,
                    currentSiteLocation: loc,
                    lastSiteLocation: nil,
                    isSiteRepeatLocation: 0   // no history offline, default 0
                )
            }
            completion(out)
            return
        }

        let db = Firestore.firestore()
        let dailyRef = db.collection("users").document(uid)
            .collection("site_changes").document("daily")
            .collection("items")

        let isoDay = ISO8601DateFormatter()
        isoDay.timeZone = TimeZone(secondsFromGMT: 0)
        isoDay.formatOptions = [.withFullDate]

        // Extend window back by 1 day to learn previous location
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        let oneDayBefore = cal.date(byAdding: .day, value: -1, to: cal.startOfDay(for: startUtc)) ?? startUtc

        let qStart = isoDay.string(from: oneDayBefore)
        let qEnd   = isoDay.string(from: endUtc)

        dailyRef
            .whereField("dateUtc", isGreaterThanOrEqualTo: qStart)
            .whereField("dateUtc", isLessThanOrEqualTo: qEnd)
            .order(by: "dateUtc")
            .getDocuments { snap, err in
                // Build rows we got (may be empty)
                var rows: [(date: Date, days: Int, loc: String)] = []
                if let docs = snap?.documents, err == nil {
                    for d in docs {
                        let data = d.data()
                        guard let dstr = data["dateUtc"] as? String,
                              let ddate = isoDay.date(from: dstr) else { continue }
                        let days = data["daysSinceChange"] as? Int ?? 0
                        let loc  = data["location"] as? String ?? "Not selected"
                        rows.append((ddate, days, loc))
                    }
                } else {
                    // Firestore error → soft fallback to cached state for whole window
                    let loc = self.storedLocation(uid: uid) ?? "Not selected"
                    let days = Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                    let hours = hourlyBins(start: startUtc, end: endUtc)
                    var out: [Date: SiteCTX] = [:]
                    for h in hours {
                        out[h] = SiteCTX(
                            hourStartUtc: h,
                            daysSinceSiteChange: days,
                            currentSiteLocation: loc,
                            lastSiteLocation: nil,
                            isSiteRepeatLocation: 0
                        )
                    }
                    completion(out)
                    return
                }

                // Index by UTC midnight
                let byDay: [Date: (days: Int, loc: String)] =
                    Dictionary(uniqueKeysWithValues: rows.map { ($0.date, ($0.days, $0.loc)) })

                // Determine (current, previous) using the last 2 daily rows we fetched
                let sorted = rows.sorted { $0.date < $1.date }
                let currentLoc: String = sorted.last?.loc
                    ?? self.storedLocation(uid: uid)
                    ?? "Not selected"
                let previousLoc: String? = (sorted.count >= 2) ? sorted[sorted.count - 2].loc : nil
                let isRepeatFlag: Int = (previousLoc != nil && previousLoc == currentLoc) ? 1 : 0

                // Render per-hour
                let hours = hourlyBins(start: startUtc, end: endUtc)
                var out: [Date: SiteCTX] = [:]
                for h in hours {
                    let sod = cal.startOfDay(for: h)
                    let tuple = byDay[sod]
                    let days = tuple?.days ?? Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                    let loc  = tuple?.loc  ?? currentLoc
                    out[h] = SiteCTX(
                        hourStartUtc: h,
                        daysSinceSiteChange: days,
                        currentSiteLocation: loc,
                        lastSiteLocation: previousLoc,
                        isSiteRepeatLocation: isRepeatFlag  // <-- always 0/1
                    )
                }
                completion(out)
            }
    }
}

/// Simple UTC-hour bin generator used above.
fileprivate func hourlyBins(start: Date, end: Date) -> [Date] {
    var out: [Date] = []
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(secondsFromGMT: 0)!
    var cur = cal.date(from: cal.dateComponents([.year,.month,.day,.hour], from: start)) ?? start
    let stop = cal.date(from: cal.dateComponents([.year,.month,.day,.hour], from: end)) ?? end
    while cur < end {
        out.append(cur)
        guard let next = cal.date(byAdding: .hour, value: 1, to: cur) else { break }
        cur = next
        if cur > stop { break }
    }
    return out
}


private extension Array {
    subscript(first idx: Int) -> Element { self[idx] }
}
