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

        Firestore.firestore()
            .collection("users").document(uid)
            .collection("site_changes").document("daily")
            .collection("items")
            .order(by: "dateUtc", descending: true)   // "YYYY-MM-DD" strings sort fine
            .limit(to: 1)
            .getDocuments { snap, err in
                if let err = err {
                    print("refreshState site_changes/daily error:", err)
                    // fall back to local cache
                    let location = self.storedLocation(uid: uid) ?? "Not selected"
                    let days = Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                    DispatchQueue.main.async {
                        self.siteChangeLocation = location
                        self.daysSinceSiteChange = days
                    }
                    return
                }
                if let data = snap?.documents.first?.data() {
                    let days = data["daysSinceChange"] as? Int ?? 0
                    let loc  = data["location"] as? String ?? "Not selected"
                    DispatchQueue.main.async {
                        self.siteChangeLocation = loc
                        self.daysSinceSiteChange = days
                    }
                } else {
                    // no rows yet → fall back to local cache
                    let location = self.storedLocation(uid: uid) ?? "Not selected"
                    let days = Self.daysSinceChange(from: self.lastChangeDate(for: uid))
                    DispatchQueue.main.async {
                        self.siteChangeLocation = location
                        self.daysSinceSiteChange = days
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
}
