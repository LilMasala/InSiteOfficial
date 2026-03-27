import Foundation
import Firebase
import FirebaseAuth
import HealthKit

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

    // MARK: - Sync entrypoint (stops spinner after fetches + writes)
    func syncHealthData(completion: @escaping () -> Void) {
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
        let defaultBackfillDays = hasDoneInitial ? -3 : -30
        let startDate: Date = (UserDefaults.standard.object(forKey: lastSyncKey) as? Date)
            ?? Calendar.current.date(byAdding: .day, value: defaultBackfillDays, to: now)!
        let endDate = now
        
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
                .loadSnapshots(since: startDate, until: endDate)) ?? []
            let baseline = try? await TherapySettingsLogManager.shared
                .getActiveTherapyProfile(at: startDate.addingTimeInterval(-1))
            let resolver = ActiveProfileResolver(
                snapshots: snapshots,
                windowStart: startDate,
                windowEnd: endDate,
                baseline: baseline
            )

            // Fire backfills in parallel (do not hold UI spinner on these)
            self.backfillTherapySettingsByHour(from: startDate, to: endDate, tz: tz)
            self.backfillSiteChangeDaily(from: startDate, to: endDate, tz: tz)

            // ---- Blood Glucose ----
            fetches.enter()
            let bgInner = DispatchGroup()
            fetcher.fetchAllBgData(start: startDate, end: endDate, group: bgInner) { result in
                defer { fetches.leave() }
                switch result {
                case .success(let (hourly, avg, pct)):
                    
                    bg_hourly = hourly
                    bg_avg    = avg
                    bg_pct    = pct

                    // hourly
                    let hourlyEnriched = hourly.map { ($0, resolver.profileId(at: $0.startDate)) }
                    writes.enter()
                    self.uploader.uploadHourlyBgData(hourlyEnriched) { writes.leave() }

                    // avg
                    let avgEnriched = avg.map { ($0, resolver.profileId(at: $0.startDate)) }
                    writes.enter()
                    self.uploader.uploadAverageBgData(avgEnriched) { writes.leave() }

                    // pct
                    let pctEnriched = pct.map { ($0, resolver.profileId(at: $0.startDate)) }
                    writes.enter()
                    self.uploader.uploadHourlyBgPercentages(pctEnriched) { writes.leave() }

                    // uROC
                    let uroc = BgAnalytics.computeHourlyURoc(hourlyBgData: hourly, targetBG: 110)
                    
                    bg_uroc = uroc
                    let urocEnriched = uroc.map { ($0, resolver.profileId(at: $0.startDate)) }
                    writes.enter()
                    self.uploader.uploadHourlyBgURoc(urocEnriched) { writes.leave() }

                case .failure(let err):
                    print("[sync] BG error:", err.localizedDescription)
                }
            }

            // ---- Heart Rate ----
            fetches.enter()
            let hrInner = DispatchGroup()
            fetcher.fetchHeartRateData(start: startDate, end: endDate, group: hrInner) { result in
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

            // ---- Exercise ----
            fetches.enter()
            let exInner = DispatchGroup()
            fetcher.fetchExerciseData(start: startDate, end: endDate, group: exInner) { result in
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

            // ---- Menstrual ----
            fetches.enter()
            self.fetcher.fetchMenstrualData(start: startDate, end: endDate) { result in
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

            // ---- Body Mass ----
            fetches.enter()
            let bmInner = DispatchGroup()
            self.fetcher.fetchBodyMassData(start: startDate, end: endDate, group: bmInner) { result in
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

            // ---- Resting HR ----
            fetches.enter()
            self.fetcher.fetchRestingHeartRate(start: startDate, end: endDate) { result in
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

            // ---- Sleep ----
            fetches.enter()
            self.fetcher.fetchSleepDurations(start: startDate, end: endDate) { result in
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

            // ---- Energy ----
            fetches.enter()
            let enInner = DispatchGroup()
            self.fetcher.fetchEnergyData(start: startDate, end: endDate, group: enInner) { result in
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

            // Phase 2: after all fetches complete, wait for all writes; then finish.
            fetches.notify(queue: .global()) {
                // ---- Build CTXs safely on a background queue ----
                // Define the hour span for join (UTC-rounded)
                func floorToHourUTC(_ d: Date) -> Date {
                    var cal = Calendar(identifier: .gregorian)
                    cal.timeZone = TimeZone(secondsFromGMT: 0)!
                    let comps = cal.dateComponents([.year,.month,.day,.hour], from: d)
                    return cal.date(from: comps)!
                }
                let span: ClosedRange<Date> = floorToHourUTC(startDate)...floorToHourUTC(endDate)

                // BG CTX
                let bgCtx: [Date: BGCTX] = buildBGCTXByHour(
                    hourly: bg_hourly,
                    avg: bg_avg,
                    pct: bg_pct,
                    uroc: bg_uroc,
                    hourValues: nil // plug in if you later fetch raw hourly values
                )

                // HR CTX
                let hrCtx: [Date: HRCTX] = buildHRCTXByHour(
                    hourlyHR: hr_hourly,
                    restingDaily: hr_restingDaily
                )

                // Energy CTX
                let energyCtx: [Date: EnergyCTX] = buildEnergyCTXByHour(
                    hourly: energy_hourly
                )

                // Sleep CTX
                let sleepCtx: [Date: SleepCTX] = buildSleepCTXByHour(
                    hourlySpan: span,
                    daily: sleep_daily,
                    mainWindows: nil,                   // plug in when you have main sleep windows
                    targetSleepMinPerNight: 7.5 * 60.0
                )

                // Exercise CTX
                let exerciseCtx: [Date: ExerciseCTX] = buildExerciseCTXByHour(
                    hourly: ex_hourly
                )

                // Menstrual CTX (per-hour expansion from per-day map)
                let menstrualCtx: [Date: MenstrualCTX] = buildMenstrualCtxByHour(
                    daily: menstrual_daily,
                    startUtc: span.lowerBound,
                    endUtc: span.upperBound
                )
                
                // define span earlier as you already do
                let moodEvents = MoodCache.shared.load() // or fetch from Firestore if you prefer
                let moodCtx: [Date: MoodCTX] = buildMoodCTXByHour(span: span, events: moodEvents, maxCarryHours: 24)
                
                var latestFrame: FeatureFrameHourly?

                // Site CTX: leave empty for now (we can add a builder that expands your daily rows to hours)
                writes.enter() // make frames upload part of the "writes" phase
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
                        mood: moodCtx
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
                

                // ---- Finish after all writes (old + frames) complete ----
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
                            if !hasDoneInitial { UserDefaults.standard.set(true, forKey: firstRunKey) }
                            completion()
                        }
                    }
                }
            }

        }
    }
}

private extension DataManager {
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
