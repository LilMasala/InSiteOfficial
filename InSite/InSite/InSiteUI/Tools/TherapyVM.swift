// TherapyVM.swift
import SwiftUI
import Combine

@MainActor
final class TherapyVM: ObservableObject {
    // Inputs/state
    @Published private(set) var profiles: [DiabeticProfile] = []
    @Published private(set) var activeProfile: DiabeticProfile?
    @Published private(set) var currentHourRange: HourRange?

    // Outputs for UI
    @Published private(set) var summaryText: String = "—"
    @Published private(set) var currentBasal: Double = 0
    @Published private(set) var currentISF: Double = 0
    @Published private(set) var currentCarbRatio: Double = 0
    @Published private(set) var currentProfileName: String = "—"
    
    @Published private(set) var sparklineBasal: [Double]? = nil
    @Published private(set) var sparklineISF:   [Double]? = nil
    @Published private(set) var sparklineCR:    [Double]? = nil

    private let store = ProfileDataStore()
    private var timerCancellable: AnyCancellable?
    private var fgObserver: Any?

    init() {
        reload()

        // Recompute “now” every minute (cheap)…
        timerCancellable = Timer.publish(every: 60, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.recompute() }

        // …and when app returns to foreground (hour may have rolled over)
        fgObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.recompute()
            }
        }
    }

    deinit {
        timerCancellable?.cancel()
        if let fgObserver { NotificationCenter.default.removeObserver(fgObserver) }
    }

    
    private func updateSparklines() {
        guard let p = activeProfile else {
            sparklineBasal = nil; sparklineISF = nil; sparklineCR = nil
            return
        }
        // Sort by start hour so the sparkline flows left→right in time
        let ranges = p.hourRanges.sorted { $0.startMinute < $1.startMinute }

        // Use the range values directly (the tile normalizes them)
        sparklineBasal = ranges.map { $0.basalRate }
        sparklineISF   = ranges.map { $0.insulinSensitivity }
        sparklineCR    = ranges.map { $0.carbRatio }
    }

    func reload() {
        profiles = store.loadProfiles()
        if let id = store.loadActiveProfileID(),
           let p = profiles.first(where: { $0.id == id }) {
            activeProfile = p
            recompute() // your existing method that sets currentHourRange, currentBasal, etc.
        } else {
            activeProfile = nil
        }
        updateSparklines()
    }

    func selectProfile(id: String) {
        guard let p = profiles.first(where: { $0.id == id }) else { return }
        activeProfile = p
        store.saveActiveProfileID(id)   // persist selection
        recompute()
    }

    // MARK: - Core compute
    func recompute(reference date: Date = Date()) {
        guard let p = activeProfile else {
            summaryText       = "—"
            currentHourRange  = nil
            currentBasal      = 0
            currentISF        = 0
            currentCarbRatio  = 0
            currentProfileName = "—"
            return
        }

        currentProfileName = p.name

        let components = Calendar.current.dateComponents([.hour, .minute], from: date)
        let minuteOfDay = (components.hour ?? 0) * 60 + (components.minute ?? 0)
        currentHourRange = Self.range(for: minuteOfDay, in: p.hourRanges)

        if let r = currentHourRange {
            currentBasal      = r.basalRate
            currentISF        = r.insulinSensitivity
            currentCarbRatio  = r.carbRatio
            summaryText = "\(p.name) · \(r.timeLabel)"
        } else {
            currentBasal = 0; currentISF = 0; currentCarbRatio = 0
            summaryText = "\(p.name) · No active range"
        }
    }

    // MARK: - Helpers (same logic you used elsewhere)
    static func range(for minuteOfDay: Int, in ranges: [HourRange]) -> HourRange? {
        return ranges
            .filter { $0.contains(minuteOfDay: minuteOfDay) }
            .sorted { span($0) < span($1) }
            .first
    }

    private static func span(_ r: HourRange) -> Int {
        r.durationMinutes
    }
}
