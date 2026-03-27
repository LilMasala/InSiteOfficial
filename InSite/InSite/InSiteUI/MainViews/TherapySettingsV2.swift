import SwiftUI
import UIKit
import FirebaseAuth
import Foundation
// MARK: - Models (same data integrity as your current app)



struct DiabeticProfile: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var name: String
    var hourRanges: [HourRange]
}

struct HourRange: Codable, Identifiable, Equatable {
    var id = UUID()
    var startMinute: Int   // 0...1439 inclusive
    var endMinute: Int     // 1...1440 exclusive
    var carbRatio: Double
    var basalRate: Double
    var insulinSensitivity: Double

    init(
        id: UUID = UUID(),
        startMinute: Int,
        endMinute: Int,
        carbRatio: Double,
        basalRate: Double,
        insulinSensitivity: Double
    ) {
        self.id = id
        self.startMinute = max(0, min(1439, startMinute))
        self.endMinute = max(self.startMinute + 15, min(1440, endMinute))
        self.carbRatio = carbRatio
        self.basalRate = basalRate
        self.insulinSensitivity = insulinSensitivity
    }

    init(
        id: UUID = UUID(),
        startHour: Int,
        endHour: Int,
        carbRatio: Double,
        basalRate: Double,
        insulinSensitivity: Double
    ) {
        let startMinute = max(0, min(23, startHour)) * 60
        let endMinute = min(1440, (max(0, min(23, endHour)) + 1) * 60)
        self.init(
            id: id,
            startMinute: startMinute,
            endMinute: max(startMinute + 60, endMinute),
            carbRatio: carbRatio,
            basalRate: basalRate,
            insulinSensitivity: insulinSensitivity
        )
    }

    var startHour: Int {
        get { startMinute / 60 }
        set { startMinute = max(0, min(23, newValue)) * 60 }
    }

    var endHour: Int {
        get { max(startHour, ((endMinute - 1) / 60)) }
        set { endMinute = min(1440, (max(0, min(23, newValue)) + 1) * 60) }
    }

    var durationMinutes: Int {
        endMinute - startMinute
    }

    func contains(minuteOfDay: Int) -> Bool {
        startMinute <= minuteOfDay && minuteOfDay < endMinute
    }

    var timeLabel: String {
        "\(Self.formatMinute(startMinute))-\(Self.formatMinute(endMinute))"
    }

    static func formatMinute(_ minute: Int) -> String {
        let clamped = max(0, min(1440, minute))
        let hour = min(clamped, 1439) / 60
        let minutePart = clamped % 60
        var components = DateComponents()
        components.hour = hour
        components.minute = minutePart
        let date = Calendar.current.date(from: components) ?? Date()
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        if clamped == 1440 {
            return "12:00 AM"
        }
        return formatter.string(from: date)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case startMinute
        case endMinute
        case carbRatio
        case basalRate
        case insulinSensitivity
        case startHour
        case endHour
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        carbRatio = try container.decode(Double.self, forKey: .carbRatio)
        basalRate = try container.decode(Double.self, forKey: .basalRate)
        insulinSensitivity = try container.decode(Double.self, forKey: .insulinSensitivity)

        if let startMinute = try container.decodeIfPresent(Int.self, forKey: .startMinute),
           let endMinute = try container.decodeIfPresent(Int.self, forKey: .endMinute) {
            self.startMinute = max(0, min(1439, startMinute))
            self.endMinute = max(self.startMinute + 15, min(1440, endMinute))
        } else {
            let startHour = try container.decodeIfPresent(Int.self, forKey: .startHour) ?? 0
            let endHour = try container.decodeIfPresent(Int.self, forKey: .endHour) ?? 23
            self.startMinute = max(0, min(23, startHour)) * 60
            self.endMinute = max(self.startMinute + 60, min(1440, (max(0, min(23, endHour)) + 1) * 60))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(startMinute, forKey: .startMinute)
        try container.encode(endMinute, forKey: .endMinute)
        try container.encode(carbRatio, forKey: .carbRatio)
        try container.encode(basalRate, forKey: .basalRate)
        try container.encode(insulinSensitivity, forKey: .insulinSensitivity)
    }
}

// MARK: - Storage (unchanged behavior)

final class ProfileDataStore {
    private enum DefaultsKey {
        static let profiles = "profilesData"
        static let activeProfile = "activeProfileID"
    }

    private func key(for base: String, uid: String? = Auth.auth().currentUser?.uid) -> String? {
        guard let uid = uid, !uid.isEmpty else { return nil }
        return "\(base)_\(uid)"
    }

    private var profilesKey: String? { key(for: DefaultsKey.profiles) }
    private var activeProfileKey: String? { key(for: DefaultsKey.activeProfile) }

    func saveProfiles(_ profiles: [DiabeticProfile]) {
        guard let profilesKey else { return }
        let encoder = JSONEncoder()
        if let encoded = try? encoder.encode(profiles) {
            UserDefaults.standard.set(encoded, forKey: profilesKey)
        }
    }

    func loadProfiles() -> [DiabeticProfile] {
        if let profilesKey,
           let saved = UserDefaults.standard.object(forKey: profilesKey) as? Data {
            let decoder = JSONDecoder()
            if let loaded = try? decoder.decode([DiabeticProfile].self, from: saved) {
                return loaded
            }
        }
        // default profile if none saved
        let def = HourRange(startHour: 0, endHour: 23, carbRatio: 10.0, basalRate: 0.10, insulinSensitivity: 50.0)
        return [DiabeticProfile(name: "Default", hourRanges: [def])]
    }

    func saveActiveProfileID(_ id: String) {
        guard let activeProfileKey else { return }
        UserDefaults.standard.set(id, forKey: activeProfileKey)
    }

    func loadActiveProfileID() -> String? {
        guard let activeProfileKey else { return nil }
        return UserDefaults.standard.string(forKey: activeProfileKey)
    }

    func clearActiveProfileID() {
        guard let activeProfileKey else { return }
        UserDefaults.standard.removeObject(forKey: activeProfileKey)
    }

    func clearData(for uid: String?) {
        let d = UserDefaults.standard
        if let profilesKey = key(for: DefaultsKey.profiles, uid: uid) {
            d.removeObject(forKey: profilesKey)
        }
        if let activeProfileKey = key(for: DefaultsKey.activeProfile, uid: uid) {
            d.removeObject(forKey: activeProfileKey)
        }
    }

    @discardableResult
    func hydrate(from snapshot: TherapySnapshot) -> [DiabeticProfile] {
        var profiles = loadProfiles()
        let imported = DiabeticProfile(
            id: snapshot.profileId,
            name: snapshot.profileName,
            hourRanges: snapshot.hourRanges.sorted {
                $0.startMinute < $1.startMinute || ($0.startMinute == $1.startMinute && $0.endMinute < $1.endMinute)
            }
        )

        if let index = profiles.firstIndex(where: { $0.id == imported.id }) {
            profiles[index] = imported
        } else {
            profiles.append(imported)
        }

        saveProfiles(profiles)
        saveActiveProfileID(imported.id)
        return profiles
    }
}

// MARK: - Main Screen

struct TherapySettings: View {
    // If you have ThemeManager in the app, this picks up your accent
    @EnvironmentObject private var themeManager: ThemeManager

    @State private var profiles: [DiabeticProfile] = []
    @State private var selectedIndex: Int = 0

    @State private var showRangeSheet = false
    @State private var editingRange: HourRange? = nil

    @State private var showNewProfileSheet = false
    @State private var tempProfileName = ""
    
    // In TherapySettings
    @State private var profileSwitchTick = 0
    @State private var saveTick = 0
    
    @State private var hueShift: Double = 0
    
    @State private var appeared = false
    
    @State private var slideDir: SlideDir = .right
    @State private var isHydratingRemoteSnapshot = false
    @State private var hydrationErrorMessage: String?
    @State private var suppressLocalFallback = false






    private let store = ProfileDataStore()

    private var accent: Color {
        // Use your app theme accent if available, else a sane default
        themeManager.theme.accent
    }

    private var currentProfile: DiabeticProfile? {
        guard profiles.indices.contains(selectedIndex) else { return nil }
        return profiles[selectedIndex]
    }

    public init() {}

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if isHydratingRemoteSnapshot {
                        Card {
                            HStack(spacing: 12) {
                                ProgressView()
                                Text("Loading the latest therapy profile…")
                                    .font(.subheadline)
                                Spacer()
                            }
                        }
                    } else if let hydrationErrorMessage {
                        Card {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Therapy profile couldn’t be refreshed")
                                    .font(.subheadline.weight(.semibold))
                                Text(hydrationErrorMessage)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    if !suppressLocalFallback {
                        headerCard
                            .opacity(appeared ? 1 : 0)
                            .offset(y: appeared ? 0 : 8)
                            .animation(.spring(response: 0.32, dampingFraction: 0.9).delay(0.02), value: appeared)

                        profileChips
                            .opacity(appeared ? 1 : 0)
                            .offset(y: appeared ? 0 : 8)
                            .animation(.spring(response: 0.32, dampingFraction: 0.9).delay(0.06), value: appeared)
                        rangeGrid
                            .opacity(appeared ? 1 : 0)
                            .offset(y: appeared ? 0 : 8)
                            .animation(.spring(response: 0.4, dampingFraction: 0.9).delay(0.06), value: appeared)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
            .background(bg.ignoresSafeArea())
            .navigationTitle("Therapy Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("New profile", systemImage: "plus") {
                            tempProfileName = ""; showNewProfileSheet = true
                        }
                        if let p = currentProfile {
                            Button("Duplicate “\(p.name)”", systemImage: "doc.on.doc") {
                                duplicateProfile(at: selectedIndex)
                            }
                            if profiles.count > 1 {
                                Button("Delete current", systemImage: "trash", role: .destructive) {
                                    deleteProfile(at: selectedIndex)
                                }
                            }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $showRangeSheet) {
                RangeEditorSheet(
                    accent: accent,
                    existing: $profiles[selectedIndex].hourRanges,
                    editing: $editingRange
                ) { newRange in
                    upsert(range: newRange)
                }
            }
            .sheet(isPresented: $showNewProfileSheet) {
                NewProfileSheet(accent: accent, name: $tempProfileName) { name in
                    addProfile(named: name)
                }
            }
            .onAppear(perform: loadInitial)
            .onAppear {
                if !appeared { appeared = true }
            }
            .task {
                await hydrateFromLatestRemoteSnapshot()
            }

            .onChange(of: profiles) { store.saveProfiles($0) }
        }
    }
}

fileprivate extension TherapySettings {
    var scales: TherapyScales {
        let allRanges = profiles.flatMap { $0.hourRanges }
        func mm(_ keyPath: KeyPath<HourRange, Double>, fallback: (Double, Double)) -> MetricScale {
            guard !allRanges.isEmpty else { return .init(min: fallback.0, max: fallback.1) }
            let vals = allRanges.map { $0[keyPath: keyPath] }
            let mn = vals.min() ?? fallback.0
            let mx = vals.max() ?? fallback.1
            // if no variance, expand a tiny bit so bars aren’t zero-width
            return mn == mx ? .init(min: mn * 0.9, max: mx * 1.1) : .init(min: mn, max: mx)
        }
        return TherapyScales(
            basal: mm(\.basalRate,          fallback: (0, 1.2)),
            cr:    mm(\.carbRatio,          fallback: (5, 25)),
            isf:   mm(\.insulinSensitivity, fallback: (20, 120))
        )
    }
}


// MARK: - Subviews (fileprivate)


fileprivate enum SlideDir { case left, right }


fileprivate extension Color {
    /// Returns a new Color with its hue shifted by `degrees` (wraps around 0…360).
    func hueShifted(_ degrees: Double) -> Color {
        let ui = UIColor(self)
        var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        guard ui.getHue(&h, saturation: &s, brightness: &b, alpha: &a) else { return self }
        let delta = CGFloat(degrees / 360.0)
        var newH = h + delta
        if newH < 0 { newH += 1 }
        if newH > 1 { newH -= 1 }
        return Color(hue: Double(newH), saturation: Double(s), brightness: Double(b), opacity: Double(a))
    }
}

fileprivate struct MetricScale {
    var min: Double
    var max: Double
    var span: Double { Swift.max(Swift.min(max - min, .greatestFiniteMagnitude), 0.000001) }
    func norm(_ x: Double) -> Double { Swift.min(1, Swift.max(0, (x - min) / span)) }
}

fileprivate struct TherapyScales {
    let basal: MetricScale
    let cr: MetricScale
    let isf: MetricScale
}



fileprivate struct AddRangeCard: View {
    let accent: Color
    let action: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(accent.opacity(0.12))
                        .frame(width: 34, height: 34)
                    Image(systemName: "plus")
                        .font(.headline)
                        .foregroundStyle(accent)
                }
                Text("Add hour range")
                    .font(.subheadline.weight(.semibold))
                Text("Start–End, Basal, CR, ISF")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(accent.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(
                        LinearGradient(colors: [accent.opacity(0.9), accent.opacity(0.35)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing),
                        style: StrokeStyle(lineWidth: 1.6, dash: [6, 6], dashPhase: pulse ? 12 : 0)
                    )
                    .animation(reduceMotion ? nil : .linear(duration: 1.6).repeatForever(autoreverses: false),
                               value: pulse)
            )
            .shadow(color: .black.opacity(0.06), radius: 8, y: 4)
        }
        .buttonStyle(.plain)
        .onAppear { pulse = true }
        .accessibilityLabel("Add hour range")
        .scaleEffect(pulse ? 1.01 : 1.0)
        .animation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true), value: pulse)

        
    }
}


fileprivate extension TherapySettings {
    var bg: some View {
        ZStack {
            LinearGradient(
                colors: [
                    themeManager.theme.bgStart.hueShifted(hueShift),
                    themeManager.theme.bgEnd.hueShifted(hueShift * 0.6)
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
            .animation(.easeInOut(duration: 0.5), value: hueShift)

            RadialGradient(colors: [.black.opacity(0.08), .clear],
                           center: .center, startRadius: 0, endRadius: 900)
            .blendMode(.multiply)
        }
    }


    var headerCard: some View {
        Card {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Circle()
                  .fill(accent.opacity(0.9))
                  .frame(width: 10, height: 10)
                  .scaleEffect(currentProfile == nil ? 1 : 1.15)
                  .animation(.spring(response: 0.25, dampingFraction: 0.8), value: currentProfile?.id)


                VStack(alignment: .leading, spacing: 2) {
                    Text(currentProfile?.name ?? "—")
                      .font(.headline)
                      .contentTransition(.numericText())  // iOS17+; otherwise .opacity
                      .animation(.easeInOut(duration: 0.25), value: currentProfile?.id)
                    Text(summaryText(for: currentProfile)).font(.footnote).foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
        .tint(accent)
        .accessibilityElement(children: .contain)
    }

    var profileChips: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                Text("Profiles").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(profiles.indices, id: \.self) { i in
                            Chip(title: profiles[i].name, selected: i == selectedIndex, tint: accent)
                                .onTapGesture {
                                    guard selectedIndex != i else { return }
                                    let prev = selectedIndex
                                    slideDir = (i > prev) ? .right : .left   // pick direction based on movement
                                    selectedIndex = i
                                    profileSwitchTick &+= 1

                                    withAnimation(.easeInOut(duration: 0.5)) {
                                        hueShift = (hueShift == 0 ? 0.02 : 0)   // (optional) your bg drift
                                    }
                                    store.saveActiveProfileID(profiles[i].id)
                                    haptic(.selection)
                                    Task { try? await TherapySettingsLogManager.shared.logTherapySettingsChange(profile: profiles[i], timestamp: Date()) }
                                }

                                .contextMenu {
                                    Button("Rename", systemImage: "pencil") { renameProfile(at: i) }
                                    Button("Duplicate", systemImage: "doc.on.doc") { duplicateProfile(at: i) }
                                    if profiles.count > 1 {
                                        Button("Delete", systemImage: "trash", role: .destructive) { deleteProfile(at: i) }
                                    }
                                }
                        }
                        Button {
                            tempProfileName = ""; showNewProfileSheet = true
                        } label: {
                            Chip(title: "New", selected: false, tint: accent, withPlus: true)
                        }
                    }
                }
            }
        }
    }

    var rangeGrid: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Hour Ranges")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 4)

            if let p = currentProfile, !p.hourRanges.isEmpty {
                
                let insertionEdge: Edge = (slideDir == .right) ? .trailing : .leading
                let removalEdge: Edge   = (slideDir == .right) ? .leading  : .trailing
                let tileTransition = AnyTransition.asymmetric(
                    insertion: .move(edge: insertionEdge).combined(with: .opacity),
                    removal:   .move(edge: removalEdge).combined(with: .opacity)
                )
                
                
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 12),
                                    GridItem(.flexible(), spacing: 12)], spacing: 12) {

                    ForEach(Array(p.hourRanges.enumerated()), id: \.element.id) { idx, r in
                        RangeCard(range: r, accent: accent, scales: scales)
                            .transition(tileTransition)
                            .animation(
                                .spring(response: 0.32, dampingFraction: 0.9).delay(0.02 * Double(idx)),
                                value: profileSwitchTick
                            )
                            .onTapGesture { editingRange = r; showRangeSheet = true }
                            .contextMenu {
                                Button("Duplicate", systemImage: "doc.on.doc") { duplicate(range: r) }
                                Button("Delete", systemImage: "trash", role: .destructive) { delete(range: r) }
                            }
                    }

                    AddRangeCard(accent: accent) {
                        editingRange = nil
                        showRangeSheet = true
                        haptic(.selection)
                    }
                    .transition(tileTransition)
                    .animation(.spring(response: 0.32, dampingFraction: 0.9).delay(0.02 * Double(p.hourRanges.count)),
                               value: profileSwitchTick)
                }

            } else {
                Card {
                    VStack(spacing: 8) {
                        Text("No ranges yet").font(.subheadline.weight(.semibold))
                        Text("Add your first time window with basal, carb ratio, and ISF.")
                            .font(.footnote).foregroundStyle(.secondary)
                        Button {
                            editingRange = nil
                            showRangeSheet = true
                        } label: { Label("Add Hour Range", systemImage: "plus") }
                            .buttonStyle(.borderedProminent)
                            .tint(accent)
                            .padding(.top, 2)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }
}

// MARK: - Actions (fileprivate)

fileprivate extension TherapySettings {
    func loadInitial() {
        profiles = store.loadProfiles()
        if let activeID = store.loadActiveProfileID(),
           let idx = profiles.firstIndex(where: { $0.id == activeID }) {
            selectedIndex = idx
        } else {
            selectedIndex = min(selectedIndex, max(profiles.count - 1, 0))
            if let first = profiles.first { store.saveActiveProfileID(first.id) }
            else { store.clearActiveProfileID() }
        }
    }

    func hydrateFromLatestRemoteSnapshot() async {
        guard Auth.auth().currentUser?.isAnonymous == false else { return }
        isHydratingRemoteSnapshot = true
        hydrationErrorMessage = nil
        suppressLocalFallback = false
        defer { isHydratingRemoteSnapshot = false }

        do {
            if let snapshot = try await TherapySettingsLogManager.shared.getLatestValidTherapySnapshot() {
                print(
                    "[TherapySettings] hydrating snapshot profile_id=\(snapshot.profileId) timestamp=\(ISO8601DateFormatter().string(from: snapshot.timestamp)) hour_ranges=\(snapshot.hourRanges.count)"
                )
                guard !snapshot.hourRanges.isEmpty else {
                    let message = "Remote therapy snapshot \(snapshot.profileId) was found but contains no schedule blocks."
                    hydrationErrorMessage = message
                    suppressLocalFallback = true
                    print("[TherapySettings] hydration failure reason=\(message)")
                    return
                }
                profiles = store.hydrate(from: snapshot)
                if let idx = profiles.firstIndex(where: { $0.id == snapshot.profileId }) {
                    selectedIndex = idx
                }
                print("[TherapySettings] hydration success profile_count=\(profiles.count) selected_index=\(selectedIndex)")
            }
        } catch {
            hydrationErrorMessage = error.localizedDescription
            suppressLocalFallback = true
            print("[TherapySettings] hydration failed error=\(error.localizedDescription)")
        }
    }

    func addProfile(named name: String) {
        let p = DiabeticProfile(name: name.isEmpty ? "Profile \(profiles.count + 1)" : name, hourRanges: [])
        profiles.append(p)
        store.saveProfiles(profiles)
        selectedIndex = profiles.count - 1
        store.saveActiveProfileID(p.id)
        haptic(.success)
    }

    func renameProfile(at index: Int) {
        guard profiles.indices.contains(index) else { return }
        // Simple inline prompt:
        // You can swap for a sheet if you prefer; keeping it minimal here.
        let old = profiles[index].name
        let newName = (old.isEmpty ? "Profile" : old)
        // For a quick rename UX, you might present a custom sheet.
        // Placeholder rename to keep compile-time safety:
        profiles[index].name = newName
    }

    func duplicateProfile(at index: Int) {
        guard profiles.indices.contains(index) else { return }
        var copy = profiles[index]
        copy.id = UUID().uuidString
        copy.name = copy.name + " Copy"
        profiles.append(copy)
        store.saveProfiles(profiles)
        haptic(.success)
    }

    func deleteProfile(at index: Int) {
        guard profiles.indices.contains(index) else { return }
        profiles.remove(at: index)
        if profiles.isEmpty {
            selectedIndex = 0
            store.clearActiveProfileID()
        } else {
            selectedIndex = min(selectedIndex, profiles.count - 1)
            store.saveActiveProfileID(profiles[selectedIndex].id)
        }
        store.saveProfiles(profiles)
        haptic(.warning)
    }

    func upsert(range: HourRange) {
        guard var p = currentProfile, let idx = profiles.firstIndex(of: p) else { return }
        if let existingIdx = p.hourRanges.firstIndex(where: { $0.id == range.id }) {
            p.hourRanges[existingIdx] = range
            
        } else {
            p.hourRanges.append(range)
        }
        p.hourRanges.sort { $0.startMinute < $1.startMinute || ($0.startMinute == $1.startMinute && $0.endMinute < $1.endMinute) }
        profiles[idx] = p
        saveTick &+= 1
        haptic(.success)
    }

    func duplicate(range: HourRange) {
        guard var p = currentProfile, let idx = profiles.firstIndex(of: p) else { return }
        var copy = range
        copy.id = UUID()
        p.hourRanges.append(copy)
        p.hourRanges.sort { $0.startMinute < $1.startMinute || ($0.startMinute == $1.startMinute && $0.endMinute < $1.endMinute) }
        profiles[idx] = p
        haptic(.success)
    }

    func delete(range: HourRange) {
        guard var p = currentProfile, let idx = profiles.firstIndex(of: p) else { return }
        p.hourRanges.removeAll { $0.id == range.id }
        profiles[idx] = p
        haptic(.warning)
    }

    func summaryText(for profile: DiabeticProfile?) -> String {
        guard let p = profile, !p.hourRanges.isEmpty else { return "No ranges added yet" }
        let crs = p.hourRanges.map(\.carbRatio)
        let crMin = crs.min() ?? 0
        let crMax = crs.max() ?? 0
        let avgBasal = (p.hourRanges.map(\.basalRate).reduce(0, +)) / Double(p.hourRanges.count)
        let isfAvg = (p.hourRanges.map(\.insulinSensitivity).reduce(0, +)) / Double(p.hourRanges.count)
        return String(format: "Avg Basal %.2f U/hr · CR %.0f–%.0f g/U · ISF %.0f", avgBasal, crMin, crMax, isfAvg)
    }
}

// MARK: - Range Card

fileprivate struct RangeCard: View {
    let range: HourRange
    let accent: Color
    // New: pass scales in
    var scales: TherapyScales

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Time header
            HStack {
                Text(range.timeLabel)
                    .font(.headline)
                Spacer()
            }

            // Bars (Basal, CR, ISF)
            RangeMetricBar(
                title: "Basal",
                value: range.basalRate,
                unit: "U/hr",
                scale: scales.basal,
                accent: accent,  // keep base accent
                decimals: 2
            )

            RangeMetricBar(
                title: "Carb Ratio",
                value: range.carbRatio,
                unit: "g/U",
                scale: scales.cr,
                accent: accent.hueShifted(22).opacity(0.9),
                decimals: 0
            )

            RangeMetricBar(
                title: "Sensitivity",
                value: range.insulinSensitivity,
                unit: "mg/dL/U",
                scale: scales.isf,
                accent: accent.hueShifted(-18).opacity(0.9),

                decimals: 0
            )
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(LinearGradient(colors: [accent.opacity(0.12), accent.opacity(0.06)],
                                     startPoint: .topLeading, endPoint: .bottomTrailing))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(accent.opacity(0.25), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 10, y: 6)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(range.timeLabel). Basal \(range.basalRate, specifier: "%.2f") U per hour. Carb ratio \(range.carbRatio, specifier: "%.0f") grams per unit. Sensitivity \(range.insulinSensitivity, specifier: "%.0f") milligrams per deciliter per unit.")
    }
}


fileprivate struct RangeMetricBar: View {
    let title: String
    let value: Double
    let unit: String
    let scale: MetricScale
    let accent: Color
    let decimals: Int

    @ScaledMetric private var barHeight: CGFloat = 10

    private var fraction: Double { scale.norm(value) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Title
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)

            // Bar + fixed right label
            HStack(alignment: .center, spacing: 8) {
                GeometryReader { geo in
                    let w = max(0, geo.size.width)
                    let h = barHeight
                    let fillW = max(4, CGFloat(fraction) * w)

                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: min(6, h/2), style: .continuous)
                            .fill(Color.primary.opacity(0.08))
                            .frame(height: h)

                        RoundedRectangle(cornerRadius: min(6, h/2), style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [accent.opacity(0.85), accent.opacity(0.45)],
                                    startPoint: .leading, endPoint: .trailing
                                )
                            )
                            .frame(width: fillW, height: h)
                            .animation(.easeInOut(duration: 0.45), value: fraction)
                    }
                }
                .frame(height: barHeight) // constrain height; width is flexible

                // Fixed-color value label to the right
                Text(formatted(value, decimals: decimals) + " \(unit)")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary) // change to .primary if you prefer
                    .lineLimit(1)
                    .minimumScaleFactor(0.9)
            }
            .frame(minHeight: barHeight) // keeps alignment tidy
        }
    }

    private func formatted(_ v: Double, decimals: Int) -> String {
        let f = NumberFormatter()
        f.minimumFractionDigits = decimals
        f.maximumFractionDigits = decimals
        return f.string(from: NSNumber(value: v)) ?? String(format: "%.\(decimals)f", v)
    }
}



// MARK: - Range Editor Sheet (pretty but simple; overlap-safe)

fileprivate struct RangeEditorSheet: View {
    let accent: Color
    @Binding var existing: [HourRange]      // existing ranges in the profile
    @Binding var editing: HourRange?        // if nil → adding new

    var onSave: (HourRange) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var startMinute = 0
    @State private var endMinute = 60
    @State private var carbRatio = 10.0
    @State private var basalRate = 0.10
    @State private var insulinSensitivity = 50.0

    // MARK: - Validation
    private let snapStep = 15

    private var availableStarts: [Int] {
        let taken = takenMinutes(excluding: editing)
        return stride(from: 0, to: 1440, by: snapStep).filter { !taken.contains($0) }
    }
    private func availableEnds(from start: Int) -> [Int] {
        let taken = takenMinutes(excluding: editing)
        var values: [Int] = []
        var cursor = start + snapStep
        while cursor <= 1440 {
            let occupied = stride(from: start, to: cursor, by: snapStep).contains { taken.contains($0) }
            if occupied { break }
            values.append(cursor)
            cursor += snapStep
        }
        return values
    }
    private func takenMinutes(excluding edit: HourRange?) -> Set<Int> {
        var set = Set<Int>()
        for r in existing {
            if let edit, r.id == edit.id { continue }
            for minute in stride(from: r.startMinute, to: r.endMinute, by: snapStep) {
                set.insert(minute)
            }
        }
        return set
    }
    private var canSave: Bool {
        availableStarts.contains(startMinute) &&
        availableEnds(from: startMinute).contains(endMinute)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Friendly header
                HStack(spacing: 10) {
                    Circle().fill(accent).frame(width: 10, height: 10)
                    Text(editing == nil ? "Add Hour Range" : "Edit Hour Range")
                        .font(.headline)
                    Spacer()
                }
                .padding(.horizontal)

                Form {
                    Section("Time") {
                        if availableStarts.isEmpty && editing == nil {
                            Text("All time slots are already covered by existing ranges.")
                                .foregroundStyle(.secondary)
                        } else {
                            Picker("Start", selection: $startMinute) {
                                ForEach(availableStarts, id: \.self) { Text(fmtMinute($0)).tag($0) }
                            }
                            .onChange(of: startMinute) { s in
                                let ends = availableEnds(from: s)
                                if let first = ends.first, !ends.contains(endMinute) { endMinute = first }
                            }

                            let ends = availableEnds(from: startMinute)
                            Picker("End", selection: $endMinute) {
                                ForEach(ends, id: \.self) { Text(fmtMinute($0)).tag($0) }
                            }
                        }
                    }

                    Section("Therapy") {
                        ValueRow(label: "Basal (U/hr)", value: $basalRate, format: .number.precision(.fractionLength(2)))
                        ValueRow(label: "Carb Ratio (g/U)", value: $carbRatio, format: .number.precision(.fractionLength(0)))
                        ValueRow(label: "Sensitivity (mg/dL/U)", value: $insulinSensitivity, format: .number.precision(.fractionLength(0)))
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(editing == nil ? "Add" : "Save") {
                        var r = editing ?? HourRange(startMinute: startMinute, endMinute: endMinute, carbRatio: carbRatio, basalRate: basalRate, insulinSensitivity: insulinSensitivity)
                        r.startMinute = startMinute
                        r.endMinute = endMinute
                        r.carbRatio = carbRatio
                        r.basalRate = basalRate
                        r.insulinSensitivity = insulinSensitivity
                        onSave(r)
                        // After onSave(r) in RangeEditorSheet, just before dismiss()
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()

                        dismiss()
                    }
                    .disabled(!canSave)
                }
            }
            .onAppear(perform: seed)
        }
        .tint(accent)
    }

    private func seed() {
        if let r = editing {
            startMinute = snap(r.startMinute)
            endMinute = snap(r.endMinute)
            carbRatio = r.carbRatio
            basalRate = r.basalRate
            insulinSensitivity = r.insulinSensitivity
        } else {
            if let first = availableStarts.first {
                startMinute = first
                endMinute = availableEnds(from: first).first ?? (first + snapStep)
            }
        }
    }

    private func snap(_ minute: Int) -> Int {
        let clamped = max(0, min(1440, minute))
        return (clamped / snapStep) * snapStep
    }

    private func fmtMinute(_ minute: Int) -> String {
        HourRange.formatMinute(minute)
    }
}

// MARK: - Add Profile Sheet (tiny + friendly)

fileprivate struct NewProfileSheet: View {
    let accent: Color
    @Binding var name: String
    var onCreate: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Profile name (e.g., Daytime, Travel, Workout)", text: $name)
                        .textInputAutocapitalization(.words)
                }
            }
            .navigationTitle("New Profile")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { onCreate(name.trimmingCharacters(in: .whitespaces)); dismiss() }
                        .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        .tint(accent)
    }
}

// MARK: - Small building blocks (fileprivate)

fileprivate struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(14)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color.white.opacity(0.12)))
            .shadow(color: .black.opacity(0.08), radius: 10, y: 6)
    }
}

fileprivate struct Chip: View {
    let title: String
    let selected: Bool
    let tint: Color
    var withPlus: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            if withPlus { Image(systemName: "plus").imageScale(.small) }
            Text(title).font(.subheadline.weight(.semibold))
        }
        .padding(.vertical, 8).padding(.horizontal, 12)
        .background(
            Capsule()
                .fill(selected ? tint.opacity(0.25) : Color.white.opacity(0.10))
                .overlay(Capsule().stroke(selected ? tint.opacity(0.45) : Color.black.opacity(0.06)))
                .shadow(color: selected ? tint.opacity(0.18) : .clear, radius: 8, y: 3)
        )
        .scaleEffect(selected ? 1.02 : 1.0)
        .animation(.spring(response: 0.22, dampingFraction: 0.9), value: selected)
    }
}

fileprivate struct ValueRow<T: Strideable & BinaryFloatingPoint>: View where T.Stride: BinaryFloatingPoint {
    let label: String
    @Binding var value: T
    let format: FloatingPointFormatStyle<T>

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField("",
                      value: $value,
                      format: format)
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 120)
        }
    }
}

// MARK: - Haptic sugar (safe on older iOS)

fileprivate func haptic(_ kind: HapticKind) {
    if #available(iOS 17.0, *) {
        switch kind {
        case .success: UIImpactFeedbackGenerator(style: .light).impactOccurred(intensity: 0.9)
        case .selection: UISelectionFeedbackGenerator().selectionChanged()
        case .warning: UIImpactFeedbackGenerator(style: .rigid).impactOccurred(intensity: 1.0)
        }
    } else {
        switch kind {
        case .success: UIImpactFeedbackGenerator(style: .light).impactOccurred()
        case .selection: UISelectionFeedbackGenerator().selectionChanged()
        case .warning: UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }
    }
}

fileprivate enum HapticKind { case success, selection, warning }

// MARK: - Preview

#Preview {
    NavigationStack {
        TherapySettings()
            .environmentObject(ThemeManager())
    }
}
