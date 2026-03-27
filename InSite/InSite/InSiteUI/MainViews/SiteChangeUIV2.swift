import SwiftUI
import FirebaseAuth
import FirebaseFirestore
import UIKit

struct SiteChangeUIV2: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @ObservedObject private var sharedData = SiteChangeData.shared
    @StateObject private var historyStore = SiteChangeHistoryStore()

    @State private var showLogSheet = false
    @State private var selectedLocation: SitePlacement?

    private var accent: Color { themeManager.theme.accent }

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                heroCard
                cycleTimelineCard
                recentHistoryCard
            }
            .padding(.horizontal, 16)
            .padding(.top, 20)
            .padding(.bottom, 28)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
        .background(
            BreathingBackground(theme: themeManager.theme)
                .ignoresSafeArea()
        )
        .navigationTitle("Site Change")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await historyStore.refresh()
        }
        .sheet(isPresented: $showLogSheet) {
            SiteChangeLogSheet(
                accent: accent,
                selectedLocation: $selectedLocation
            ) { location in
                logSiteChange(location: location)
            }
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
    }

    private var heroCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(heroValueText)
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                    Text(heroSubtitleText)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                neutralStatusBadge
            }

            HStack(spacing: 12) {
                statChip(title: "Current location", value: sharedData.siteChangeLocation)
                statChip(title: "Last change", value: latestChangeText)
            }

            Button {
                selectedLocation = SitePlacement.from(label: sharedData.siteChangeLocation)
                showLogSheet = true
            } label: {
                HStack {
                    Image(systemName: "plus.circle.fill")
                        .imageScale(.medium)
                    Text("Change Site")
                        .font(.headline)
                    Spacer()
                    Text(selectedLocationLabel)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.82))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 15)
                .background(
                    LinearGradient(
                        colors: [accent.opacity(0.95), accent.opacity(0.7)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: 18, style: .continuous)
                )
                .foregroundStyle(.white)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Changed site now. Current selected location \(selectedLocationLabel)")
        }
        .siteCardStyle(accent: accent)
    }

    private var cycleTimelineCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Current cycle")
                .font(.headline)

            Text("Your current site and the most recent logged changes, in time order.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if cycleMarkers.isEmpty {
                Text("Cycle details appear after site-change events sync from Firebase.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                HStack(alignment: .center, spacing: 10) {
                    ForEach(Array(cycleMarkers.enumerated()), id: \.offset) { index, marker in
                        VStack(spacing: 8) {
                            Capsule()
                                .fill(markerFill(for: marker))
                                .frame(width: 18, height: marker.isToday ? 44 : 30)
                                .overlay {
                                    if marker.isToday {
                                        Capsule()
                                            .stroke(Color.white.opacity(0.7), lineWidth: 1.5)
                                    }
                                }
                            Text(marker.label)
                                .font(.caption2)
                                .foregroundStyle(marker.isToday ? .primary : .secondary)
                                .multilineTextAlignment(.center)
                                .frame(width: 42)
                        }
                        if index < cycleMarkers.count - 1 {
                            Rectangle()
                                .fill(Color.primary.opacity(0.08))
                                .frame(height: 2)
                        }
                    }
                }
            }

            HStack(spacing: 10) {
                statChip(title: "Days on site", value: "\(sharedData.daysSinceSiteChange)")
                statChip(title: "Last logged", value: latestChangeRelativeText)
            }
        }
        .siteCardStyle(accent: accent)
    }

    private var recentHistoryCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Recent history")
                    .font(.headline)
                Spacer()
                if historyStore.isLoading {
                    ProgressView()
                        .tint(accent)
                }
            }

            if let errorMessage = historyStore.errorMessage {
                Text(errorMessage)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else if historyStore.events.isEmpty {
                Text("Your recent site locations will appear here after you log a change.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(historyStore.events.prefix(8)) { event in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(event.location)
                                    .font(.subheadline.weight(.semibold))
                                    .lineLimit(2)
                                Text(event.timestamp.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(width: 150, alignment: .leading)
                            .padding(14)
                            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                        }
                    }
                }

                VStack(spacing: 10) {
                    ForEach(Array(historyStore.events.prefix(5).enumerated()), id: \.element.id) { index, event in
                        HStack(alignment: .top, spacing: 12) {
                            VStack(spacing: 4) {
                                Circle()
                                    .fill(accent.opacity(index == 0 ? 0.95 : 0.45))
                                    .frame(width: 10, height: 10)
                                if index < min(historyStore.events.count, 5) - 1 {
                                    Rectangle()
                                        .fill(Color.primary.opacity(0.1))
                                        .frame(width: 2, height: 28)
                                }
                            }

                            VStack(alignment: .leading, spacing: 2) {
                                Text(event.location)
                                    .font(.subheadline.weight(.semibold))
                                Text(event.timestamp.formatted(date: .abbreviated, time: .shortened))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                        }
                    }
                }
            }
        }
        .siteCardStyle(accent: accent)
    }

    private var heroValueText: String {
        "\(sharedData.daysSinceSiteChange) day\(sharedData.daysSinceSiteChange == 1 ? "" : "s") on current site"
    }

    private var heroSubtitleText: String {
        if sharedData.siteChangeLocation == "Not selected" {
            return "No current site location is logged yet."
        }
        return "Currently tracking \(sharedData.siteChangeLocation.lowercased()) as your active site."
    }

    private var latestChangeText: String {
        guard let date = latestChangeDate else { return "Not logged" }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private var latestChangeRelativeText: String {
        guard let date = latestChangeDate else { return "No event" }
        return RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
    }

    private var selectedLocationLabel: String {
        selectedLocation?.label ?? sharedData.siteChangeLocation
    }

    private var neutralStatusBadge: some View {
        Text(sharedData.siteChangeLocation == "Not selected" ? "Tracking off" : "Tracking on")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(accent.opacity(0.14), in: Capsule())
            .foregroundStyle(accent)
    }

    private var latestChangeDate: Date? {
        historyStore.events.first?.timestamp ?? sharedData.latestSiteChangeDate
    }

    private var cycleMarkers: [CycleMarker] {
        let recentEvents = Array(historyStore.events.prefix(5))
        guard !recentEvents.isEmpty else { return [] }

        return recentEvents
            .reversed()
            .enumerated()
            .map { index, event in
                let isLatest = index == recentEvents.count - 1
                return CycleMarker(
                    label: isLatest ? "Now" : shortCycleLabel(for: event.timestamp),
                    isFilled: true,
                    isToday: isLatest,
                    isCurrentWindow: isLatest
                )
            }
    }

    private func shortCycleLabel(for date: Date) -> String {
        let days = Calendar.current.dateComponents([.day], from: Calendar.current.startOfDay(for: date), to: Calendar.current.startOfDay(for: Date())).day ?? 0
        if days <= 0 {
            return "Today"
        }
        if days == 1 {
            return "1d ago"
        }
        return "\(days)d ago"
    }

    private func markerFill(for marker: CycleMarker) -> Color {
        if marker.isToday {
            return accent
        }
        if marker.isFilled {
            return accent.opacity(0.6)
        }
        return Color.primary.opacity(0.08)
    }

    private func statChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .lineLimit(2)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func logSiteChange(location: SitePlacement) {
        let label = location.label
        sharedData.setSiteChange(location: label)
        HealthDataUploader().recordSiteChange(location: label, localTz: .current, backfillDays: 14)
        UIImpactFeedbackGenerator(style: .light).impactOccurred()

        Task {
            await historyStore.refresh()
        }
    }
}

#Preview {
    NavigationStack {
        SiteChangeUIV2()
            .environmentObject(ThemeManager())
    }
}

private final class SiteChangeHistoryStore: ObservableObject {
    @Published var events: [SiteChangeHistoryEvent] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    func refresh() async {
        guard let uid = Auth.auth().currentUser?.uid, !uid.isEmpty else {
            await MainActor.run {
                events = []
                errorMessage = nil
                isLoading = false
            }
            return
        }

        await MainActor.run {
            isLoading = true
            errorMessage = nil
        }

        do {
            let snapshot = try await Firestore.firestore()
                .collection("users").document(uid)
                .collection("site_changes").document("events")
                .collection("items")
                .order(by: "createdAt", descending: true)
                .limit(to: 12)
                .getDocuments()

            let loaded = snapshot.documents.compactMap { document -> SiteChangeHistoryEvent? in
                let data = document.data()
                let location = (data["location"] as? String) ?? "Unknown"
                let timestamp = (data["timestamp"] as? Timestamp)?.dateValue()
                    ?? (data["createdAt"] as? Timestamp)?.dateValue()
                    ?? Self.parseClientTimestamp(data["clientTimestamp"])
                guard let timestamp else { return nil }
                return SiteChangeHistoryEvent(id: document.documentID, location: location, timestamp: timestamp)
            }

            await MainActor.run {
                events = loaded
                isLoading = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Recent site history is unavailable right now."
                isLoading = false
            }
        }
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

private struct SiteChangeHistoryEvent: Identifiable, Equatable {
    let id: String
    let location: String
    let timestamp: Date
}

private struct CycleMarker {
    let label: String
    let isFilled: Bool
    let isToday: Bool
    let isCurrentWindow: Bool
}

private enum SitePlacement: String, CaseIterable, Identifiable {
    case leftArm = "Left Arm"
    case rightArm = "Right Arm"
    case leftAbdomen = "Left Abdomen"
    case rightAbdomen = "Right Abdomen"
    case leftButt = "Left Butt"
    case rightButt = "Right Butt"
    case leftThigh = "Left Thigh"
    case rightThigh = "Right Thigh"

    var id: String { rawValue }
    var label: String { rawValue }

    var group: String {
        switch self {
        case .leftArm, .rightArm:
            return "Arm"
        case .leftAbdomen, .rightAbdomen:
            return "Abdomen"
        case .leftButt, .rightButt:
            return "Butt"
        case .leftThigh, .rightThigh:
            return "Thigh"
        }
    }

    static func from(label: String) -> SitePlacement? {
        allCases.first { $0.label.caseInsensitiveCompare(label) == .orderedSame }
    }
}

private struct SiteChangeLogSheet: View {
    let accent: Color
    @Binding var selectedLocation: SitePlacement?
    let onConfirm: (SitePlacement) -> Void
    @Environment(\.dismiss) private var dismiss

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]
    private let groups = SitePlacementGroup.all

    var body: some View {
        NavigationStack {
            ScrollView { sheetContent }
            .navigationTitle("Log Site Change")
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .bottom) { confirmationBar }
        }
    }

    private var sheetContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Choose the location for the site you just changed.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            ForEach(groups) { group in
                placementSection(group)
            }
        }
        .padding(20)
    }

    private func placementSection(_ group: SitePlacementGroup) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(group.title)
                .font(.headline)

            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(group.placements) { placement in
                    placementButton(placement)
                }
            }
        }
    }

    private func placementButton(_ placement: SitePlacement) -> some View {
        let isSelected = selectedLocation == placement
        let fillColor = isSelected ? accent.opacity(0.12) : Color.primary.opacity(0.05)
        let strokeColor = isSelected ? accent.opacity(0.5) : Color.primary.opacity(0.08)

        return Button {
            selectedLocation = placement
        } label: {
            HStack {
                Text(placement.label)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(accent)
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(fillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(strokeColor, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Select \(placement.label)")
    }

    private var confirmationBar: some View {
        let buttonTitle = selectedLocation == nil ? "Choose a location" : "Confirm \(selectedLocation?.label ?? "")"
        let fillColor = selectedLocation == nil ? Color.primary.opacity(0.08) : accent
        let foreground = selectedLocation == nil ? Color.secondary : Color.white

        return Button {
            if let selectedLocation {
                onConfirm(selectedLocation)
                dismiss()
            }
        } label: {
            Text(buttonTitle)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(fillColor)
                )
                .foregroundStyle(foreground)
        }
        .buttonStyle(.plain)
        .disabled(selectedLocation == nil)
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}

private struct SitePlacementGroup: Identifiable {
    let title: String
    let placements: [SitePlacement]

    var id: String { title }

    static let all: [SitePlacementGroup] = [
        SitePlacementGroup(title: "Abdomen", placements: [.leftAbdomen, .rightAbdomen]),
        SitePlacementGroup(title: "Arm", placements: [.leftArm, .rightArm]),
        SitePlacementGroup(title: "Butt", placements: [.leftButt, .rightButt]),
        SitePlacementGroup(title: "Thigh", placements: [.leftThigh, .rightThigh])
    ]
}

private extension View {
    func siteCardStyle(accent: Color) -> some View {
        self
            .padding(18)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(accent.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: accent.opacity(0.08), radius: 18, y: 10)
    }
}
