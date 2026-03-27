import SwiftUI
import Foundation

public struct CommunityHub: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @State private var appeared = false

    public init() {}

    private var accent: Color { themeManager.theme.accent }

    public var body: some View {
        ZStack {
            // Same animated gradient background you use on Home
            BreathingBackground(theme: themeManager.theme).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 16) {
                    headerCard
                        .opacity(appeared ? 1 : 0)
                        .offset(y: appeared ? 0 : 8)
                        .animation(.spring(response: 0.35, dampingFraction: 0.9), value: appeared)

                    VStack(spacing: 12) {
                        NavigationLink {
                            CommunityBoardView(accent: accent)
                        } label: {
                            FeatureTile(
                                icon: "text.bubble.fill",
                                title: "Community Board",
                                subtitle: "Anonymous posts • upvotes • time filters",
                                accent: accent,
                                status: .live
                            )
                        }
                        .buttonStyle(.plain)

                        NavigationLink {
                            CrosswordsHome(accent: accent)
                        } label: {
                            FeatureTile(
                                icon: "square.grid.3x3.fill.square",
                                title: "Crosswords",
                                subtitle: "Daily puzzle is live • standard mode is rolling out",
                                accent: accent,
                                status: .rollingOut
                            )
                        }
                        .buttonStyle(.plain)

                        FeatureTile(
                            icon: "bubble.left.and.exclamationmark.bubble.right.fill",
                            title: "Feedback Board",
                            subtitle: "Structured product feedback is coming next",
                            accent: accent,
                            status: .comingSoon
                        )

                        FeatureTile(
                            icon: "person.2.wave.2.fill",
                            title: "Mentorship",
                            subtitle: "Peer guidance threads are staged for a later release",
                            accent: accent,
                            status: .comingSoon
                        )
                    }

                    aboutCard
                }
                .padding(.horizontal, 16)
                .padding(.top, 16)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
        }
        .navigationTitle("Community")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { if !appeared { appeared = true } }
    }

    // MARK: - Pieces

    private var headerCard: some View {
        Card {
            HStack(spacing: 10) {
                Circle()
                    .fill(accent)
                    .frame(width: 10, height: 10)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Welcome").font(.headline)
                    Text("The board is live now. Games and guided support are rolling out in phases.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
        .tint(accent)
    }

    private var aboutCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                Text("About")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("The Community Board is the live social layer today. Crosswords are shipping in stages, starting with the daily community puzzle and clue submission flow. Feedback and mentorship surfaces are intentionally marked as upcoming so nothing feels half-finished.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Shared building blocks

fileprivate struct FeatureTile: View {
    enum Status {
        case live
        case rollingOut
        case comingSoon

        var label: String {
            switch self {
            case .live: return "Live"
            case .rollingOut: return "Rolling Out"
            case .comingSoon: return "Coming Soon"
            }
        }
    }

    var icon: String
    var title: String
    var subtitle: String
    var accent: Color
    var status: Status

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle().fill(accent.opacity(0.12))
                Image(systemName: icon)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(accent)
            }
            .frame(width: 48, height: 48)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(title).font(.headline)
                    statusBadge
                }
                Text(subtitle).font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            if status != .comingSoon {
                Image(systemName: "chevron.right")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(.primary.opacity(0.06), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 10, y: 6)
        .opacity(status == .comingSoon ? 0.92 : 1.0)
    }

    private var statusBadge: some View {
        Text(status.label)
            .font(.caption2.weight(.semibold))
            .textCase(.uppercase)
            .tracking(0.6)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(statusColor.opacity(0.14), in: Capsule())
            .foregroundStyle(statusColor)
    }

    private var statusColor: Color {
        switch status {
        case .live:
            return accent
        case .rollingOut:
            return .orange
        case .comingSoon:
            return .secondary
        }
    }
}

fileprivate struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(14)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color.primary.opacity(0.06)))
            .shadow(color: .black.opacity(0.08), radius: 10, y: 6)
    }
}

// MARK: - Previews
struct CommunityHub_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack { CommunityHub() }
                .environmentObject(ThemeManager())
                .environment(\.colorScheme, .light)

            NavigationStack { CommunityHub() }
                .environmentObject(ThemeManager())
                .environment(\.colorScheme, .dark)
        }
    }
}
