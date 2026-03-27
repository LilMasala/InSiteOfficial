import SwiftUI
import UIKit // for UIColor bridge

// MARK: - Public Screen

public struct SiteChangeUI: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @ObservedObject private var sharedData = SiteChangeData.shared

    @State private var showAlert = false
    @State private var pendingSelection: (region: Region, isLeft: Bool)?

    private var palette: RegionPalette { .init(base: themeManager.theme.accent) }
    private var gridCols: [GridItem] { [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)] }

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(spacing: 16) {

                // Header card
                SectionCard(title: "Choose location", icon: "mappin.and.ellipse") {
                    HStack(spacing: 12) {
                        Image("BearBlue")
                            .resizable().scaledToFill()
                            .frame(width: 44, height: 44)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(RoundedRectangle(cornerRadius: 10).stroke(.white.opacity(0.18), lineWidth: 1))
                            .shadow(color: .black.opacity(0.08), radius: 6, y: 3)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Pick your current infusion site").font(.footnote).foregroundStyle(.secondary)
                            Text("Tap a side below").font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                }

                // Status card
                SectionCard(title: "Current", icon: "circle.dashed") {
                    HStack(spacing: 12) {
                        ChangeProgressRing(daysSince: sharedData.daysSinceSiteChange, tint: themeManager.theme.accent)
                        Text(sharedData.siteChangeLocation.isEmpty ? "Not selected" : sharedData.siteChangeLocation)
                            .font(.subheadline)
                        Spacer()
                        let daysLeft = max(0, 3 - sharedData.daysSinceSiteChange)
                        Text("\(daysLeft) day\(daysLeft == 1 ? "" : "s") until change")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }

                // Region sections
                VStack(spacing: 12) {
                    siteSection(title: "Arm", icon: "hand.raised.fill", region: .arm)
                    siteSection(title: "Abdomen", icon: "figure.mind.and.body", region: .abdomen)
                    siteSection(title: "Butt", icon: "figure.seated.side.right", region: .butt)
                    siteSection(title: "Thigh", icon: "figure.walk", region: .thigh)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 20)
            .frame(maxWidth: 700)
            .frame(maxWidth: .infinity)
        }
        .background(
            ZStack {
                LinearGradient(colors: [themeManager.theme.bgStart, themeManager.theme.bgEnd],
                               startPoint: .topLeading, endPoint: .bottomTrailing)
                    .ignoresSafeArea()
                // subtle vignette to lift cards
                RadialGradient(colors: [.black.opacity(0.10), .clear],
                               center: .center, startRadius: 0, endRadius: 900)
                    .blendMode(.multiply)
                    .ignoresSafeArea()
            }
        )
        .navigationTitle("Site Change")
        .alert("Change Site", isPresented: $showAlert) {
            Button("Change", role: .destructive) {
                guard let p = pendingSelection else { return }
                let loc = p.region.label(isLeft: p.isLeft)
                sharedData.setSiteChange(location: loc)
                HealthDataUploader().recordSiteChange(location: loc, localTz: .current, backfillDays: 14)
            }
            Button("Cancel", role: .cancel) { pendingSelection = nil }
        } message: {
            Text("Change site to \(pendingSelection.map { $0.region.label(isLeft: $0.isLeft) } ?? "")?")
        }
    }

    @ViewBuilder
    private func siteSection(title: String, icon: String, region: Region) -> some View {
        SectionCard(title: title, icon: icon) {
            LazyVGrid(columns: gridCols, alignment: .leading, spacing: 10) {
                let left  = region.label(isLeft: true)
                let right = region.label(isLeft: false)
                let tint  = palette.tint(region)

                SiteOptionChip(title: left,  tint: tint,
                               selected: sharedData.siteChangeLocation == left) {
                    pendingSelection = (region, true); showAlert = true
                }
                SiteOptionChip(title: right, tint: tint,
                               selected: sharedData.siteChangeLocation == right) {
                    pendingSelection = (region, false); showAlert = true
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        SiteChangeUI()
            .environmentObject(ThemeManager())
            .environmentObject(SiteChangeData.shared)
    }
}

// MARK: - fileprivate Helpers & UI

fileprivate enum Region: CaseIterable {
    case arm, abdomen, butt, thigh

    static var ordered: [Region] { [.arm, .abdomen, .butt, .thigh] }

    var title: String {
        switch self {
        case .arm: return "Arm"
        case .abdomen: return "Abdomen"
        case .butt: return "Butt"
        case .thigh: return "Thigh"
        }
    }

    func label(isLeft: Bool) -> String {
        "\(isLeft ? "Left" : "Right") \(title)"
    }
}

fileprivate struct RegionPalette {
    let base: Color
    func tint(_ r: Region) -> Color {
        switch r {
        case .arm:     return base
        case .abdomen: return base.hueShifted(30)   // +30°
        case .butt:    return base.hueShifted(-25)  // -25°
        case .thigh:   return base.hueShifted(60)   // +60°
        }
    }
}

fileprivate struct UI {
    static let corner: CGFloat = 16
    static let chipCorner: CGFloat = 14
    static let cardStroke = Color.white.opacity(0.13)
    static let chipStroke = Color.black.opacity(0.07)
    static let chipFill   = Color.white.opacity(0.06)
    static let chipFillSel = Color.white.opacity(0.12)
}

fileprivate struct SectionCard<Content: View>: View {
    var title: String
    var icon: String
    @EnvironmentObject private var themeManager: ThemeManager
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .imageScale(.medium)
                    .foregroundStyle(themeManager.theme.accent.opacity(0.8))
                Text(title)
                    .font(.headline)
                Spacer()
            }
            content
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: UI.corner, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            themeManager.theme.accent.opacity(0.15),
                            themeManager.theme.accent.opacity(0.08)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: UI.corner, style: .continuous)
                .stroke(themeManager.theme.accent.opacity(0.25), lineWidth: 1)
        )
        .shadow(color: themeManager.theme.accent.opacity(0.15), radius: 10, y: 6)
    }
}


fileprivate struct SiteOptionChip: View {
    let title: String
    let tint: Color
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Circle()
                    .fill(selected ? tint : tint.opacity(0.35))
                    .frame(width: 8, height: 8)
                    .shadow(color: tint.opacity(selected ? 0.45 : 0), radius: selected ? 6 : 0)

                Text(title)
                    .font(.callout.weight(.semibold))

                Spacer()
            }
            .padding(.vertical, 13)
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity, minHeight: 48)
            .background(
                RoundedRectangle(cornerRadius: UI.chipCorner, style: .continuous)
                    .fill(selected ? UI.chipFillSel : UI.chipFill)
                    .shadow(color: .black.opacity(0.06), radius: 6, y: 3)
                    .overlay(
                        RoundedRectangle(cornerRadius: UI.chipCorner, style: .continuous)
                            .stroke(UI.chipStroke, lineWidth: 1)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: UI.chipCorner, style: .continuous)
                            .stroke(
                                LinearGradient(
                                    colors: [tint.opacity(selected ? 0.9 : 0.55),
                                             tint.opacity(selected ? 0.45 : 0.25)],
                                    startPoint: .topLeading, endPoint: .bottomTrailing
                                ),
                                lineWidth: selected ? 1.6 : 1.0
                            )
                    )
                    .shadow(color: tint.opacity(selected ? 0.18 : 0), radius: selected ? 10 : 0)
            )
            .scaleEffect(selected ? 1.01 : 1.0)
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: UI.chipCorner))
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

fileprivate struct ChangeProgressRing: View {
    var daysSince: Int
    var tint: Color
    private var progress: Double { min(1, Double(daysSince) / 3.0) }

    var body: some View {
        ZStack {
            Circle().stroke(Color.primary.opacity(0.12), lineWidth: 6)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(colors: [tint.opacity(0.9), tint.opacity(0.4), tint.opacity(0.9)], center: .center),
                    style: StrokeStyle(lineWidth: 6, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 28, height: 28)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Days since change \(daysSince) of 3")
    }
}

// MARK: - fileprivate Utilities

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
