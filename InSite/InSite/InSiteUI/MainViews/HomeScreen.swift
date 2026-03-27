import Foundation
import FirebaseAuth
import HealthKit
import HealthKitUI
import SwiftUI
import Combine

//Home Screen

struct HealthAuthView: View {
    var body: some View {
        HomeScreen(showSignInView: .constant(false))
    }
}
struct ContentPreview: PreviewProvider {
  static var previews: some View {
    HealthAuthView()
      .environmentObject(ThemeManager())
  }
}

private extension Double {
    var homeSignedPercentString: String {
        let percent = Int((self * 100).rounded())
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }
}

extension View {
    func pressable(scale: CGFloat = 0.98) -> some View {
        modifier(Pressable(scale: scale))
    }
}
struct Pressable: ViewModifier {
    @GestureState private var pressed = false
    var scale: CGFloat
    func body(content: Content) -> some View {
        content
            .scaleEffect(pressed ? scale : 1)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($pressed) { _, state, _ in state = true }
            )
    }
}

private struct InnerWindow<Content: View>: View {
    var diameter: CGFloat
    var insetFactor: CGFloat = 0.72   // 72% of circle width; tweak 0.70–0.78
    @ViewBuilder var content: Content

    var body: some View {
        let w = diameter * insetFactor
        VStack(spacing: 6) {
            content
        }
        .multilineTextAlignment(.center)
        .frame(width: w, height: w * 0.72, alignment: .center)   // rectangle “window”
        .padding(.horizontal, 2)
    }
}




private struct MoodCTAOrb: View {
    var accent: Color
    var title: String

    @ScaledMetric private var size: CGFloat = 60
    @State private var breathe: CGFloat = 0
    @State private var shimmer: CGFloat = 0

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(accent.opacity(0.18))
                Circle()
                    .stroke(AngularGradient(
                        gradient: Gradient(colors: [accent.opacity(0.9), accent.opacity(0.3), accent.opacity(0.9)]),
                        center: .center),
                        lineWidth: 2.5
                    )
                    .rotationEffect(.degrees(Double(shimmer)))
                Circle()
                    .fill(accent.gradient)
                    .frame(width: size * 0.65, height: size * 0.65)
                    .scaleEffect(1 + 0.02 * breathe)
                    .shadow(color: accent.opacity(0.35), radius: 8, x: 0, y: 4)
                    .overlay(
                        Image(systemName: "face.smiling")
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.95))
                    )
            }
            .frame(width: size, height: size)
            .onAppear {
                withAnimation(.easeInOut(duration: 3.4).repeatForever(autoreverses: true)) {
                    breathe = 1
                }
                withAnimation(.linear(duration: 6.0).repeatForever(autoreverses: false)) {
                    shimmer = 360
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text("Open mood check-in").font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right").foregroundStyle(.secondary)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}





// MARK: - Therapy Carousel Tile

struct TherapyCarouselTile: View {
    enum Metric: Int, CaseIterable { case basal, isf, cr }

    var activeRange: HourRange?
    var accent: Color

    // Current values
    var basalUph: Double
    var isf: Double
    var carbRatio: Double

    // Optional 3–7 point day-patterns for tiny sparklines (normalized in-view)
    var basalPattern: [Double]? = nil
    var isfPattern: [Double]? = nil
    var crPattern: [Double]? = nil

    

    // Layout
    @ScaledMetric private var diameter: CGFloat = 140
    @ScaledMetric private var ringStroke: CGFloat = 8
    @ScaledMetric private var trackStroke: CGFloat = 10
    @ScaledMetric private var nowDot: CGFloat = 6
    @ScaledMetric private var pieInset: CGFloat = 18

    // Carousel state
    @State private var index: Int = 0
    @State private var dragOffset: CGFloat = 0
    @State private var lastInteraction: Date = Date()
    @State private var autoAdvanceTick: Int = 0

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var currentHour: Int {
        Calendar.current.component(.hour, from: Date())
    }

    private var currentMetric: Metric {
        Metric.allCases[index % Metric.allCases.count]
    }

    var body: some View {
        CircleTileBase(diameter: diameter) {
            ZStack {
                // Track ring
                Circle().stroke(Color.primary.opacity(0.08), lineWidth: trackStroke)

                // Active window arc
                if let r = activeRange {
                    TherapyArc(range: r)
                        .stroke(
                            LinearGradient(colors: [accent.opacity(0.85), accent.opacity(0.45)],
                                           startPoint: .leading, endPoint: .trailing),
                            style: StrokeStyle(lineWidth: ringStroke, lineCap: .round)
                        )
                        .frame(width: diameter * 0.78, height: diameter * 0.78)
                }

//
                let innerW = diameter * 0.74     // safe width inside ring
                let innerH = diameter * 0.62     // safe height; tweak 0.58–0.66

                VStack(spacing: 8) {

                  // Slide content (title + value+unit)
                  SlideStack(index: $index, dragOffset: $dragOffset, reduceMotion: reduceMotion) {
                    ForEach(Array(Metric.allCases.enumerated()), id: \.offset) { _, m in
                      VStack(spacing: 6) {
                        Text(title(for: m))
                          .font(.caption2.weight(.semibold))
                          .foregroundStyle(.secondary)
                          .lineLimit(1)

                        HStack(spacing: 6) {
                          Text(value(for: m))
                            .font(.title3.weight(.semibold))
                            .minimumScaleFactor(0.75)
                          Text(unit(for: m))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .baselineOffset(2)
                        }
                        .lineLimit(1)
                      }
                      .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                  }
                  .simultaneousGesture(swipeGesture)   // don’t block NavigationLink
                  .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: index)

                  // Dots
                  HStack(spacing: 8) {
                    ForEach(0..<Metric.allCases.count, id: \.self) { i in
                      IndicatorDot(active: i == index, accent: accent)
                    }
                  }
                  .padding(.bottom, 2) // tiny air above ring
                }
                .frame(width: innerW, height: innerH)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityLabel)
        .onAppear {
            lastInteraction = Date()
        }
        .onReceive(timer) { _ in
            guard !reduceMotion else { return }
            // Pause auto-advance ~7s after last interaction
            if Date().timeIntervalSince(lastInteraction) >= 7 {
                withAnimation(.easeInOut(duration: 0.2)) {
                    index = (index + 1) % Metric.allCases.count
                }
            }
            autoAdvanceTick &+= 1
        }
    }

    // MARK: - Timer (every ~3.7s)
    private var timer: Publishers.Autoconnect<Timer.TimerPublisher> {
        Timer.publish(every: 3.7, on: .main, in: .common).autoconnect()
    }

    // MARK: - Swipe
    private var swipeGesture: some Gesture {
        DragGesture(minimumDistance: 10, coordinateSpace: .local)
            .onChanged { v in
                dragOffset = v.translation.width
            }
            .onEnded { v in
                defer {
                    dragOffset = 0
                    lastInteraction = Date() // pause auto-advance
                }
                let threshold: CGFloat = 40
                if v.translation.width < -threshold {
                    withAnimation(.easeInOut(duration: reduceMotion ? 0 : 0.18)) {
                        index = (index + 1) % Metric.allCases.count
                    }
                } else if v.translation.width > threshold {
                    withAnimation(.easeInOut(duration: reduceMotion ? 0 : 0.18)) {
                        index = (index - 1 + Metric.allCases.count) % Metric.allCases.count
                    }
                }
            }
    }

    // MARK: - Helpers
    private func value(for metric: Metric) -> String {
        switch metric {
        case .basal: return String(format: "%.2f", basalUph)
        case .isf:   return String(format: "%.0f", isf)
        case .cr:    return String(format: "%.1f", carbRatio)
        }
    }
    private func unit(for metric: Metric) -> String {
        switch metric {
        case .basal: return "U/hr"
        case .isf:   return "mg/dL/U"
        case .cr:    return "g/U"
        }
    }
    private func title(for metric: Metric) -> String {
        switch metric {
        case .basal: return "Basal now"
        case .isf:   return "ISF now"
        case .cr:    return "CR now"
        }
    }
    private func sparkline(for metric: Metric) -> [Double]? {
        switch metric {
        case .basal: return normalized(basalPattern)
        case .isf:   return normalized(isfPattern)
        case .cr:    return normalized(crPattern)
        }
    }
    private func normalized(_ arr: [Double]?) -> [Double]? {
        guard let arr, arr.count >= 3 else { return nil }
        let minV = arr.min() ?? 0, maxV = arr.max() ?? 1
        let denom = max(maxV - minV, .leastNonzeroMagnitude)
        return arr.map { ($0 - minV) / denom }
    }

    private var accessibilityLabel: String {
        let m = currentMetric
        let valueText = value(for: m) + " " + unit(for: m)
        if let r = activeRange {
            return "Therapy. \(title(for: m)), \(valueText). Active \(r.timeLabel)."
        } else {
            return "Therapy. \(title(for: m)), \(valueText)."
        }
    }
}

// MARK: - Slide view
// Update SlideView signature
private struct SlideView: View {
    var metric: TherapyCarouselTile.Metric
    var value: String
    var unit: String
    var title: String
    var accent: Color
    var sparkline: [Double]?
    var innerDiameter: CGFloat   // ← new

    var body: some View {
        InnerWindow(diameter: innerDiameter) {   // ← use passed size
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)

            HStack(spacing: 6) {
                Text(value).font(.title3.weight(.semibold))
                    .minimumScaleFactor(0.7)
                Text(unit).font(.caption2).foregroundStyle(.secondary)
                    .baselineOffset(2)
            }
            .lineLimit(1)

            if let spark = sparkline, spark.count > 1 {
                Sparkline(points: spark)
                    .stroke(accent.opacity(0.55),
                            style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                    .frame(height: 14)
                    .padding(.top, 2)
            } else {
                Color.clear.frame(height: 14).padding(.top, 2)
            }
        }
        .padding(.bottom, 4) // a little extra lift off the ring
    }
}


// MARK: - SlideStack (simple pager with drag offset)

private struct SlideStack<Content: View>: View {
    @Binding var index: Int
    @Binding var dragOffset: CGFloat
    var reduceMotion: Bool
    @ViewBuilder var content: () -> Content

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            HStack(spacing: 0) {
                content()
                    .frame(width: width)
            }
            .offset(x: -CGFloat(index) * width + dragOffset)
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.18), value: index)
            .animation(reduceMotion ? nil : .interactiveSpring(response: 0.25, dampingFraction: 0.9), value: dragOffset)
        }
        .clipped()
    }
}

// MARK: - Indicator

private struct IndicatorDot: View {
    var active: Bool
    var accent: Color
    var body: some View {
        Circle()
            .fill(active ? accent.opacity(0.95) : accent.opacity(0.25))
            .frame(width: 6, height: 6)
    }
}

// MARK: - Sparkline Shape (0...1 points)

private struct Sparkline: Shape {
    var points: [Double]
    func path(in rect: CGRect) -> Path {
        guard points.count > 1 else { return Path() }
        var p = Path()
        let stepX = rect.width / CGFloat(points.count - 1)
        let ys = points.map { rect.height * (1 - CGFloat($0)) } // invert (0 at bottom)
        p.move(to: CGPoint(x: 0, y: ys[0]))
        for i in 1..<points.count {
            p.addLine(to: CGPoint(x: CGFloat(i) * stepX, y: ys[i]))
        }
        return p
    }
}







struct ActivityItem: Identifiable {
    enum Kind { case site, sync, therapy, note }
    let id = UUID()
    let kind: Kind
    let title: String
    let detail: String?
    let time: Date
}

private struct ActivityTimeline: View {
    var items: [ActivityItem]
    var accent: Color

    @ScaledMetric private var dot: CGFloat = 8
    @ScaledMetric private var pad: CGFloat = 12
    @ScaledMetric private var lineW: CGFloat = 2

    var body: some View {
        VStack(alignment: .leading, spacing: pad) {
            Text("Recent activity")
                .font(.headline)
                .padding(.bottom, 2)

            VStack(alignment: .leading, spacing: pad) {
                ForEach(items) { item in
                    HStack(alignment: .top, spacing: 12) {
                        // timeline rail + dot
                        VStack {
                            Circle()
                                .fill(color(for: item.kind, accent: accent))
                                .frame(width: dot, height: dot)
                                .overlay(Circle().stroke(.white.opacity(0.3), lineWidth: 1))
                            Rectangle()
                                .fill(
                                    LinearGradient(colors: [
                                        color(for: item.kind, accent: accent).opacity(0.5),
                                        .clear
                                    ], startPoint: .top, endPoint: .bottom)
                                )
                                .frame(width: lineW)
                                .opacity(0.5)
                                .padding(.top, 2)
                            Spacer(minLength: 0)
                        }

                        // content
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Image(systemName: icon(for: item.kind))
                                    .imageScale(.small)
                                    .foregroundStyle(color(for: item.kind, accent: accent))
                                Text(item.title)
                                    .font(.subheadline.weight(.semibold))
                            }
                            if let detail = item.detail, !detail.isEmpty {
                                Text(detail)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                    .minimumScaleFactor(0.9)
                            }
                            Text(timeString(item.time))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 0)
                    }
                }
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func icon(for kind: ActivityItem.Kind) -> String {
        switch kind {
        case .site: return "bandage.fill"
        case .sync: return "arrow.triangle.2.circlepath"
        case .therapy: return "cross.case.fill"
        case .note: return "note.text"
        }
    }
    private func color(for kind: ActivityItem.Kind, accent: Color) -> Color {
        switch kind {
        case .site: return accent
        case .sync: return accent.opacity(0.9)
        case .therapy: return accent.opacity(0.8)
        case .note: return .secondary
        }
    }
    private func timeString(_ date: Date) -> String {
        let f = DateFormatter()
        f.timeStyle = .short
        f.doesRelativeDateFormatting = true
        return f.string(from: date)
    }
}



// MARK: - HOME

struct HomeScreen: View {
    @Binding var showSignInView: Bool
    

    // Models
    @ObservedObject private var site = SiteChangeData.shared
    
    @StateObject private var therapyVM = TherapyVM()

    
    @State private var lastSyncText = "Synced recently"
    @State private var therapySummary = "Profile 1 · Basal 0.9–1.1"
    @State private var isSyncing = false
    @State private var syncTick = 0
    @State private var activeBanner: HomeBanner?
    @State private var latestTherapySnapshot: TherapySnapshot?
    @State private var showBackfillSetupSheet = false
    @State private var backfillConfiguration = HealthBackfillConfiguration.defaults()
    @State private var hasRequestedHealthAuthorization = false
    @ObservedObject private var chameliaDashboard = ChameliaDashboardStore.shared
    @StateObject private var chameliaInsights = ChameliaInsightsStore()
    @EnvironmentObject private var themeManager: ThemeManager
    private var theme: HomeTheme { themeManager.theme }  // computed proxy
    // Layout
    @ScaledMetric private var gridMin: CGFloat = 220
    private var columns: [GridItem] { [GridItem(.adaptive(minimum: gridMin), spacing: 16)] }

    private var recentActivityItems: [ActivityItem] {
        var items: [ActivityItem] = []

        if let siteDate = site.latestSiteChangeDate,
           site.siteChangeLocation != "Not selected" {
            items.append(.init(
                kind: .site,
                title: "Site changed",
                detail: site.siteChangeLocation,
                time: siteDate
            ))
        }

        if let userId = Auth.auth().currentUser?.uid,
           let lastSyncDate = UserDefaults.standard.object(forKey: "LastSyncDate.\(userId)") as? Date {
            items.append(.init(
                kind: .sync,
                title: "Health sync completed",
                detail: "Health data up to date",
                time: lastSyncDate
            ))
        }

        if let latestTherapySnapshot {
            items.append(.init(
                kind: .therapy,
                title: "Therapy profile updated",
                detail: latestTherapySnapshot.profileName,
                time: latestTherapySnapshot.timestamp
            ))
        }

        if let dashboardDate = chameliaDashboard.state.lastUpdatedAt {
            let detail: String?
            if let recommendation = chameliaDashboard.state.recommendation {
                if let firstSummary = recommendation.segmentSummaries.first {
                    let changes = [firstSummary.isf, firstSummary.cr, firstSummary.basal]
                        .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                        .joined(separator: " · ")
                    detail = changes.isEmpty ? firstSummary.label : "\(firstSummary.label) · \(changes)"
                } else if let firstStructure = recommendation.structureSummaries.first {
                    detail = firstStructure
                } else if let predicted = recommendation.predictedOutcomes {
                    detail = "Predicted TIR \(predicted.deltaTIR.homeSignedPercentString) · % low \(predicted.deltaPercentLow.homeSignedPercentString)"
                } else {
                    detail = "Predicted cost improvement \(recommendation.predictedImprovement.formatted(.number.precision(.fractionLength(2))))"
                }
            } else if let status = chameliaDashboard.state.status {
                detail = status.graduated ? "Recommendation system is ready" : "Still learning your patterns"
            } else {
                detail = nil
            }

            items.append(.init(
                kind: .note,
                title: "Chamelia updated",
                detail: detail,
                time: dashboardDate
            ))
        }

        let sorted = items.sorted { $0.time > $1.time }
        if sorted.isEmpty {
            return [
                .init(
                    kind: .note,
                    title: "No recent activity yet",
                    detail: "Sync HealthKit or update a therapy profile to start a timeline.",
                    time: Date()
                )
            ]
        }

        return Array(sorted.prefix(4))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                BreathingBackground(theme: theme)
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 20) {

                        // HERO HEADER
                        HeroHeader(accent: theme.accent, lastSyncText: lastSyncText)

                        // CIRCULAR TILES
                        FlowLayout(alignment: .center, spacing: 16) {
                            // Site Change (countdown to 3 days)
                            NavigationLink { SiteChangeUIV2() } label: {
                                                            SiteChangeTile(
                                                                daysSince: site.daysSinceSiteChange,
                                                                location: site.siteChangeLocation,
                                                                accent: theme.accent
                                                            )
                                                            .floatAndPulse(seed: 0.11)
//                                                            .pressable()
                                                        }
                            .buttonStyle(.plain)
                            
                            // Community (board + crosswords)
                            NavigationLink {
                                CommunityHub()   // ← no accent param now
                            } label: {
                                CommunityTile(accent: theme.accent).floatAndPulse(seed: 0.21)
                            }
                            .buttonStyle(.plain)



                            // Therapy (24h ticks + active window hint)
                            NavigationLink { TherapySettings() } label: {
                                TherapyCarouselTile(
                                        activeRange: therapyVM.currentHourRange,
                                        accent: theme.accent,
                                        basalUph: therapyVM.currentBasal,
                                        isf: therapyVM.currentISF,
                                        carbRatio: therapyVM.currentCarbRatio,
                                        basalPattern: therapyVM.sparklineBasal,   // optional: [Double]? or nil
                                        isfPattern: therapyVM.sparklineISF,       // optional: [Double]? or nil
                                        crPattern: therapyVM.sparklineCR          // optional: [Double]? or nil
                                    )
                                .floatAndPulse(seed: 0.57)
                            }
                            .buttonStyle(.plain)

                            NavigationLink {
                                if let recommendation = chameliaDashboard.state.recommendation {
                                    RecommendationView(
                                        recommendation: recommendation,
                                        recId: chameliaDashboard.state.recId,
                                        status: chameliaDashboard.state.status,
                                        currentHourRanges: therapyVM.activeProfile?.hourRanges,
                                        onApply: applyCurrentRecommendation,
                                        onSkip: skipCurrentRecommendation
                                    )
                                    .environmentObject(themeManager)
                                } else if shadowProgressStatus.graduated {
                                    ChameliaInsightsView(store: chameliaInsights)
                                        .environmentObject(themeManager)
                                } else {
                                    ZStack {
                                        BreathingBackground(theme: theme)
                                            .ignoresSafeArea()
                                        ShadowProgressView(status: shadowProgressStatus)
                                            .environmentObject(themeManager)
                                            .padding(16)
                                    }
                                    .navigationTitle("Shadow Progress")
                                    .navigationBarTitleDisplayMode(.inline)
                                }
                            } label: {
                                RecommendationTile(
                                    status: chameliaDashboard.state.status,
                                    recommendation: chameliaDashboard.state.recommendation,
                                    accent: theme.accent,
                                    isRefreshing: chameliaDashboard.isRefreshing,
                                    errorMessage: chameliaDashboard.latestErrorMessage
                                )
                                .floatAndPulse(seed: 0.73)
                            }
                            .buttonStyle(.plain)

                            // Sync (liquid-ish wave when syncing)
                            Button {
                                guard !isSyncing else { return }
                                if DataManager.shared.shouldPromptForInitialBackfill() {
                                    backfillConfiguration = DataManager.shared.initialBackfillConfiguration()
                                    showBackfillSetupSheet = true
                                    return
                                }
                                isSyncing = true
                                DataManager.shared.syncHealthData {
                                    isSyncing = false
                                    refreshLastSyncText()
                                    syncTick &+= 1
                                    Task {
                                        await refreshChameliaDashboard()
                                        await refreshChameliaInsights()
                                        await refreshLatestTherapySnapshot()
                                    }
                                }
                            } label: {
                                SyncTile(
                                    isSyncing: isSyncing,
                                    accent: theme.accent
                                ).floatAndPulse(seed: 0.97)
                            }
                            .buttonStyle(.plain)
                            .applySuccessHaptic(trigger: syncTick)
                        }
                        .padding(.horizontal, 16)

                        if chameliaInsights.snapshot != nil || chameliaInsights.isLoading || chameliaDashboard.state.status?.graduated == true {
                            NavigationLink {
                                ChameliaInsightsView(store: chameliaInsights)
                                    .environmentObject(themeManager)
                            } label: {
                                ChameliaInsightsEntryCard(
                                    snapshot: chameliaInsights.snapshot,
                                    fallbackStatus: chameliaDashboard.state.status,
                                    accent: theme.accent,
                                    isLoading: chameliaInsights.isLoading,
                                    errorMessage: chameliaInsights.errorMessage
                                )
                            }
                            .buttonStyle(.plain)
                            .padding(.horizontal, 16)
                        }

                        ActivityTimeline(items: recentActivityItems, accent: theme.accent)
                            .padding(.horizontal, 16)

                        // CTA → Mood
                        NavigationLink {
                            MoodPicker()
                        } label: {
                            MoodCTAOrb(accent: theme.accent, title: "How are you feeling?")
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)

                    }
                    .padding(.top, 12)
                }
            }
            .overlay(alignment: .top) {
                if let activeBanner {
                    FloatingBanner(banner: activeBanner)
                        .padding(.top, 10)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .navigationTitle("Home")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink {
                        SettingsView(showSignInView: $showSignInView)
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .imageScale(.medium)
                            .accessibilityLabel("Settings")
                            .foregroundStyle(theme.accent)
                    }
                }
            }
            .task {
                            await handleInitialHealthAuthorizationFlow()
                            therapyVM.reload()   // ensure it loads on first appear
                            await refreshChameliaDashboard()
                            await refreshLatestTherapySnapshot()
                            await refreshChameliaInsights()
                            refreshLastSyncText()
                        }
            .onAppear {
                refreshLastSyncText()
            }
            .onChange(of: chameliaDashboard.latestErrorMessage) { message in
                guard let message, !message.isEmpty else { return }
                presentBanner(message, tone: .warning)
                chameliaDashboard.clearTransientError()
            }
            .sheet(isPresented: $showBackfillSetupSheet) {
                HealthBackfillSetupSheet(
                    configuration: $backfillConfiguration,
                    accent: theme.accent,
                    isSyncing: isSyncing,
                    onStart: startInitialBackfillSync
                )
            }
        }
    }
}

// MARK: - HERO HEADER

private struct HeroHeader: View {
    var accent: Color
    @ScaledMetric(relativeTo: .title3) private var orbSize: CGFloat = 28
    @ScaledMetric(relativeTo: .title3) private var bearSize: CGFloat = 42

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var orbBreath: CGFloat = 0

    var lastSyncText: String

    var body: some View {
        HStack(spacing: 12) {

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text("InSite").font(.title2.weight(.semibold))
                    // Mood orb
                    Circle()
                        .fill(accent.gradient)
                        .frame(width: orbSize, height: orbSize)
                        .scaleEffect(reduceMotion ? 1 : 1 + 0.02 * orbBreath)
                        .shadow(color: accent.opacity(0.35), radius: 6, x: 0, y: 2)
                        .accessibilityHidden(true)
                        .onAppear {
                            guard !reduceMotion else { return }
                            withAnimation(.easeInOut(duration: 3.5).repeatForever(autoreverses: true)) {
                                orbBreath = 1
                            }
                        }
                }
                Text(lastSyncText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Last sync: \(lastSyncText)")
            }

            Spacer()
        }
        .padding(.horizontal, 16)
    }
}

private struct HomeBanner: Equatable {
    enum Tone {
        case success
        case warning
        case neutral

        var color: Color {
            switch self {
            case .success: return .green
            case .warning: return .orange
            case .neutral: return .secondary
            }
        }

        var icon: String {
            switch self {
            case .success: return "checkmark.circle.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .neutral: return "brain.head.profile"
            }
        }
    }

    var message: String
    var tone: Tone
}

private struct HealthBackfillSetupSheet: View {
    @Binding var configuration: HealthBackfillConfiguration
    let accent: Color
    let isSyncing: Bool
    let onStart: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Choose how much Health data to import the first time. InSite will request up to this many days for each category and backfill Firebase before normal incremental syncs begin.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    ForEach(HealthBackfillDataType.allCases) { type in
                        backfillRow(for: type)
                    }
                }
                .padding(20)
            }
            .navigationTitle("Initial Health Backfill")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Not now") {
                        dismiss()
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                Button {
                    onStart()
                } label: {
                    HStack {
                        if isSyncing {
                            ProgressView()
                                .tint(.white)
                        }
                        Text(isSyncing ? "Backfilling…" : "Start Backfill")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(accent)
                    )
                    .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
                .disabled(isSyncing)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }
        }
    }

    private func backfillRow(for type: HealthBackfillDataType) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(type.title)
                .font(.headline)
            Text(type.detail)
                .font(.caption)
                .foregroundStyle(.secondary)

            Stepper(value: binding(for: type), in: 0...365, step: 5) {
                HStack {
                    Text("\(configuration.days(for: type)) days")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    if configuration.days(for: type) == 0 {
                        Text("Skip")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(14)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func binding(for type: HealthBackfillDataType) -> Binding<Int> {
        Binding(
            get: { configuration.days(for: type) },
            set: { configuration.setDays($0, for: type) }
        )
    }
}

private struct FloatingBanner: View {
    let banner: HomeBanner

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: banner.tone.icon)
                .foregroundStyle(banner.tone.color)
            Text(banner.message)
                .font(.footnote.weight(.medium))
                .foregroundStyle(.primary)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: 520)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(
            Capsule()
                .stroke(banner.tone.color.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.12), radius: 14, x: 0, y: 6)
        .padding(.horizontal, 16)
    }
}







// MARK: - SITE CHANGE TILE (countdown ring)

private struct SiteChangeTile: View {
    var daysSince: Int
    var location: String
    var accent: Color

    @ScaledMetric private var diameter: CGFloat = 140
    @ScaledMetric private var stroke: CGFloat = 10

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var rotation: Double = 0

    private var progress: Double {
        let goal = 3.0
        return min(1.0, Double(daysSince) / goal)
    }

    var body: some View {
        CircleTileBase(diameter: diameter) {
            ZStack {
                // Track
                Circle()
                    .stroke(Color.primary.opacity(0.08), lineWidth: stroke)

                // Progress ring with slow rotate
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        AngularGradient(colors: [
                            accent.opacity(0.85),
                            accent.opacity(0.35),
                            accent.opacity(0.85)
                        ], center: .center),
                        style: StrokeStyle(lineWidth: stroke, lineCap: .round)
                    )
                    .rotationEffect(.degrees(reduceMotion ? 0 : rotation))

                InnerWindow(diameter: diameter) {
                    Text("Days Since")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    Text("\(daysSince)")
                        .font(.system(.largeTitle, design: .rounded).weight(.semibold))
                        .minimumScaleFactor(0.7)
                        .lineLimit(1)

                    if !location.isEmpty {
                        Text(location)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                            .truncationMode(.tail)
                            .layoutPriority(1)
                    }
                }
                .padding(.horizontal, 8)
            }
        }
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.linear(duration: 18).repeatForever(autoreverses: false)) {
                rotation = 360
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(daysSince) days since site change at \(location)")
    }
}


    // MARK: - Shapes & tiny views

private struct PieSlice: Shape {
        var startAngle: Angle
        var endAngle: Angle

        func path(in rect: CGRect) -> Path {
            var p = Path()
            let center = CGPoint(x: rect.midX, y: rect.midY)
            let r = min(rect.width, rect.height) / 2
            p.move(to: center)
            p.addArc(center: center, radius: r, startAngle: startAngle, endAngle: endAngle, clockwise: false)
            p.closeSubpath()
            return p
        }
    }

private struct NowMarker: Shape {
        var hour: Int
        var animatableData: CGFloat {
            get { CGFloat(hour) }
            set { hour = Int(newValue) }
        }
        func path(in rect: CGRect) -> Path {
            var p = Path()
            let radius = min(rect.width, rect.height) / 2
            let angle = CGFloat(hour) / 24.0 * 2 * .pi - .pi / 2
            let c = CGPoint(x: rect.midX, y: rect.midY)
            let pt = CGPoint(x: c.x + cos(angle) * radius, y: c.y + sin(angle) * radius)
            p.addEllipse(in: CGRect(x: pt.x - 3, y: pt.y - 3, width: 6, height: 6))
            return p
        }
    }

private struct LegendDot: View {
        var color: Color
        var body: some View {
            Circle().fill(color).frame(width: 6, height: 6)
        }
    }

// Arc for a given HourRange (supports wraparound)
private struct TherapyArc: Shape {
    var range: HourRange

    private func angle(for minute: Double) -> Angle {
        Angle(degrees: (minute / 1440.0) * 360.0 - 90.0)
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2

        let startAngle = angle(for: Double(range.startMinute))
        let endAngle   = angle(for: Double(range.endMinute))
        path.addArc(center: center, radius: radius, startAngle: startAngle, endAngle: endAngle, clockwise: false)
        return path
    }
}


// MARK: - SYNC TILE (liquid-ish wave)

private struct SyncTile: View {
    var isSyncing: Bool
    var accent: Color

    @ScaledMetric private var diameter: CGFloat = 140

    var body: some View {
        CircleTileBase(diameter: diameter) {
            ZStack {
                if isSyncing {
                    LiquidWave(color: accent)
                } else {
                    Circle().fill(accent.opacity(0.10))
                }

                VStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(accent)
                    Text(isSyncing ? "Syncing…" : "Health data")
                        .font(.subheadline)
                    Text(isSyncing ? "Please wait" : "Sync & status")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(isSyncing ? "Sync in progress" : "Sync health data")
    }
}

// MARK: - BUILDING BLOCKS

struct CircleTileBase<Content: View>: View {
    @ScaledMetric private var pad: CGFloat = 14
    @ScaledMetric private var corner: CGFloat = 24
    var diameter: CGFloat
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @ViewBuilder var content: Content

    var body: some View {
        ZStack {
            Circle()
                .fill(.ultraThinMaterial)
                .overlay(Circle().stroke(Color.primary.opacity(0.06), lineWidth: 1))
                .shadow(color: Color.black.opacity(0.10), radius: 10, x: 0, y: 4)

            content.padding(pad)
        }
        .frame(width: diameter, height: diameter)
        .contentShape(Circle())
        .scaleEffect(reduceMotion ? 1 : 1.0) // future micro-interactions here
    }
}

fileprivate struct Card<Inner: View>: View {
    @ScaledMetric private var corner: CGFloat = 16
    @ScaledMetric private var pad: CGFloat = 14
    @ViewBuilder var content: Inner

    var body: some View {
        VStack { content }
            .padding(pad)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: corner))
    }
}

private struct ActivityRow: View {
    var text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        HStack(spacing: 8) {
            Circle().frame(width: 6, height: 6).foregroundStyle(.secondary)
            Text(text).font(.subheadline)
            Spacer()
        }
    }
}

// MARK: - BACKGROUND (breathing, subtle)
struct BreathingBackground: View {
    var theme: HomeTheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hueShift: Double = 0

    var body: some View {
        LinearGradient(
            colors: [
                theme.bgStart.hueShifted(hueShift),
                theme.bgEnd.hueShifted(hueShift * 0.6)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 8.0).repeatForever(autoreverses: true)) {
                hueShift = 0.02 // ±2% hue drift
            }
        }
    }
}

// MARK: - LIQUID WAVE (inside the sync tile)

private struct LiquidWave: View {
    var color: Color
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            ZStack {
                Circle()
                    .fill(color.opacity(0.12))

                // Wave clipped to circle
                WaveShape(amplitude: size * 0.06, wavelength: size * 0.7, phase: phase)
                    .fill(color.opacity(0.35))
                    .clipShape(Circle())
                    .offset(y: size * 0.15)

                WaveShape(amplitude: size * 0.04, wavelength: size * 0.55, phase: phase * 1.3)
                    .fill(color.opacity(0.25))
                    .clipShape(Circle())
                    .offset(y: size * 0.12)
            }
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.linear(duration: 2.2).repeatForever(autoreverses: false)) {
                    phase = .pi * 2
                }
            }
        }
    }
}

private struct WaveShape: Shape {
    var amplitude: CGFloat
    var wavelength: CGFloat
    var phase: CGFloat

    var animatableData: CGFloat {
        get { phase }
        set { phase = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let midY = rect.midY
        path.move(to: CGPoint(x: 0, y: midY))
        let step = max(1, wavelength / 20)
        for x in stride(from: 0, through: rect.width, by: step) {
            let relative = x / wavelength
            let y = midY + sin(relative * .pi * 2 + phase) * amplitude
            path.addLine(to: CGPoint(x: x, y: y))
        }
        // close at bottom
        path.addLine(to: CGPoint(x: rect.width, y: rect.height))
        path.addLine(to: CGPoint(x: 0, y: rect.height))
        path.closeSubpath()
        return path
    }
}

// MARK: - UTILITIES

//private extension View {
//    /// Success notification haptic when `trigger` changes (iOS 17+).
//    @ViewBuilder
//    func applySuccessHaptic(trigger: Int) -> some View {
//        if #available(iOS 17.0, *) {
//            self.sensoryFeedback(.success, trigger: trigger)
//        } else {
//            self
//        }
//    }
//}

private extension Color {
    func hueShifted(_ delta: Double) -> Color {
        // super-lightweight approximation (good enough for tiny shifts)
        // for precise HSB math, keep your existing mapping utilities
        return self.opacity(1.0) // placeholder to avoid heavy conversions here
    }
}

private extension HomeScreen {
    func handleInitialHealthAuthorizationFlow() async {
        guard !hasRequestedHealthAuthorization else { return }
        hasRequestedHealthAuthorization = true

        let authorized = await withCheckedContinuation { continuation in
            DataManager.shared.requestAuthorization { success in
                continuation.resume(returning: success)
            }
        }

        guard authorized, DataManager.shared.shouldPromptForInitialBackfill() else { return }
        await MainActor.run {
            backfillConfiguration = DataManager.shared.initialBackfillConfiguration()
            showBackfillSetupSheet = true
        }
    }

    func startInitialBackfillSync() {
        guard !isSyncing else { return }
        isSyncing = true
        let configuration = backfillConfiguration
        DataManager.shared.syncHealthData(initialBackfill: configuration) {
            isSyncing = false
            showBackfillSetupSheet = false
            refreshLastSyncText()
            syncTick &+= 1
            Task {
                await refreshChameliaDashboard()
                await refreshChameliaInsights()
                await refreshLatestTherapySnapshot()
            }
        }
    }

    var shadowProgressStatus: GraduationStatus {
        chameliaDashboard.state.status
            ?? GraduationStatus(
                graduated: false,
                nDays: 0,
                winRate: 0,
                safetyViolations: 0,
                consecutiveDays: 0
            )
    }

    func refreshChameliaDashboard() async {
        await chameliaDashboard.bootstrapCurrentUser()
    }

    func refreshChameliaInsights() async {
        await chameliaInsights.refresh(
            userId: Auth.auth().currentUser?.uid,
            fallbackStatus: chameliaDashboard.state.status
        )
    }

    func refreshLatestTherapySnapshot() async {
        latestTherapySnapshot = try? await TherapySettingsLogManager.shared.getActiveTherapyProfile(at: Date())
    }

    func refreshLastSyncText() {
        guard let userId = Auth.auth().currentUser?.uid,
              let lastSyncDate = UserDefaults.standard.object(forKey: "LastSyncDate.\(userId)") as? Date else {
            lastSyncText = "Sync when you're ready"
            return
        }
        lastSyncText = "Synced \(relativeTimestamp(from: lastSyncDate))"
    }

    func relativeTimestamp(from date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: Date())
    }

    func presentBanner(_ message: String, tone: HomeBanner.Tone) {
        withAnimation(.spring(response: 0.32, dampingFraction: 0.86)) {
            activeBanner = HomeBanner(message: message, tone: tone)
        }
        Task {
            try? await Task.sleep(nanoseconds: 2_600_000_000)
            await MainActor.run {
                guard activeBanner?.message == message else { return }
                withAnimation(.spring(response: 0.32, dampingFraction: 0.9)) {
                    activeBanner = nil
                }
            }
        }
    }

    func applyCurrentRecommendation() {
        guard let user = Auth.auth().currentUser else { return }
        guard let recommendation = chameliaDashboard.state.recommendation else { return }
        guard recommendation.actionLevel < 3 else {
            presentBanner("This recommendation can be previewed but not applied in InSite yet.", tone: .warning)
            return
        }

        switch recommendation.recommendationScope {
        case "patch_existing":
            if let targetId = recommendation.targetProfileId {
                applyRecommendation(recommendation, toProfileId: targetId, userId: user.uid, successMessage: "Recommendation applied to the selected therapy profile.")
            } else {
                applyRecommendationToActiveProfile(recommendation, userId: user.uid)
            }
        case "create_new":
            let regime = recommendation.detectedRegime?
                .split(separator: "_")
                .map { $0.capitalized }
                .joined(separator: " ")
                ?? "regime"
            presentBanner("Chamelia suggests a new \(regime.lowercased()) profile. That creation flow is coming soon.", tone: .neutral)
            return
        default:
            applyRecommendationToActiveProfile(recommendation, userId: user.uid)
        }
    }

    func skipCurrentRecommendation() {
        guard let user = Auth.auth().currentUser else { return }
        let recId = chameliaDashboard.state.recId
        let latestSignals = chameliaDashboard.state.latestSignals
        chameliaDashboard.clearRecommendation(userId: user.uid)
        lastSyncText = "Recommendation skipped"
        presentBanner("Recommendation skipped for now.", tone: .neutral)

        Task {
            do {
                if let recId {
                    try await ChameliaEngine.shared.recordOutcome(
                        patientId: user.uid,
                        recId: Int(recId),
                        response: "reject",
                        signals: latestSignals,
                        cost: 0
                    )
                }
                _ = try await ChameliaStateManager.shared.saveToFirebase(userId: user.uid)
            } catch {
                await MainActor.run {
                    presentBanner(readableMessage(for: error), tone: .warning)
                }
            }
        }
    }

    func readableMessage(for error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    func activeProfileIndex(in profiles: [DiabeticProfile], store: ProfileDataStore) -> Int? {
        if let activeId = store.loadActiveProfileID(),
           let index = profiles.firstIndex(where: { $0.id == activeId }) {
            return index
        }
        return profiles.indices.first
    }

    func applyRecommendationToActiveProfile(_ recommendation: RecommendationPackage, userId: String) {
        let store = ProfileDataStore()
        let profiles = store.loadProfiles()
        guard let targetIndex = activeProfileIndex(in: profiles, store: store) else { return }
        let targetId = profiles[targetIndex].id
        applyRecommendation(
            recommendation,
            toProfileId: targetId,
            userId: userId,
            successMessage: "Recommendation applied to your active therapy profile."
        )
    }

    func applyRecommendation(_ recommendation: RecommendationPackage, toProfileId profileId: String, userId: String, successMessage: String) {
        let recId = chameliaDashboard.state.recId
        let latestSignals = chameliaDashboard.state.latestSignals

        let store = ProfileDataStore()
        var profiles = store.loadProfiles()
        guard let targetIndex = profiles.firstIndex(where: { $0.id == profileId }) ?? activeProfileIndex(in: profiles, store: store) else { return }

        let existingProfile = profiles[targetIndex]
        let updatedRanges: [HourRange]
        do {
            updatedRanges = try applyRecommendationAction(recommendation.action, to: existingProfile.hourRanges)
        } catch {
            presentBanner(readableMessage(for: error), tone: .warning)
            return
        }

        let updatedProfile = DiabeticProfile(
            id: existingProfile.id,
            name: existingProfile.name,
            hourRanges: updatedRanges
        )
        profiles[targetIndex] = updatedProfile

        store.saveProfiles(profiles)
        therapyVM.reload()
        lastSyncText = "Recommendation applied"
        latestTherapySnapshot = TherapySnapshot(
            timestamp: Date(),
            profileId: updatedProfile.id,
            profileName: updatedProfile.name,
            hourRanges: updatedProfile.hourRanges
        )
        chameliaDashboard.clearRecommendation(userId: userId)
        presentBanner(successMessage, tone: .success)

        Task {
            do {
                _ = try? await TherapySettingsLogManager.shared.logTherapySettingsChange(profile: updatedProfile)
                if let recId {
                    try await ChameliaEngine.shared.recordOutcome(
                        patientId: userId,
                        recId: Int(recId),
                        response: "accept",
                        signals: latestSignals,
                        cost: 0
                    )
                }
                _ = try await ChameliaStateManager.shared.saveToFirebase(userId: userId)
            } catch {
                await MainActor.run {
                    presentBanner(readableMessage(for: error), tone: .warning)
                }
            }
        }
    }

    func applyDeltas(_ deltas: [String: Double], to range: HourRange) -> HourRange {
        var updated = range
        if let basalDelta = deltas["basal_delta"] {
            updated.basalRate = max(0, updated.basalRate * (1 + basalDelta))
        }
        if let isfDelta = deltas["isf_delta"] {
            updated.insulinSensitivity = max(0, updated.insulinSensitivity * (1 + isfDelta))
        }
        if let crDelta = deltas["cr_delta"] {
            updated.carbRatio = max(0, updated.carbRatio * (1 + crDelta))
        }
        return updated
    }

    func applyRecommendationAction(_ action: TherapyAction, to ranges: [HourRange]) throws -> [HourRange] {
        if action.segmentDeltas.isEmpty && action.structuralEdits.isEmpty {
            return ranges.map { applyDeltas(action.deltas, to: $0) }
        }

        var segments = ranges
            .sorted { $0.startMinute < $1.startMinute }
            .map { SegmentWorkItem(segmentId: stableSegmentId(for: $0), range: $0) }

        if !action.structuralEdits.isEmpty {
            segments = try applyStructureEdits(action.structuralEdits, to: segments)
        }
        if !action.segmentDeltas.isEmpty {
            segments = applySegmentDeltas(action.segmentDeltas, to: segments)
        }

        return segments
            .sorted { $0.range.startMinute < $1.range.startMinute }
            .map(\.range)
    }

    func applyStructureEdits(_ edits: [StructureEditPayload], to segments: [SegmentWorkItem]) throws -> [SegmentWorkItem] {
        var current = segments

        for edit in edits {
            switch edit.editType.lowercased() {
            case "split":
                guard let idx = current.firstIndex(where: { $0.segmentId == edit.targetSegmentId }) else { continue }
                let target = current[idx]
                let splitMinute = roundedSplitMinute(for: target.range, splitAtMinute: edit.splitAtMinute)
                guard splitMinute > target.range.startMinute, splitMinute < target.range.endMinute else {
                    throw ChameliaError.serverError(0, "Chamelia suggested an unsupported split for \(edit.targetSegmentId).")
                }

                let left = HourRange(
                    id: target.range.id,
                    startMinute: target.range.startMinute,
                    endMinute: splitMinute,
                    carbRatio: target.range.carbRatio,
                    basalRate: target.range.basalRate,
                    insulinSensitivity: target.range.insulinSensitivity
                )
                let right = HourRange(
                    id: UUID(),
                    startMinute: splitMinute,
                    endMinute: target.range.endMinute,
                    carbRatio: target.range.carbRatio,
                    basalRate: target.range.basalRate,
                    insulinSensitivity: target.range.insulinSensitivity
                )

                current.remove(at: idx)
                current.insert(SegmentWorkItem(segmentId: "\(edit.targetSegmentId)_b", range: right), at: idx)
                current.insert(SegmentWorkItem(segmentId: "\(edit.targetSegmentId)_a", range: left), at: idx)

            case "merge":
                guard
                    let firstIdx = current.firstIndex(where: { $0.segmentId == edit.targetSegmentId }),
                    let neighborId = edit.neighborSegmentId,
                    let secondIdx = current.firstIndex(where: { $0.segmentId == neighborId })
                else { continue }

                let first = current[firstIdx]
                let second = current[secondIdx]
                let ordered = [first, second].sorted { $0.range.startMinute < $1.range.startMinute }
                let a = ordered[0]
                let b = ordered[1]
                guard a.range.endMinute == b.range.startMinute else { continue }

                let merged = HourRange(
                    id: a.range.id,
                    startMinute: a.range.startMinute,
                    endMinute: b.range.endMinute,
                    carbRatio: (a.range.carbRatio + b.range.carbRatio) / 2,
                    basalRate: (a.range.basalRate + b.range.basalRate) / 2,
                    insulinSensitivity: (a.range.insulinSensitivity + b.range.insulinSensitivity) / 2
                )

                current.removeAll { $0.segmentId == a.segmentId || $0.segmentId == b.segmentId }
                current.append(
                    SegmentWorkItem(
                        segmentId: "\(a.segmentId)__\(b.segmentId)",
                        range: merged
                    )
                )
                current.sort { $0.range.startMinute < $1.range.startMinute }

            case "add", "remove":
                throw ChameliaError.serverError(0, "This type of structure edit is not supported in InSite yet.")
            default:
                continue
            }
        }

        return current
    }

    func applySegmentDeltas(_ deltas: [SegmentDeltaPayload], to segments: [SegmentWorkItem]) -> [SegmentWorkItem] {
        var current = segments

        for delta in deltas {
            guard let idx = current.firstIndex(where: { $0.segmentId == delta.segmentId }) else { continue }
            var range = current[idx].range
            if delta.basalDelta != 0 {
                range.basalRate = max(0, range.basalRate * (1 + delta.basalDelta))
            }
            if delta.isfDelta != 0 {
                range.insulinSensitivity = max(0, range.insulinSensitivity * (1 + delta.isfDelta))
            }
            if delta.crDelta != 0 {
                range.carbRatio = max(0, range.carbRatio * (1 + delta.crDelta))
            }
            current[idx].range = range
        }

        return current
    }

    func roundedSplitMinute(for range: HourRange, splitAtMinute: Int?) -> Int {
        let snapStep = 15
        let fallbackMinute = range.startMinute + (range.durationMinutes / 2)
        let splitMinute = splitAtMinute ?? fallbackMinute
        let rounded = Int(round(Double(splitMinute) / Double(snapStep))) * snapStep
        return max(range.startMinute + snapStep, min(range.endMinute - snapStep, rounded))
    }

    func stableSegmentId(for range: HourRange) -> String {
        "\(range.startMinute)-\(range.endMinute)"
    }
}

private struct SegmentWorkItem {
    var segmentId: String
    var range: HourRange
}

struct Template: View {
    var body: some View {
        Text("Hello")
    }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    HomeScreen(showSignInView: .constant(false))
      .environmentObject(ThemeManager())
  }
}
