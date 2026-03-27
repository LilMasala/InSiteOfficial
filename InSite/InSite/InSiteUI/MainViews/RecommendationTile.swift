import SwiftUI

struct RecommendationTile: View {
    let status: GraduationStatus?
    let recommendation: RecommendationPackage?
    let accent: Color
    let isRefreshing: Bool
    let errorMessage: String?

    @ScaledMetric private var diameter: CGFloat = 140
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false
    @State private var hum = false

    private var isReady: Bool {
        (status?.graduated ?? false) && recommendation != nil
    }

    private var isLiveWithoutRecommendation: Bool {
        (status?.graduated ?? false) && recommendation == nil
    }

    private var isUnavailable: Bool {
        errorMessage != nil && status == nil && recommendation == nil
    }

    private var progress: Double {
        guard let status else { return 0.08 }
        let dayProgress = Double(min(status.nDays, 21)) / 21.0
        let streakProgress = Double(min(status.consecutiveDays, 7)) / 7.0
        let winProgress = min(status.winRate / 0.6, 1.0)
        let safetyProgress = status.safetyViolations == 0 ? 1.0 : 0.0
        return min(1.0, max(0.08, (dayProgress + streakProgress + winProgress + safetyProgress) / 4.0))
    }

    private var readinessPercent: Int {
        Int((progress * 100).rounded())
    }

    private var tileState: TileState {
        if isUnavailable { return .unavailable }
        if isRefreshing && status == nil && recommendation == nil { return .syncing }
        if isReady { return .ready }
        if isLiveWithoutRecommendation { return .live }
        if progress >= 0.82 { return .nearlyReady }
        return .learning
    }

    var body: some View {
        CircleTileBase(diameter: diameter) {
            ZStack {
                Circle()
                    .stroke(Color.primary.opacity(0.08), lineWidth: 10)

                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        AngularGradient(
                            colors: tileState.ringColors(accent: accent),
                            center: .center
                        ),
                        style: StrokeStyle(lineWidth: 8, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .scaleEffect(pulse ? 1.01 : 0.99)

                VStack {
                    Spacer(minLength: 0)
                    stateValue
                    stateCaption
                    Spacer(minLength: 0)
                }
                .multilineTextAlignment(.center)
                .frame(width: diameter * 0.7, height: diameter * 0.56)
                .padding(.horizontal, 8)
            }
        }
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 2.8).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }

    private var stateValue: some View {
        Text(tileState.primaryValue(status: status, readinessPercent: readinessPercent))
            .font(.system(.title2, design: .rounded).weight(.bold))
            .foregroundStyle(tileState.titleColor(accent: accent))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
    }

    private var stateCaption: some View {
        VStack(spacing: 4) {
            Text(tileState.eyebrow)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Text(tileState.title)
                .font(.headline.weight(.bold))
                .foregroundStyle(tileState.titleColor(accent: accent))
            Text(tileState.subtitle(status: status, errorMessage: errorMessage))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(1)
        }
    }

    private var accessibilityLabel: String {
        switch tileState {
        case .ready:
            return "Chamelia ready. Recommendation available now."
        case .live:
            return "Chamelia live. Monitoring and evaluating between recommendations."
        case .nearlyReady:
            return "Chamelia nearly ready. Readiness \(readinessPercent) percent."
        case .learning:
            if let status {
                return "Chamelia learning. \(status.nDays) days in shadow, win rate \(Int((status.winRate * 100).rounded())) percent, \(status.consecutiveDays) consecutive good days."
            }
            return "Chamelia learning. Sync to start building readiness."
        case .syncing:
            return "Chamelia syncing."
        case .unavailable:
            return "Chamelia temporarily unavailable."
        }
    }
}

private extension RecommendationTile {
    enum TileState {
        case ready
        case live
        case nearlyReady
        case learning
        case syncing
        case unavailable

        var eyebrow: String {
            switch self {
            case .ready: return "Chamelia"
            case .live: return "Chamelia"
            case .nearlyReady: return "Shadow"
            case .learning: return "Shadow"
            case .syncing: return "Chamelia"
            case .unavailable: return "Chamelia"
            }
        }

        var title: String {
            switch self {
            case .ready: return "Ready"
            case .live: return "Live"
            case .nearlyReady: return "Shadow"
            case .learning: return "Shadow"
            case .syncing: return "Syncing"
            case .unavailable: return "Paused"
            }
        }

        func titleColor(accent: Color) -> Color {
            switch self {
            case .ready: return .green
            case .live: return .green
            case .nearlyReady: return accent
            case .learning: return accent
            case .syncing: return accent
            case .unavailable: return .orange
            }
        }

        func primaryValue(status: GraduationStatus?, readinessPercent: Int) -> String {
            switch self {
            case .ready:
                return "\(readinessPercent)%"
            case .live:
                return status.map { "\($0.nDays)d" } ?? "Live"
            case .nearlyReady:
                return "\(readinessPercent)%"
            case .learning:
                return status.map { "\($0.nDays)d" } ?? "..."
            case .syncing:
                return "..."
            case .unavailable:
                return "!"
            }
        }

        func ringColors(accent: Color) -> [Color] {
            switch self {
            case .ready:
                return [Color.green.opacity(0.9), accent.opacity(0.7), Color.green.opacity(0.9)]
            case .live:
                return [Color.green.opacity(0.88), accent.opacity(0.5), Color.green.opacity(0.88)]
            case .nearlyReady:
                return [accent.opacity(0.95), Color.green.opacity(0.55), accent.opacity(0.95)]
            case .learning:
                return [accent.opacity(0.85), accent.opacity(0.28), accent.opacity(0.85)]
            case .syncing:
                return [accent.opacity(0.3), accent.opacity(0.85), accent.opacity(0.3)]
            case .unavailable:
                return [Color.orange.opacity(0.7), Color.red.opacity(0.45), Color.orange.opacity(0.7)]
            }
        }

        func subtitle(status: GraduationStatus?, errorMessage: String?) -> String {
            switch self {
            case .ready:
                return "Recommendation waiting"
            case .live:
                return status?.lastDecisionReason ?? "Monitoring between actions"
            case .nearlyReady:
                if let status {
                    return "\(status.consecutiveDays)/7 streak"
                }
                return "Shadow criteria nearly met"
            case .learning:
                if let status {
                    return "\(Int((status.winRate * 100).rounded()))% win rate"
                }
                return "Starting up"
            case .syncing:
                return "Refreshing"
            case .unavailable:
                return "Unavailable"
            }
        }
    }
}

#Preview {
    ZStack {
        BreathingBackground(theme: .defaultTeal).ignoresSafeArea()
        HStack(spacing: 20) {
            RecommendationTile(
                status: GraduationStatus(graduated: false, nDays: 10, winRate: 0.48, safetyViolations: 0, consecutiveDays: 3),
                recommendation: nil,
                accent: .teal,
                isRefreshing: false,
                errorMessage: nil
            )
            RecommendationTile(
                status: GraduationStatus(graduated: true, nDays: 23, winRate: 0.7, safetyViolations: 0, consecutiveDays: 9),
                recommendation: RecommendationPackage(
                    action: TherapyAction(kind: "therapy_adjustment", deltas: ["isf_delta": 0.05]),
                    predictedImprovement: 0.08,
                    confidence: 0.72,
                    effectSize: 0.11,
                    cvarValue: 0.2,
                    burnoutAttribution: nil
                ),
                accent: .teal,
                isRefreshing: false,
                errorMessage: nil
            )
        }
    }
}
