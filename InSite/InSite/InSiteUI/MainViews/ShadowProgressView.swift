import SwiftUI

struct ShadowProgressView: View {
    let status: GraduationStatus

    @EnvironmentObject private var themeManager: ThemeManager

    private var accent: Color { themeManager.theme.accent }
    private var winRatePercent: Int { Int((status.winRate * 100).rounded()) }
    private var estimatedDaysRemaining: Int {
        let minDaysRemaining = max(0, 21 - status.nDays)
        let streakRemaining = max(0, 7 - status.consecutiveDays)
        let winRateGap = max(0, 0.6 - status.winRate)
        let winRateDays = Int((winRateGap * 20).rounded(.up))
        return max(minDaysRemaining, streakRemaining, winRateDays)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 14) {
                ZStack {
                    Circle()
                        .fill(accent.opacity(0.15))
                    Circle()
                        .stroke(accent.opacity(0.35), lineWidth: 1)
                    Image(systemName: "hourglass.bottomhalf.filled")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(accent)
                }
                .frame(width: 52, height: 52)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Shadow Progress")
                        .font(.title3.weight(.semibold))
                    Text("Chamelia is learning quietly before surfacing recommendations.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                metricRow(title: "Days in shadow", value: "\(status.nDays)/21", tint: accent)
                metricRow(title: "Win rate", value: "\(winRatePercent)%", tint: winRateColor)
                metricRow(title: "Consecutive good days", value: "\(status.consecutiveDays)/7", tint: accent.opacity(0.9))
                metricRow(title: "Safety violations", value: "\(status.safetyViolations)", tint: safetyColor)
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Estimated until first recommendation")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    Text(estimatedDaysRemaining == 0 ? "Ready soon" : "~\(estimatedDaysRemaining) day\(estimatedDaysRemaining == 1 ? "" : "s")")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(accent)
                }

                GeometryReader { proxy in
                    let progress = min(1.0, max(0.0, Double(status.nDays) / 21.0))
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(Color.primary.opacity(0.08))
                        Capsule()
                            .fill(
                                LinearGradient(
                                    colors: [accent.opacity(0.75), accent],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: proxy.size.width * progress)
                    }
                }
                .frame(height: 12)
            }

            VStack(alignment: .leading, spacing: 8) {
                criterionRow(title: "21 shadow days", met: status.nDays >= 21)
                criterionRow(title: "Win rate at least 60%", met: status.winRate >= 0.6)
                criterionRow(title: "Zero safety violations", met: status.safetyViolations == 0)
                criterionRow(title: "7 consecutive good days", met: status.consecutiveDays >= 7)
            }
        }
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(
            "Shadow progress. \(status.nDays) days in shadow, win rate \(winRatePercent) percent, \(status.consecutiveDays) consecutive good days, \(status.safetyViolations) safety violations."
        )
    }

    private var winRateColor: Color {
        status.winRate >= 0.6 ? .green : .orange
    }

    private var safetyColor: Color {
        status.safetyViolations == 0 ? .green : .red
    }

    @ViewBuilder
    private func metricRow(title: String, value: String, tint: Color) -> some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(tint)
        }
    }

    @ViewBuilder
    private func criterionRow(title: String, met: Bool) -> some View {
        HStack(spacing: 10) {
            Image(systemName: met ? "checkmark.circle.fill" : "circle.dotted")
                .foregroundStyle(met ? Color.green : accent.opacity(0.6))
            Text(title)
                .font(.subheadline)
            Spacer()
        }
        .accessibilityLabel("\(title), \(met ? "met" : "pending")")
    }
}

#Preview {
    ZStack {
        BreathingBackground(theme: .defaultTeal).ignoresSafeArea()
        ShadowProgressView(
            status: GraduationStatus(
                graduated: false,
                nDays: 12,
                winRate: 0.54,
                safetyViolations: 0,
                consecutiveDays: 4
            )
        )
        .environmentObject(ThemeManager())
        .padding()
    }
}
