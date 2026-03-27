import SwiftUI
import UIKit

struct RecommendationView: View {
    let recommendation: RecommendationPackage
    let recId: Int64?
    let status: GraduationStatus?
    let currentHourRanges: [HourRange]?
    let onApply: () -> Void
    let onSkip: () -> Void

    init(
        recommendation: RecommendationPackage,
        recId: Int64?,
        status: GraduationStatus?,
        currentHourRanges: [HourRange]? = nil,
        onApply: @escaping () -> Void,
        onSkip: @escaping () -> Void
    ) {
        self.recommendation    = recommendation
        self.recId             = recId
        self.status            = status
        self.currentHourRanges = currentHourRanges
        self.onApply           = onApply
        self.onSkip            = onSkip
    }

    @EnvironmentObject private var themeManager: ThemeManager
    @State private var showLevel2Confirmation = false

    private var accent: Color { themeManager.theme.accent }
    private var predictedImprovementPercent: Int {
        Int((recommendation.predictedImprovement * 100).rounded())
    }
    private var confidencePercent: Int {
        Int((recommendation.confidence * 100).rounded())
    }
    private var effectSizePercent: Int {
        Int((recommendation.effectSize * 100).rounded())
    }
    private var usesSegmentSummaries: Bool {
        !changedSegmentSummaries.isEmpty
    }
    private var usesStructureSummaries: Bool {
        !recommendation.structureSummaries.isEmpty
    }
    private var requiresExtraConfirmation: Bool {
        recommendation.actionLevel == 2
    }
    private var canApplyInApp: Bool {
        recommendation.actionLevel < 3
    }
    private var changedSegmentSummaries: [RecommendationSegmentSummary] {
        recommendation.segmentSummaries.filter { summary in
            [summary.isf, summary.cr, summary.basal].contains { !$0.isVisuallyUnchanged }
        }
    }

    var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme)
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 18) {
                    recommendationContextCard
                    predictedTradeoffCard
                    confidenceCard
                    changesCard
                    structurePreviewCard
                    burnoutCard
                    shadowContextCard
                    actionButtons
                }
                .padding(16)
            }
        }
        .navigationTitle("Recommendation")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog(
            "Apply new time-block recommendation?",
            isPresented: $showLevel2Confirmation,
            titleVisibility: .visible
        ) {
            Button("Apply Suggested Time-Block Changes") {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                onApply()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This recommendation changes the structure of your therapy schedule. Review the preview carefully before applying.")
        }
    }

    private var predictedTradeoffCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Predicted tradeoff", systemImage: "waveform.path.ecg")
                .font(.headline)
                .foregroundStyle(accent)

            if let predicted = recommendation.predictedOutcomes {
                Text(predicted.deltaTIR >= 0 ? "Expected +\(Int((predicted.deltaTIR * 100).rounded()))% TIR" : "Expected \(Int((predicted.deltaTIR * 100).rounded()))% TIR")
                    .font(.system(.largeTitle, design: .rounded).weight(.bold))

                Text("These are rollout-based estimates, not guarantees. Lower % low and % high are better; lower average BG is only good if lows do not rise.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    detailPill(title: "% Low", value: signedPercentString(predicted.deltaPercentLow), tint: predicted.deltaPercentLow <= 0 ? .green : .orange)
                    detailPill(title: "% High", value: signedPercentString(predicted.deltaPercentHigh), tint: predicted.deltaPercentHigh <= 0 ? .green : .orange)
                    detailPill(title: "Avg BG", value: signedNumber(predicted.deltaAverageBG), tint: predicted.deltaAverageBG <= 0 ? .green : .orange)
                }

                if let uncertainty = recommendation.predictedUncertainty {
                    HStack(spacing: 8) {
                        detailPill(title: "TIR ±", value: percentMagnitudeString(uncertainty.tirStd), tint: .secondary)
                        detailPill(title: "% Low ±", value: percentMagnitudeString(uncertainty.percentLowStd), tint: .secondary)
                        detailPill(title: "% High ±", value: percentMagnitudeString(uncertainty.percentHighStd), tint: .secondary)
                    }
                }
            } else {
                Text("Expected better cost profile")
                    .font(.system(.largeTitle, design: .rounded).weight(.bold))

                Text("This recommendation currently includes a predicted CVaR/cost improvement, but not an explicit TIR / % low / % high forecast.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                detailPill(title: "Predicted cost", value: signedPercentish(predictedImprovementPercent), tint: predictedImprovementPercent >= 0 ? .green : .orange)
            }
        }
        .cardStyle()
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Predicted tradeoff details for this recommendation.")
    }

    @ViewBuilder
    private var recommendationContextCard: some View {
        if let contextLine = recommendationContextLine {
            HStack(spacing: 10) {
                Image(systemName: "sparkles.rectangle.stack.fill")
                    .foregroundStyle(accent)
                Text(contextLine)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.primary)
                Spacer()
            }
            .cardStyle()
            .accessibilityElement(children: .combine)
            .accessibilityLabel(contextLine)
        }
    }

    private var confidenceCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Confidence")
                    .font(.headline)
                Spacer()
                Text("\(confidencePercent)%")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(confidenceColor.opacity(0.16), in: Capsule())
                    .foregroundStyle(confidenceColor)
            }

            Text("Composite κ·ρ·η score")
                .font(.subheadline.weight(.medium))

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(confidenceColor.gradient)
                        .frame(width: proxy.size.width * max(0, min(1, recommendation.confidence)))
                }
            }
            .frame(height: 12)

            HStack {
                detailPill(title: "Effect size", value: "\(effectSizePercent)%")
                detailPill(title: "CVaR", value: recommendation.cvarValue.formatted(.number.precision(.fractionLength(2))))
            }

            if let breakdown = recommendation.confidenceBreakdown {
                HStack(spacing: 8) {
                    detailPill(title: "Familiarity", value: percentMagnitudeString(breakdown.familiarity), tint: .secondary)
                    detailPill(title: "Concordance", value: percentMagnitudeString(breakdown.concordance), tint: .secondary)
                    detailPill(title: "Calibration", value: percentMagnitudeString(breakdown.calibration), tint: .secondary)
                }
            }
        }
        .cardStyle()
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Confidence \(confidencePercent) percent. Effect size \(effectSizePercent) percent.")
    }

    private func signedNumber(_ value: Double) -> String {
        let rounded = Int(value.rounded())
        if rounded > 0 { return "+\(rounded)" }
        return "\(rounded)"
    }

    private func signedPercentString(_ value: Double) -> String {
        let percent = Int((value * 100).rounded())
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }

    private func signedPercentish(_ value: Int) -> String {
        value > 0 ? "+\(value)" : "\(value)"
    }

    private func percentMagnitudeString(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private var changesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What changes")
                .font(.headline)

            if usesSegmentSummaries {
                ForEach(changedSegmentSummaries) { summary in
                    segmentSummaryRow(summary)
                }
            } else {
                changeRow(title: "Insulin sensitivity", delta: recommendation.action.deltas["isf_delta"])
                changeRow(title: "Carb ratio", delta: recommendation.action.deltas["cr_delta"])
                changeRow(title: "Basal", delta: recommendation.action.deltas["basal_delta"])
            }
        }
        .cardStyle()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(usesSegmentSummaries ? "What changes. Segment specific adjustments." : "What changes. Insulin sensitivity, carb ratio, and basal deltas.")
    }

    private var recommendationContextLine: String? {
        guard let detectedRegime = recommendation.detectedRegime else { return nil }
        let regime = formatRegimeLabel(detectedRegime)
        switch recommendation.recommendationScope {
        case "create_new":
            return "Detected pattern: \(regime). Suggested: create a \(regime.lowercased()) profile."
        case "patch_existing":
            return "Detected pattern: \(regime). Suggested: update an existing \(regime.lowercased()) profile."
        default:
            return "Detected pattern: \(regime). Suggested: update your current profile."
        }
    }

    @ViewBuilder
    private var structurePreviewCard: some View {
        if usesStructureSummaries {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Structure preview", systemImage: "square.split.2x1")
                        .font(.headline)
                        .foregroundStyle(.orange)
                    Spacer()
                    Text("Level \(recommendation.actionLevel)")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.orange.opacity(0.14), in: Capsule())
                        .foregroundStyle(.orange)
                }

                ForEach(Array(recommendation.structureSummaries.enumerated()), id: \.offset) { _, summary in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .padding(.top, 2)
                        Text(summary)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                Text("Time-block edits always require an extra confirmation before apply.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .cardStyle()
        }
    }

    private var burnoutCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Burnout risk")
                    .font(.headline)
                Spacer()
                Text(burnoutLabel)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(burnoutColor)
            }

            if let burnout = recommendation.burnoutAttribution {
                Text("Upper confidence bound is \((burnout.upperCI * 100).formatted(.number.precision(.fractionLength(1))))% over \(burnout.horizon) days.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text("Attributable delta \((burnout.deltaHat * 100).formatted(.number.precision(.fractionLength(1))))%, treated \((burnout.pTreated * 100).formatted(.number.precision(.fractionLength(1))))%, baseline \((burnout.pBaseline * 100).formatted(.number.precision(.fractionLength(1))))%.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("No burnout attribution payload was returned for this recommendation.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .cardStyle()
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Burnout risk is \(burnoutLabel).")
    }

    @ViewBuilder
    private var shadowContextCard: some View {
        if let status {
            if status.graduated {
                ChameliaStatusSummaryCard(status: status, accent: accent)
            } else {
                ShadowProgressView(status: status)
            }
        }
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            Button {
                guard canApplyInApp else { return }
                if requiresExtraConfirmation {
                    showLevel2Confirmation = true
                } else {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    onApply()
                }
            } label: {
                Label(applyLabel, systemImage: applyIcon)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(RecommendationActionButtonStyle(fill: accent, foreground: .white))
            .disabled(!canApplyInApp)
            .opacity(canApplyInApp ? 1 : 0.6)
            .accessibilityLabel(canApplyInApp ? "Apply recommendation" : "Preview only, cannot apply in app")

            Button {
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
                onSkip()
            } label: {
                Label("Skip", systemImage: "arrowshape.turn.up.right")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(RecommendationActionButtonStyle(fill: Color.primary.opacity(0.08), foreground: .primary))
            .accessibilityLabel("Skip recommendation")
        }
    }

    private var applyLabel: String {
        switch recommendation.actionLevel {
        case 2:
            return "Review Before Apply"
        case 3...:
            return "Preview Only"
        default:
            return "Apply"
        }
    }

    private var applyIcon: String {
        switch recommendation.actionLevel {
        case 2:
            return "checkmark.shield.fill"
        case 3...:
            return "eye.fill"
        default:
            return "checkmark.circle.fill"
        }
    }

    @ViewBuilder
    private func changeRow(title: String, delta: Double?) -> some View {
        HStack {
            Text(title)
                .font(.subheadline)
            Spacer()
            Text(formattedDelta(delta))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(deltaColor(delta))
        }
        .accessibilityLabel("\(title), \(formattedDelta(delta))")
    }

    private func segmentSummaryRow(_ summary: RecommendationSegmentSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(summary.label)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(summary.segmentId)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                segmentFieldPill(title: "ISF", value: summary.isf)
                segmentFieldPill(title: "CR", value: summary.cr)
                segmentFieldPill(title: "Basal", value: summary.basal)
            }

            settingsDiffRow(for: summary)
        }
        .padding(14)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    @ViewBuilder
    private func settingsDiffRow(for summary: RecommendationSegmentSummary) -> some View {
        let hourRange = currentHourRanges?.first { $0.timeLabel == summary.label }
        let deltas    = recommendation.action.segmentDeltas.first { $0.segmentId == summary.segmentId }
        if let hr = hourRange, let d = deltas {
            HStack(spacing: 10) {
                if !summary.isf.isVisuallyUnchanged {
                    diffPill(
                        title: "ISF",
                        before: hr.insulinSensitivity,
                        after:  hr.insulinSensitivity * (1 + d.isfDelta),
                        unit:   "mg/dL/U"
                    )
                }
                if !summary.cr.isVisuallyUnchanged {
                    diffPill(
                        title: "CR",
                        before: hr.carbRatio,
                        after:  hr.carbRatio * (1 + d.crDelta),
                        unit:   "g/U"
                    )
                }
                if !summary.basal.isVisuallyUnchanged {
                    diffPill(
                        title: "Basal",
                        before: hr.basalRate,
                        after:  hr.basalRate * (1 + d.basalDelta),
                        unit:   "U/hr"
                    )
                }
            }
        }
    }

    private func diffPill(title: String, before: Double, after: Double, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Text(formatTherapyValue(before, unit: unit))
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Image(systemName: "arrow.right")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(formatTherapyValue(after, unit: unit))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
            }
            Text(unit)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func formatTherapyValue(_ value: Double, unit: String) -> String {
        // Use 2 decimal places for basal (small numbers), 0 for ISF/CR (larger integers in practice)
        let decimals = unit == "U/hr" ? 2 : 0
        if decimals == 0 {
            return String(Int(value.rounded()))
        }
        return String(format: "%.2f", value)
    }

    private func segmentFieldPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(value.isVisuallyUnchanged ? .secondary : .primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func detailPill(title: String, value: String, tint: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func formattedDelta(_ delta: Double?) -> String {
        guard let delta else { return "Unchanged" }
        let percent = Int((delta * 100).rounded())
        if percent == 0 { return "Unchanged" }
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }

    private func formatRegimeLabel(_ raw: String) -> String {
        raw
            .split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    private func deltaColor(_ delta: Double?) -> Color {
        guard let delta else { return .secondary }
        if delta == 0 { return .secondary }
        return delta > 0 ? accent : .orange
    }

    private var confidenceColor: Color {
        switch recommendation.confidence {
        case ..<0.45: return .red
        case ..<0.7: return .orange
        default: return .green
        }
    }

    private var burnoutLabel: String {
        guard let upperCI = recommendation.burnoutAttribution?.upperCI else { return "Unavailable" }
        switch upperCI {
        case ..<0.025: return "Low"
        case ..<0.05: return "Watch"
        default: return "High"
        }
    }

    private var burnoutColor: Color {
        switch burnoutLabel {
        case "Low": return .green
        case "Watch": return .orange
        case "High": return .red
        default: return .secondary
        }
    }
}

private extension String {
    var isVisuallyUnchanged: Bool {
        trimmingCharacters(in: .whitespacesAndNewlines).lowercased().contains("unchanged")
    }
}

private struct RecommendationActionButtonStyle: ButtonStyle {
    let fill: Color
    let foreground: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(fill.opacity(configuration.isPressed ? 0.82 : 1), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .foregroundStyle(foreground)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.88), value: configuration.isPressed)
    }
}

private extension View {
    func cardStyle() -> some View {
        padding(18)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
    }
}

#Preview {
    NavigationStack {
        RecommendationView(
            recommendation: RecommendationPackage(
                action: TherapyAction(kind: "therapy_adjustment", deltas: ["isf_delta": 0.05, "cr_delta": 0.0, "basal_delta": 0.0]),
                predictedImprovement: 0.08,
                confidence: 0.74,
                effectSize: 0.12,
                cvarValue: 0.18,
                burnoutAttribution: BurnoutAttribution(deltaHat: 0.01, pTreated: 0.04, pBaseline: 0.03, upperCI: 0.03, horizon: 30)
            ),
            recId: 42,
            status: GraduationStatus(graduated: true, nDays: 24, winRate: 0.68, safetyViolations: 0, consecutiveDays: 8),
            onApply: {},
            onSkip: {}
        )
        .environmentObject(ThemeManager())
    }
}
