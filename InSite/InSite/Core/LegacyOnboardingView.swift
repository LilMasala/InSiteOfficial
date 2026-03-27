import SwiftUI
import UIKit

extension Notification.Name {
    static let requestChameliaOnboarding = Notification.Name("requestChameliaOnboarding")
}

enum ChameliaOnboardingStore {
    private static let completedKey = "HasSeenChameliaOnboarding"

    static func isCompleted() -> Bool {
        UserDefaults.standard.bool(forKey: completedKey)
    }

    static func markCompleted(_ completed: Bool) {
        UserDefaults.standard.set(completed, forKey: completedKey)
    }

    static func savePreferences(_ preferences: ChameliaPreferences, for userId: String) {
        guard let data = try? JSONEncoder().encode(preferences) else { return }
        UserDefaults.standard.set(data, forKey: preferencesKey(for: userId))
    }

    static func loadPreferences(for userId: String) -> ChameliaPreferences? {
        guard let data = UserDefaults.standard.data(forKey: preferencesKey(for: userId)) else { return nil }
        return try? JSONDecoder().decode(ChameliaPreferences.self, from: data)
    }

    private static func preferencesKey(for userId: String) -> String {
        "ChameliaPreferences.\(userId)"
    }
}

struct OnboardingView: View {
    @Binding var isPresented: Bool

    @EnvironmentObject private var themeManager: ThemeManager

    @State private var stepIndex = 0
    @State private var aggressiveness = 0.5
    @State private var hypoglycemiaFear = 0.6
    @State private var burdenSensitivity = 0.6
    @State private var persona = "balanced"
    @State private var isSubmitting = false
    @State private var errorMessage: String?

    private let engine = ChameliaEngine.shared

    private let aggressivenessOptions: [(String, Double)] = [
        ("Conservative", 0.2),
        ("Balanced", 0.5),
        ("Proactive", 0.8)
    ]
    private let fearOptions: [(String, Double)] = [
        ("Very", 0.9),
        ("Somewhat", 0.6),
        ("Not much", 0.3)
    ]
    private let burdenOptions: [(String, Double)] = [
        ("Daily", 0.3),
        ("Weekly", 0.6),
        ("Only significant changes", 0.9)
    ]
    private let personaOptions: [(String, String)] = [
        ("Balanced default", "balanced"),
        ("Athlete", "athlete"),
        ("High stress", "high_stress"),
        ("Sensitive to lows", "hypo_cautious")
    ]

    var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme)
                .ignoresSafeArea()

            VStack(spacing: 18) {
                header
                progressBar
                card
                footer
            }
            .padding(18)
        }
        .interactiveDismissDisabled(isSubmitting)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Set up Chamelia")
                .font(.system(.largeTitle, design: .rounded).weight(.bold))
            Text("Tell InSite how assertive and how frequent recommendations should feel before Chamelia starts learning your pattern.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var progressBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Step \(stepIndex + 1) of 4")
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(stepTitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule().fill(themeManager.theme.accent.gradient)
                        .frame(width: proxy.size.width * (Double(stepIndex + 1) / 4.0))
                }
            }
            .frame(height: 12)
        }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 16) {
            Group {
                switch stepIndex {
                case 0:
                    selectionStep(
                        title: "How aggressive should recommendations be?",
                        subtitle: "Choose the starting balance between caution and momentum.",
                        options: aggressivenessOptions.map { ($0.0, $0.0 == selectedAggressivenessLabel) },
                        action: { label in aggressiveness = aggressivenessOptions.first(where: { $0.0 == label })?.1 ?? aggressiveness }
                    )
                case 1:
                    selectionStep(
                        title: "How worried are you about low blood sugar?",
                        subtitle: "This tunes how heavily Chamelia prioritizes hypo avoidance.",
                        options: fearOptions.map { ($0.0, $0.0 == selectedFearLabel) },
                        action: { label in hypoglycemiaFear = fearOptions.first(where: { $0.0 == label })?.1 ?? hypoglycemiaFear }
                    )
                case 2:
                    selectionStep(
                        title: "How often do you want recommendations?",
                        subtitle: "Higher burden sensitivity means fewer, more meaningful nudges.",
                        options: burdenOptions.map { ($0.0, $0.0 == selectedBurdenLabel) },
                        action: { label in burdenSensitivity = burdenOptions.first(where: { $0.0 == label })?.1 ?? burdenSensitivity }
                    )
                default:
                    selectionStep(
                        title: "Pick a starting persona",
                        subtitle: "You can refine this later once real data accumulates.",
                        options: personaOptions.map { ($0.0, $0.1 == persona) },
                        action: { label in persona = personaOptions.first(where: { $0.0 == label })?.1 ?? persona }
                    )
                }
            }
            .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity), removal: .move(edge: .leading).combined(with: .opacity)))

            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .accessibilityLabel("Onboarding error. \(errorMessage)")
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
        .animation(.spring(response: 0.35, dampingFraction: 0.88), value: stepIndex)
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Button(stepIndex == 0 ? "Not now" : "Back") {
                if stepIndex == 0 {
                    ChameliaOnboardingStore.markCompleted(false)
                    isPresented = false
                } else {
                    stepIndex -= 1
                }
            }
            .buttonStyle(OnboardingSecondaryButtonStyle())
            .accessibilityLabel(stepIndex == 0 ? "Dismiss onboarding" : "Go back")

            Button {
                if stepIndex == 3 {
                    submit()
                } else {
                    stepIndex += 1
                }
            } label: {
                HStack {
                    if isSubmitting {
                        ProgressView()
                            .tint(.white)
                    }
                    Text(stepIndex == 3 ? "Finish" : "Continue")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(OnboardingPrimaryButtonStyle(fill: themeManager.theme.accent))
            .disabled(isSubmitting)
            .accessibilityLabel(stepIndex == 3 ? "Finish onboarding" : "Continue to next step")
        }
    }

    private var stepTitle: String {
        switch stepIndex {
        case 0: return "Aggressiveness"
        case 1: return "Hypo caution"
        case 2: return "Cadence"
        default: return "Persona"
        }
    }

    private var selectedAggressivenessLabel: String {
        aggressivenessOptions.first(where: { $0.1 == aggressiveness })?.0 ?? "Balanced"
    }

    private var selectedFearLabel: String {
        fearOptions.first(where: { $0.1 == hypoglycemiaFear })?.0 ?? "Somewhat"
    }

    private var selectedBurdenLabel: String {
        burdenOptions.first(where: { $0.1 == burdenSensitivity })?.0 ?? "Weekly"
    }

    @ViewBuilder
    private func selectionStep(
        title: String,
        subtitle: String,
        options: [(String, Bool)],
        action: @escaping (String) -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title)
                .font(.title3.weight(.semibold))
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            ForEach(options, id: \.0) { option, isSelected in
                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    action(option)
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isSelected ? themeManager.theme.accent : .secondary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(option)
                                .font(.headline)
                            Text(detailText(for: option))
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(isSelected ? themeManager.theme.accent.opacity(0.12) : Color.primary.opacity(0.05))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(isSelected ? themeManager.theme.accent.opacity(0.45) : Color.primary.opacity(0.06), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(option)\(isSelected ? ", selected" : "")")
            }
        }
    }

    private func detailText(for option: String) -> String {
        switch option {
        case "Conservative": return "Smaller moves with more caution."
        case "Balanced": return "Moderate changes when confidence is decent."
        case "Proactive": return "Faster adaptation when evidence supports it."
        case "Very": return "Strongly prioritize avoiding lows."
        case "Somewhat": return "Keep hypo avoidance important but balanced."
        case "Not much": return "Accept more assertive tradeoffs."
        case "Daily": return "More frequent nudges with lower burden weight."
        case "Weekly": return "A middle ground between responsiveness and quiet."
        case "Only significant changes": return "Prefer fewer recommendations unless evidence is strong."
        case "Balanced default": return "General-purpose starting point for most users."
        case "Athlete": return "Useful if exercise swings dominate your week."
        case "High stress": return "Useful if stress and schedule variability hit glucose often."
        case "Sensitive to lows": return "Useful if lows dominate decision making."
        default: return ""
        }
    }

    private func submit() {
        errorMessage = nil
        isSubmitting = true

        Task {
            do {
                let authUser = try AuthManager.shared.getAuthenticatedUser()
                let preferences = ChameliaPreferences(
                    aggressiveness: aggressiveness,
                    hypoglycemiaFear: hypoglycemiaFear,
                    burdenSensitivity: burdenSensitivity,
                    persona: persona
                )
                try await engine.initialize(patientId: authUser.uid, preferences: preferences)
                ChameliaOnboardingStore.savePreferences(preferences, for: authUser.uid)
                ChameliaOnboardingStore.markCompleted(true)
                await MainActor.run {
                    isSubmitting = false
                    isPresented = false
                }
            } catch {
                await MainActor.run {
                    isSubmitting = false
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
}

private struct OnboardingPrimaryButtonStyle: ButtonStyle {
    let fill: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(fill.opacity(configuration.isPressed ? 0.82 : 1), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .foregroundStyle(.white)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.88), value: configuration.isPressed)
    }
}

private struct OnboardingSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Color.primary.opacity(configuration.isPressed ? 0.12 : 0.08), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .foregroundStyle(.primary)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.88), value: configuration.isPressed)
    }
}

#Preview {
    OnboardingView(isPresented: .constant(true))
        .environmentObject(ThemeManager())
}
