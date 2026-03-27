import SwiftUI
import FirebaseAuth

extension Notification.Name {
    static let requestChameliaQuestionnaireOnboarding = Notification.Name("requestChameliaQuestionnaireOnboarding")
}

enum ChameliaQuestionnaireStore {
    static let completionKey = "ChameliaOnboardingComplete"
    private static let answersKey = "ChameliaQuestionnaireAnswers"
    private static let draftKey = "ChameliaQuestionnairePreferenceDraft"

    private static func key(_ base: String, userId: String? = Auth.auth().currentUser?.uid) -> String? {
        guard let userId, !userId.isEmpty else { return nil }
        return "\(base).\(userId)"
    }

    static func isCompleted(userId: String? = Auth.auth().currentUser?.uid) -> Bool {
        guard let key = key(completionKey, userId: userId) else { return false }
        return UserDefaults.standard.bool(forKey: key)
    }

    static func setCompleted(_ value: Bool, userId: String? = Auth.auth().currentUser?.uid) {
        guard let key = key(completionKey, userId: userId) else { return }
        UserDefaults.standard.set(value, forKey: key)
    }

    static func resetCompletion(userId: String? = Auth.auth().currentUser?.uid) {
        guard let key = key(completionKey, userId: userId) else { return }
        UserDefaults.standard.set(false, forKey: key)
    }

    static func loadAnswers(userId: String? = Auth.auth().currentUser?.uid) -> QuestionnaireAnswers {
        guard let key = key(answersKey, userId: userId),
              let data = UserDefaults.standard.data(forKey: key),
              let answers = try? JSONDecoder().decode(QuestionnaireAnswers.self, from: data) else {
            return QuestionnaireAnswers()
        }
        return answers
    }

    static func saveAnswers(_ answers: QuestionnaireAnswers, userId: String? = Auth.auth().currentUser?.uid) {
        guard let key = key(answersKey, userId: userId) else { return }
        guard let data = try? JSONEncoder().encode(answers) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }

    static func loadPreferenceDraft(userId: String? = Auth.auth().currentUser?.uid) -> QuestionnairePreferenceDraft {
        guard let key = key(draftKey, userId: userId),
              let data = UserDefaults.standard.data(forKey: key),
              let draft = try? JSONDecoder().decode(QuestionnairePreferenceDraft.self, from: data) else {
            return QuestionnairePreferenceDraft()
        }
        return draft
    }

    static func savePreferenceDraft(_ draft: QuestionnairePreferenceDraft, userId: String? = Auth.auth().currentUser?.uid) {
        guard let key = key(draftKey, userId: userId) else { return }
        guard let data = try? JSONEncoder().encode(draft) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}

struct QuestionnaireOnboardingView: View {
    @Binding var isPresented: Bool
    var onCompleted: ((String) -> Void)? = nil

    @EnvironmentObject private var themeManager: ThemeManager

    @State private var answers = QuestionnaireAnswers()
    @State private var preferenceDraft = QuestionnairePreferenceDraft()
    @State private var stepIndex = 0
    @State private var isSubmitting = false
    @State private var errorMessage: String?
    @State private var hasLoadedDraft = false
    @State private var bedtimeSelection = QuestionnaireOnboardingView.defaultBedtimeDate(for: nil)

    private let screenCount = 9

    var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme)
                .ignoresSafeArea()

            VStack(spacing: 18) {
                header
                content
                footer
            }
            .padding(18)
        }
        .task {
            guard !hasLoadedDraft else { return }
            let userId = Auth.auth().currentUser?.uid
            answers = ChameliaQuestionnaireStore.loadAnswers(userId: userId)
            preferenceDraft = ChameliaQuestionnaireStore.loadPreferenceDraft(userId: userId)
            bedtimeSelection = Self.defaultBedtimeDate(for: answers.bedtimeCategory)
            hasLoadedDraft = true
        }
        .interactiveDismissDisabled(isSubmitting)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(stepIndex == 0 ? "Meet Chamelia" : "Step \(stepIndex + 1) of \(screenCount)")
                    .font(.headline)
                Spacer()
                Text(screenTitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                ForEach(0..<screenCount, id: \.self) { index in
                    Capsule()
                        .fill(index <= stepIndex ? themeManager.theme.accent : Color.primary.opacity(0.12))
                        .frame(height: 8)
                }
            }
        }
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                switch stepIndex {
                case 0: welcomeScreen
                case 1: sleepScreen
                case 2: exerciseScreen
                case 3: eatingPartOneScreen
                case 4: eatingPartTwoScreen
                case 5: stressScreen
                case 6: menstrualScreen
                case 7: aboutYouScreen
                default: confirmationScreen
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(20)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
            .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity), removal: .move(edge: .leading).combined(with: .opacity)))
            .animation(.spring(response: 0.35, dampingFraction: 0.88), value: stepIndex)
        }
    }

    private var footer: some View {
        VStack(spacing: 12) {
            if stepIndex < screenCount - 1 {
                Button("Skip this screen") {
                    if stepIndex == 0 {
                        goForward()
                    } else {
                        goForward()
                    }
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Skip this screen")
            }

            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 12) {
                Button(stepIndex == 0 ? "Not now" : "Back") {
                    if stepIndex == 0 {
                        isPresented = false
                    } else {
                        stepIndex -= 1
                    }
                }
                .buttonStyle(QuestionnaireSecondaryButtonStyle())

                Button {
                    if stepIndex == screenCount - 1 {
                        submit()
                    } else {
                        goForward()
                    }
                } label: {
                    HStack {
                        if isSubmitting {
                            ProgressView().tint(.white)
                        }
                        Text(stepIndex == screenCount - 1 ? "Start Learning" : "Next")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(QuestionnairePrimaryButtonStyle(fill: themeManager.theme.accent))
                .disabled(isSubmitting)
            }
        }
    }

    private var welcomeScreen: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Meet Chamelia")
                .font(.system(.largeTitle, design: .rounded).weight(.bold))
            Text("Chamelia learns your patterns and suggests insulin pump adjustments to help improve your time in range. It watches silently for at least 21 days before making any recommendations — it earns the right to help.")
                .font(.body)
                .foregroundStyle(.secondary)
            Text("This questionnaire is optional. It only gives Chamelia a better starting prior before your real CGM and HealthKit data take over.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var sleepScreen: some View {
        questionScreen(
            title: "Tell us about your sleep",
            subtitle: "These answers help seed Chamelia's early-day recovery and regularity assumptions."
        ) {
            bedtimePicker
            singleChoice("How many hours do you usually sleep?", selection: $answers.sleepHours, options: [
                (.under6, "Less than 6"), (.sixSeven, "6–7 hours"), (.sevenEight, "7–8 hours"), (.over8, "8+ hours")
            ])
            singleChoice("How consistent is your sleep schedule?", selection: $answers.sleepConsistency, options: [
                (.very, "Very consistent (±30 min)"), (.fairly, "Fairly consistent (±1 hour)"), (.variable, "Pretty variable (±2 hours)"), (.irregular, "Completely irregular")
            ])
            singleChoice("Do you feel rested when you wake up?", selection: $answers.restedFeeling, options: [
                (.usually, "Almost always"), (.sometimes, "Sometimes"), (.rarely, "Rarely")
            ])
        }
    }

    private var exerciseScreen: some View {
        questionScreen(
            title: "Tell us about your activity",
            subtitle: "We'll use the objective parts here and let HealthKit fill in the rest over time."
        ) {
            singleChoice("How often do you exercise?", selection: $answers.exerciseFreq, options: [
                (.never, "Never"), (.lightWeek, "1–2x per week"), (.modWeek, "3–5x per week"), (.daily, "Daily or more")
            ])

            if answers.exerciseFreq != .never, answers.exerciseFreq != nil {
                singleChoice("What kind of exercise?", selection: $answers.exerciseType, options: [
                    (.cardio, "Cardio"), (.strength, "Strength training"), (.mixed, "Mixed"), (.light, "Light movement")
                ])
                Text("We skip subjective effort and fitness ratings here. HealthKit signals like resting heart rate, body mass, and activity will be more trustworthy.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var eatingPartOneScreen: some View {
        questionScreen(
            title: "Tell us about your eating habits",
            subtitle: "Part 1 of 2"
        ) {
            singleChoice("What time do you usually eat your first meal?", selection: $answers.firstMealTime, options: [
                (.early, "Before 7am"), (.normal, "7am–9am"), (.late, "9am–11am"), (.skip, "After 11am / skip breakfast")
            ])
            singleChoice("Do you skip breakfast often?", selection: $answers.breakfastSkip, options: [
                (.never, "Almost never"), (.sometimes, "Sometimes"), (.often, "Often"), (.always, "Almost always")
            ])
            singleChoice("Do you skip lunch often?", selection: $answers.lunchSkip, options: [
                (.never, "Almost never"), (.sometimes, "Sometimes"), (.often, "Often"), (.always, "Almost always")
            ])
            singleChoice("How many times a day do you eat?", selection: $answers.mealFrequency, options: [
                (.oneTwo, "1–2"), (.three, "3"), (.fourFive, "4–5"), (.sixPlus, "6 or more")
            ])
        }
    }

    private var eatingPartTwoScreen: some View {
        questionScreen(
            title: "Tell us about your eating habits",
            subtitle: "Part 2 of 2"
        ) {
            singleChoice("How consistent are your meal times?", selection: $answers.mealConsistency, options: [
                (.very, "Very consistent"), (.fairly, "Fairly consistent"), (.variable, "Pretty variable"), (.chaotic, "Very irregular / chaotic")
            ])
            singleChoice("How would you describe your typical portion sizes?", selection: $answers.portionSize, options: [
                (.small, "Small"), (.average, "About average"), (.generous, "Generous"), (.large, "Large")
            ])
            singleChoice("How would you describe your diet?", selection: $answers.dietType, options: [
                (.lowCarb, "Low carb / keto"), (.moderate, "Moderate carb"), (.highCarb, "High carb"), (.veryVariable, "Very variable")
            ])
            singleChoice("What time do you usually eat your last meal or snack?", selection: $answers.lastMealTime, options: [
                (.early, "Before 7pm"), (.normal, "7pm–9pm"), (.late, "After 9pm"), (.veryLate, "Very late / midnight snacks")
            ])
        }
    }

    private var stressScreen: some View {
        questionScreen(
            title: "A few questions about stress and mood",
            subtitle: "This helps Chamelia start with a better guess for non-physiology variability."
        ) {
            singleChoice("How stressed are you most days?", selection: $answers.stressLevel, options: [
                (.rarely, "Rarely stressed"), (.sometimes, "Sometimes stressed"), (.often, "Often stressed"), (.always, "Almost always stressed")
            ])
            singleChoice("When you're stressed, does your blood sugar tend to go up?", selection: $answers.stressBgEffect, options: [
                (.noticeably, "Yes, noticeably"), (.little, "A little"), (.notReally, "Not really"), (.unsure, "Not sure")
            ])
            singleChoice("How much does your mood vary from day to day?", selection: $answers.moodVariability, options: [
                (.stable, "Very stable"), (.some, "Some variation"), (.variable, "Quite variable"), (.very, "Very variable")
            ])
            singleChoice("Do you work a regular schedule?", selection: $answers.scheduleType, options: [
                (.regular, "Regular 9–5 type"), (.shift, "Shift work"), (.variable, "Very variable / freelance")
            ])
        }
    }

    private var menstrualScreen: some View {
        questionScreen(
            title: "Menstrual cycle & blood sugar",
            subtitle: "Skip this screen if it doesn't apply."
        ) {
            singleChoice("Do you have a menstrual cycle?", selection: $answers.cyclePresence, options: [
                (.regular, "Yes, fairly regular"), (.irregular, "Yes, but irregular"), (.no, "No / prefer not to say")
            ])

            if answers.cyclePresence == .regular || answers.cyclePresence == .irregular {
                singleChoice("Does your blood sugar tend to change around your period?", selection: $answers.cycleBgEffect, options: [
                    (.noticeably, "Yes, noticeably"), (.little, "A little"), (.notReally, "Not really"), (.unsure, "Not sure")
                ])
                singleChoice("Do you eat more or feel hungrier in the week before your period?", selection: $answers.cycleHunger, options: [
                    (.noticeably, "Yes, noticeably"), (.little, "A little"), (.notReally, "Not really")
                ])
                singleChoice("Does your mood noticeably change before your period?", selection: $answers.cycleMood, options: [
                    (.noticeably, "Yes, noticeably"), (.little, "A little"), (.notReally, "Not really")
                ])
            }
        }
    }

    private var aboutYouScreen: some View {
        questionScreen(
            title: "A few last things to help Chamelia start well",
            subtitle: "This combines the trust/cadence preferences with the prior-building questions."
        ) {
            singleChoice("Have you ever been told your insulin sensitivity is unusually high or low?", selection: $answers.insulinSensitivity, options: [
                (.high, "High sensitivity"), (.normal, "Normal / average"), (.low, "Low sensitivity"), (.unsure, "Not sure")
            ])
            singleChoice("When you eat carbs, do they spike your blood sugar a lot?", selection: $answers.carbSpike, options: [
                (.lot, "Yes, a lot more than expected"), (.average, "About average"), (.notMuch, "Not much / carb tolerant"), (.unsure, "Not sure")
            ])
            singleChoice("How willing are you to make bigger changes to your pump settings?", selection: $answers.aggressiveness, options: [
                (.veryCautious, "Very cautious"), (.moderate, "Moderate"), (.willing, "Willing"), (.veryWilling, "Very willing")
            ])
            singleChoice("How worried are you about low blood sugar?", selection: $preferenceDraft.hypoglycemiaFear, options: [
                (.very, "Very"), (.somewhat, "Somewhat"), (.notMuch, "Not much")
            ])
            singleChoice("How often do you want recommendations?", selection: $preferenceDraft.recommendationCadence, options: [
                (.daily, "Daily"), (.weekly, "Weekly"), (.significant, "Only significant changes")
            ])
            singleChoice("How precisely do you implement pump setting changes when you make them?", selection: $answers.complianceLevel, options: [
                (.exact, "Exactly as set"), (.close, "Pretty closely"), (.rough, "Roughly"), (.forget, "Sometimes forget / approximate")
            ])
            singleChoice("Be honest — how often would you check recommendations?", selection: $answers.checkFrequency, options: [
                (.everyDay, "Every day without fail"), (.mostDays, "Most days"), (.sometimes, "Whenever I remember"), (.rarely, "Probably rarely")
            ])
            singleChoice("How much do you trust an AI system to help with your diabetes management?", selection: $answers.trustLevel, options: [
                (.skeptical, "Very skeptical"), (.cautious, "Cautiously open"), (.trusting, "Fairly trusting"), (.veryTrusting, "Very trusting")
            ])
        }
    }

    private var confirmationScreen: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("You're all set")
                .font(.system(.largeTitle, design: .rounded).weight(.bold))
            Text("Chamelia will use your answers as a starting point. Your real data from the CGM will refine this over time — within a few weeks the questionnaire barely matters.")
                .font(.body)
                .foregroundStyle(.secondary)
            summaryRow("Sleep", value: sleepSummary)
            summaryRow("Activity", value: activitySummary)
            summaryRow("Meals", value: mealSummary)
            summaryRow("Stress", value: stressSummary)
            summaryRow("Preferences", value: preferenceSummary)
        }
    }

    private var screenTitle: String {
        switch stepIndex {
        case 0: return "Welcome"
        case 1: return "Sleep"
        case 2: return "Exercise"
        case 3: return "Eating 1"
        case 4: return "Eating 2"
        case 5: return "Stress & Mood"
        case 6: return "Menstrual"
        case 7: return "About You"
        default: return "Confirm"
        }
    }

    private var sleepSummary: String {
        if answers.bedtimeCategory != nil {
            return Self.bedtimeFormatter.string(from: bedtimeSelection)
        }
        return "Using defaults"
    }

    private var activitySummary: String {
        if let freq = answers.exerciseFreq {
            return freq.rawValue.replacingOccurrences(of: "_", with: " ")
        }
        return "Using defaults"
    }

    private var mealSummary: String {
        if let diet = answers.dietType {
            return diet.rawValue.replacingOccurrences(of: "_", with: " ")
        }
        return "Using defaults"
    }

    private var stressSummary: String {
        if let stress = answers.stressLevel {
            return stress.rawValue.replacingOccurrences(of: "_", with: " ")
        }
        return "Using defaults"
    }

    private var preferenceSummary: String {
        [
            answers.aggressiveness?.rawValue.replacingOccurrences(of: "_", with: " "),
            preferenceDraft.hypoglycemiaFear?.rawValue.replacingOccurrences(of: "_", with: " "),
            preferenceDraft.recommendationCadence?.rawValue.replacingOccurrences(of: "_", with: " ")
        ]
        .compactMap { $0 }
        .joined(separator: " · ")
        .isEmpty ? "Using defaults" : [
            answers.aggressiveness?.rawValue.replacingOccurrences(of: "_", with: " "),
            preferenceDraft.hypoglycemiaFear?.rawValue.replacingOccurrences(of: "_", with: " "),
            preferenceDraft.recommendationCadence?.rawValue.replacingOccurrences(of: "_", with: " ")
        ]
        .compactMap { $0 }
        .joined(separator: " · ")
    }

    private func goForward() {
        errorMessage = nil
        stepIndex = min(screenCount - 1, stepIndex + 1)
    }

    private func submit() {
        errorMessage = nil
        isSubmitting = true
        let answersSnapshot = answers
        let preferencesSnapshot = preferenceDraft

        Task {
            do {
                let authUser = try AuthManager.shared.getAuthenticatedUser()
                let result = QuestionnaireToPriors.compute(answersSnapshot)
                let physicalPriorsJSON: [String: [Double]]
                if answersSnapshot.hasAnyAnswer {
                    physicalPriorsJSON = result.physicalPriors.mapValues { [$0.mean, $0.std] }
                } else {
                    physicalPriorsJSON = [:]
                }
                let prefs = ChameliaPreferences(
                    aggressiveness: result.aggressiveness,
                    hypoglycemiaFear: QuestionnaireToPriors.hypoglycemiaFearValue(preferencesSnapshot.hypoglycemiaFear),
                    burdenSensitivity: QuestionnaireToPriors.burdenSensitivityValue(preferencesSnapshot.recommendationCadence),
                    persona: "questionnaire_derived",
                    physicalPriors: physicalPriorsJSON
                )
                try await ChameliaEngine.shared.initialize(patientId: authUser.uid, preferences: prefs)
                ChameliaQuestionnaireStore.saveAnswers(answersSnapshot, userId: authUser.uid)
                ChameliaQuestionnaireStore.savePreferenceDraft(preferencesSnapshot, userId: authUser.uid)
                ChameliaQuestionnaireStore.setCompleted(true, userId: authUser.uid)
                await MainActor.run {
                    onCompleted?(authUser.uid)
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

    @ViewBuilder
    private func questionScreen<Content: View>(
        title: String,
        subtitle: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title3.weight(.semibold))
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func summaryRow(_ title: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .frame(width: 94, alignment: .leading)
            Text(value)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var bedtimePicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("What time do you usually go to bed?")
                .font(.headline)
            DatePicker(
                "Usual bedtime",
                selection: $bedtimeSelection,
                displayedComponents: .hourAndMinute
            )
            .datePickerStyle(.wheel)
            .labelsHidden()
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color.primary.opacity(0.05))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
            .onChange(of: bedtimeSelection) { newValue in
                answers.bedtimeCategory = Self.bedtimeCategory(for: newValue)
            }

            Text("We'll use this as a rough starting prior, then hand off to real sleep data once it exists.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private func singleChoice<T: Hashable & Codable>(
        _ title: String,
        selection: Binding<T?>,
        options: [(T, String)]
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
            ForEach(Array(options.enumerated()), id: \.offset) { _, option in
                let isSelected = selection.wrappedValue == option.0
                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    selection.wrappedValue = option.0
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isSelected ? themeManager.theme.accent : .secondary)
                        Text(option.1)
                            .font(.subheadline)
                            .multilineTextAlignment(.leading)
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
                .accessibilityLabel("\(title). \(option.1)\(isSelected ? ", selected" : "")")
            }
        }
    }

    private static let bedtimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }()

    private static func bedtimeCategory(for date: Date) -> BedtimeCategory {
        let hour = Calendar.current.component(.hour, from: date)
        switch hour {
        case ..<22:
            return .early
        case 22..<24:
            return .normal
        case 0..<2:
            return .late
        default:
            return .veryLate
        }
    }

    private static func defaultBedtimeDate(for category: BedtimeCategory?) -> Date {
        let hour: Int
        switch category {
        case .early:
            hour = 21
        case .normal:
            hour = 23
        case .late:
            hour = 1
        case .veryLate:
            hour = 3
        case nil:
            hour = 23
        }
        return Calendar.current.date(bySettingHour: hour, minute: 0, second: 0, of: Date()) ?? Date()
    }
}

private struct QuestionnairePrimaryButtonStyle: ButtonStyle {
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

private struct QuestionnaireSecondaryButtonStyle: ButtonStyle {
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
    QuestionnaireOnboardingView(isPresented: .constant(true))
        .environmentObject(ThemeManager())
}
