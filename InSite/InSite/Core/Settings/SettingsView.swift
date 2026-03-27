//
//  SettingsView.swift
//  InSite
//
//  Created by Anand Parikh on 9/12/24.
//

import SwiftUI
import FirebaseAuth

struct SettingsView: View {
    @StateObject private var viewModel = SettingsViewModel()
    @Binding var showSignInView: Bool
    @State private var chameliaLevel2Enabled = false
    @ObservedObject private var nightscoutStore = NightscoutConnectionStore.shared
    @State private var showNightscoutSheet = false
    @State private var isValidatingNightscout = false
    @State private var nightscoutBanner: String?

    var body: some View {
        List {
            Section("Chamelia") {
                Button {
                    ChameliaQuestionnaireStore.resetCompletion(userId: Auth.auth().currentUser?.uid)
                    NotificationCenter.default.post(name: .requestChameliaQuestionnaireOnboarding, object: nil)
                } label: {
                    HStack {
                        Text("Chamelia Preferences")
                        Spacer()
                        Text(preferencePreview)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Toggle("Allow Chamelia to suggest new time blocks", isOn: $chameliaLevel2Enabled)
            }

            Section {
                Button {
                    showNightscoutSheet = true
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(nightscoutStore.state.isConnected ? "Nightscout Connected" : "Connect Nightscout")
                            Text(nightscoutSubtitle)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }
                }

                if let summary = nightscoutStore.state.summary {
                    LabeledContent("Recent TIR") {
                        Text(summary.calibration?.recentTIR.formatted(.percent.precision(.fractionLength(0))) ?? "Unavailable")
                    }
                    LabeledContent("% Low") {
                        Text(summary.calibration?.recentPercentLow.formatted(.percent.precision(.fractionLength(1))) ?? "Unavailable")
                    }
                    LabeledContent("% High") {
                        Text(summary.calibration?.recentPercentHigh.formatted(.percent.precision(.fractionLength(1))) ?? "Unavailable")
                    }
                    LabeledContent("Profile") {
                        Text(summary.latestProfileName ?? "No active profile")
                    }
                    LabeledContent("IOB / COB") {
                        let iob = summary.latestIOB.map { "\($0.formatted(.number.precision(.fractionLength(1)))) U" } ?? "–"
                        let cob = summary.latestCOB.map { "\($0.formatted(.number.precision(.fractionLength(0)))) g" } ?? "–"
                        Text("\(iob) / \(cob)")
                    }
                    if let lastValidatedAt = nightscoutStore.state.lastValidatedAt {
                        LabeledContent("Last validated") {
                            Text(lastValidatedAt.formatted(date: .abbreviated, time: .shortened))
                        }
                    }
                }

                if let message = nightscoutStore.state.lastErrorMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }

                if nightscoutStore.state.isConnected {
                    Button(isValidatingNightscout ? "Refreshing Nightscout…" : "Refresh Nightscout Snapshot") {
                        Task { await refreshNightscoutValidation() }
                    }
                    .disabled(isValidatingNightscout)

                    Button("Disconnect Nightscout", role: .destructive) {
                        nightscoutStore.disconnect()
                    }
                }
            } header: {
                Text("Nightscout")
            } footer: {
                Text("Nightscout can backfill recent CGM, profile, and treatment context so Chamelia can cold-start from real recent data instead of asking you to estimate TIR manually.")
            }

            Button("Log Out") {
                Task {
                    do {
                        try await viewModel.logOut()
                        showSignInView = true
                    } catch {
                        print(error)
                    }
                }
            }
            
            if viewModel.authProviders.contains(.email) {
                Section("Email Functions") {
                    Button("Reset Password") {
                        Task {
                            do {
                                try await viewModel.resetPassword()
                                print("Password reset")
                            } catch {
                                print(error)
                            }
                        }
                    }
                    Button("Update Password") {
                        Task {
                            do {
                                try await viewModel.updatePassword()
                                print("Password updated")
                            } catch {
                                print(error)
                            }
                        }
                    }
                    Button("Update Email") {
                        Task {
                            do {
                                try await viewModel.updatePassword()
                                print("Email updated")
                            } catch {
                                print(error)
                            }
                        }
                    }
                }
            }
        }
        .onAppear {
            viewModel.loadAuthProviders()
            chameliaLevel2Enabled = ChameliaSettingsStore.level2Enabled()
            nightscoutStore.refresh()
        }
        .navigationTitle("Settings")
        .onChange(of: chameliaLevel2Enabled) { newValue in
            ChameliaSettingsStore.setLevel2Enabled(newValue)
        }
        .sheet(isPresented: $showNightscoutSheet) {
            NightscoutConnectionSheet(
                initialState: nightscoutStore.state,
                isValidating: isValidatingNightscout,
                onSave: { baseURLString, authMode, credential in
                    Task {
                        isValidatingNightscout = true
                        defer { isValidatingNightscout = false }
                        do {
                            try await nightscoutStore.connect(baseURLString: baseURLString, authMode: authMode, credential: credential)
                            nightscoutBanner = "Nightscout connected and validated."
                            showNightscoutSheet = false
                        } catch {
                            nightscoutBanner = error.localizedDescription
                        }
                    }
                }
            )
        }
        .alert("Nightscout", isPresented: Binding(
            get: { nightscoutBanner != nil },
            set: { if !$0 { nightscoutBanner = nil } }
        )) {
            Button("OK", role: .cancel) { nightscoutBanner = nil }
        } message: {
            Text(nightscoutBanner ?? "")
        }
    }

    private var preferencePreview: String {
        let userId = Auth.auth().currentUser?.uid
        let answers = ChameliaQuestionnaireStore.loadAnswers(userId: userId)
        let draft = ChameliaQuestionnaireStore.loadPreferenceDraft(userId: userId)

        let parts = [
            answers.aggressiveness?.rawValue.replacingOccurrences(of: "_", with: " "),
            draft.hypoglycemiaFear?.rawValue.replacingOccurrences(of: "_", with: " "),
            draft.recommendationCadence?.rawValue.replacingOccurrences(of: "_", with: " ")
        ]
        .compactMap { $0 }

        return parts.isEmpty ? "Not set" : parts.joined(separator: " · ")
    }

    private var nightscoutSubtitle: String {
        if let summary = nightscoutStore.state.summary {
            let tir = summary.calibration?.recentTIR.formatted(.percent.precision(.fractionLength(0))) ?? "No TIR"
            return "\(summary.latestProfileName ?? "Validated") · \(tir) recent TIR"
        }
        if nightscoutStore.state.isConnected {
            return nightscoutStore.state.normalizedBaseURLString
        }
        return "Optional read-only sync for CGM, settings, and treatments"
    }

    private func refreshNightscoutValidation() async {
        isValidatingNightscout = true
        defer { isValidatingNightscout = false }
        do {
            try await nightscoutStore.validateConnection()
            nightscoutBanner = "Nightscout snapshot refreshed."
        } catch {
            nightscoutBanner = error.localizedDescription
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView(showSignInView: .constant(false))
    }
}

private struct NightscoutConnectionSheet: View {
    let initialState: NightscoutConnectionState
    let isValidating: Bool
    let onSave: (_ baseURLString: String, _ authMode: NightscoutAuthMode, _ credential: String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var baseURLString = ""
    @State private var authMode: NightscoutAuthMode = .accessToken
    @State private var credential = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Connection") {
                    TextField("https://your-site.herokuapp.com", text: $baseURLString)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()

                    Picker("Authentication", selection: $authMode) {
                        ForEach(NightscoutAuthMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }

                    SecureField(authMode == .accessToken ? "Access token" : "API secret", text: $credential)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("What InSite uses first") {
                    Text("Nightscout validation pulls status, recent CGM entries, recent treatments, latest profile, and latest device status. That lets InSite estimate recent TIR / % low / % high and reuse real pump context for cold-start calibration.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Nightscout")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(isValidating ? "Validating…" : "Save") {
                        onSave(baseURLString, authMode, credential)
                    }
                    .disabled(isValidating || baseURLString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || credential.isEmpty)
                }
            }
            .task {
                baseURLString = initialState.baseURLString
                authMode = initialState.authMode
            }
        }
    }
}
