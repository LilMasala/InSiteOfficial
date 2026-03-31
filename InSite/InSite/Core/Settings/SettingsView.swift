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
    @ObservedObject private var telemetrySourceStore = TelemetrySourceStore.shared
    @ObservedObject private var nightscoutStore = NightscoutConnectionStore.shared
    @ObservedObject private var tandemStore = TandemConnectionStore.shared
    @State private var showNightscoutSheet = false
    @State private var showTandemSheet = false
    @State private var isValidatingNightscout = false
    @State private var isRefreshingTandemStatus = false
    @State private var isConnectingTandem = false
    @State private var nightscoutBanner: String?
    @State private var tandemBanner: String?

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

            Section("Telemetry Source") {
                Picker("Source", selection: telemetrySelectionBinding) {
                    ForEach(TelemetrySource.allCases) { source in
                        Text(source.title).tag(source)
                    }
                }

                Text(telemetrySourceStore.selectedSource.detail)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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

            Section {
                Button {
                    showTandemSheet = true
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(tandemPrimaryTitle)
                            Text(tandemSubtitle)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }
                }

                LabeledContent("Source") {
                    Text("Tandem direct")
                }
                LabeledContent("Status") {
                    Text(tandemStatusTitle)
                }
                LabeledContent("Region") {
                    Text(tandemStore.state.region.rawValue)
                }
                if let lastSuccessfulSync = tandemStore.state.lastSuccessfulSync {
                    LabeledContent("Last successful sync") {
                        Text(lastSuccessfulSync.formatted(date: .abbreviated, time: .shortened))
                    }
                }
                if let message = tandemVisibleErrorMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }

                Button(isRefreshingTandemStatus ? "Refreshing Tandem Status…" : "Refresh Tandem Status") {
                    Task { await refreshTandemStatus() }
                }
                .disabled(isRefreshingTandemStatus || !tandemStore.state.shouldAttemptStatusRefresh)
            } header: {
                Text("Tandem")
            } footer: {
                Text("Tandem Direct uses a separate adapter service, triggers canonical Firestore imports, and keeps downstream therapy hydration source-agnostic.")
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
            telemetrySourceStore.refresh()
            nightscoutStore.refresh()
            tandemStore.refresh()
            Task { await refreshTandemStatusIfNeeded() }
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
        .sheet(isPresented: $showTandemSheet) {
            TandemConnectionSheet(
                initialState: tandemStore.state,
                isConnecting: isConnectingTandem,
                onSave: { adapterBaseURLString, email, password, region, pumpSerialNumber in
                    Task {
                        isConnectingTandem = true
                        defer { isConnectingTandem = false }
                        do {
                            try await tandemStore.connect(
                                adapterBaseURLString: adapterBaseURLString,
                                email: email,
                                password: password,
                                region: region,
                                pumpSerialNumber: pumpSerialNumber
                            )
                            tandemBanner = "Tandem connected."
                            showTandemSheet = false
                        } catch {
                            tandemBanner = error.localizedDescription
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
        .alert("Tandem", isPresented: Binding(
            get: { tandemBanner != nil },
            set: { if !$0 { tandemBanner = nil } }
        )) {
            Button("OK", role: .cancel) { tandemBanner = nil }
        } message: {
            Text(tandemBanner ?? "")
        }
    }

    private var preferencePreview: String {
        let userId = Auth.auth().currentUser?.uid
        let answers = ChameliaQuestionnaireStore.loadAnswers(userId: userId)
        let draft = ChameliaQuestionnaireStore.loadPreferenceDraft(userId: userId)

        let parts = [
            answers.aggressiveness?.rawValue.replacingOccurrences(of: "_", with: " "),
            draft.hypoglycemiaFear?.rawValue.replacingOccurrences(of: "_", with: " "),
            draft.recommendationCadence?.rawValue.replacingOccurrences(of: "_", with: " "),
            draft.minimumAcceptableISFChange.map { "ISF \($0.rawValue)" },
            draft.minimumAcceptableCRChange.map { "CR \($0.rawValue)" },
            draft.minimumAcceptableBasalChange.map { "Basal \($0.rawValue)" }
        ]
        .compactMap { $0 }

        return parts.isEmpty ? "Not set" : parts.joined(separator: " · ")
    }

    private var telemetrySelectionBinding: Binding<TelemetrySource> {
        Binding(
            get: { telemetrySourceStore.selectedSource },
            set: { telemetrySourceStore.setSelectedSource($0) }
        )
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

    private var tandemSubtitle: String {
        if !tandemStore.state.isAdapterConfigured {
            return "Not configured yet"
        }
        if !tandemStore.state.hasAttemptedConnection {
            return "Adapter configured. Enter Tandem credentials to connect."
        }
        switch tandemStore.state.status {
        case .connected:
            if let lastSuccessfulSync = tandemStore.state.lastSuccessfulSync {
                return "Tandem direct · synced \(lastSuccessfulSync.formatted(date: .abbreviated, time: .shortened))"
            }
            return "Tandem direct · ready to sync"
        case .needsReauth:
            return "Tandem direct · re-auth required"
        case .failed:
            return tandemStore.state.lastErrorMessage ?? "Tandem direct · connection failed"
        case .disconnected:
            return "Not connected"
        }
    }

    private var tandemPrimaryTitle: String {
        if tandemStore.state.isConnected {
            return "Tandem Connected"
        }
        if !tandemStore.state.isAdapterConfigured || !tandemStore.state.hasAttemptedConnection {
            return "Connect Tandem"
        }
        return "Connect Tandem"
    }

    private var tandemStatusTitle: String {
        if !tandemStore.state.isAdapterConfigured {
            return "Not configured"
        }
        if !tandemStore.state.hasAttemptedConnection {
            return "Not connected"
        }
        return tandemStore.state.status.title
    }

    private var tandemVisibleErrorMessage: String? {
        guard tandemStore.state.hasAttemptedConnection else { return nil }
        return tandemStore.state.lastErrorMessage
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

    private func refreshTandemStatusIfNeeded() async {
        guard telemetrySourceStore.selectedSource == .tandemDirect else { return }
        guard tandemStore.state.shouldAttemptStatusRefresh else { return }
        await refreshTandemStatus()
    }

    private func refreshTandemStatus() async {
        guard !isRefreshingTandemStatus else { return }
        isRefreshingTandemStatus = true
        defer { isRefreshingTandemStatus = false }
        do {
            try await tandemStore.refreshStatus()
        } catch {
            tandemBanner = error.localizedDescription
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

private struct TandemConnectionSheet: View {
    let initialState: TandemConnectionState
    let isConnecting: Bool
    let onSave: (_ adapterBaseURLString: String, _ email: String, _ password: String, _ region: TandemRegion, _ pumpSerialNumber: String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var adapterBaseURLString = ""
    @State private var email = ""
    @State private var password = ""
    @State private var region: TandemRegion = .us
    @State private var pumpSerialNumber = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Account") {
                    TextField("Email", text: $email)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.emailAddress)
                        .autocorrectionDisabled()

                    SecureField("Password", text: $password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Picker("Region", selection: $region) {
                        ForEach(TandemRegion.allCases) { value in
                            Text(value.rawValue).tag(value)
                        }
                    }

                    TextField("Pump serial number (optional)", text: $pumpSerialNumber)
                        .textInputAutocapitalization(.characters)
                        .autocorrectionDisabled()
                }

                Section("Adapter") {
                    TextField("http://127.0.0.1:8091", text: $adapterBaseURLString)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                }

                Section("What InSite does") {
                    Text("InSite sends Tandem credentials only to the adapter service, asks it to write canonical Firestore telemetry, then continues normal app hydration from Firestore.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Tandem")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(isConnecting ? "Connecting…" : "Connect") {
                        onSave(adapterBaseURLString, email, password, region, pumpSerialNumber)
                    }
                    .disabled(
                        isConnecting ||
                        adapterBaseURLString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                        email.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                        password.isEmpty
                    )
                }
            }
            .task {
                adapterBaseURLString = initialState.adapterBaseURLString
                region = initialState.region
                pumpSerialNumber = initialState.pumpSerialNumber
            }
        }
    }
}
