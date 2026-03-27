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
        }
        .navigationTitle("Settings")
        .onChange(of: chameliaLevel2Enabled) { newValue in
            ChameliaSettingsStore.setLevel2Enabled(newValue)
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
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView(showSignInView: .constant(false))
    }
}
