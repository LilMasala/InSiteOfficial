//
//  SettingsViewModel.swift
//  InSite
//
//  Created by Anand Parikh on 12/26/24.
//

import Foundation
import SwiftUI
import FirebaseAuth

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var authProviders: [AuthProviderOption] = []
    
    func loadAuthProviders() {
        if let providers = try? AuthManager.shared.getProvider() {
            authProviders = providers
        }
    }
    func logOut() async throws {
        let departingUid = Auth.auth().currentUser?.uid
        // Best-effort save before signing out so state is not lost
        if let uid = departingUid {
            try? await ChameliaStateManager.shared.saveToFirebase(userId: uid)
        }
        try AuthManager.shared.signOut()
        ProfileDataStore().clearData(for: departingUid)
        SiteChangeData.shared.clearData(for: departingUid)
        DataManager.shared.handleLogout(for: departingUid)
        ChameliaDashboardStore.shared.clearSession()
    }
    
    func resetPassword() async throws {
        let authUser = try AuthManager.shared.getAuthenticatedUser()
        guard let email = authUser.email else {
            throw URLError(.fileDoesNotExist)
        }
        try await AuthManager.shared.resetPassword(email: email)
    }
    
    func updateEmail() async throws {
        let email = "hello123@testing.com"
        try await AuthManager.shared.updateEmail(email: email)
    }
    
    func updatePassword() async throws {
        let password = "HelloTrying123!"
        try await AuthManager.shared.updatePassword(password: password)
    }
}
