//
//  SignInEmailViewModel.swift
//  InSite
//
//  Created by Anand Parikh on 12/26/24.
//

import Foundation

import SwiftUI

@MainActor
final class SignInEmailViewModel: ObservableObject {
    @Published var email = ""
    @Published var password = ""
    
    func signUp() async throws {
        guard !email.isEmpty, !password.isEmpty else {
            print("No email or password is found")
            return
        }
        
        let authDataResult = try await AuthManager.shared.createUser(email: email, password: password)
        let user = DBUser(auth: authDataResult)
        try await UserManager.shared.createNewUser(user: user)
        
    }
    
    func signIn() async throws {
        guard !email.isEmpty, !password.isEmpty else {
            print("No email or password is found")
            return
        }
        
        try await AuthManager.shared.signInUser(email: email, password: password)
     
    }
}
