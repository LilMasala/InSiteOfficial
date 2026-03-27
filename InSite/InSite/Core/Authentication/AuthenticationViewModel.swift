//
//  AuthenticationViewModel.swift
//  InSite
//
//  Created by Anand Parikh on 12/26/24.
//

import SwiftUI
import GoogleSignIn
import GoogleSignInSwift
import AuthenticationServices
import CryptoKit
import Foundation

@MainActor
final class AuthViewModel: NSObject, ObservableObject {
    
    private var currentNonce: String?
    @Published var didSignInWithApple: Bool = false
    
    func signInGoogle() async throws {
        let helper = SignInGoogleHelper()
        let tokens = try await helper.signIn()
        let authDataResult = try await AuthManager.shared.signInWithGoogle(tokens: tokens)
        let user = DBUser(auth: authDataResult)
        try await UserManager.shared.createNewUser(user: user)
        
    }
    
    func signInAnonymous() async throws {
        let authDataResult = try await AuthManager.shared.signInAnonymous()
        let user = DBUser(auth: authDataResult)
        try await UserManager.shared.createNewUser(user: user)
        
    }


    func signInApple() async throws {
        startSignInWithAppleFlow()
    }
    @available(iOS 13, *)
    func startSignInWithAppleFlow() {
        
        guard let topVC = Utilities.shared.topViewController() else {
            return
        }
        
        
      let nonce = randomNonceString()
      currentNonce = nonce
      let appleIDProvider = ASAuthorizationAppleIDProvider()
      let request = appleIDProvider.createRequest()
      request.requestedScopes = [.fullName, .email]
      request.nonce = sha256(nonce)

      let authorizationController = ASAuthorizationController(authorizationRequests: [request])
      authorizationController.delegate = self
      authorizationController.presentationContextProvider = topVC
      authorizationController.performRequests()
    }
    private func randomNonceString(length: Int = 32) -> String {
      precondition(length > 0)
      var randomBytes = [UInt8](repeating: 0, count: length)
      let errorCode = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)
      if errorCode != errSecSuccess {
        fatalError(
          "Unable to generate nonce. SecRandomCopyBytes failed with OSStatus \(errorCode)"
        )
      }

      let charset: [Character] =
        Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")

      let nonce = randomBytes.map { byte in
        // Pick a random character from the set, wrapping around if needed.
        charset[Int(byte) % charset.count]
      }

      return String(nonce)
    }
    
    @available(iOS 13, *)
    private func sha256(_ input: String) -> String {
      let inputData = Data(input.utf8)
      let hashedData = SHA256.hash(data: inputData)
      let hashString = hashedData.compactMap {
        String(format: "%02x", $0)
      }.joined()

      return hashString
    }


        
    
}



@available(iOS 13.0, *)
extension AuthViewModel: ASAuthorizationControllerDelegate {

  func authorizationController(controller: ASAuthorizationController, didCompleteWithAuthorization authorization: ASAuthorization) {
      
      guard
        let appleIDCredential = authorization.credential as? ASAuthorizationAppleIDCredential,
        let appleIDToken = appleIDCredential.identityToken,
        let idTokenString = String(data: appleIDToken, encoding: .utf8),
        let nonce = currentNonce else {
        print("error")
        return
      }
        
      let tokens = SignInWithAppleResult(token: idTokenString, nonce: nonce)
      
      Task {
          do {
              let authDataResult = try await AuthManager.shared.signInWithApple(tokens: tokens)
              let user = DBUser(auth: authDataResult)
              try await UserManager.shared.createNewUser(user: user)
              
              didSignInWithApple = true
          } catch {
          }
    }
  }

  func authorizationController(controller: ASAuthorizationController, didCompleteWithError error: Error) {
    // Handle error.
    print("Sign in with Apple errored: \(error)")
  }

}
