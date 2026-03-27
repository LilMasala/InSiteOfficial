//
//  SignInEmailView.swift
//  InSite
//
//  Created by Anand Parikh on 9/11/24.
//
import Foundation
import SwiftUI

struct SignInEmailView: View {
    @StateObject private var viewModel = SignInEmailViewModel()
    @Binding var showSignInView: Bool
    var body: some View {
        GeometryReader { geometry in
            VStack {
                TextField("Email...",text:$viewModel.email)
                    .padding()
                    .background(Color.gray.opacity(0.4))
                    .cornerRadius(geometry.size.height * 0.02)
                SecureField("Password...",text:$viewModel.password)
                    .padding()
                    .background(Color.gray.opacity(0.4))
                    .cornerRadius(geometry.size.height * 0.02)
                Button {
                    Task {
                        do {
                            try await viewModel.signUp()
                            showSignInView = false
                            return
                        } catch {
                            print(error)
                        }
                        do {
                            try await viewModel.signIn()
                            showSignInView = false
                            return
                        } catch {
                            print(error)
                        }
                    }

                } label: {
                    Text("Sign in")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(height: geometry.size.height * 0.07)
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .cornerRadius(geometry.size.height * 0.02)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("Sign in with email")
        }
    }
}

struct SignInEmailView_Previews: PreviewProvider {
    static var previews: some View {
        SignInEmailView(showSignInView:.constant(false))
    }
}
