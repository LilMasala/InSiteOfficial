//
//  InSiteApp.swift
//  InSite
//
//  Created by Anand Parikh on 12/13/23.
//
import SwiftUI
import FirebaseCore
import FirebaseAppCheck
import FirebaseAuth


class AppDelegate: NSObject, UIApplicationDelegate {
  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {

    #if DEBUG
    AppCheck.setAppCheckProviderFactory(AppCheckDebugProviderFactory())
    FirebaseConfiguration.shared.setLoggerLevel(.debug)
    #endif

    FirebaseApp.configure()

    // Detect unexpected auth invalidation (account disabled, token revoked, etc.)
    // This fires mid-session when Firebase cannot refresh the user's token.
    Auth.auth().addStateDidChangeListener { _, user in
      guard user == nil else { return }
      NotificationCenter.default.post(name: .chameliaAuthUserBecameUnauthenticated, object: nil)
    }

    #if DEBUG
    AppCheck.appCheck().token(forcingRefresh: true) { token, error in
      if let error = error {
        print("app check token fetch failed:", error)
      } else {
        print("app check token fetch succeeded:", token?.token ?? "nil")
      }
    }
    #endif

    return true
  }
}


@main
struct YourApp: App {
  @StateObject private var themeManager = ThemeManager()
  @Environment(\.scenePhase) private var scenePhase
  @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate

  var body: some Scene {
    WindowGroup {
      RootView()
        .environmentObject(themeManager)
    }
    .onChange(of: scenePhase) { newPhase in
      guard newPhase == .background else { return }
      guard let uid = Auth.auth().currentUser?.uid else { return }

      Task {
        try? await ChameliaStateManager.shared.saveToFirebase(userId: uid)
      }
    }
  }
}
