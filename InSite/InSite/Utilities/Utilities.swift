//
//  Utilities.swift
//  InSite
//
//  Created by Anand Parikh on 9/18/24.
//

import Foundation
import FirebaseAuth

enum ChameliaSettingsStore {
    private static let level2EnabledKey = "ChameliaLevel2Enabled"

    private static func key(_ base: String, userId: String?) -> String? {
        guard let userId, !userId.isEmpty else { return nil }
        return "\(base).\(userId)"
    }

    static func level2Enabled(userId: String? = Auth.auth().currentUser?.uid) -> Bool {
        guard let key = key(level2EnabledKey, userId: userId) else { return false }
        return UserDefaults.standard.bool(forKey: key)
    }

    static func setLevel2Enabled(_ value: Bool, userId: String? = Auth.auth().currentUser?.uid) {
        guard let key = key(level2EnabledKey, userId: userId) else { return }
        UserDefaults.standard.set(value, forKey: key)
    }
}
import UIKit


final class Utilities {
    static let shared = Utilities()
    private init() {}

    @MainActor
    func topViewController(controller: UIViewController? = nil) -> UIViewController? {
        
        let controller = controller ?? UIApplication.shared.keyWindow?.rootViewController
        
        if let navigationController = controller as? UINavigationController {
            return topViewController(controller: navigationController.visibleViewController)
        }
        if let tabController = controller as? UITabBarController {
            if let selected = tabController.selectedViewController {
                return topViewController(controller: selected)
            }
        }
        if let presented = controller?.presentedViewController {
            return topViewController(controller: presented)
        }
        return controller
    }
    
}
