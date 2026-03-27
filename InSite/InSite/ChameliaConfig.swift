import Foundation

enum ChameliaConfig {
    // Keep the Cloud Run URL centralized here; do not duplicate it elsewhere.
    static let baseURL = URL(string: "https://chamelia-136217612465.us-central1.run.app")!
    static let timeoutSeconds: TimeInterval = 30
}
