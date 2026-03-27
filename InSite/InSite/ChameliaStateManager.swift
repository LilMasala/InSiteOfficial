import Foundation

enum ChameliaStateManagerError: Error {
    case invalidResponse
    case notFound
    case unexpectedStatusCode(Int)
    case invalidPayload
}

private actor ChameliaStateUserLock {
    private var busyUsers: Set<String> = []
    private var waiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    func acquire(for userId: String) async {
        guard busyUsers.contains(userId) else {
            busyUsers.insert(userId)
            return
        }

        await withCheckedContinuation { continuation in
            waiters[userId, default: []].append(continuation)
        }
    }

    func release(for userId: String) {
        guard var userWaiters = waiters[userId], !userWaiters.isEmpty else {
            busyUsers.remove(userId)
            return
        }

        let next = userWaiters.removeFirst()
        waiters[userId] = userWaiters.isEmpty ? nil : userWaiters
        next.resume()
    }
}

struct ChameliaStateResponse {
    let statusCode: Int
    let payload: [String: Any]
}

final class ChameliaStateManager {
    static let shared = ChameliaStateManager()

    private let session: URLSession
    private let userLock = ChameliaStateUserLock()

    init(session: URLSession = .shared) {
        self.session = session
    }

    func saveToFirebase(userId: String) async throws -> ChameliaStateResponse {
        try await withUserLock(userId: userId) {
            try await post(path: "/chamelia_save_patient", userId: userId)
        }
    }

    func loadFromFirebase(userId: String) async throws -> ChameliaStateResponse {
        try await withUserLock(userId: userId) {
            try await post(path: "/chamelia_load_patient", userId: userId)
        }
    }

    private func post(path: String, userId: String) async throws -> ChameliaStateResponse {
        var request = URLRequest(url: ChameliaConfig.baseURL.appending(path: path))
        request.httpMethod = "POST"
        request.timeoutInterval = ChameliaConfig.timeoutSeconds
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["patient_id": userId])

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ChameliaStateManagerError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if httpResponse.statusCode == 404 {
                throw ChameliaStateManagerError.notFound
            }
            throw ChameliaStateManagerError.unexpectedStatusCode(httpResponse.statusCode)
        }

        let payloadObject = try JSONSerialization.jsonObject(with: data)
        guard let payload = payloadObject as? [String: Any] else {
            throw ChameliaStateManagerError.invalidPayload
        }

        return ChameliaStateResponse(statusCode: httpResponse.statusCode, payload: payload)
    }

    private func withUserLock(
        userId: String,
        operation: () async throws -> ChameliaStateResponse
    ) async throws -> ChameliaStateResponse {
        await userLock.acquire(for: userId)
        defer {
            Task {
                await userLock.release(for: userId)
            }
        }

        return try await operation()
    }
}
