import SwiftUI
import Combine
import Foundation
import FirebaseAuth
import FirebaseFirestore

// Simple shared types & the tile (kept here so Home imports only Community)
public struct CommunityTile: View {
    public var accent: Color
    @ScaledMetric private var diameter: CGFloat = 140

    public init(accent: Color) { self.accent = accent }

    public var body: some View {
        CircleTileBase(diameter: diameter) {
            VStack(spacing: 8) {
                ZStack {
                    Circle().fill(accent.opacity(0.10))
                    Image(systemName: "person.3.sequence.fill")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(accent)
                }
                .frame(height: 54)

                Text("Community")
                    .font(.subheadline.weight(.semibold))

                Text("Board • Crosswords")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
        }
        .accessibilityLabel("Community: board and crosswords")
    }
}

// MARK: - Board models

public enum BoardWindow: String, CaseIterable, Identifiable {
    case today = "Today", week = "Last 7 Days", month = "Last 30 Days"
    public var id: String { rawValue }
    public var interval: TimeInterval {
        switch self {
        case .today: return 60*60*24
        case .week:  return 60*60*24*7
        case .month: return 60*60*24*30
        }
    }
}

public struct CommunityPost: Identifiable, Hashable {
    public let id: String
    public var text: String
    public var createdAt: Date
    public var upvotes: Int
    public var comments: Int
    public var viewerHasUpvoted: Bool

    init(
        id: String = UUID().uuidString,
        text: String,
        createdAt: Date,
        upvotes: Int,
        comments: Int,
        viewerHasUpvoted: Bool = false
    ) {
        self.id = id
        self.text = text
        self.createdAt = createdAt
        self.upvotes = upvotes
        self.comments = comments
        self.viewerHasUpvoted = viewerHasUpvoted
    }
}

private struct CommunityPostRecord {
    let text: String
    let createdAt: Date
    let upvotes: Int
    let comments: Int
    let upvoterIds: [String]
    let authorId: String
    let dayBucket: String
}

public final class CommunityBoardVM: ObservableObject {
    @Published public var window: BoardWindow = .today
    @Published public var posts: [CommunityPost] = []

    private let db = Firestore.firestore()
    private let postsCollection = Firestore.firestore().collection("community_posts")

    public init() {}

    public func upvote(_ post: CommunityPost) {
        guard let idx = posts.firstIndex(of: post) else { return }
        guard !posts[idx].viewerHasUpvoted else { return }
        posts[idx].upvotes += 1
        posts[idx].viewerHasUpvoted = true
        Task { await persistUpvote(postId: post.id, add: true) }
    }

    public var filteredSorted: [CommunityPost] {
        let cutoff = Date().addingTimeInterval(-window.interval)
        let filtered = posts
            .filter { $0.createdAt >= cutoff }
        if sortTop {
            return filtered.sorted { lhs, rhs in
                (lhs.upvotes, lhs.createdAt) > (rhs.upvotes, rhs.createdAt)
            }
        } else {
            return filtered.sorted { $0.createdAt > $1.createdAt }
        }
    }

    func removeUpvote(_ post: CommunityPost) {
        guard let idx = posts.firstIndex(of: post) else { return }
        guard posts[idx].viewerHasUpvoted else { return }
        posts[idx].upvotes = max(0, posts[idx].upvotes - 1)
        posts[idx].viewerHasUpvoted = false
        Task { await persistUpvote(postId: post.id, add: false) }
    }

    var sortTop: Bool = true {
        didSet {}
    }

    @MainActor
    func refresh() async {
        let cutoff = Date().addingTimeInterval(-window.interval)
        let currentUid = Auth.auth().currentUser?.uid

        do {
            let snapshot = try await postsCollection
                .whereField("created_at", isGreaterThanOrEqualTo: Timestamp(date: cutoff))
                .order(by: "created_at", descending: true)
                .getDocuments()

            posts = snapshot.documents.compactMap { document in
                guard let text = document.data()["text"] as? String else { return nil }
                let createdAt = (document.data()["created_at"] as? Timestamp)?.dateValue() ?? Date.distantPast
                let upvotes = document.data()["upvotes"] as? Int ?? 0
                let comments = document.data()["comments"] as? Int ?? 0
                let upvoterIds = document.data()["upvoter_ids"] as? [String] ?? []
                return CommunityPost(
                    id: document.documentID,
                    text: text,
                    createdAt: createdAt,
                    upvotes: upvotes,
                    comments: comments,
                    viewerHasUpvoted: currentUid.map(upvoterIds.contains) ?? false
                )
            }
        } catch {
            print("[CommunityBoardVM] refresh failed: \(error)")
        }
    }

    @MainActor
    func submitPost(text: String) async {
        guard let uid = Auth.auth().currentUser?.uid else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let tempId = "temp-\(UUID().uuidString)"
        posts.insert(
            CommunityPost(
                id: tempId,
                text: trimmed,
                createdAt: Date(),
                upvotes: 0,
                comments: 0,
                viewerHasUpvoted: false
            ),
            at: 0
        )

        do {
            let dayBucket = Self.dayBucket(for: Date())
            let todayCount = try await postsCollection
                .whereField("author_id", isEqualTo: uid)
                .whereField("day_bucket", isEqualTo: dayBucket)
                .count
                .getAggregation(source: .server)

            if todayCount.count.intValue >= 3 {
                posts.removeAll { $0.id == tempId }
                print("[CommunityBoardVM] rate limit reached for user \(uid)")
                return
            }

            let docRef = postsCollection.document()
            try await docRef.setData([
                "text": trimmed,
                "created_at": FieldValue.serverTimestamp(),
                "upvotes": 0,
                "comments": 0,
                "upvoter_ids": [],
                "author_id": uid,
                "day_bucket": dayBucket
            ], merge: true)

            posts.removeAll { $0.id == tempId }
            await refresh()
        } catch {
            posts.removeAll { $0.id == tempId }
            print("[CommunityBoardVM] submit failed: \(error)")
        }
    }

    private func persistUpvote(postId: String, add: Bool) async {
        guard let uid = Auth.auth().currentUser?.uid else { return }
        guard !postId.hasPrefix("temp-") else { return }
        let docRef = postsCollection.document(postId)

        do {
            let snapshot = try await docRef.getDocument()
            let existingUpvotes = snapshot.data()?["upvotes"] as? Int ?? 0
            let upvoterIds = snapshot.data()?["upvoter_ids"] as? [String] ?? []
            var nextIds = Set(upvoterIds)
            var nextUpvotes = existingUpvotes

            if add {
                if nextIds.insert(uid).inserted {
                    nextUpvotes += 1
                }
            } else if nextIds.remove(uid) != nil {
                nextUpvotes = max(0, existingUpvotes - 1)
            }

            try await docRef.updateData([
                "upvotes": nextUpvotes,
                "upvoter_ids": Array(nextIds)
            ])
        } catch {
            print("[CommunityBoardVM] upvote write failed: \(error)")
            await refresh()
        }
    }

    private static func dayBucket(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }
}

// MARK: - Crosswords models

//public struct QA: Identifiable, Hashable {
//    public let id = UUID()
//    public var clue: String
//    public var answer: String
//}
//
//public final class CrosswordMakerVM: ObservableObject {
//    @Published public var pairs: [QA] = []
//    @Published public var clue: String = ""
//    @Published public var answer: String = ""
//
//    public func addPair() {
//        let c = clue.trimmingCharacters(in: .whitespacesAndNewlines)
//        let a = answer.trimmingCharacters(in: .whitespacesAndNewlines)
//        guard !c.isEmpty, !a.isEmpty else { return }
//        pairs.append(QA(clue: c, answer: a.uppercased().replacingOccurrences(of: " ", with: "")))
//        clue = ""; answer = ""
//    }
//
//    public var previewRows: [String] {
//        pairs.map { "\($0.answer) — \($0.clue)" }
//    }
//}
