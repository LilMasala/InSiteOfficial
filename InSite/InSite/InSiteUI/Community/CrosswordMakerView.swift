import SwiftUI
import FirebaseAuth
import FirebaseFirestore

public struct QA: Identifiable, Hashable {
    public let id = UUID()
    public var clue: String
    public var answer: String
    public var owner: String = "me" // later: user uid / anon hash
}

public final class CrosswordMakerVM: ObservableObject {
    @Published public var pairs: [QA] = []
    @Published public var clue: String = ""
    @Published public var answer: String = ""
    @Published public var submissionMessage: String?

    private let db = Firestore.firestore()

    func addPair() {
        let c = clue.trimmingCharacters(in: .whitespacesAndNewlines)
        let a = answer.trimmingCharacters(in: .letters.inverted).uppercased()
        guard !c.isEmpty, !a.isEmpty else { return }
        // v1 SFW guardrail (client-side): filter a few obvious words; real filter lives server-side
        let banned = ["NSFW","SLUR","XXX"] // replace w/ real list/Cloud Function
        guard !banned.contains(where: { c.uppercased().contains($0) || a.contains($0) }) else { return }
        pairs.append(QA(clue: c, answer: a))
        clue = ""; answer = ""
    }

    func remove(at offsets: IndexSet) { pairs.remove(atOffsets: offsets) }

    var previewRows: [String] { pairs.map { "\($0.answer) — \($0.clue)" } }

    @MainActor
    func submitToCommunityPool() async {
        guard let uid = Auth.auth().currentUser?.uid else {
            submissionMessage = "Sign in to submit clues."
            return
        }
        guard !pairs.isEmpty else {
            submissionMessage = "Add at least one clue first."
            return
        }

        let payload = pairs.map {
            [
                "clue": $0.clue,
                "answer": $0.answer,
                "owner": $0.owner
            ]
        }

        do {
            let docRef = db.collection("users")
                .document(uid)
                .collection("crossword_submissions")
                .document()

            try await docRef.setData([
                "entries": payload,
                "moderation_status": "pending_review",
                "submitted_at": FieldValue.serverTimestamp()
            ], merge: true)

            submissionMessage = "Submitted for review."
            pairs.removeAll()
        } catch {
            submissionMessage = "Submission failed."
            print("[CrosswordMakerVM] submit failed: \(error)")
        }
    }
}

public struct CrosswordMakerView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    public var accent: Color
    @StateObject private var vm = CrosswordMakerVM()

    public init(accent: Color) { self.accent = accent }

    public var body: some View {
        VStack(spacing: 12) {
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 10) {
                        Circle().fill(accent).frame(width: 10, height: 10)
                        Text("Add clue/answer").font(.headline)
                        Spacer()
                        Text("\(vm.pairs.count) added")
                            .font(.caption).foregroundStyle(.secondary)
                    }

                    VStack(spacing: 8) {
                        TextField("Clue (e.g., 'Common hypo snack')", text: $vm.clue)
                            .textFieldStyle(.roundedBorder)

                        TextField("Answer (letters only, e.g., 'JUICEBOX')", text: $vm.answer)
                            .textInputAutocapitalization(.characters)
                            .textFieldStyle(.roundedBorder)

                        HStack {
                            Spacer()
                            Button {
                                vm.addPair()
                            } label: {
                                Label("Add", systemImage: "plus.circle.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(accent)
                            .disabled(vm.clue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                                      vm.answer.trimmingCharacters(in: .letters.inverted).isEmpty)
                        }
                    }
                }
            }
            .tint(accent)

            if vm.pairs.isEmpty {
                if #available(iOS 17.0, *) {
                    ContentUnavailableView(
                        "No entries yet",
                        systemImage: "character.textbox",
                        description: Text("Add clue/answer pairs and we’ll assemble a simple preview.\nLater: auto-grid + sharing.")
                    )
                    .padding()
                }
            } else {
                Card {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Your entries")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)

                        ForEach(vm.pairs) { qa in
                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(qa.answer).font(.headline).monospaced()
                                Text(qa.clue).font(.subheadline).foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.vertical, 4)
                        }
                        .onDelete(perform: vm.remove)

                        Divider().padding(.vertical, 4)

                        // “Export” to daily pool (in-memory; later → Firestore)
                        HStack {
                            Text("Looks good?").font(.subheadline)
                            Spacer()
                            Button {
                                Task { await vm.submitToCommunityPool() }
                            } label: {
                                Label("Submit to Community Pool", systemImage: "paperplane.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(accent)
                        }

                        if let submissionMessage = vm.submissionMessage {
                            Text(submissionMessage)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 8)
    }
}

// simple glass card used above
fileprivate struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .padding(14)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(.primary.opacity(0.06)))
            .shadow(color: .black.opacity(0.06), radius: 10, y: 6)
    }
}


struct CrosswordMakerView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack {
                CrosswordMakerView(accent: .blue)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .light)

            NavigationStack {
                CrosswordMakerView(accent: .teal)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .dark)
        }
    }
}
