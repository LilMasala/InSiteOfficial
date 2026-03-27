import SwiftUI
import Combine
import FirebaseFirestore

// 1) HiddenKeyCapture: a single invisible field that grabs keystrokes
struct HiddenKeyCapture: UIViewRepresentable {
    final class Coordinator: NSObject, UITextFieldDelegate {
        var onChar: (Character) -> Void
        var onBackspace: () -> Void
        init(onChar: @escaping (Character)->Void, onBackspace: @escaping ()->Void) {
            self.onChar = onChar; self.onBackspace = onBackspace
        }
        func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
            if string.isEmpty { onBackspace(); return false }
            // filter to letters only, uppercase
            if let ch = string.uppercased().first, ch.isLetter { onChar(ch) }
            return false // don't actually insert into the hidden field
        }
    }

    var makeFirstResponder: Bool
    var onChar: (Character) -> Void
    var onBackspace: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onChar: onChar, onBackspace: onBackspace) }

    func makeUIView(context: Context) -> UITextField {
        let tf = UITextField(frame: .zero)
        tf.autocorrectionType = .no
        tf.autocapitalizationType = .allCharacters
        tf.spellCheckingType = .no
        tf.smartDashesType = .no
        tf.smartQuotesType = .no
        tf.keyboardType = .asciiCapable
        tf.isHidden = true
        tf.delegate = context.coordinator
        return tf
    }

    func updateUIView(_ uiView: UITextField, context: Context) {
        if makeFirstResponder, uiView.window != nil, !uiView.isFirstResponder {
            uiView.becomeFirstResponder()
        }
    }
}


struct DailyCrosswordView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    var accent: Color

    @StateObject private var vm = MiniCrosswordVM()

    init(accent: Color) { self.accent = accent }

    var body: some View {
        VStack(spacing: 12) {
            header

            // Grid
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .fill(.ultraThinMaterial)
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.primary.opacity(0.06)))
                VStack(spacing: 2) {
                    ForEach(0..<vm.size, id: \.self) { r in
                        HStack(spacing: 2) {
                            ForEach(0..<vm.size, id: \.self) { c in
                                CellView(cell: vm.cellAt(r, c),
                                         isSelected: vm.isSelected(r, c),
                                         accent: accent)
                                .onTapGesture { vm.select(r, c) }
                            }
                        }
                    }
                    HiddenKeyCapture(
                        makeFirstResponder: true,
                        onChar: { vm.type($0) },
                        onBackspace: { vm.backspace() }
                    )
                    .frame(width: 0, height: 0)
                    .accessibilityHidden(true)
                }
                .padding(8)
            }
            .aspectRatio(1, contentMode: .fit)
            .shadow(color: .black.opacity(0.06), radius: 10, y: 6)

            // Clues
            Card {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Clues")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(vm.clues.enumeratedArray(), id: \.element.id) { (i, clue) in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text("\(i+1).").font(.caption.weight(.bold)).foregroundStyle(accent)
                            Text(clue.text).font(.body)
                            Spacer()
                        }
                    }
                }
            }
        }
        .padding(.vertical, 8)
        .task { await vm.spawnDaily() }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Daily Mini")
                    .font(.headline)
                Text(vm.dateTitle)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            ProgressView(value: vm.progress)
                .progressViewStyle(.circular)
                .tint(accent)
            VStack(alignment: .trailing, spacing: 2) {
                Text(vm.timerText)
                    .font(.title3.monospacedDigit().weight(.semibold))
                Text(vm.isRunning ? "Tap to pause" : "Tap to resume")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .onTapGesture { vm.toggleTimer() }
        }
    }
}

// MARK: - VM + Models

struct MiniCell: Identifiable, Equatable {
    let id = UUID()
    var row: Int
    var col: Int
    var isBlock: Bool
    var solution: Character? // expected letter
    var entry: Character?    // user letter
}

struct MiniClue: Identifiable {
    let id = UUID()
    var text: String
    var answer: String
}

final class MiniCrosswordVM: ObservableObject {
    @Published private(set) var grid: [[MiniCell]] = []
    @Published private(set) var clues: [MiniClue] = []
    @Published var selected: (r:Int, c:Int)? = nil
    @Published var isRunning: Bool = false
    @Published private(set) var elapsed: TimeInterval = 0

    let size = 9
    private let db = Firestore.firestore()
    
    
    init() {
            // seed with all blocks so cellAt() is safe pre-spawn
            self.grid = (0..<size).map { r in
                (0..<size).map { c in MiniCell(row: r, col: c, isBlock: true, solution: nil, entry: nil) }
            }
            self.clues = []
            self.selected = nil
            self.isRunning = false
            self.elapsed = 0
        }

        deinit { timerC?.cancel() }
    
    private var timerC: AnyCancellable?
    private var startTime: Date?

    var dateTitle: String {
        let f = DateFormatter(); f.dateStyle = .full
        return f.string(from: Date())
    }

    // 0.0 ... 1.0
    var progress: Double {
        let fill = grid.flatMap { $0 }.filter { !$0.isBlock && $0.solution != nil }
        guard !fill.isEmpty else { return 0 }
        let correct = fill.filter { $0.entry == $0.solution }.count
        return Double(correct) / Double(fill.count)
    }

    var timerText: String {
        let t = Int(elapsed)
        let m = t / 60
        let s = t % 60
        return String(format: "%02d:%02d", m, s)
    }

    // MARK: - Lifecycle
    @MainActor
    func spawnDaily() async {
        // Merge standard + community pool (stubbed)
        let standard: [QA] = [
            .init(clue: "Rapid insulin brand", answer: "NOVOLOG"),
            .init(clue: "Glucose check", answer: "FINGER"),
            .init(clue: "Low snack", answer: "JUICEBOX"),
            .init(clue: "Sensor brand", answer: "DEXCOM"),
            .init(clue: "Basal mode", answer: "SLEEP")
        ]
        let community = await fetchApprovedCommunityPairs()

        let merged = pickCommunityFair(community, targetShare: 0.3) + standard.shuffled()
        buildMini(from: merged)

        // timer
        elapsed = 0
        startTimer()
    }

    private func fetchApprovedCommunityPairs() async -> [QA] {
        do {
            let snapshot = try await db.collectionGroup("crossword_submissions")
                .whereField("moderation_status", isEqualTo: "approved")
                .getDocuments()

            return snapshot.documents.flatMap { document -> [QA] in
                let owner = document.reference.parent.parent?.documentID ?? "community"
                let entries = document.data()["entries"] as? [[String: Any]] ?? []
                let mapped: [QA] = entries.compactMap { entry -> QA? in
                    guard
                        let clue = entry["clue"] as? String,
                        let answer = entry["answer"] as? String,
                        !clue.isEmpty,
                        !answer.isEmpty
                    else {
                        return nil
                    }
                    return QA(clue: clue, answer: answer, owner: owner)
                }
                return mapped
            }
        } catch {
            print("[MiniCrosswordVM] approved clue fetch failed: \(error)")
            return []
        }
    }

    func pickCommunityFair(_ pool: [QA], targetShare: Double) -> [QA] {
        // at most one per owner
        var seen = Set<String>()
        let unique = pool.shuffled().filter { qa in
            guard !seen.contains(qa.owner) else { return false }
            seen.insert(qa.owner); return true
        }
        // target ~share of total words (aim ~4-6 for a mini)
        let target = max(2, Int(Double(6) * targetShare))
        return Array(unique.prefix(target))
    }

    func buildMini(from list: [QA]) {
        // super-simple horizontal placer (v1 UI only; replace later)
        var cells: [[MiniCell]] = (0..<size).map { r in
            (0..<size).map { c in MiniCell(row: r, col: c, isBlock: true, solution: nil, entry: nil) }
        }

        var used: [MiniClue] = []
        var row = 0
        for qa in list {
            let word = qa.answer.filter(\.isLetter)
            guard word.count >= 3 else { continue }
            if row >= size { break }
            if word.count > size { continue }
            // place at row, left-aligned (v1)
            for c in 0..<word.count {
                cells[row][c].isBlock = false
                cells[row][c].solution = Array(word)[c]
            }
            // leave a vertical gap row as separator
            used.append(MiniClue(text: qa.clue, answer: word))
            row += 2
        }

        // sprinkle random blocks in empty rows
        for r in 0..<size {
            for c in 0..<size {
                if cells[r][c].isBlock { /* already true */ }
            }
        }

        grid = cells
        clues = used
        selected = firstWritable()
    }

    private func firstWritable() -> (Int,Int)? {
        for r in 0..<size { for c in 0..<size {
            if !grid[r][c].isBlock, grid[r][c].solution != nil { return (r,c) }
        } }
        return nil
    }

    // MARK: - Selection/Input
    func cellAt(_ r: Int, _ c: Int) -> MiniCell { grid[r][c] }

    func isSelected(_ r: Int, _ c: Int) -> Bool {
        guard let s = selected else { return false }
        return s.r == r && s.c == c
    }

    func select(_ r: Int, _ c: Int) {
        guard !grid[r][c].isBlock, grid[r][c].solution != nil else { return }
        selected = (r,c)
    }

    func type(_ ch: Character) {
        guard let s = selected else { return }
        guard !grid[s.r][s.c].isBlock else { return }
        grid[s.r][s.c].entry = Character(ch.uppercased())
        advance()
        objectWillChange.send()
    }

    func backspace() {
        guard let s = selected else { return }
        guard !grid[s.r][s.c].isBlock else { return }
        grid[s.r][s.c].entry = nil
        objectWillChange.send()
    }

    func clearCurrent() {
        guard let s = selected else { return }
        // clear the whole across word on that row (v1)
        for c in 0..<size {
            if !grid[s.r][c].isBlock, grid[s.r][c].solution != nil {
                grid[s.r][c].entry = nil
            }
        }
        selected = (s.r, 0)
        objectWillChange.send()
    }

    private func advance() {
        guard let s = selected else { return }
        var c = s.c + 1
        while c < size {
            if !grid[s.r][c].isBlock, grid[s.r][c].solution != nil {
                selected = (s.r, c); return
            }
            c += 1
        }
    }

    // MARK: - Timer
    func startTimer() {
        isRunning = true
        startTime = Date()
        timerC?.cancel()
        timerC = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()
            .sink { [weak self] _ in
                guard let self = self, self.isRunning, let start = self.startTime else { return }
                self.elapsed = Date().timeIntervalSince(start)
            }
    }

    func toggleTimer() {
        if isRunning {
            isRunning = false
        } else {
            startTime = Date().addingTimeInterval(-elapsed)
            isRunning = true
        }
    }
}

// MARK: - Views

fileprivate struct CellView: View {
    var cell: MiniCell
    var isSelected: Bool
    var accent: Color

    var body: some View {
        ZStack {
            Rectangle()
                .fill(cell.isBlock ? Color.primary.opacity(0.15) : Color.white.opacity(0.9))
                .overlay(
                    RoundedRectangle(cornerRadius: 3)
                        .stroke(isSelected ? accent : Color.primary.opacity(0.12), lineWidth: isSelected ? 2 : 1)
                )
            if !cell.isBlock {
                Text(cell.entry.map { String($0) } ?? "")
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(.primary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .aspectRatio(1, contentMode: .fit)
    }
}



fileprivate extension Array {
    func enumeratedArray() -> [(offset: Int, element: Element)] {
        self.enumerated().map { (offset: $0.offset, element: $0.element) }
    }
}

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


struct DailyCrosswordView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack {
                DailyCrosswordView(accent: .purple)
                    .environmentObject(ThemeManager())
                    .padding()
            }
            .environment(\.colorScheme, .light)

            NavigationStack {
                DailyCrosswordView(accent: .orange)
                    .environmentObject(ThemeManager())
                    .padding()
            }
            .environment(\.colorScheme, .dark)
        }
    }
}
