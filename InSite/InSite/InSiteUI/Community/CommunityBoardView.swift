import SwiftUI

// MARK: - Community Board (glassy + live preview + correct upvote)
public struct CommunityBoardView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @StateObject private var vm = CommunityBoardVM()

    public var accent: Color
    @State private var composerText = ""
    @State private var sortTop = true
    @State private var charLimit = 280
    @State private var appeared = false

    public init(accent: Color) { self.accent = accent }

    public var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 16) {
                    header

                    // Composer + preview
                    VStack(spacing: 12) {
                        ComposerBar(
                            text: $composerText,
                            accent: accent,
                            charLimit: charLimit,
                            onSend: { Task { await handleSend() } }
                        )

                        if !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text("Preview")
                                .font(.caption).foregroundStyle(.secondary)
                                .padding(.horizontal, 16)
                                .padding(.top, 2)
                                .frame(maxWidth: .infinity, alignment: .leading)

                            PostCard(
                                post: .init(text: composerText, createdAt: Date(), upvotes: 0, comments: 0),
                                accent: accent,
                                onUpvote: {}, onRemoveUpvote: {},
                                preview: true
                            )
                            .padding(.horizontal, 12)
                            .transition(.opacity.combined(with: .move(edge: .top)))
                        }
                    }
                    .padding(.top, 4)

                    // Feed
                    if vm.filteredSorted.isEmpty {
                        if #available(iOS 17.0, *) {
                            ContentUnavailableView(
                                "Nothing here yet",
                                systemImage: "bubble.left.and.bubble.right",
                                description: Text("Be the first to post something. It’s anonymous!")
                            )
                            .padding(.top, 24)
                            .padding(.horizontal, 16)
                        }
                    } else {
                        LazyVStack(spacing: 12) {
                            ForEach(vm.filteredSorted) { post in
                                PostCard(
                                    post: post,
                                    accent: accent,
                                    onUpvote: { vm.upvote(post) },
                                    onRemoveUpvote: { vm.removeUpvote(post) },
                                    preview: false
                                )
                                .padding(.horizontal, 16)
                                .transition(.opacity.combined(with: .move(edge: .bottom)))
                            }
                        }
                        .animation(.easeInOut(duration: 0.2), value: vm.window)
                        .animation(.easeInOut(duration: 0.2), value: vm.posts)
                        .padding(.bottom, 24)
                    }
                }
                .padding(.top, 12)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
            .refreshable { await vm.refresh() }
        }
        .navigationTitle("Community Board")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Picker("Time Window", selection: $vm.window) {
                        ForEach(BoardWindow.allCases) { w in Text(w.rawValue).tag(w) }
                    }
                    Toggle(isOn: $sortTop) {
                        Label("Sort by top", systemImage: "arrow.up.arrow.down")
                    }
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
                .onChange(of: sortTop) { _ in
                    vm.sortTop = sortTop
                }
            }
        }
        .task {
            await vm.refresh()
        }
        .onAppear { if !appeared { appeared = true } }
    }

    // MARK: - Header
    private var header: some View {
        HStack(spacing: 12) {
            Picker("Window", selection: $vm.window) {
                ForEach(BoardWindow.allCases) { w in Text(w.rawValue).tag(w) }
            }
            .pickerStyle(.segmented)

            Spacer(minLength: 0)

            HStack(spacing: 6) {
                Image(systemName: sortTop ? "chart.bar.fill" : "clock.fill")
                    .foregroundStyle(accent)
                Text(sortTop ? "Top" : "New")
                    .font(.footnote.weight(.semibold))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    // MARK: - Send
    private func handleSend() async {
        let text = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
            composerText = ""
        }
        await vm.submitPost(text: text)
    }
}

// MARK: - ComposerBar (unchanged from your version, kept pretty)
fileprivate struct ComposerBar: View {
    @Binding var text: String
    var accent: Color
    var charLimit: Int
    var onSend: () -> Void

    @FocusState private var focused: Bool
    @State private var breathe: CGFloat = 0

    var remaining: Int { max(0, charLimit - text.count) }

    var body: some View {
        VStack(spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                ZStack {
                    Circle().fill(accent.opacity(0.12))
                    Image(systemName: "person.fill.viewfinder").foregroundStyle(accent)
                }
                .frame(width: 36, height: 36)
                .scaleEffect(1 + (focused ? 0.02 * breathe : 0))

                VStack(spacing: 6) {
                    ZStack(alignment: .topLeading) {
                        if text.isEmpty {
                            Text("Share something (anonymous)…")
                                .foregroundStyle(.secondary)
                                .padding(.vertical, 10)
                                .padding(.horizontal, 12)
                        }
                        TextEditor(text: $text)
                            .padding(8)
                            .frame(minHeight: 46, maxHeight: 120)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(.ultraThinMaterial)
                                    .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.primary.opacity(0.08)))
                            )
                            .focused($focused)
                            .onChange(of: text) { newVal in
                                if newVal.count > charLimit { text = String(newVal.prefix(charLimit)) }
                            }
                    }

                    HStack {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                Chip("Win:", accent) { insert(" Small W today: ") }
                                Chip("Tip:", accent) { insert(" Pro tip: ") }
                                Chip("Q:", accent)   { insert(" Question: ") }
                            }
                            .padding(.leading, 2)
                        }
                        Spacer()
                        Text("\(remaining)")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(remaining <= 20 ? .red.opacity(0.8) : .secondary)
                    }
                }

                Button {
                    onSend()
                    focused = false
                } label: {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 17, weight: .semibold))
                        .padding(10)
                        .background(remaining == charLimit ? Color.gray.opacity(0.2) : accent.opacity(0.2))
                        .clipShape(Circle())
                        .overlay(Circle().stroke(accent.opacity(0.35), lineWidth: 1))
                }
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .onAppear {
            withAnimation(.easeInOut(duration: 3.4).repeatForever(autoreverses: true)) { breathe = 1 }
        }
    }

    private func insert(_ snippet: String) {
        if text.isEmpty { text = snippet } else { text.append(snippet) }
    }
}

fileprivate struct Chip: View {
    var label: String
    var accent: Color
    var action: () -> Void
    init(_ label: String, _ accent: Color, action: @escaping () -> Void) {
        self.label = label; self.accent = accent; self.action = action
    }
    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.caption.weight(.semibold))
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
                .background(accent.opacity(0.12), in: Capsule())
                .overlay(Capsule().stroke(accent.opacity(0.25), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Post Card (NO local +1; uses onUpvote/onRemoveUpvote)
fileprivate struct PostCard: View {
    var post: CommunityPost
    var accent: Color
    var onUpvote: () -> Void
    var onRemoveUpvote: () -> Void
    var preview: Bool

    @State private var bounce = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Top row
            HStack(spacing: 10) {
                Circle().fill(accent.opacity(0.12))
                    .frame(width: 28, height: 28)
                    .overlay(Image(systemName: "person.fill").foregroundStyle(accent))
                Text("Anonymous").font(.subheadline.weight(.semibold))
                Spacer()
                Text(preview ? "now" : relative(post.createdAt))
                    .font(.caption).foregroundStyle(.secondary)
            }

            // Content
            Text(post.text).font(.body).fixedSize(horizontal: false, vertical: true)

            // Actions
            HStack(spacing: 14) {
                Button {
                    guard !preview else { return }
                    if #available(iOS 17.0, *), !post.viewerHasUpvoted { UIApplication.shared.prepareFeedback() }
                    bounce = true
                    if post.viewerHasUpvoted { onRemoveUpvote() } else { onUpvote() }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) { bounce = false }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: post.viewerHasUpvoted ? "arrow.up.square.fill" : "arrow.up.square")
                        // IMPORTANT: show only source of truth (no local +1)
                        Text("\(post.upvotes)").monospacedDigit()
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(post.viewerHasUpvoted ? accent : accent.opacity(0.9))
                .scaleEffect(bounce ? 1.08 : 1.0)

                Label("\(post.comments)", systemImage: "text.bubble")
                    .foregroundStyle(.secondary)

                Spacer()

                if !preview {
                    Menu {
                        Button(role: .destructive) { /* TODO: report */ } label: {
                            Label("Report", systemImage: "flag")
                        }
                    } label: { Image(systemName: "ellipsis").foregroundStyle(.secondary) }
                }
            }
            .font(.footnote)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(.primary.opacity(0.06), lineWidth: 1))
        .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 4)
    }

    private func relative(_ date: Date) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .short
        return f.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Tiny feedback helper
extension UIApplication {
    func prepareFeedback() {
        if #available(iOS 13.0, *) {
            let g = UINotificationFeedbackGenerator()
            g.notificationOccurred(.success)
        }
    }
}

// MARK: - Previews
struct CommunityBoardView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack {
                CommunityBoardView(accent: .blue)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .light)

            NavigationStack {
                CommunityBoardView(accent: .pink)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .dark)
        }
    }
}
