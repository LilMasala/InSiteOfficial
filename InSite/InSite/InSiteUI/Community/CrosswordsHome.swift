import SwiftUI

public struct CrosswordsHome: View {
    @EnvironmentObject private var themeManager: ThemeManager
    public var accent: Color
    @State private var tab = 1 // start on Community Maker by default

    public init(accent: Color) { self.accent = accent }

    public var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme).ignoresSafeArea()

            VStack(spacing: 0) {
                Picker("Mode", selection: $tab) {
                    Text("Standard").tag(0)
                    Text("Community Maker").tag(1)
                    Text("Daily Puzzle").tag(2)
                }
                .pickerStyle(.segmented)
                .padding()

                Group {
                    if tab == 0 {
                        StandardCrosswordPlaceholder(accent: accent)
                    } else if tab == 1 {
                        CrosswordMakerView(accent: accent)
                    } else {
                        DailyCrosswordView(accent: accent)
                    }
                }
                .padding(.horizontal, 12)
                .frame(maxWidth: 820)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Crosswords")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct StandardCrosswordPlaceholder: View {
    var accent: Color

    var body: some View {
        VStack(spacing: 18) {
            Spacer(minLength: 0)

            VStack(spacing: 18) {
                ZStack {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    accent.opacity(0.24),
                                    accent.opacity(0.08)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 88, height: 88)

                    Image(systemName: "square.grid.3x3.fill")
                        .font(.system(size: 34, weight: .semibold))
                        .foregroundStyle(accent)
                }

                VStack(spacing: 8) {
                    Text("Standard Crossword")
                        .font(.title3.weight(.bold))

                    Text("Coming soon")
                        .font(.caption.weight(.semibold))
                        .textCase(.uppercase)
                        .tracking(1.0)
                        .foregroundStyle(accent)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(accent.opacity(0.12), in: Capsule())

                    Text("The daily community puzzle is live now. A full standard crossword provider is still being finalized.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                VStack(alignment: .leading, spacing: 12) {
                    placeholderRow(
                        title: "What’s ready",
                        detail: "Create clues in Community Maker and play the curated daily puzzle."
                    )
                    placeholderRow(
                        title: "What’s pending",
                        detail: "Provider choice for larger standard crosswords."
                    )
                    placeholderRow(
                        title: "Likely next",
                        detail: "PuzzleMe integration or an internal generator, depending on product direction."
                    )
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(24)
            .frame(maxWidth: 520)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(.primary.opacity(0.08), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.08), radius: 18, x: 0, y: 10)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func placeholderRow(title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(accent.opacity(0.9))
                .frame(width: 8, height: 8)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(detail)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}



struct CrosswordsHome_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack {
                CrosswordsHome(accent: .blue)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .light)

            NavigationStack {
                CrosswordsHome(accent: .pink)
                    .environmentObject(ThemeManager())
            }
            .environment(\.colorScheme, .dark)
        }
    }
}
