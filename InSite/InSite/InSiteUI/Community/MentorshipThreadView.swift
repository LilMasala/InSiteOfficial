import SwiftUI

struct MentorshipThreadView: View {
    @EnvironmentObject private var themeManager: ThemeManager

    var body: some View {
        ZStack {
            BreathingBackground(theme: themeManager.theme).ignoresSafeArea()

            VStack(spacing: 18) {
                Spacer(minLength: 0)

                rolloutCard

                Spacer(minLength: 0)
            }
            .padding(20)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle("Mentorship")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var rolloutCard: some View {
        VStack(spacing: 18) {
            ZStack {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(themeManager.theme.accent.opacity(0.14))
                    .frame(width: 88, height: 88)

                Image(systemName: "person.2.wave.2.fill")
                    .font(.system(size: 34, weight: .semibold))
                    .foregroundStyle(themeManager.theme.accent)
            }

            VStack(spacing: 8) {
                Text("Mentorship")
                    .font(.title3.weight(.bold))

                Text("Coming Soon")
                    .font(.caption.weight(.semibold))
                    .textCase(.uppercase)
                    .tracking(1.0)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.secondary.opacity(0.12), in: Capsule())
                    .foregroundStyle(.secondary)

                Text("Peer support threads and matched guidance are staged for a later release. For now, the Community Board is the live place to ask questions and share what is working.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(.primary.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 18, x: 0, y: 10)
    }
}

#Preview {
    NavigationStack {
        MentorshipThreadView()
            .environmentObject(ThemeManager())
    }
}
