//
//  PeekContainer.swift
//  InSite
//
//  Created by You on 9/24/25.
//

import SwiftUI

/// Wrap any tile/content to enable a long-press "peek" that shows a magnified preview.
/// - Long press (~0.28s) to peek
/// - Tap anywhere outside to dismiss
/// - The `peek` view is your actual destination UI rendered inside a glassy card.
struct PeekContainer<Content: View, Peek: View>: View {
    // MARK: Inputs
    private let content: () -> Content
    private let peek: () -> Peek

    // MARK: Config (tweak if needed)
    private let peekCorner: CGFloat
    private let peekSize: CGSize?          // nil = size to fit
    private let backdropOpacity: Double

    // MARK: State
    @State private var isPeeking = false
    @GestureState private var isPressing = false

    // MARK: Init
    init(
        peekCorner: CGFloat = 28,
        peekSize: CGSize? = CGSize(width: 280, height: 280),
        backdropOpacity: Double = 0.15,
        @ViewBuilder content: @escaping () -> Content,
        @ViewBuilder peek: @escaping () -> Peek
    ) {
        self.peekCorner = peekCorner
        self.peekSize = peekSize
        self.backdropOpacity = backdropOpacity
        self.content = content
        self.peek = peek
    }

    var body: some View {
        content()
            // Press hint ring
            .overlay(alignment: .center) {
                if isPressing && !isPeeking {
                    Circle()
                        .strokeBorder(.secondary.opacity(0.30), lineWidth: 2)
                        .frame(width: 56, height: 56)
                        .scaleEffect(1.12)
                        .opacity(0.7)
                        .transition(.opacity)
                        .animation(.easeOut(duration: 0.2), value: isPressing)
                }
            }
            // Long press to open
            .gesture(
                LongPressGesture(minimumDuration: 0.28)
                    .updating($isPressing) { _, state, _ in state = true }
                    .onEnded { _ in
                        withAnimation(.spring(response: 0.28, dampingFraction: 0.9)) {
                            isPeeking = true
                        }
                    }
            )
            // Peek overlay
            .overlay {
                if isPeeking {
                    ZStack {
                        // 1) Backdrop that eats taps (tap outside to dismiss)
                        Color.black.opacity(backdropOpacity)
                            .ignoresSafeArea()
                            .contentShape(Rectangle())
                            .onTapGesture {
                                withAnimation(.spring(response: 0.28, dampingFraction: 0.95)) {
                                    isPeeking = false
                                }
                            }

                        // 2) Magnifier card with your destination UI
                        Group {
                            peek()
                                .clipShape(RoundedRectangle(cornerRadius: peekCorner, style: .continuous))
                                .overlay(
                                    RoundedRectangle(cornerRadius: peekCorner, style: .continuous)
                                        .stroke(.white.opacity(0.25), lineWidth: 1)
                                )
                                .background(
                                    RoundedRectangle(cornerRadius: peekCorner, style: .continuous)
                                        .fill(.ultraThinMaterial)
                                )
                                .shadow(color: .black.opacity(0.25), radius: 20, x: 0, y: 10)
                        }
                        .frame(width: peekSize?.width, height: peekSize?.height)
                        .padding(16)
                        .transition(.scale.combined(with: .opacity))
                        .onTapGesture {
                            // Taps inside do nothing (lets you add buttons/links in the peek)
                        }
                    }
                    .animation(.spring(response: 0.32, dampingFraction: 0.9), value: isPeeking)
                    .applyPeekHaptic(trigger: isPeeking)  // soft haptic on open (iOS 17+)
                }
            }
    }
}

// MARK: - iOS 17+ haptic helper (no-op on earlier iOS)
private extension View {
    @ViewBuilder
    func applyPeekHaptic(trigger: Bool) -> some View {
        if #available(iOS 17.0, *) {
            self.sensoryFeedback(.impact(flexibility: .soft, intensity: 0.85), trigger: trigger)
        } else {
            self
        }
    }
}
