//
//  FloatPulseModifier.swift
//  InSite
//
//  Created by Anand Parikh on 9/24/25.
//

import Foundation
import SwiftUI

struct FloatPulseModifier: ViewModifier {
    var seed: Double          // use a different seed per tile
    var amplitude: CGFloat = 6
    var pulse: CGFloat = 0.015
    var period: Double = 6.0  // seconds

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var t: TimeInterval = 0

    func body(content: Content) -> some View {
        TimelineView(.animation) { timeline in
            let now = timeline.date.timeIntervalSinceReferenceDate
            let dt = reduceMotion ? 0 : now + seed * 123.0

            let x = amplitude * CGFloat(sin((dt / period) * .pi * 2))
            let y = amplitude * CGFloat(cos((dt / (period * 1.3)) * .pi * 2))
            let s = 1 + (reduceMotion ? 0 : pulse * CGFloat(sin((dt / (period * 0.9)) * .pi * 2)))

            content
                .offset(x: x, y: y)
                .scaleEffect(s)
        }
    }
}

extension View {
    func floatAndPulse(seed: Double, amplitude: CGFloat = 6, pulse: CGFloat = 0.015, period: Double = 6) -> some View {
        modifier(FloatPulseModifier(seed: seed, amplitude: amplitude, pulse: pulse, period: period))
    }
}
