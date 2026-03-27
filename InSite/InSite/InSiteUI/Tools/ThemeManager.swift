//
//  ThemeManager.swift
//  InSite
//
//  Created by Anand Parikh on 9/24/25.
//

import Foundation
import SwiftUI

// MARK: - THEME (inject mood color here later)
struct HomeTheme: Codable, Equatable {
    var accentR: Double
    var accentG: Double
    var accentB: Double
    var bgStartA: Double
    var bgEndA: Double

    // Convenience computed colors
    var accent: Color { Color(red: accentR, green: accentG, blue: accentB) }
    var onAccent: Color { // simple contrast
        // relative luminance
        let L = 0.2126*accentR + 0.7152*accentG + 0.0722*accentB
        return L > 0.6 ? .black : .white
    }
    var bgStart: Color { accent.opacity(bgStartA) }
    var bgEnd: Color { accent.opacity(bgEndA) }

    static let defaultTeal = HomeTheme(
        accentR: 0.22, accentG: 0.74, accentB: 0.70,
        bgStartA: 0.15, bgEndA: 0.06
    )
}


final class ThemeManager: ObservableObject {
    @Published var theme: HomeTheme = .defaultTeal {
        didSet { persist() }
    }

    private let key = "insite.theme.v1"

    init() { load() }

    func update(from mood: MoodPoint) {
        // Map your MoodPoint → Color using your existing hsbFor()
        let c = colorFor(mood) // defined below
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        UIColor(c).getRed(&r, green: &g, blue: &b, alpha: &a)
        theme = HomeTheme(
            accentR: r.double, accentG: g.double, accentB: b.double,
            bgStartA: 0.15, bgEndA: 0.06
        )
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(theme) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
    private func load() {
        if let data = UserDefaults.standard.data(forKey: key),
           let t = try? JSONDecoder().decode(HomeTheme.self, from: data) {
            theme = t
        }
    }
}

// Tiny helpers
private extension CGFloat { var double: Double { Double(self) } }


// Make these global so both MoodPicker & ThemeManager can use them
func hsbFor(_ p: MoodPoint) -> (h: Double, s: Double, b: Double) {
    let x = p.valence, y = p.arousal
    var hue = atan2(y, x) / (2 * .pi)
    hue = hue < 0 ? hue + 1 : hue
    let shiftedHue = fmod(hue + 0.20, 1.0)
    let r = min(1.0, max(0.0, Double(hypot(x, y))))
    let s = 0.25 + 0.65 * r
    let b = 0.85 - 0.25 * r
    return (shiftedHue, s, b)
}

func colorFor(_ p: MoodPoint) -> Color {
    let (h, s, b) = hsbFor(p)
    return Color(hue: h, saturation: s, brightness: b)
}
