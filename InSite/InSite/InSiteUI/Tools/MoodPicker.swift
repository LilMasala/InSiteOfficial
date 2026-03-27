import SwiftUI


//Codable means able to be converted to and from serialized representaitons
//equatable means that i can check wether or not two instances are equal or comparable using the ==
struct MoodPoint: Equatable, Codable {
    var valence: Double   // [-1, 1]  (pleasant ↔ unpleasant)
    var arousal: Double   // [-1, 1]  (energetic ↔ calm)
    var timestamp: Date = Date()
}

extension View {
    /// Success notification haptic when `trigger` changes (iOS 17+).
    @ViewBuilder
    func applySuccessHaptic(trigger: Int) -> some View {
        if #available(iOS 17.0, *) {
            self.sensoryFeedback(.success, trigger: trigger)
        } else {
            self
        }
    }
}

// Low-key ripple animation for backgrounds / wheels
struct RippleWaves: View {
    var color: Color
    var isActive: Bool = true          // you can drive this from isDragging, etc.
    var period: Double = 4.0           // seconds per ripple cycle
    var rippleCount: Int = 3           // overlapping waves

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        // Disable when Reduce Motion is on or not active
        if !isActive || reduceMotion {
            EmptyView()
        } else {
            TimelineView(.animation(minimumInterval: 1/30, paused: false)) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                Canvas { ctx, size in
                    let minSide = min(size.width, size.height)
                    let center = CGPoint(x: size.width/2, y: size.height/2)
                    let maxR = minSide * 0.48      // stay inside the wheel

                    // Draw N ripples with phase offsets
                    for i in 0..<rippleCount {
                        let phase = (t / period + Double(i) / Double(rippleCount)).truncatingRemainder(dividingBy: 1)
                        // Ease radius & fade (start small & opaque → large & transparent)
                        let eased = pow(phase, 0.8)           // tweak feel
                        let radius = maxR * CGFloat(eased)
                        let alpha = 0.18 * (1 - phase)        // fade out

                        var path = Path()
                        path.addEllipse(in: CGRect(
                            x: center.x - radius,
                            y: center.y - radius,
                            width:  radius * 2,
                            height: radius * 2
                        ))

                        ctx.stroke(path,
                                   with: .color(color.opacity(alpha)),
                                   lineWidth: max(1, minSide * 0.006))
                    }
                }
                .compositingGroup()
                .allowsHitTesting(false)
                .accessibilityHidden(true)
                .blur(radius: 0.5) // super subtle
                .animation(.linear(duration: period).repeatForever(autoreverses: false), value: t)
            }
        }
    }
}


struct MoodPicker: View {
    @Environment(\.dismiss) private var dismiss
    @State private var saveTick = 0          // for haptic
    @State private var showCheck = false     // checkmark pop
    @State private var showRipple = false    // expanding ring

    @EnvironmentObject private var themeManager: ThemeManager
    @State private var point = MoodPoint(valence: 0.35, arousal: 0.24)
    @State private var isDragging = false
    @State private var hapticTick = 0
    

    // Derived color from the 2D point
    private var moodColor: Color {
        colorFor(point)
    }
    private var textOnTint: Color {
        // Choose black/white by brightness for contrast
        let (_, _, b) = hsbFor(point)
        return b > 0.65 ? .black : .white
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("How do you feel?")
                .font(.title2.weight(.semibold))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
                .accessibilityAddTraits(.isHeader)

            MoodWheel(point: $point, isDragging: $isDragging, hapticTick: $hapticTick, accent: moodColor)
                .padding(.horizontal)

            // Readout
            VStack(spacing: 4) {
                Text(primaryLabel(for: point))
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.primary)
                Text(secondaryLabel(for: point))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            Button {
                // 1) persist theme
                themeManager.update(from: point)

                // 2) animate success
                withAnimation(.spring(response: 0.24, dampingFraction: 0.85)) {
                    showCheck = true
                }
                withAnimation(.easeOut(duration: 0.35)) {
                    showRipple = true
                }
                saveTick &+= 1  // haptic

                // 3) dismiss shortly after the pop starts
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) {
                    dismiss()
                    // tidy up for next time (in case user comes back)
                    showCheck = false
                    showRipple = false
                }
            } label: {
                Text("Save mood")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(moodColor)
            .foregroundStyle(textOnTint)
            .padding(.horizontal)
            .applySuccessHaptic(trigger: saveTick)
            .overlay(alignment: .center) {
                ZStack {
                    // Ripple ring
                    if showRipple {
                        Circle()
                            .stroke(moodColor.opacity(0.55), lineWidth: 2)
                            .scaleEffect(1.25)
                            .opacity(0)
                            .transition(.identity) // driven entirely by explicit animation above
                            .onAppear {
                                // start small & visible, then we animate to big & transparent
                            }
                    }
                    // Checkmark pop
                    if showCheck {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundStyle(textOnTint)
                            .padding(.vertical, 4)
                            .background(
                                Circle()
                                    .fill(moodColor)
                                    .shadow(radius: 6, y: 2)
                            )
                            .scaleEffect(1.0)
                            .transition(.scale.combined(with: .opacity))
                    }
                }
            }
        }
        .padding(.top, 24)
        .background(backgroundFor(point).ignoresSafeArea())
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .principal) { Text("Mood").font(.headline) } }
        .animation(.easeInOut(duration: 0.25), value: moodColor)  // smooth color updates
    }

    // MARK: - Labels

    private func primaryLabel(for p: MoodPoint) -> String {
        let r = hypot(p.valence, p.arousal)
        let strong = r > 0.75
        switch (p.valence, p.arousal) {
        case (let x, let y) where x >= 0 && y >= 0: return strong ? "Excited" : "Pleasant"
        case (let x, let y) where x >= 0 && y  < 0: return strong ? "Calm"    : "Content"
        case (let x, let y) where x  < 0 && y >= 0: return strong ? "Tense"   : "Uneasy"
        default:                                    return strong ? "Sad"     : "Down"
        }
    }

    private func secondaryLabel(for p: MoodPoint) -> String {
        "Pleasantness \(Int((p.valence * 100).rounded())) • Energy \(Int((p.arousal * 100).rounded()))"
    }

    // MARK: - Color Mapping

    /// Convert (x,y) → (hue,sat,bright)
    /// - Hue: from angle (atan2), shifted so right/pleasant ≈ green/teal, left/unpleasant ≈ magenta/red.
    /// - Sat: increases with radius (stronger feelings = richer color).
    /// - Bright: slightly higher near center to keep labels readable.
    private func hsbFor(_ p: MoodPoint) -> (h: Double, s: Double, b: Double) {
        let x = p.valence
        let y = p.arousal
        let angle = atan2(y, x)           // -π...π (0 at +x)
        var hue = (angle / (2 * .pi))     // -0.5...0.5
        hue = hue < 0 ? hue + 1 : hue     // 0...1

        // Shift hue so +x (pleasant) ~ 0.45 (teal-green), -x (unpleasant) ~ 0.9 (magenta)
        let shiftedHue = fmod(hue + 0.20, 1.0)

        // Radius 0..1
        let r = min(1.0, max(0.0, Double(hypot(x, y))))
        // Saturation grows with radius; keep a floor for gentle color even near center
        let s = 0.25 + 0.65 * r
        // Brightness slightly higher near center (so text remains readable)
        let b = 0.85 - 0.25 * r

        return (shiftedHue, s, b)
    }

    private func colorFor(_ p: MoodPoint) -> Color {
        let (h, s, b) = hsbFor(p)
        return Color(hue: h, saturation: s, brightness: b)
    }

    private func backgroundFor(_ p: MoodPoint) -> some View {
        let c = colorFor(p)
        return RadialGradient(
            gradient: Gradient(colors: [
                c.opacity(0.15),
                c.opacity(0.06)
            ]),
            center: .center,
            startRadius: 0,
            endRadius: 700
        )
    }
}

// MARK: - Wheel

private struct MoodWheel: View {
    @Binding var point: MoodPoint
    @Binding var isDragging: Bool
    @Binding var hapticTick: Int
    let accent: Color

    @ScaledMetric(relativeTo: .title) private var knobSize: CGFloat = 28
    @ScaledMetric private var gridStroke: CGFloat = 1

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            let radius = size / 2

            ZStack {
                Circle()
                    .fill(backgroundGradient(size: size))
                    .overlay(crosshairs)
                    .overlay(rings(count: 4))
                    .overlay(glow(size: size))
                
                RippleWaves(color: accent, isActive: true)

                

                knob
                    .position(knobPosition(in: radius, center: CGPoint(x: geo.size.width/2, y: geo.size.height/2)))
            }
            .frame(width: size, height: size)
            .contentShape(Circle())
            .gesture(dragGesture(in: radius, center: CGPoint(x: geo.size.width/2, y: geo.size.height/2)))
            .animation(.spring(response: 0.25, dampingFraction: 0.8), value: isDragging)
            .applyImpactHaptic(trigger: hapticTick)
        }
        .aspectRatio(1, contentMode: .fit)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Mood wheel")
        .accessibilityValue("Pleasantness \(Int(point.valence*100)), Energy \(Int(point.arousal*100))")
    }

    private func backgroundGradient(size: CGFloat) -> RadialGradient {
        RadialGradient(
            gradient: Gradient(colors: [accent.opacity(0.10), accent.opacity(0.20)]),
            center: .center,
            startRadius: size * 0.05,
            endRadius: size * 0.55
        )
    }

    private var crosshairs: some View {
        GeometryReader { geo in
            let s = min(geo.size.width, geo.size.height)
            Path { path in
                path.move(to: CGPoint(x: s/2, y: 0))
                path.addLine(to: CGPoint(x: s/2, y: s))
                path.move(to: CGPoint(x: 0, y: s/2))
                path.addLine(to: CGPoint(x: s, y: s/2))
            }
            .stroke(Color.primary.opacity(0.08), lineWidth: gridStroke)
        }
    }

    private func rings(count: Int) -> some View {
        GeometryReader { geo in
            let s = min(geo.size.width, geo.size.height)
            ZStack {
                ForEach(1...count, id: \.self) { i in
                    Circle()
                        .stroke(Color.primary.opacity(0.06), lineWidth: gridStroke)
                        .frame(width: s * CGFloat(i) / CGFloat(count),
                               height: s * CGFloat(i) / CGFloat(count))
                }
            }
        }
    }

    private func glow(size: CGFloat) -> some View {
        Circle()
            .stroke(accent.opacity(0.35), lineWidth: 6)
            .blur(radius: 10)
            .padding(6)
    }

    private var knob: some View {
        ZStack {
            Circle().fill(.ultraThinMaterial)
                .overlay(Circle().stroke(.white.opacity(0.9), lineWidth: 1))
            Circle().stroke(accent, lineWidth: 2).padding(2)
        }
        .frame(width: knobSize, height: knobSize)
        .shadow(color: .black.opacity(isDragging ? 0.25 : 0.12), radius: isDragging ? 6 : 3, x: 0, y: 2)
        .accessibilityHidden(true)
    }

    private func knobPosition(in radius: CGFloat, center: CGPoint) -> CGPoint {
        let x = center.x + CGFloat(point.valence) * radius * 0.9
        let y = center.y - CGFloat(point.arousal) * radius * 0.9
        return CGPoint(x: x, y: y)
    }

    private func dragGesture(in radius: CGFloat, center: CGPoint) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                isDragging = true
                let v = CGVector(dx: value.location.x - center.x, dy: center.y - value.location.y)
                let clamped = clampToCircle(v, max: radius * 0.9)
                let normalized = CGVector(dx: max(-1, min(1, clamped.dx / (radius * 0.9))),
                                          dy: max(-1, min(1, clamped.dy / (radius * 0.9))))
                point.valence = Double(normalized.dx)
                point.arousal = Double(normalized.dy)
                hapticTick &+= 1
            }
            .onEnded { _ in isDragging = false }
    }

    private func clampToCircle(_ v: CGVector, max r: CGFloat) -> CGVector {
        let mag = sqrt(v.dx*v.dx + v.dy*v.dy)
        guard mag > r && mag > 0 else { return v }
        let scale = r / mag
        return CGVector(dx: v.dx * scale, dy: v.dy * scale)
    }
}

// MARK: - iOS 17 haptics (safe on older iOS)

private extension View {
    @ViewBuilder
    func applyImpactHaptic(trigger: Int) -> some View {
        if #available(iOS 17.0, *) {
            self.sensoryFeedback(.impact(flexibility: .soft, intensity: 0.7), trigger: trigger)
        } else {
            self
        }
    }
}

#Preview {
  NavigationStack { MoodPicker() }
    .environmentObject(ThemeManager())
}
