import SwiftUI

/// A simple wrapping (flow) layout for SwiftUI (iOS 16+).
struct FlowLayout: Layout {
    var alignment: HorizontalAlignment = .leading
    var spacing: CGFloat = 12

    // Measure total size needed
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var currentRowWidth: CGFloat = 0
        var currentRowHeight: CGFloat = 0
        var totalHeight: CGFloat = 0

        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if currentRowWidth > 0, currentRowWidth + spacing + size.width > maxWidth {
                totalHeight += currentRowHeight + spacing
                currentRowWidth = 0
                currentRowHeight = 0
            }
            currentRowWidth += (currentRowWidth == 0 ? 0 : spacing) + size.width
            currentRowHeight = max(currentRowHeight, size.height)
        }
        totalHeight += currentRowHeight
        return CGSize(width: proposal.width ?? currentRowWidth, height: totalHeight)
    }

    // Place each subview
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let maxWidth = bounds.width
        var rows: [[(index: Int, size: CGSize)]] = [[]]
        var currentRowWidth: CGFloat = 0
        var currentRowHeight: CGFloat = 0

        // Build rows
        for (i, sub) in subviews.enumerated() {
            let size = sub.sizeThatFits(.unspecified)
            let nextWidth = (currentRowWidth == 0 ? size.width : currentRowWidth + spacing + size.width)
            if nextWidth > maxWidth, !rows.last!.isEmpty {
                rows.append([])
                currentRowWidth = 0
                currentRowHeight = 0
            }
            rows[rows.count - 1].append((i, size))
            currentRowWidth = (currentRowWidth == 0 ? size.width : currentRowWidth + spacing + size.width)
            currentRowHeight = max(currentRowHeight, size.height)
        }

        // Place rows with alignment
        var y: CGFloat = bounds.minY
        for row in rows where !row.isEmpty {
            let rowWidth = row.map(\.size.width).reduce(0, +) + CGFloat(row.count - 1) * spacing
            let startX: CGFloat
            switch alignment {
            case .leading:  startX = bounds.minX
            case .center:   startX = bounds.minX + (maxWidth - rowWidth) / 2
            case .trailing: startX = bounds.maxX - rowWidth
            default:        startX = bounds.minX
            }

            let rowHeight = row.map(\.size.height).max() ?? 0
            var x = startX
            for (i, size) in row {
                subviews[i].place(
                    at: CGPoint(x: x + size.width / 2, y: y + rowHeight / 2),
                    anchor: .center,
                    proposal: ProposedViewSize(width: size.width, height: size.height)
                )
                x += size.width + spacing
            }
            y += rowHeight + spacing
        }
    }
}
