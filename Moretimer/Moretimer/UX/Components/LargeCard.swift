import SwiftUI

struct LargeCard: View {
    let imageData: Data?
    var placeholderIcon: String = AppIcon.bookFilled
    var subtext: String?
    let title: String
    var description: String?
    var tint: Color = .clear
    var width: CGFloat = 160
    var height: CGFloat = 240

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            imageContent
            textOverlay
        }
        .frame(width: width, height: height)
        .clipShape(.rect(cornerRadius: 16))
        .glassEffect(.regular.tint(tint.opacity(0.15)), in: .rect(cornerRadius: 16))
    }

    @ViewBuilder
    private var imageContent: some View {
        if let data = imageData {
            imageFromData(data, contentMode: .fill)
                .frame(width: width, height: height)
                .clipped()
        } else {
            Image(systemName: placeholderIcon)
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
                .frame(width: width, height: height)
        }
    }

    private var textOverlay: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let subtext {
                Text(subtext)
                    .font(.caption)
                    .foregroundStyle(.white.secondary)
            }

            Text(title)
                .font(.headline)
                .foregroundStyle(.white)
                .lineLimit(2)

            if let description {
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.8))
                    .lineLimit(1)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.6)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }
}
