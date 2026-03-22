import SwiftUI

struct SmallCard: View {
    let imageData: Data?
    var placeholderIcon: String = AppIcon.threads
    let title: String
    var subtitle: String?
    var tint: Color = .clear
    var size: CGFloat = 120

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            imageContent
            textContent
        }
        .frame(width: size)
    }

    @ViewBuilder
    private var imageContent: some View {
        Group {
            if let data = imageData {
                imageFromData(data, contentMode: .fill)
                    .frame(width: size, height: size)
                    .clipped()
            } else {
                Image(systemName: placeholderIcon)
                    .font(.system(size: 28))
                    .foregroundStyle(.secondary)
                    .frame(width: size, height: size)
            }
        }
        .clipShape(.rect(cornerRadius: 12))
        .glassEffect(.regular.tint(tint.opacity(0.15)), in: .rect(cornerRadius: 12))
    }

    private var textContent: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)

            if let subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }
}
