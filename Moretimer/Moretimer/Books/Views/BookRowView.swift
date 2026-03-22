import SwiftUI

struct BookRowView: View {
    let book: BookEntity

    var body: some View {
        HStack(spacing: 12) {
            bookCoverImage
                .frame(width: 50, height: 70)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 4) {
                Text(book.title)
                    .font(.headline)
                Text(book.author)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let progress = book.readingProgress {
                    ProgressView(value: progress.percentage, total: 100)
                        .tint(.accentColor)
                    Text("\(Int(progress.percentage))% read")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var bookCoverImage: some View {
        if let firstImage = book.images.first,
           let data = firstImage.imageData {
            imageFromData(data, contentMode: .fill)
        } else {
            RoundedRectangle(cornerRadius: 6)
                .fill(.quaternary)
                .overlay {
                    Image(systemName: AppIcon.book)
                        .foregroundStyle(.secondary)
                }
        }
    }
}
