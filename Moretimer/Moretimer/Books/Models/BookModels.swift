import Foundation
import SwiftUI
import SwiftData

@Model
final class BookEntity {
    var title: String
    var author: String
    var publisher: String?
    var language: String?
    var identifier: String?
    var date: String?
    var rights: String?
    var sourceFile: String
    var importedAt: Date
    var chapterCount: Int
    var imageCount: Int

    @Relationship(deleteRule: .cascade, inverse: \ChapterEntity.book)
    var chapters: [ChapterEntity] = []

    @Relationship(deleteRule: .cascade, inverse: \ImageRefEntity.book)
    var images: [ImageRefEntity] = []

    @Relationship(deleteRule: .cascade, inverse: \ReadingProgress.book)
    var readingProgress: ReadingProgress?

    var sortedChapters: [ChapterEntity] {
        chapters.sorted { $0.index < $1.index }
    }

    var coverImage: ImageRefEntity? {
        images.first(where: {
            $0.path.localizedCaseInsensitiveContains("cover") ||
            $0.imageId.localizedCaseInsensitiveContains("cover")
        }) ?? images.first
    }

    init(
        title: String,
        author: String,
        publisher: String? = nil,
        language: String? = nil,
        identifier: String? = nil,
        date: String? = nil,
        rights: String? = nil,
        sourceFile: String,
        chapterCount: Int,
        imageCount: Int
    ) {
        self.title = title
        self.author = author
        self.publisher = publisher
        self.language = language
        self.identifier = identifier
        self.date = date
        self.rights = rights
        self.sourceFile = sourceFile
        self.importedAt = Date()
        self.chapterCount = chapterCount
        self.imageCount = imageCount
    }
}

@Model
final class ChapterEntity {
    var title: String
    var index: Int
    var book: BookEntity?

    @Relationship(deleteRule: .cascade, inverse: \ParagraphEntity.chapter)
    var paragraphs: [ParagraphEntity] = []

    var sortedParagraphs: [ParagraphEntity] {
        paragraphs.sorted { $0.order < $1.order }
    }

    /// Paragraphs excluding the first one if it duplicates the chapter title.
    var displayParagraphs: [ParagraphEntity] {
        let sorted = paragraphs.sorted { $0.order < $1.order }
        guard let first = sorted.first else { return sorted }
        let firstText = normalizeWhitespace(first.fullText)
        let titleText = normalizeWhitespace(title)
        if firstText == titleText {
            return Array(sorted.dropFirst())
        }
        return sorted
    }

    init(title: String, index: Int) {
        self.title = title
        self.index = index
    }
}

@Model
final class ParagraphEntity {
    var textSegments: [String]
    var order: Int
    var chapter: ChapterEntity?

    @Relationship(deleteRule: .cascade, inverse: \ImageRefEntity.paragraph)
    var images: [ImageRefEntity] = []

    var fullText: String {
        textSegments.joined()
    }

    init(textSegments: [String], order: Int) {
        self.textSegments = textSegments
        self.order = order
    }
}

@Model
final class ImageRefEntity {
    var path: String
    var imageId: String
    var mediaType: String
    @Attribute(.externalStorage) var imageData: Data?

    var book: BookEntity?
    var paragraph: ParagraphEntity?

    init(path: String, imageId: String, mediaType: String, imageData: Data? = nil) {
        self.path = path
        self.imageId = imageId
        self.mediaType = mediaType
        self.imageData = imageData
    }

    func toJSON() -> ImageRefJSON {
        ImageRefJSON(
            path: path,
            id: imageId,
            mediaType: mediaType,
            data: imageData?.base64EncodedString() ?? ""
        )
    }
}

@Model
final class ReadingProgress {
    var currentChapterIndex: Int
    var currentParagraphIndex: Int
    var lastReadAt: Date
    var book: BookEntity?

    var percentage: Double {
        guard let book, book.chapterCount > 0 else { return 0 }
        let totalParagraphs = book.chapters.reduce(0) { $0 + $1.paragraphs.count }
        guard totalParagraphs > 0 else { return 0 }

        var paragraphsBefore = 0
        for chapter in book.sortedChapters {
            if chapter.index < currentChapterIndex {
                paragraphsBefore += chapter.paragraphs.count
            } else if chapter.index == currentChapterIndex {
                paragraphsBefore += min(currentParagraphIndex, chapter.paragraphs.count)
                break
            }
        }
        return Double(paragraphsBefore) / Double(totalParagraphs) * 100
    }

    init(currentChapterIndex: Int = 0, currentParagraphIndex: Int = 0) {
        self.currentChapterIndex = currentChapterIndex
        self.currentParagraphIndex = currentParagraphIndex
        self.lastReadAt = Date()
    }
}

// MARK: - Export

extension BookEntity {
    func toJSON() -> BookOutputJSON {
        let imageRefs = images.map { $0.toJSON() }

        let chapters = sortedChapters.map { chapter in
            ChapterJSON(
                title: chapter.title,
                index: chapter.index,
                paragraphs: chapter.sortedParagraphs.map { para in
                    ParagraphJSON(
                        text: para.textSegments,
                        images: para.images.map { $0.toJSON() }
                    )
                }
            )
        }

        return BookOutputJSON(
            book: BookJSON(chapters: chapters, images: imageRefs),
            metadata: BookMetadataJSON(
                title: title,
                author: author,
                publisher: publisher,
                language: language,
                identifier: identifier,
                date: date,
                rights: rights
            ),
            extraction: ExtractionMetadataJSON(
                extractedAt: ISO8601DateFormatter().string(from: importedAt),
                extractionDurationMs: 0,
                chapterCount: chapterCount,
                imageCount: imageCount,
                sourceFile: sourceFile
            )
        )
    }
}

// MARK: - AppStorage Keys

enum BookReaderKeys {
    static let fontSize = "bookReaderFontSize"
    static let mode = "bookReaderMode"
    static let fontDesign = "bookReaderFontDesign"
}

// MARK: - Helpers

nonisolated private func normalizeWhitespace(_ string: String) -> String {
    string.components(separatedBy: .whitespacesAndNewlines)
        .filter { !$0.isEmpty }
        .joined(separator: " ")
}

// MARK: - Image from Data

@ViewBuilder
func imageFromData(_ data: Data, contentMode: ContentMode = .fit) -> some View {
    #if canImport(UIKit)
    if let img = UIImage(data: data) {
        Image(uiImage: img)
            .resizable()
            .aspectRatio(contentMode: contentMode)
    }
    #elseif canImport(AppKit)
    if let img = NSImage(data: data) {
        Image(nsImage: img)
            .resizable()
            .aspectRatio(contentMode: contentMode)
    }
    #endif
}
