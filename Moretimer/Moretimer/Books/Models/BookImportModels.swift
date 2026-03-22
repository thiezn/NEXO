import Foundation

// MARK: - Codable structs mirroring Rust JSON output (deserialization only)

nonisolated struct BookOutputJSON: Codable, Sendable {
    let book: BookJSON
    let metadata: BookMetadataJSON
    let extraction: ExtractionMetadataJSON
}

nonisolated struct BookJSON: Codable, Sendable {
    let chapters: [ChapterJSON]
    let images: [ImageRefJSON]
}

nonisolated struct ChapterJSON: Codable, Sendable {
    let title: String
    let index: Int
    let paragraphs: [ParagraphJSON]
}

nonisolated struct ParagraphJSON: Codable, Sendable {
    let text: [String]
    let images: [ImageRefJSON]
}

nonisolated struct ImageRefJSON: Codable, Sendable {
    let path: String
    let id: String
    let mediaType: String
    let data: String?

    enum CodingKeys: String, CodingKey {
        case path, id, data
        case mediaType = "media_type"
    }
}

nonisolated struct BookMetadataJSON: Codable, Sendable {
    let title: String?
    let author: String?
    let publisher: String?
    let language: String?
    let identifier: String?
    let date: String?
    let rights: String?
}

nonisolated struct ExtractionMetadataJSON: Codable, Sendable {
    let extractedAt: String
    let extractionDurationMs: Int
    let chapterCount: Int
    let imageCount: Int
    let sourceFile: String

    enum CodingKeys: String, CodingKey {
        case extractedAt = "extracted_at"
        case extractionDurationMs = "extraction_duration_ms"
        case chapterCount = "chapter_count"
        case imageCount = "image_count"
        case sourceFile = "source_file"
    }
}
