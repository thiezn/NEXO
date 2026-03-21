import Foundation
import SwiftData

struct BookImportService {

    @MainActor
    static func importFromData(
        _ data: Data,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        let decoded = try JSONDecoder().decode(BookOutputJSON.self, from: data)
        return try importFromJSON(decoded, into: modelContext)
    }

    @MainActor
    static func importFromJSON(
        _ output: BookOutputJSON,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        return try mapToEntities(output, into: modelContext)
    }

    @MainActor
    static func importBook(
        from jsonURL: URL,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        let data = try Data(contentsOf: jsonURL)
        return try importFromData(data, into: modelContext)
    }

    @MainActor
    static func importBundledBook(into modelContext: ModelContext) throws -> BookEntity {
        guard let jsonURL = Bundle.main.url(
            forResource: "book",
            withExtension: "json",
            subdirectory: "mzzl_meiden"
        ) else {
            throw ImportError.bundledBookNotFound
        }
        return try importBook(from: jsonURL, into: modelContext)
    }

    @MainActor
    static func importFromFilePicker(
        url: URL,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        guard url.startAccessingSecurityScopedResource() else {
            throw ImportError.accessDenied
        }
        defer { url.stopAccessingSecurityScopedResource() }
        return try importBook(from: url, into: modelContext)
    }

    @MainActor
    private static func mapToEntities(
        _ output: BookOutputJSON,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        let meta = output.metadata

        let bookEntity = BookEntity(
            title: meta.title ?? "Untitled",
            author: meta.author ?? "Unknown",
            publisher: meta.publisher,
            language: meta.language,
            identifier: meta.identifier,
            date: meta.date,
            rights: meta.rights,
            sourceFile: output.extraction.sourceFile,
            chapterCount: output.extraction.chapterCount,
            imageCount: output.extraction.imageCount
        )
        modelContext.insert(bookEntity)

        // Build image data cache from book-level catalog (base64 decode)
        var imageDataCache: [String: Data] = [:]

        for imageJSON in output.book.images {
            let imageData = Data(base64Encoded: imageJSON.data)
            let imageEntity = ImageRefEntity(
                path: imageJSON.path,
                imageId: imageJSON.id,
                mediaType: imageJSON.mediaType,
                imageData: imageData
            )
            if let imageData {
                imageDataCache[imageJSON.path] = imageData
            }
            imageEntity.book = bookEntity
            modelContext.insert(imageEntity)
        }

        // Import chapters and paragraphs
        for chapterJSON in output.book.chapters.sorted(by: { $0.index < $1.index }) {
            let chapterEntity = ChapterEntity(title: chapterJSON.title, index: chapterJSON.index)
            chapterEntity.book = bookEntity
            modelContext.insert(chapterEntity)

            for (pIdx, paraJSON) in chapterJSON.paragraphs.enumerated() {
                let paraEntity = ParagraphEntity(textSegments: paraJSON.text, order: pIdx)
                paraEntity.chapter = chapterEntity
                modelContext.insert(paraEntity)

                for imgJSON in paraJSON.images {
                    let imgData = imageDataCache[imgJSON.path] ?? Data(base64Encoded: imgJSON.data)
                    let imgEntity = ImageRefEntity(
                        path: imgJSON.path,
                        imageId: imgJSON.id,
                        mediaType: imgJSON.mediaType,
                        imageData: imgData
                    )
                    imgEntity.paragraph = paraEntity
                    modelContext.insert(imgEntity)
                }
            }
        }

        // Create initial reading progress
        let progress = ReadingProgress()
        progress.book = bookEntity
        modelContext.insert(progress)

        try modelContext.save()
        return bookEntity
    }

    enum ImportError: LocalizedError {
        case bundledBookNotFound
        case accessDenied

        var errorDescription: String? {
            switch self {
            case .bundledBookNotFound: "Bundled book resource not found in app bundle"
            case .accessDenied: "Could not access the selected file"
            }
        }
    }
}
