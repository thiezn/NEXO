import Foundation
import OSLog
import SwiftData

struct BookImportService {

    @MainActor
    static func importFromData(
        _ data: Data,
        into modelContext: ModelContext
    ) throws -> BookEntity {
        let decoded: BookOutputJSON
        do {
            decoded = try JSONDecoder().decode(BookOutputJSON.self, from: data)
        } catch {
            Logger.book.error("JSON decode failed: \(error, privacy: .public)")
            throw error
        }
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
            forResource: "mzzl_meiden",
            withExtension: "json"
        ) else {
            logBundleDiagnostics()
            throw ImportError.bundledBookNotFound
        }
        Logger.book.info("Found bundled book at: \(jsonURL.path, privacy: .public)")
        return try importBook(from: jsonURL, into: modelContext)
    }

    private static func logBundleDiagnostics() {
        let bundle = Bundle.main
        Logger.book.error("Failed to find mzzl_meiden.json in bundle")
        Logger.book.error("Bundle path: \(bundle.bundlePath, privacy: .public)")
        Logger.book.error("Resource path: \(bundle.resourcePath ?? "nil", privacy: .public)")
        Logger.book.error("Bundle identifier: \(bundle.bundleIdentifier ?? "nil", privacy: .public)")

        guard let resourcePath = bundle.resourcePath else { return }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else {
            Logger.book.error("Could not list bundle resource directory")
            return
        }

        let jsonFiles = items.filter { $0.hasSuffix(".json") }
        Logger.book.error("JSON files in bundle root: \(jsonFiles, privacy: .public)")

        let directories = items.filter { item in
            var isDir: ObjCBool = false
            fm.fileExists(atPath: resourcePath + "/" + item, isDirectory: &isDir)
            return isDir.boolValue
        }
        Logger.book.error("Directories in bundle: \(directories, privacy: .public)")

        for dir in directories {
            if let subItems = try? fm.contentsOfDirectory(atPath: resourcePath + "/" + dir) {
                let subJson = subItems.filter { $0.hasSuffix(".json") }
                if !subJson.isEmpty {
                    Logger.book.error("JSON in \(dir, privacy: .public)/: \(subJson, privacy: .public)")
                }
            }
        }

        Logger.book.error("All bundle items (\(items.count) total): \(items, privacy: .public)")
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
            let imageData = imageJSON.data.flatMap { Data(base64Encoded: $0) }
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
                    let imgData = imageDataCache[imgJSON.path] ?? imgJSON.data.flatMap { Data(base64Encoded: $0) }
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
