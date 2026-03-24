import SwiftUI
import SwiftData

struct BookReaderView: View {
    let book: BookEntity
    @Environment(\.modelContext) private var modelContext
    @State private var scrollPosition: ScrollPosition
    @State private var showSettings = false
    @State private var currentTitle: String
    @State private var saveTask: Task<Void, Never>?
    @State private var cachedPages: [BookPage] = []
    @State private var lastPageSize: CGSize = .zero

    @AppStorage(BookReaderKeys.fontSize) private var fontSize: Double = 17
    @AppStorage(BookReaderKeys.mode) private var readingMode: ReadingMode = .continuous
    @AppStorage(BookReaderKeys.fontDesign) private var fontDesignOption: FontDesignOption = .default

    init(book: BookEntity) {
        self.book = book
        self._currentTitle = State(initialValue: book.title)

        if let progress = book.readingProgress,
           let chapter = book.sortedChapters.first(where: { $0.index == progress.currentChapterIndex }) {
            let paras = chapter.sortedParagraphs
            if progress.currentParagraphIndex < paras.count {
                self._scrollPosition = State(initialValue: ScrollPosition(
                    id: paras[progress.currentParagraphIndex].persistentModelID,
                    anchor: .top
                ))
                return
            }
        }
        self._scrollPosition = State(initialValue: ScrollPosition())
    }

    var body: some View {
        Group {
            switch readingMode {
            case .continuous:
                continuousView
            case .paged:
                pagedView
            }
        }
        .navigationTitle(currentTitle)
        .toolbarTitleDisplayMode(.inline)
        .toolbar {
            DetailToolbarContent(
                primaryAction: MenuAction(title: "Settings", icon: AppIcon.textSettings) {
                    showSettings = true
                }
            )
        }
        .sheet(isPresented: $showSettings) {
            BookSettingsSheet(
                readingMode: $readingMode,
                fontSize: $fontSize,
                fontDesignOption: $fontDesignOption
            )
            .presentationDetents([.medium])
        }
        .onDisappear { saveProgress() }
    }

    // MARK: - Continuous Mode

    private var continuousView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach(book.sortedChapters) { chapter in
                    ChapterView(
                        chapter: chapter,
                        fontSize: fontSize,
                        fontDesign: fontDesignOption.fontDesign
                    )
                }
            }
            .scrollTargetLayout()
            .padding(.horizontal)
        }
        .scrollPosition($scrollPosition)
        .onScrollTargetVisibilityChange(idType: PersistentIdentifier.self) { ids in
            guard let firstVisibleID = ids.first else { return }
            onVisibleParagraphChanged(firstVisibleID)
        }
    }

    // MARK: - Paged Mode

    private var pagedView: some View {
        GeometryReader { geo in
            let size = geo.size
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(cachedPages) { page in
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(page.items) { item in
                                switch item.kind {
                                case .chapterTitle(let title):
                                    Text(title)
                                        .font(.system(.title2, design: fontDesignOption.fontDesign))
                                        .fontWeight(.bold)
                                        .padding(.top, 8)
                                case .text(let text):
                                    Text(text)
                                        .font(.system(size: fontSize, design: fontDesignOption.fontDesign))
                                        .lineSpacing(fontSize * 0.4)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                        .containerRelativeFrame(.vertical, alignment: .top)
                    }
                }
                .scrollTargetLayout()
            }
            .scrollTargetBehavior(.paging)
            .scrollPosition($scrollPosition)
            .onChange(of: size) { recomputePages(for: size) }
            .onChange(of: fontSize) { recomputePages(for: size) }
            .onChange(of: fontDesignOption) { recomputePages(for: size) }
            .onAppear { recomputePages(for: size) }
        }
    }

    private func recomputePages(for size: CGSize) {
        guard size.height > 0, size.width > 0 else { return }
        cachedPages = computePages(
            pageHeight: size.height,
            pageWidth: size.width - 32
        )
    }

    // MARK: - Page Computation

    private func computePages(pageHeight: CGFloat, pageWidth: CGFloat) -> [BookPage] {
        var pages: [BookPage] = []
        var currentItems: [PageItem] = []
        var currentHeight: CGFloat = 0
        let verticalSpacing: CGFloat = 12
        let titleTopPadding: CGFloat = 8
        var globalItemIndex = 0

        func flushPage() {
            if !currentItems.isEmpty {
                pages.append(BookPage(index: pages.count, items: currentItems))
                currentItems = []
                currentHeight = 0
            }
        }

        func addItem(_ item: PageItem, height: CGFloat) {
            let spacingNeeded = currentItems.isEmpty ? 0 : verticalSpacing
            if currentHeight + spacingNeeded + height > pageHeight && !currentItems.isEmpty {
                flushPage()
            }
            let spacing = currentItems.isEmpty ? 0 : verticalSpacing
            currentHeight += spacing + height
            currentItems.append(item)
        }

        #if canImport(UIKit)
        let titleFont = UIFont.systemFont(ofSize: UIFont.preferredFont(forTextStyle: .title2).pointSize, weight: .bold)
        let bodyFont = UIFont.systemFont(ofSize: fontSize)
        let bodyLineSpacing = fontSize * 0.4

        for chapter in book.sortedChapters {
            flushPage()
            let titleHeight = measureText(chapter.title, font: titleFont, lineSpacing: 0, width: pageWidth) + titleTopPadding
            let titleItem = PageItem(index: globalItemIndex, kind: .chapterTitle(chapter.title))
            globalItemIndex += 1
            addItem(titleItem, height: titleHeight)

            for paragraph in chapter.displayParagraphs {
                let text = paragraph.fullText
                let textHeight = measureText(text, font: bodyFont, lineSpacing: bodyLineSpacing, width: pageWidth)

                if textHeight <= pageHeight {
                    let item = PageItem(index: globalItemIndex, kind: .text(text), paragraphID: paragraph.persistentModelID)
                    globalItemIndex += 1
                    addItem(item, height: textHeight)
                } else {
                    let chunks = splitTextToFit(text, maxHeight: pageHeight, font: bodyFont, lineSpacing: bodyLineSpacing, width: pageWidth)
                    for (i, chunk) in chunks.enumerated() {
                        let chunkHeight = measureText(chunk, font: bodyFont, lineSpacing: bodyLineSpacing, width: pageWidth)
                        let id = i == 0 ? paragraph.persistentModelID : nil
                        let item = PageItem(index: globalItemIndex, kind: .text(chunk), paragraphID: id)
                        globalItemIndex += 1
                        addItem(item, height: chunkHeight)
                    }
                }
            }
        }
        flushPage()
        #endif

        if pages.isEmpty {
            pages.append(BookPage(index: 0, items: []))
        }
        return pages
    }

    #if canImport(UIKit)
    private func measureText(_ text: String, font: UIFont, lineSpacing: CGFloat, width: CGFloat) -> CGFloat {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = lineSpacing
        let attributes: [NSAttributedString.Key: Any] = [.font: font, .paragraphStyle: style]
        let attrString = NSAttributedString(string: text, attributes: attributes)
        let rect = attrString.boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            context: nil
        )
        return ceil(rect.height)
    }

    private func splitTextToFit(_ text: String, maxHeight: CGFloat, font: UIFont, lineSpacing: CGFloat, width: CGFloat) -> [String] {
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        var chunks: [String] = []
        var current = ""

        for word in words {
            let candidate = current.isEmpty ? String(word) : current + " " + word
            let height = measureText(candidate, font: font, lineSpacing: lineSpacing, width: width)
            if height > maxHeight && !current.isEmpty {
                chunks.append(current)
                current = String(word)
            } else {
                current = candidate
            }
        }
        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks.isEmpty ? [text] : chunks
    }
    #endif

    // MARK: - Progress

    private func onVisibleParagraphChanged(_ firstVisibleID: PersistentIdentifier) {
        for chapter in book.sortedChapters {
            let paragraphs = chapter.sortedParagraphs
            if let pIdx = paragraphs.firstIndex(where: { $0.persistentModelID == firstVisibleID }) {
                if currentTitle != chapter.title {
                    currentTitle = chapter.title
                }
                if let progress = book.readingProgress,
                   progress.currentChapterIndex != chapter.index || progress.currentParagraphIndex != pIdx {
                    progress.currentChapterIndex = chapter.index
                    progress.currentParagraphIndex = pIdx
                    progress.lastReadAt = Date()
                    debounceSave()
                }
                return
            }
        }
    }

    private func debounceSave() {
        saveTask?.cancel()
        saveTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            try? modelContext.save()
        }
    }

    private func saveProgress() {
        saveTask?.cancel()
        try? modelContext.save()
    }
}

// MARK: - Paged Mode Types

struct BookPage: Identifiable {
    let index: Int
    var items: [PageItem]
    var id: Int { index }
}

struct PageItem: Identifiable {
    let index: Int
    let kind: PageItemKind
    var paragraphID: PersistentIdentifier?
    var id: Int { index }

    init(index: Int, kind: PageItemKind, paragraphID: PersistentIdentifier? = nil) {
        self.index = index
        self.kind = kind
        self.paragraphID = paragraphID
    }
}

enum PageItemKind {
    case chapterTitle(String)
    case text(String)
}
