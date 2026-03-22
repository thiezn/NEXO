import SwiftUI
import SwiftData

struct BookReaderView: View {
    let book: BookEntity
    @Environment(\.modelContext) private var modelContext
    @State private var scrollPosition = ScrollPosition()
    @State private var showSettings = false
    @State private var currentTitle: String
    @State private var saveTask: Task<Void, Never>?

    @AppStorage(BookReaderKeys.fontSize) private var fontSize: Double = 17
    @AppStorage(BookReaderKeys.mode) private var readingMode: ReadingMode = .continuous
    @AppStorage(BookReaderKeys.fontDesign) private var fontDesignOption: FontDesignOption = .default

    init(book: BookEntity) {
        self.book = book
        self._currentTitle = State(initialValue: book.title)
    }

    private var sortedChapters: [ChapterEntity] {
        book.sortedChapters
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
        #if !os(macOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
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
        .task(id: book.persistentModelID) {
            await restoreProgress()
        }
        .onDisappear { saveProgress() }
    }

    // MARK: - Continuous Mode

    private var continuousView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach(sortedChapters) { chapter in
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
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(sortedChapters) { chapter in
                    VStack(alignment: .leading, spacing: 16) {
                        Text(chapter.title)
                            .font(.system(.title2, design: fontDesignOption.fontDesign))
                            .fontWeight(.bold)
                            .padding(.top, 24)

                        ForEach(chapter.sortedParagraphs) { paragraph in
                            ParagraphView(
                                paragraph: paragraph,
                                fontSize: fontSize,
                                fontDesign: fontDesignOption.fontDesign
                            )
                            .id(paragraph.persistentModelID)
                        }
                    }
                    .padding(.horizontal)
                    .containerRelativeFrame(.vertical, alignment: .top)
                }
            }
            .scrollTargetLayout()
        }
        .scrollTargetBehavior(.paging)
        .scrollPosition($scrollPosition)
    }

    // MARK: - Progress

    private func onVisibleParagraphChanged(_ firstVisibleID: PersistentIdentifier) {
        let chapters = sortedChapters
        for chapter in chapters {
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

    private func restoreProgress() async {
        guard let progress = book.readingProgress else { return }

        if let chapter = sortedChapters.first(where: { $0.index == progress.currentChapterIndex }) {
            let paragraphs = chapter.sortedParagraphs
            if progress.currentParagraphIndex < paragraphs.count {
                try? await Task.sleep(for: .milliseconds(100))
                scrollPosition.scrollTo(
                    id: paragraphs[progress.currentParagraphIndex].persistentModelID,
                    anchor: .top
                )
            }
        }
    }
}
