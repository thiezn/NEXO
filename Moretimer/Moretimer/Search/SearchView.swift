import SwiftUI
import SwiftData

enum SearchContentType: String, CaseIterable, Identifiable {
    case all = "All"
    case books = "Books"
    case threads = "Threads"

    var id: String { rawValue }
}

struct SearchView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile

    @State private var searchText = ""
    @State private var selectedType: SearchContentType = .all
    @State private var bookResults: [BookEntity] = []
    @State private var threadResults: [ThreadEntity] = []
    @State private var searchTask: Task<Void, Never>?

    var body: some View {
        List {
            Picker("Content Type", selection: $selectedType) {
                ForEach(SearchContentType.allCases) { type in
                    Text(type.rawValue).tag(type)
                }
            }
            .pickerStyle(.segmented)
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)

            if selectedType == .all || selectedType == .books {
                BookSearchResultsView(books: bookResults)
            }

            if selectedType == .all || selectedType == .threads {
                ThreadSearchResultsView(threads: threadResults)
            }

            if searchText.isEmpty {
                ContentUnavailableView("Search Moretimer",
                    systemImage: AppIcon.search,
                    description: Text("Search across books and threads"))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
            } else if bookResults.isEmpty && threadResults.isEmpty {
                ContentUnavailableView.search(text: searchText)
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
            }
        }
        .navigationTitle("Search")
        .searchable(text: $searchText, prompt: "Books, threads...")
        .toolbar {
            TopLevelToolbarContent(
                avatarData: userProfile.avatarImageData,
                avatarCrop: userProfile.avatarCropData,
                avatarInitials: userProfile.initials,
                onAvatarTap: { navManager.presentSheet(.settings) }
            )
        }
        .onChange(of: searchText) { _, newValue in
            searchTask?.cancel()
            searchTask = Task {
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled else { return }
                performSearch(query: newValue)
            }
        }
        .onChange(of: selectedType) { _, _ in
            performSearch(query: searchText)
        }
    }

    private func performSearch(query: String) {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            bookResults = []
            threadResults = []
            return
        }

        if selectedType == .all || selectedType == .books {
            let bookPredicate = #Predicate<BookEntity> { book in
                book.title.localizedStandardContains(trimmed) ||
                book.author.localizedStandardContains(trimmed)
            }
            let bookDescriptor = FetchDescriptor<BookEntity>(
                predicate: bookPredicate,
                sortBy: [SortDescriptor(\.importedAt, order: .reverse)]
            )
            bookResults = (try? modelContext.fetch(bookDescriptor)) ?? []
        } else {
            bookResults = []
        }

        if selectedType == .all || selectedType == .threads {
            let threadPredicate = #Predicate<ThreadEntity> { thread in
                thread.title.localizedStandardContains(trimmed) ||
                thread.category.localizedStandardContains(trimmed)
            }
            let threadDescriptor = FetchDescriptor<ThreadEntity>(
                predicate: threadPredicate,
                sortBy: [SortDescriptor(\.lastMessageAt, order: .reverse)]
            )
            threadResults = (try? modelContext.fetch(threadDescriptor)) ?? []
        } else {
            threadResults = []
        }
    }
}

#Preview(traits: .modifier(PreviewAppEnvironment())) {
    NavigationStack {
        SearchView()
    }
}
