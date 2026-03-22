import SwiftUI
import SwiftData
import UniformTypeIdentifiers

struct BooksListView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile
    @Query(sort: \BookEntity.importedAt, order: .reverse) private var books: [BookEntity]
    @State private var showFileImporter = false
    @State private var importError: String?
    @State private var showError = false

    var body: some View {
        List {
            ForEach(books) { book in
                NavigationLink(value: AppDestination.book(book.persistentModelID)) {
                    BookRowView(book: book)
                }
                .draggable(book.toJSON())
            }
            .onDelete(perform: deleteBooks)
        }
        .navigationTitle("Library")
        .toolbar {
            #if os(iOS)
            ToolbarItem(placement: .navigationBarTrailing) {
                EditButton()
            }
            #endif
            TopLevelToolbarContent(
                avatarData: userProfile.avatarImageData,
                avatarInitials: userProfile.initials,
                onAvatarTap: { navManager.presentSheet(.settings) },
                listSections: [[
                    MenuAction(title: "Import Book", icon: AppIcon.importBook) {
                        showFileImporter = true
                    },
                    MenuAction(title: "Load Sample Book", icon: AppIcon.book) {
                        loadSampleBook()
                    },
                ]]
            )
        }
        .dropDestination(for: BookOutputJSON.self) { droppedBooks, _ in
            for bookJSON in droppedBooks {
                do {
                    _ = try BookImportService.importFromJSON(bookJSON, into: modelContext)
                } catch {
                    importError = error.localizedDescription
                    showError = true
                }
            }
            return !droppedBooks.isEmpty
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.json, .bookJSON],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result)
        }
        .alert("Import Error", isPresented: $showError) {
            Button("OK") { }
        } message: {
            Text(importError ?? "Unknown error")
        }
        .overlay {
            if books.isEmpty {
                EmptyStateView(
                    "No Books",
                    systemImage: AppIcon.books,
                    description: "Add a book to get started.",
                    actionLabel: "Load Sample Book"
                ) {
                    loadSampleBook()
                }
            }
        }
    }

    private func loadSampleBook() {
        do {
            _ = try BookImportService.importBundledBook(into: modelContext)
        } catch {
            importError = error.localizedDescription
            showError = true
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            do {
                _ = try BookImportService.importFromFilePicker(url: url, into: modelContext)
            } catch {
                importError = error.localizedDescription
                showError = true
            }
        case .failure(let error):
            importError = error.localizedDescription
            showError = true
        }
    }

    private func deleteBooks(at offsets: IndexSet) {
        withAnimation {
            for index in offsets {
                modelContext.delete(books[index])
            }
        }
    }
}
