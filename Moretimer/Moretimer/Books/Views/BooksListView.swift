import SwiftUI
import SwiftData
import UniformTypeIdentifiers

struct BooksListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \BookEntity.importedAt, order: .reverse) private var books: [BookEntity]
    @State private var showFileImporter = false
    @State private var importError: String?
    @State private var showError = false

    var body: some View {
        List {
            ForEach(books) { book in
                NavigationLink(value: book.persistentModelID) {
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
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button("Import Book...", systemImage: "doc.badge.plus") {
                        showFileImporter = true
                    }
                    if books.isEmpty {
                        Button("Load Sample Book", systemImage: "book") {
                            loadSampleBook()
                        }
                    }
                } label: {
                    Label("Add", systemImage: "plus")
                }
            }
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
                ContentUnavailableView {
                    Label("No Books", systemImage: "books.vertical")
                } description: {
                    Text("Add a book to get started.")
                } actions: {
                    Button("Load Sample Book") {
                        loadSampleBook()
                    }
                    .buttonStyle(.borderedProminent)
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
