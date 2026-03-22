import SwiftUI
import SwiftData

struct BookSearchResultsView: View {
    let books: [BookEntity]

    var body: some View {
        if !books.isEmpty {
            Section("Books") {
                ForEach(books) { book in
                    NavigationLink(value: AppDestination.book(book.persistentModelID)) {
                        BookRowView(book: book)
                    }
                }
            }
        }
    }
}
