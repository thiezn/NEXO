//
//  BookSearchResultsView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct BookSearchResultsView: View {
    @Environment(NavigationManager.self) private var navManager
    let books: [BookEntity]

    var body: some View {
        if !books.isEmpty {
            Section("Books") {
                ForEach(books) { book in
                    Button {
                        navManager.navigateToBook(book.persistentModelID)
                    } label: {
                        BookRowView(book: book)
                    }
                    .tint(.primary)
                }
            }
        }
    }
}
