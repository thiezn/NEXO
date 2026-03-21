//
//  ContentView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 21/03/2026.
//

import SwiftUI
import SwiftData


struct ContentView: View {
    var body: some View {
        BooksListView()
    }
}

#Preview {
    ContentView()
        .modelContainer(for: [
            BookEntity.self,
            ChapterEntity.self,
            ParagraphEntity.self,
            ImageRefEntity.self,
            ReadingProgress.self,
        ], inMemory: true)
}
