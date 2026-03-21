//
//  MoretimerApp.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 21/03/2026.
//

import SwiftUI
import SwiftData

@main
struct MoretimerApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self,
            BookEntity.self,
            ChapterEntity.self,
            ParagraphEntity.self,
            ImageRefEntity.self,
            ReadingProgress.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
