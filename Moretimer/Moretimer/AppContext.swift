//
//  AppContext.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftData
import SwiftUI
import OSLog

/// Centralized Application context for ModelContainer, LocalDataStorage, and API clients.
struct AppContext {
    
    /// The Application Context
    struct Context {

        /// The SwiftData ModelContainer
        let container: ModelContainer

        /// The app-wide ErrorManager
        let errorManager: ErrorManager

        /// The app-wide ThemeManager
        let themeManager: ThemeManager

        /// The User Profile Manager
        let userProfileManager: UserProfileManager

        /// The Learning Service
        let learningService: LearningService

        /// The NEXO Gateway Service
        let nexoService: NexoService
    }

    /// Builds a shared application environment with all models registered.
    /// Autosave is disabled to allow explicit saves.
    ///
    /// If the SwiftData store is incompatible with the current schema (e.g. after
    /// a model property type change), the store files are deleted and recreated
    /// from scratch. Important data is offloaded to the REST API so local data
    /// loss is acceptable during development.
    @MainActor
    static func make() throws -> Context {

        Logger.storage.log("Building AppContext ModelContainer")

        let schema = Schema([
            // Books
            BookEntity.self,
            ChapterEntity.self,
            ParagraphEntity.self,
            ImageRefEntity.self,
            ReadingProgress.self,

            // Threads
            ThreadEntity.self,
            MessageEntity.self,

            // Questions & Answers
            QuestionEntity.self,
            AnswerEntity.self,

            // Learning
            LearningDeck.self,
            LearningSession.self,
            LearningResult.self,
        ])
        let config = ModelConfiguration("default", schema: schema)
        
        // WARNING, this will delete ALL local storage!
        // Can be useful if we need to change models. We don't have a migration schema.
        // deleteStoreFiles(at: config.url)
        
        let container: ModelContainer
        do {
            container = try ModelContainer(for: schema, configurations: [config])
        } catch {
            Logger.storage.error("ModelContainer creation failed, deleting store and retrying: \(error)")
            deleteStoreFiles(at: config.url)
            container = try ModelContainer(for: schema, configurations: [config])
        }

        container.mainContext.autosaveEnabled = false

        Logger.storage.log("Building AppContext ErrorManager")
        let errorManager = ErrorManager()

        Logger.storage.log("Building AppContext ThemeManager")
        let themeManager = ThemeManager()

        Logger.storage.log("Building AppContext UserProfileManager")
        let userProfileManager = UserProfileManager()

        Logger.storage.log("Building AppContext LearningService")
        let learningService = LearningService(modelContext: container.mainContext)

        Logger.storage.log("Building AppContext NexoService")
        let nexoService = NexoService(errorManager: errorManager)

        return Context(
            container: container,
            errorManager: errorManager,
            themeManager: themeManager,
            userProfileManager: userProfileManager,
            learningService: learningService,
            nexoService: nexoService
        )
    }

    // Shared cached environment for app-wide reuse
    @MainActor
    private static var cachedContext: Context?

    /// Returns the shared environment, creating it once on first access.
    @MainActor
    static func shared() throws -> Context {
        if let cachedContext {
            return cachedContext
        }

        let context = try make()
        cachedContext = context

        return context
    }

    /// Delete the SQLite store and its WAL/SHM companion files.
    ///
    /// Called when the existing store is incompatible with the current schema
    /// and cannot be auto-migrated. A fresh store will be created on retry.
    private static func deleteStoreFiles(at url: URL) {
        let fm = FileManager.default
        for ext in ["", "-wal", "-shm"] {
            let fileURL = URL(fileURLWithPath: url.path + ext)
            try? fm.removeItem(at: fileURL)
        }
    }
}
