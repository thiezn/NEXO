//
//  PreviewModifiers.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//
// Modifiers for our previews so we can easily inject required environment variables and modelContext
// https://www.donnywals.com/using-previewmodifier-to-build-a-previewing-environment/
// https://developer.apple.com/documentation/SwiftUI/PreviewModifier

import SwiftUI
import SwiftData

struct PreviewAppEnvironment: PreviewModifier {

    struct Context {
        let container: ModelContainer
        let errorManager: ErrorManager
        let themeManager: ThemeManager
        let userProfileManager: UserProfileManager
        let learningService: LearningService
        let nexoService: NexoService
    }

    @MainActor
    static func makeSharedContext() throws -> Context {
        let appContext = try AppContext.shared()
        return Context(
            container: appContext.container,
            errorManager: appContext.errorManager,
            themeManager: appContext.themeManager,
            userProfileManager: appContext.userProfileManager,
            learningService: appContext.learningService,
            nexoService: appContext.nexoService
        )
    }

    func body(content: Content, context: Context) -> some View {
        content
            .modelContainer(context.container)
            .environment(NavigationManager())
            .resolveThemeColors()
            .environment(context.themeManager)
            .environment(context.errorManager)
            .environment(context.userProfileManager)
            .environment(context.learningService)
            .environment(context.nexoService)
    }
}
