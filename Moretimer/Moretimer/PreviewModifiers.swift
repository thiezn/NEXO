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
    }

    @MainActor
    static func makeSharedContext() throws -> Context {
        let appContext = try AppContext.shared()
        return Context(
            container: appContext.container,
            errorManager: appContext.errorManager
        )
    }

    func body(content: Content, context: Context) -> some View {
        content
            .modelContainer(context.container)
            .environment(NavigationManager())
            .environment(ThemeManager())
            .environment(context.errorManager)
            .environment(UserProfileManager())
    }
}
