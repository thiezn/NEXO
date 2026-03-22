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

    #if os(macOS)
        @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate
    #else
        @UIApplicationDelegateAdaptor private var appDelegate: AppDelegate
    #endif

    @State private var container: ModelContainer
    @State private var errorManager: ErrorManager
    @State private var themeManager: ThemeManager
    @State private var userProfileManager: UserProfileManager

    init() {
        do {
            let context = try AppContext.shared()
            self._container = State(initialValue: context.container)
            self._errorManager = State(initialValue: context.errorManager)
            self._themeManager = State(initialValue: context.themeManager)
            self._userProfileManager = State(initialValue: context.userProfileManager)
        } catch {
            fatalError("Failed to initialize AppContext after store reset: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            WindowSceneView()
                .environment(errorManager)
                .environment(themeManager)
                .environment(userProfileManager)
                .preferredColorScheme(themeManager.preferredColorScheme)
        }
        .modelContainer(container)
    }
}


/// The WindowSceneView helps separate the NavigationManager() objects into
/// separate state for each opened window.
struct WindowSceneView: View {
    @State private var navManager = NavigationManager()
    @Environment(ThemeManager.self) private var themeManager
    @Environment(ErrorManager.self) private var errorManager
    @Environment(UserProfileManager.self) private var userProfile

    var body: some View {
        MainTabView()
            .environment(navManager)
            .resolveThemeColors()
            .loadingErrorOverlay()
            .sheet(item: $navManager.currentSheet) { sheet in
                switch sheet {
                case .settings:
                    NavigationStack {
                        SettingsView()
                    }
                    .resolveThemeColors()
                    .environment(themeManager)
                    .environment(errorManager)
                    .environment(userProfile)
                }
            }
    }
}
