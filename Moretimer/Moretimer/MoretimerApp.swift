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
    @State private var themeManager = ThemeManager()
    @State private var userProfileManager = UserProfileManager()

    init() {
        do {
            let context = try AppContext.shared()
            self._container = State(initialValue: context.container)
            self._errorManager = State(initialValue: context.errorManager)
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
                .tint(themeManager.colors.accent)
        }
        .modelContainer(container)
    }
}


/// The WindowSceneView helps separate the NavigationManager() objects into
/// separate state for each opened window.
struct WindowSceneView: View {
    @State private var navManager = NavigationManager()

    var body: some View {
        MainTabView()
            .environment(navManager)
            .loadingErrorOverlay()
            .sheet(item: $navManager.currentSheet) { sheet in
                switch sheet {
                case .settings:
                    NavigationStack {
                        SettingsView()
                    }
                }
            }
    }
}
