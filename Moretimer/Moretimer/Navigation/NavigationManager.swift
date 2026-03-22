import SwiftUI
import SwiftData

// MARK: - Tab Definition

enum AppTab: String, Hashable, Identifiable, CaseIterable, Sendable {
    case home
    case books
    case threads
    case search

    var id: String { rawValue }

    var title: String {
        switch self {
        case .home: "Home"
        case .books: "Books"
        case .threads: "Threads"
        case .search: "Search"
        }
    }

    var systemImage: String {
        switch self {
        case .home: AppIcon.home
        case .books: AppIcon.books
        case .threads: AppIcon.threads
        case .search: AppIcon.search
        }
    }
}

// MARK: - Sheet Type

enum SheetType: String, Identifiable {
    case settings

    var id: String { rawValue }
}

// MARK: - Navigation Manager

@MainActor @Observable
final class NavigationManager {

    var selectedTab: AppTab = .home

    var homePath = NavigationPath()
    var booksPath = NavigationPath()
    var threadsPath = NavigationPath()
    var searchPath = NavigationPath()

    var currentSheet: SheetType?

    // MARK: - Sheet Presentation

    func presentSheet(_ sheet: SheetType) {
        currentSheet = sheet
    }

    func dismissSheet() {
        currentSheet = nil
    }

    // MARK: - Programmatic Navigation

    func navigateToBook(_ bookID: PersistentIdentifier) {
        selectedTab = .books
        booksPath.append(AppDestination.book(bookID))
    }

    func navigateToThread(_ threadID: PersistentIdentifier) {
        selectedTab = .threads
        threadsPath.append(AppDestination.thread(threadID))
    }

    func popToRoot(tab: AppTab) {
        switch tab {
        case .home: homePath = NavigationPath()
        case .books: booksPath = NavigationPath()
        case .threads: threadsPath = NavigationPath()
        case .search: searchPath = NavigationPath()
        }
    }
}
