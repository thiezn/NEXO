//
//  NavigationManager.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

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
        case .home: "house"
        case .books: "books.vertical"
        case .threads: "bubble.left.and.bubble.right"
        case .search: "magnifyingglass"
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
        booksPath.append(bookID)
    }

    func navigateToThread(_ threadID: PersistentIdentifier) {
        selectedTab = .threads
        threadsPath.append(threadID)
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
