//
//  MainTabView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct MainTabView: View {
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile
    @Environment(\.modelContext) private var modelContext

    @Query(filter: #Predicate<ThreadEntity> { $0.isRead == false })
    private var unreadThreads: [ThreadEntity]

    var body: some View {
        @Bindable var navManager = navManager

        TabView(selection: $navManager.selectedTab) {
            Tab(value: .home) {
                NavigationStack(path: $navManager.homePath) {
                    HomeView()
                }
            } label: {
                Label {
                    Text("Home")
                } icon: {
                    profileTabIcon
                }
            }

            Tab(AppTab.books.title, systemImage: AppTab.books.systemImage, value: .books) {
                NavigationStack(path: $navManager.booksPath) {
                    BooksListView()
                        .navigationDestination(for: PersistentIdentifier.self) { bookID in
                            if let book = modelContext.model(for: bookID) as? BookEntity {
                                BookReaderView(book: book)
                            }
                        }
                }
            }

            Tab(AppTab.threads.title, systemImage: AppTab.threads.systemImage, value: .threads) {
                NavigationStack(path: $navManager.threadsPath) {
                    ThreadsListView()
                        .navigationDestination(for: PersistentIdentifier.self) { threadID in
                            if let thread = modelContext.model(for: threadID) as? ThreadEntity {
                                ThreadDetailView(thread: thread)
                            }
                        }
                }
            }
            .badge(unreadThreads.count)

            Tab(value: .search, role: .search) {
                NavigationStack(path: $navManager.searchPath) {
                    SearchView()
                }
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        .tabBarMinimizeBehavior(.onScrollDown)
    }

    @ViewBuilder
    private var profileTabIcon: some View {
        if let data = userProfile.avatarImageData {
            imageFromData(data, contentMode: .fill)
                .frame(width: 24, height: 24)
                .clipShape(.circle)
        } else {
            Image(systemName: "person.circle.fill")
        }
    }
}

#Preview(traits: .modifier(PreviewAppEnvironment())) {
    MainTabView()
}
