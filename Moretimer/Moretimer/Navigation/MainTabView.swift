import SwiftUI
import SwiftData

struct MainTabView: View {
    @Environment(NavigationManager.self) private var navManager
    @Environment(\.modelContext) private var modelContext

    @Query(filter: #Predicate<ThreadEntity> { $0.isRead == false })
    private var unreadThreads: [ThreadEntity]

    var body: some View {
        @Bindable var navManager = navManager

        TabView(selection: $navManager.selectedTab) {
            Tab(AppTab.home.title, systemImage: AppTab.home.systemImage, value: .home) {
                NavigationStack(path: $navManager.homePath) {
                    HomeView()
                        .appNavigationDestinations()
                }
            }

            Tab(AppTab.books.title, systemImage: AppTab.books.systemImage, value: .books) {
                NavigationStack(path: $navManager.booksPath) {
                    BooksListView()
                        .appNavigationDestinations()
                }
            }

            Tab(AppTab.threads.title, systemImage: AppTab.threads.systemImage, value: .threads) {
                NavigationStack(path: $navManager.threadsPath) {
                    ThreadsListView()
                        .appNavigationDestinations()
                }
            }
            .badge(unreadThreads.count)

            Tab(value: .search, role: .search) {
                NavigationStack(path: $navManager.searchPath) {
                    SearchView()
                        .appNavigationDestinations()
                }
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        #if os(iOS)
        .tabBarMinimizeBehavior(.onScrollDown)
        #endif
    }
}

#Preview(traits: .modifier(PreviewAppEnvironment())) {
    MainTabView()
}
