import SwiftUI
import SwiftData

struct HomeView: View {
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile
    @Environment(\.themeColors) private var themeColors
    @Environment(\.appNamespace) private var namespace

    @Query(sort: \ReadingProgress.lastReadAt, order: .reverse)
    private var recentProgress: [ReadingProgress]

    @Query(sort: \ThreadEntity.lastMessageAt, order: .reverse)
    private var recentThreads: [ThreadEntity]

    private var recentBooks: [BookEntity] {
        recentProgress.prefix(10).compactMap(\.book)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if !recentBooks.isEmpty {
                    recentBooksSection
                }

                if !recentThreads.isEmpty {
                    recentThreadsSection
                }

                if recentBooks.isEmpty && recentThreads.isEmpty {
                    EmptyStateView(
                        "Welcome to Mor(e)timer",
                        systemImage: AppIcon.empty,
                        description: "Start by importing a book or creating a thread."
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.top, 60)
                }
            }
            .padding(.vertical)
        }
        .navigationTitle("Home")
        .toolbarTitleDisplayMode(.inline)
        .toolbar {
            TopLevelToolbarContent(
                avatarData: userProfile.avatarImageData,
                avatarCrop: userProfile.avatarCropData,
                avatarInitials: userProfile.initials,
                onAvatarTap: { navManager.presentSheet(.settings) }
            )
        }
    }

    // MARK: - Recent Books

    private var recentBooksSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader("Continue Reading", actionLabel: "See All") {
                navManager.selectedTab = .books
            }

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 16) {
                    ForEach(recentBooks) { book in
                        NavigationLink(value: AppDestination.book(book.persistentModelID)) {
                            LargeCard(
                                imageData: book.coverImage?.imageData,
                                placeholderIcon: AppIcon.bookFilled,
                                subtext: book.author,
                                title: book.title,
                                description: progressText(for: book),
                                tint: themeColors.accent
                            )
                        }
                        .buttonStyle(.plain)
                        .matchedTransitionSource(id: book.persistentModelID, in: namespace)
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    // MARK: - Recent Threads

    private var recentThreadsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader("Recent Conversations", actionLabel: "See All") {
                navManager.selectedTab = .threads
            }

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 16) {
                    ForEach(recentThreads.prefix(10)) { thread in
                        NavigationLink(value: AppDestination.thread(thread.persistentModelID)) {
                            SmallCard(
                                imageData: nil,
                                placeholderIcon: thread.threadCategory?.systemImage ?? AppIcon.threads,
                                title: thread.title,
                                subtitle: thread.category,
                                tint: themeColors.accent
                            )
                        }
                        .buttonStyle(.plain)
                        .matchedTransitionSource(id: thread.persistentModelID, in: namespace)
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    // MARK: - Helpers

    private func progressText(for book: BookEntity) -> String? {
        guard let progress = book.readingProgress else { return nil }
        return "\(Int(progress.percentage))% complete"
    }
}

#Preview(traits: .modifier(PreviewAppEnvironment())) {
    NavigationStack {
        HomeView()
    }
}
