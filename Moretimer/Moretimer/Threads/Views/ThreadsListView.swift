import SwiftUI
import SwiftData

struct ThreadsListView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile

    @Query(sort: \ThreadEntity.lastMessageAt, order: .reverse) private var threads: [ThreadEntity]

    var pinnedThreads: [ThreadEntity] {
        threads.filter(\.isPinned)
    }

    var unpinnedThreads: [ThreadEntity] {
        threads.filter { !$0.isPinned }
    }

    var body: some View {
        List {
            if !pinnedThreads.isEmpty {
                Section("Pinned") {
                    ForEach(pinnedThreads) { thread in
                        threadRow(thread)
                    }
                }
            }

            Section(pinnedThreads.isEmpty ? "" : "Recent") {
                ForEach(unpinnedThreads) { thread in
                    threadRow(thread)
                }
            }
        }
        .navigationTitle("Threads")
        .toolbarTitleDisplayMode(.inline)
        .toolbar {
            TopLevelToolbarContent(
                avatarData: userProfile.avatarImageData,
                avatarCrop: userProfile.avatarCropData,
                avatarInitials: userProfile.initials,
                onAvatarTap: { navManager.presentSheet(.settings) },
                listSections: [
                    ThreadCategory.allCases.map { category in
                        MenuAction(
                            title: "New \(category.displayName)",
                            icon: category.systemImage
                        ) {
                            createThread(category: category)
                        }
                    }
                ]
            )
        }
        .overlay {
            if threads.isEmpty {
                EmptyStateView(
                    "No Threads",
                    systemImage: AppIcon.threads,
                    description: "Start a conversation with your assistant.",
                    actionLabel: "New Thread"
                ) {
                    createThread(category: .general)
                }
            }
        }
    }

    @ViewBuilder
    private func threadRow(_ thread: ThreadEntity) -> some View {
        NavigationLink(value: AppDestination.thread(thread.persistentModelID)) {
            ThreadRowView(thread: thread)
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive) {
                deleteThread(thread)
            } label: {
                Label("Delete", systemImage: AppIcon.delete)
            }
        }
        .swipeActions(edge: .leading, allowsFullSwipe: true) {
            Button {
                togglePin(thread)
            } label: {
                Label(
                    thread.isPinned ? "Unpin" : "Pin",
                    systemImage: thread.isPinned ? AppIcon.unpin : AppIcon.pin
                )
            }
            .tint(.orange)

            Button {
                toggleRead(thread)
            } label: {
                Label(
                    thread.isRead ? "Mark Unread" : "Mark Read",
                    systemImage: thread.isRead ? AppIcon.markUnread : AppIcon.markRead
                )
            }
            .tint(.blue)
        }
    }

    private func createThread(category: ThreadCategory = .general) {
        let thread = ThreadEntity(title: "New \(category.displayName) Thread", category: category.rawValue)
        modelContext.insert(thread)
        try? modelContext.save()
        navManager.threadsPath.append(AppDestination.thread(thread.persistentModelID))
    }

    private func deleteThread(_ thread: ThreadEntity) {
        withAnimation {
            modelContext.delete(thread)
            try? modelContext.save()
        }
    }

    private func togglePin(_ thread: ThreadEntity) {
        withAnimation {
            thread.isPinned.toggle()
            try? modelContext.save()
        }
    }

    private func toggleRead(_ thread: ThreadEntity) {
        withAnimation {
            thread.isRead.toggle()
            try? modelContext.save()
        }
    }
}
