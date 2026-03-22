//
//  ThreadsListView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct ThreadsListView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(NavigationManager.self) private var navManager
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
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("New Thread", systemImage: "plus") {
                    createThread()
                }
            }
        }
        .overlay {
            if threads.isEmpty {
                ContentUnavailableView {
                    Label("No Threads", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("Start a conversation with your AI agent.")
                } actions: {
                    Button("New Thread") {
                        createThread()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    @ViewBuilder
    private func threadRow(_ thread: ThreadEntity) -> some View {
        NavigationLink(value: thread.persistentModelID) {
            ThreadRowView(thread: thread)
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive) {
                deleteThread(thread)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
        .swipeActions(edge: .leading, allowsFullSwipe: true) {
            Button {
                togglePin(thread)
            } label: {
                Label(
                    thread.isPinned ? "Unpin" : "Pin",
                    systemImage: thread.isPinned ? "pin.slash" : "pin"
                )
            }
            .tint(.orange)

            Button {
                toggleRead(thread)
            } label: {
                Label(
                    thread.isRead ? "Mark Unread" : "Mark Read",
                    systemImage: thread.isRead ? "envelope.badge" : "envelope.open"
                )
            }
            .tint(.blue)
        }
    }

    private func createThread() {
        let thread = ThreadEntity(title: "New Thread")
        modelContext.insert(thread)
        try? modelContext.save()
        navManager.threadsPath.append(thread.persistentModelID)
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
