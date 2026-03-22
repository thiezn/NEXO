//
//  HomeView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct HomeView: View {
    @Environment(NavigationManager.self) private var navManager
    @Environment(UserProfileManager.self) private var userProfile
    @Environment(ThemeManager.self) private var themeManager

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
                headerSection
                    .padding(.horizontal)

                if !recentBooks.isEmpty {
                    recentBooksSection
                }

                if !recentThreads.isEmpty {
                    recentThreadsSection
                }

                if recentBooks.isEmpty && recentThreads.isEmpty {
                    ContentUnavailableView {
                        Label("Welcome to Moretimer", systemImage: "sparkles")
                    } description: {
                        Text("Start by importing a book or creating a thread.")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 60)
                }
            }
            .padding(.vertical)
        }
        .navigationTitle("Home")
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(spacing: 16) {
            Button {
                navManager.presentSheet(.settings)
            } label: {
                profileImage
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 2) {
                Text(greeting)
                    .font(.title2.weight(.semibold))
                if let name = userProfile.fullName {
                    Text(name)
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
    }

    @ViewBuilder
    private var profileImage: some View {
        if let data = userProfile.avatarImageData {
            imageFromData(data, contentMode: .fill)
                .frame(width: 56, height: 56)
                .clipShape(.circle)
        } else {
            Image(systemName: "person.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
        }
    }

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 5..<12: return "Good morning"
        case 12..<17: return "Good afternoon"
        case 17..<22: return "Good evening"
        default: return "Good night"
        }
    }

    // MARK: - Recent Books

    private var recentBooksSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Continue Reading")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("See All") {
                    navManager.selectedTab = .books
                }
                .font(.subheadline)
            }
            .padding(.horizontal)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 16) {
                    ForEach(recentBooks) { book in
                        RecentBookCard(book: book, accentColor: themeManager.colors.accent)
                            .onTapGesture {
                                navManager.navigateToBook(book.persistentModelID)
                            }
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    // MARK: - Recent Threads

    private var recentThreadsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Conversations")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("See All") {
                    navManager.selectedTab = .threads
                }
                .font(.subheadline)
            }
            .padding(.horizontal)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 16) {
                    ForEach(recentThreads.prefix(10)) { thread in
                        RecentThreadCard(thread: thread, accentColor: themeManager.colors.accent)
                            .onTapGesture {
                                navManager.navigateToThread(thread.persistentModelID)
                            }
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

// MARK: - Recent Book Card

private struct RecentBookCard: View {
    let book: BookEntity
    let accentColor: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Cover image
            Group {
                if let firstImage = book.images.first, let data = firstImage.imageData {
                    imageFromData(data, contentMode: .fill)
                } else {
                    Image(systemName: "book.closed.fill")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(width: 140, height: 180)
            .clipShape(.rect(cornerRadius: 12))

            Text(book.title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(2)

            Text(book.author)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            if let progress = book.readingProgress {
                ProgressView(value: progress.percentage, total: 100)
                    .tint(accentColor)
            }
        }
        .frame(width: 140)
        .padding(12)
        .glassEffect(.regular.tint(accentColor.opacity(0.15)), in: .rect(cornerRadius: 16))
    }
}

// MARK: - Recent Thread Card

private struct RecentThreadCard: View {
    let thread: ThreadEntity
    let accentColor: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(thread.title)
                    .font(.headline)
                    .fontWeight(thread.isRead ? .semibold : .bold)
                    .lineLimit(1)

                Spacer()

                if !thread.isRead {
                    Circle()
                        .fill(.blue)
                        .frame(width: 8, height: 8)
                }
            }

            Text(thread.category)
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.secondary.opacity(0.15), in: .capsule)

            if let lastMessage = thread.lastMessage {
                Text(lastMessage.content)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            Text(thread.lastMessageAt, style: .relative)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(width: 200, height: 120)
        .padding(12)
        .glassEffect(.regular.tint(accentColor.opacity(0.15)), in: .rect(cornerRadius: 16))
    }
}

#Preview(traits: .modifier(PreviewAppEnvironment())) {
    NavigationStack {
        HomeView()
    }
}
