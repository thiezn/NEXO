//
//  ThreadRowView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI

struct ThreadRowView: View {
    let thread: ThreadEntity

    var body: some View {
        HStack(spacing: 12) {
            if thread.isPinned {
                Image(systemName: AppIcon.pin)
                    .symbolVariant(.fill)
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(thread.title)
                        .font(.headline)
                        .fontWeight(thread.isRead ? .regular : .bold)
                        .lineLimit(1)

                    if !thread.isRead {
                        Circle()
                            .fill(.blue)
                            .frame(width: 8, height: 8)
                    }

                    Spacer()

                    Text(thread.lastMessageAt, style: .relative)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                HStack {
                    CategoryBadge(text: thread.category)

                    if let lastMessage = thread.lastMessage {
                        Text(lastMessage.content)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }
}
