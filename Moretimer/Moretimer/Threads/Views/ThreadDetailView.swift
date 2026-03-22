//
//  ThreadDetailView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct ThreadDetailView: View {
    @Bindable var thread: ThreadEntity
    @Environment(\.modelContext) private var modelContext
    @State private var messageText = ""
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                GlassEffectContainer(spacing: 8) {
                    LazyVStack(spacing: 12) {
                        ForEach(thread.sortedMessages) { message in
                            MessageBubbleView(message: message)
                        }
                    }
                    .padding()
                }
            }
            .defaultScrollAnchor(.bottom)

            Divider()

            messageInputBar
        }
        .navigationTitle(thread.title)
        #if !os(macOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Settings", systemImage: "gearshape") {
                    showSettings = true
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            ThreadSettingsSheet(thread: thread)
        }
        .task {
            if !thread.isRead {
                thread.isRead = true
            }
        }
    }

    private var trimmedMessage: String {
        messageText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var messageInputBar: some View {
        HStack(spacing: 12) {
            TextField("Message...", text: $messageText, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.secondary.opacity(0.1), in: .capsule)

            Button {
                sendMessage()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(trimmedMessage.isEmpty)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private func sendMessage() {
        let content = trimmedMessage
        guard !content.isEmpty else { return }

        let message = MessageEntity(content: content, role: .user)
        message.thread = thread
        thread.messages.append(message)
        thread.lastMessageAt = Date()
        modelContext.insert(message)
        try? modelContext.save()

        messageText = ""
    }
}
