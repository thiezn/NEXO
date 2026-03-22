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

            MessageInputBar(text: $messageText) { content in
                sendMessage(content)
            }
        }
        .navigationTitle(thread.title)
        #if !os(macOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            DetailToolbarContent(
                primaryAction: MenuAction(title: "Settings", icon: AppIcon.settings) {
                    showSettings = true
                },
                listSections: [
                    questionTestActions
                ]
            )
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

    @discardableResult
    private func insertMessage(content: String, role: MessageRole) -> MessageEntity {
        let message = MessageEntity(content: content, role: role)
        message.thread = thread
        thread.messages.append(message)
        thread.lastMessageAt = Date()
        modelContext.insert(message)
        return message
    }

    private func sendMessage(_ content: String) {
        insertMessage(content: content, role: .user)
        try? modelContext.save()
    }

    // MARK: - Test Question Actions

    private var questionTestActions: [MenuAction] {
        QuestionType.allCases.map { type in
            MenuAction(title: "Add \(type.displayName)", icon: type.systemImage) {
                addTestQuestion(type: type)
            }
        } + [
            MenuAction(title: "Add All Types", icon: AppIcon.addAll) {
                addAllTestQuestions()
            }
        ]
    }

    private func addTestQuestion(type: QuestionType) {
        let message = insertMessage(content: "", role: .assistant)

        let question = QuestionEntity.sample(for: type, order: 0)
        question.message = message
        message.questions.append(question)
        modelContext.insert(question)

        try? modelContext.save()
    }

    private func addAllTestQuestions() {
        let message = insertMessage(content: "", role: .assistant)

        for (index, type) in QuestionType.allCases.enumerated() {
            let question = QuestionEntity.sample(for: type, order: index)
            question.message = message
            message.questions.append(question)
            modelContext.insert(question)
        }

        try? modelContext.save()
    }
}
