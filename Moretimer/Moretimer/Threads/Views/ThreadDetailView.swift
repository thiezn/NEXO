import SwiftUI
import SwiftData

struct ThreadDetailView: View {
    @Bindable var thread: ThreadEntity
    @Environment(\.modelContext) private var modelContext
    @Environment(LearningService.self) private var learningService
    @State private var messageText = ""
    @State private var showSettings = false
    @State private var showDeckPicker = false
    @FocusState private var isInputFocused: Bool

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
            .scrollDismissesKeyboard(.interactively)
            .defaultScrollAnchor(.bottom)
            .simultaneousGesture(TapGesture().onEnded { isInputFocused = false })

            Divider()

            MessageInputBar(text: $messageText, isFocused: $isInputFocused) { content in
                sendMessage(content)
            }
        }
        .navigationTitle(thread.title)
        .toolbarTitleDisplayMode(.inline)
        .toolbar {
            DetailToolbarContent(
                primaryAction: MenuAction(title: "Settings", icon: AppIcon.settings) {
                    showSettings = true
                },
                listSections: [
                    learningActions,
                    questionTestActions
                ]
            )
        }
        .sheet(isPresented: $showSettings) {
            ThreadSettingsSheet(thread: thread)
        }
        .sheet(isPresented: $showDeckPicker) {
            LearningDeckPickerView(thread: thread, learningService: learningService)
        }
        .task {
            if !thread.isRead {
                thread.isRead = true
            }
            if thread.isLearningThread && thread.messages.isEmpty {
                showDeckPicker = true
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

    // MARK: - Learning Actions

    private var learningActions: [MenuAction] {
        guard thread.isLearningThread else { return [] }
        return [
            MenuAction(title: "Add Cards", icon: AppIcon.deck) {
                showDeckPicker = true
            }
        ]
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
            },
            MenuAction(title: "Add Flashcard", icon: AppIcon.flashcard) {
                addTestFlashcard()
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

    private func addTestFlashcard() {
        let message = insertMessage(content: "", role: .assistant)

        let question = QuestionEntity.sampleFlashcard(order: 0)
        question.message = message
        message.questions.append(question)
        modelContext.insert(question)

        try? modelContext.save()
    }
}
