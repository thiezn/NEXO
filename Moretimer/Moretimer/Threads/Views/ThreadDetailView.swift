import SwiftUI
import SwiftData
import OSLog

struct ThreadDetailView: View {
    @Bindable var thread: ThreadEntity
    @Environment(\.modelContext) private var modelContext
    @Environment(LearningService.self) private var learningService
    @Environment(NexoService.self) private var nexoService
    @State private var messageText = ""
    @State private var showSettings = false
    @State private var showDeckPicker = false
    @State private var activeRun: ActiveRun?
    @FocusState private var isInputFocused: Bool

    private var isConnected: Bool { nexoService.connectionState.isConnected }

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

            if !isConnected {
                connectionBanner
            }

            Divider()

            MessageInputBar(text: $messageText, isFocused: $isInputFocused) { content in
                sendMessage(content)
            }
            .disabled(activeRun != nil || !isConnected)
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
            await consumeAgentEvents()
        }
    }

    // MARK: - Connection Banner

    private var connectionBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: nexoService.connectionState.statusIcon)
                .font(.caption)
            Text(nexoService.connectionState.statusText)
                .font(.caption)
        }
        .foregroundStyle(nexoService.connectionState.statusColor)
        .padding(.vertical, 4)
    }

    // MARK: - Message Sending

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
        let userMessage = insertMessage(content: content, role: .user)
        let placeholder = insertMessage(content: "", role: .assistant)
        placeholder.isThinking = true
        try? modelContext.save()

        Task {
            await sendAgentRequest(prompt: content, placeholder: placeholder)
        }
    }

    private func sendAgentRequest(prompt: String, placeholder: MessageEntity) async {
        do {
            if thread.nexoSessionId == nil {
                let session = try await nexoService.sessionCreate(name: thread.title)
                thread.nexoSessionId = session.sessionId
                try? modelContext.save()
            }

            let response = try await nexoService.agent(
                prompt: prompt,
                idempotencyKey: UUID().uuidString,
                sessionId: thread.nexoSessionId
            )
            activeRun = ActiveRun(runId: response.runId, message: placeholder)
        } catch {
            Logger.nexo.error("Agent request failed: \(error.localizedDescription)")
            finishRun(content: "Failed to send message: \(error.localizedDescription)", failed: true)
        }
    }

    // MARK: - Event Subscription

    private func consumeAgentEvents() async {
        let stream = nexoService.subscribe()
        for await frameEvent in stream {
            guard !Task.isCancelled else { break }
            guard activeRun != nil else { continue }
            guard frameEvent.event == .agent else { continue }
            guard let payload = try? frameEvent.payload(as: AgentEventPayload.self) else { continue }
            guard payload.runId == activeRun?.runId else { continue }

            handleAgentEvent(payload)
        }
    }

    private func handleAgentEvent(_ payload: AgentEventPayload) {
        guard let run = activeRun else { return }
        let message = run.message

        switch payload.status {
        case .thinking:
            message.isThinking = true
        case .streaming:
            if let content = payload.content {
                message.content = content
                message.isThinking = false
            }
        case .completed:
            if let content = payload.content, !content.isEmpty {
                message.content = content
            }
            finishRun(content: nil, failed: false)
        case .failed:
            let errorMsg = payload.error ?? payload.content ?? "Agent run failed"
            finishRun(content: errorMsg, failed: true)
        case .toolCall:
            message.isThinking = true
            if let toolName = payload.toolName {
                message.content = "Using tool: \(toolName)..."
            }
        case .queued:
            message.content = "Waiting for inference node..."
            message.isThinking = true
        case .accepted, .cancelled:
            break
        }
    }

    private func finishRun(content: String?, failed: Bool) {
        if let content, !content.isEmpty {
            activeRun?.message.content = content
        }
        activeRun?.message.isThinking = false
        if failed, let msg = activeRun?.message, msg.content.isEmpty {
            modelContext.delete(msg)
        }
        activeRun = nil
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

// MARK: - Active Run

private struct ActiveRun {
    let runId: String
    let message: MessageEntity
}
