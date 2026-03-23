//
//  ThreadModels.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import SwiftData

// MARK: - Message Role

enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
}

// MARK: - Thread Category

enum ThreadCategory: String, CaseIterable, Identifiable, Sendable {
    case general = "General"
    case work = "Work"
    case personal = "Personal"
    case research = "Research"
    case creative = "Creative"
    case learning = "Learning"

    var id: String { rawValue }

    var displayName: String { rawValue }

    var systemImage: String {
        switch self {
        case .general: "bubble.left"
        case .work: "briefcase"
        case .personal: "person"
        case .research: "magnifyingglass"
        case .creative: "paintbrush"
        case .learning: AppIcon.learning
        }
    }
}

// MARK: - Thread Entity

@Model
final class ThreadEntity {
    var title: String
    var category: String
    var isPinned: Bool
    var isRead: Bool
    var createdAt: Date
    var lastMessageAt: Date

    @Relationship(deleteRule: .cascade, inverse: \MessageEntity.thread)
    var messages: [MessageEntity] = []

    var sortedMessages: [MessageEntity] {
        messages.sorted { $0.createdAt < $1.createdAt }
    }

    var lastMessage: MessageEntity? {
        messages.max { $0.createdAt < $1.createdAt }
    }

    var messageCount: Int { messages.count }

    var threadCategory: ThreadCategory? {
        get { ThreadCategory(rawValue: category) }
        set { if let newValue { category = newValue.rawValue } }
    }

    var isLearningThread: Bool { category == ThreadCategory.learning.rawValue }

    init(title: String, category: String = "General", isPinned: Bool = false) {
        self.title = title
        self.category = category
        self.isPinned = isPinned
        self.isRead = true
        self.createdAt = Date()
        self.lastMessageAt = Date()
    }
}

// MARK: - Message Entity

@Model
final class MessageEntity {
    var content: String
    var role: MessageRole
    var createdAt: Date
    var thread: ThreadEntity?

    @Relationship(deleteRule: .cascade, inverse: \QuestionEntity.message)
    var questions: [QuestionEntity] = []

    var sortedQuestions: [QuestionEntity] {
        questions.sorted { $0.order < $1.order }
    }

    var isQuestionMessage: Bool { !questions.isEmpty }

    init(content: String, role: MessageRole) {
        self.content = content
        self.role = role
        self.createdAt = Date()
    }
}
