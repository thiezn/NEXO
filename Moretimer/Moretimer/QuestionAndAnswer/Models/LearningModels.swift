//
//  LearningModels.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import SwiftData

// MARK: - Learning Deck

@Model
final class LearningDeck {
    var title: String
    var descriptionText: String?
    var createdAt: Date
    var updatedAt: Date

    @Relationship(deleteRule: .cascade, inverse: \QuestionEntity.deck)
    var cards: [QuestionEntity] = []

    @Relationship(deleteRule: .cascade, inverse: \LearningSession.deck)
    var sessions: [LearningSession] = []

    var cardCount: Int { cards.count }

    var lastSession: LearningSession? {
        sessions.max { $0.startedAt < $1.startedAt }
    }

    var averageScore: Double? {
        let completed = sessions.filter { $0.completedAt != nil }
        guard !completed.isEmpty else { return nil }
        let total = completed.reduce(0.0) { $0 + $1.scorePercentage }
        return total / Double(completed.count)
    }

    init(title: String, descriptionText: String? = nil) {
        self.title = title
        self.descriptionText = descriptionText
        self.createdAt = Date()
        self.updatedAt = Date()
    }
}

// MARK: - Learning Session

@Model
final class LearningSession {
    var startedAt: Date
    var completedAt: Date?
    var deck: LearningDeck?
    var thread: ThreadEntity?

    @Relationship(deleteRule: .cascade, inverse: \LearningResult.session)
    var results: [LearningResult] = []

    var totalCards: Int { results.count }

    var correctCount: Int {
        results.filter(\.isCorrect).count
    }

    var scorePercentage: Double {
        guard totalCards > 0 else { return 0 }
        return Double(correctCount) / Double(totalCards) * 100
    }

    var isComplete: Bool { completedAt != nil }

    init() {
        self.startedAt = Date()
    }
}

// MARK: - Learning Result

@Model
final class LearningResult {
    var session: LearningSession?
    var question: QuestionEntity?
    var isCorrect: Bool
    var answeredAt: Date
    var responseTimeSeconds: Double?

    // SM-2 spaced repetition data
    var easeFactor: Double
    var interval: Int
    var repetitions: Int
    var nextReviewDate: Date

    init(
        isCorrect: Bool,
        responseTimeSeconds: Double? = nil,
        easeFactor: Double = 2.5,
        interval: Int = 1,
        repetitions: Int = 0
    ) {
        self.isCorrect = isCorrect
        self.answeredAt = Date()
        self.responseTimeSeconds = responseTimeSeconds
        self.easeFactor = easeFactor
        self.interval = interval
        self.repetitions = repetitions
        self.nextReviewDate = Calendar.current.date(
            byAdding: .day, value: interval, to: Date()
        ) ?? Date()
    }
}

// MARK: - Recall Quality

enum RecallQuality: Int, Sendable {
    case blackout = 0
    case incorrect = 1
    case difficult = 2
    case correct = 3
    case good = 4
    case perfect = 5
}
