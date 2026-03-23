//
//  LearningService.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import SwiftData
import OSLog

@MainActor @Observable
final class LearningService {

    private let modelContext: ModelContext

    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    // MARK: - Deck Management

    func createDeck(title: String, description: String? = nil) -> LearningDeck {
        let deck = LearningDeck(title: title, descriptionText: description)
        modelContext.insert(deck)
        try? modelContext.save()
        Logger.learning.info("Created learning deck: \(title)")
        return deck
    }

    func addCard(
        to deck: LearningDeck,
        question: String,
        answer: String,
        description: String? = nil
    ) -> QuestionEntity {
        let card = QuestionEntity(
            questionText: question,
            questionType: .openEnded,
            descriptionText: description,
            order: deck.cardCount,
            staticAnswer: answer
        )
        card.deck = deck
        deck.cards.append(card)
        deck.updatedAt = Date()
        modelContext.insert(card)
        try? modelContext.save()
        return card
    }

    func deleteDeck(_ deck: LearningDeck) {
        modelContext.delete(deck)
        try? modelContext.save()
    }

    func removeCard(_ card: QuestionEntity, from deck: LearningDeck) {
        deck.cards.removeAll { $0.persistentModelID == card.persistentModelID }
        deck.updatedAt = Date()
        modelContext.delete(card)
        try? modelContext.save()
    }

    // MARK: - Session Management

    func startSession(deck: LearningDeck, in thread: ThreadEntity? = nil) -> LearningSession {
        let session = LearningSession()
        session.deck = deck
        session.thread = thread
        modelContext.insert(session)
        try? modelContext.save()
        Logger.learning.info("Started learning session for deck: \(deck.title)")
        return session
    }

    func recordResult(
        session: LearningSession,
        question: QuestionEntity,
        quality: RecallQuality,
        responseTime: Double? = nil
    ) -> LearningResult {
        let previousResults = fetchPreviousResults(for: question)
        let lastResult = previousResults.last

        let easeFactor = lastResult?.easeFactor ?? 2.5
        let interval = lastResult?.interval ?? 1
        let repetitions = lastResult?.repetitions ?? 0

        let (newEase, newInterval, newReps) = calculateSM2(
            quality: quality,
            easeFactor: easeFactor,
            interval: interval,
            repetitions: repetitions
        )

        let result = LearningResult(
            isCorrect: quality.rawValue >= 3,
            responseTimeSeconds: responseTime,
            easeFactor: newEase,
            interval: newInterval,
            repetitions: newReps
        )
        result.session = session
        result.question = question
        modelContext.insert(result)
        try? modelContext.save()
        return result
    }

    func completeSession(_ session: LearningSession) {
        session.completedAt = Date()
        try? modelContext.save()
        Logger.learning.info("Completed session. Score: \(session.scorePercentage)%")
    }

    // MARK: - Queries

    func fetchDecks() -> [LearningDeck] {
        let descriptor = FetchDescriptor<LearningDeck>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
        )
        return (try? modelContext.fetch(descriptor)) ?? []
    }

    func fetchCardsForReview(in deck: LearningDeck) -> [QuestionEntity] {
        let now = Date()
        return deck.cards.filter { card in
            let results = fetchPreviousResults(for: card)
            guard let lastResult = results.last else { return true }
            return lastResult.nextReviewDate <= now
        }
    }

    func fetchSessions(for deck: LearningDeck) -> [LearningSession] {
        deck.sessions.sorted { $0.startedAt > $1.startedAt }
    }

    // MARK: - Thread Integration

    func addCardsToThread(_ cards: [QuestionEntity], thread: ThreadEntity) {
        let message = MessageEntity(content: "", role: .assistant)
        message.thread = thread
        thread.messages.append(message)
        thread.lastMessageAt = Date()
        modelContext.insert(message)

        for (index, card) in cards.enumerated() {
            let threadCard = QuestionEntity(
                questionText: card.questionText,
                questionType: card.questionType,
                descriptionText: card.descriptionText,
                imageData: card.imageData,
                choices: card.choices,
                order: index,
                staticAnswer: card.staticAnswer
            )
            threadCard.message = message
            message.questions.append(threadCard)
            modelContext.insert(threadCard)
        }

        try? modelContext.save()
    }

    // MARK: - SM-2 Algorithm

    private func calculateSM2(
        quality: RecallQuality,
        easeFactor: Double,
        interval: Int,
        repetitions: Int
    ) -> (easeFactor: Double, interval: Int, repetitions: Int) {
        let q = Double(quality.rawValue)

        if quality.rawValue < 3 {
            return (max(1.3, easeFactor - 0.2), 1, 0)
        }

        let newEase = max(1.3, easeFactor + (0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02)))
        let newReps = repetitions + 1

        let newInterval: Int
        switch newReps {
        case 1: newInterval = 1
        case 2: newInterval = 6
        default: newInterval = Int(round(Double(interval) * newEase))
        }

        return (newEase, newInterval, newReps)
    }

    private func fetchPreviousResults(for question: QuestionEntity) -> [LearningResult] {
        let questionID = question.persistentModelID
        let descriptor = FetchDescriptor<LearningResult>(
            predicate: #Predicate { $0.question?.persistentModelID == questionID },
            sortBy: [SortDescriptor(\.answeredAt)]
        )
        return (try? modelContext.fetch(descriptor)) ?? []
    }
}
