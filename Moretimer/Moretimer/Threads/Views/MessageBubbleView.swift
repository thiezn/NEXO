//
//  MessageBubbleView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI

struct MessageBubbleView: View {
    let message: MessageEntity

    private var isUser: Bool { message.role == .user }

    var body: some View {
        if message.isQuestionMessage {
            questionContent
        } else {
            textContent
        }
    }

    private var textContent: some View {
        HStack {
            if isUser { Spacer(minLength: 60) }

            Group {
                if message.isThinking && message.content.isEmpty {
                    thinkingIndicator
                } else {
                    Text(message.content)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassEffect(
                .regular.tint(isUser ? .blue : .gray),
                in: .rect(cornerRadius: 16)
            )
            .overlay(alignment: .bottomTrailing) {
                if message.isThinking && !message.content.isEmpty {
                    streamingDot
                }
            }

            if !isUser { Spacer(minLength: 60) }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
    }

    private var thinkingIndicator: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(.secondary)
                    .frame(width: 6, height: 6)
                    .phaseAnimator([false, true]) { content, phase in
                        content.opacity(phase ? 1 : 0.3)
                    } animation: { _ in
                        .easeInOut(duration: 0.5).delay(Double(i) * 0.15)
                    }
            }
        }
        .padding(.horizontal, 4)
    }

    private var streamingDot: some View {
        Circle()
            .fill(.blue)
            .frame(width: 6, height: 6)
            .padding(4)
            .phaseAnimator([false, true]) { content, phase in
                content.opacity(phase ? 1 : 0.3)
            } animation: { _ in
                .easeInOut(duration: 0.6)
            }
    }

    @ViewBuilder
    private var questionContent: some View {
        let sorted = message.sortedQuestions
        if sorted.count == 1, let question = sorted.first {
            if question.isFlashcard {
                FlashcardView(question: question)
                    .padding(.horizontal)
            } else {
                InlineQuestionView(question: question)
            }
        } else {
            QuestionCarouselView(questions: sorted)
        }
    }
}
