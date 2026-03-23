//
//  QuestionCarouselView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct QuestionCarouselView: View {
    let questions: [QuestionEntity]
    var onSubmit: (() -> Void)?

    @State private var currentID: PersistentIdentifier?
    @Environment(\.modelContext) private var modelContext
    @Environment(\.themeColors) private var colors

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            ScrollView(.horizontal) {
                LazyHStack(spacing: 16) {
                    ForEach(questions) { question in
                        Group {
                            if question.isFlashcard {
                                FlashcardView(question: question)
                            } else {
                                QuestionCardView(question: question)
                            }
                        }
                        .containerRelativeFrame(.horizontal, count: 1, spacing: 16)
                        .scrollTransition(.interactive) { content, phase in
                            content
                                .scaleEffect(phase.isIdentity ? 1.0 : 0.92)
                                .opacity(phase.isIdentity ? 1.0 : 0.7)
                        }
                        .id(question.persistentModelID)
                    }
                }
                .scrollTargetLayout()
            }
            .scrollTargetBehavior(.viewAligned)
            .scrollPosition(id: $currentID)
            .scrollIndicators(.hidden)

            HStack {
                pageIndicator

                Spacer()

                Button("Submit") {
                    try? modelContext.save()
                    onSubmit?()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal)
        }
    }

    private var pageIndicator: some View {
        HStack(spacing: 6) {
            ForEach(questions) { question in
                let isCurrent = currentID == question.persistentModelID
                let isAnswered = question.answer != nil
                Circle()
                    .fill(
                        isCurrent
                            ? colors.accent
                            : isAnswered
                                ? colors.accent.opacity(0.5)
                                : colors.secondary.opacity(0.3)
                    )
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut(duration: 0.2), value: currentID)
            }
        }
    }
}
