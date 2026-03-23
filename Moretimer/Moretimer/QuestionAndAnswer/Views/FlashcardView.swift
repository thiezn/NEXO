//
//  FlashcardView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct FlashcardView: View {
    @Bindable var question: QuestionEntity
    var onRecall: ((RecallQuality) -> Void)?
    @Environment(\.themeColors) private var colors
    @State private var isFlipped = false

    var body: some View {
        ZStack {
            flashcardFront
                .opacity(isFlipped ? 0 : 1)
                .rotation3DEffect(
                    .degrees(isFlipped ? 180 : 0),
                    axis: (x: 0, y: 1, z: 0),
                    perspective: 0.5
                )

            flashcardBack
                .opacity(isFlipped ? 1 : 0)
                .rotation3DEffect(
                    .degrees(isFlipped ? 0 : -180),
                    axis: (x: 0, y: 1, z: 0),
                    perspective: 0.5
                )
        }
    }

    // MARK: - Front

    private var flashcardFront: some View {
        QuestionCardView(question: question) {
            withAnimation(.easeInOut(duration: 0.6)) {
                isFlipped = true
            }
        }
    }

    // MARK: - Back

    private var flashcardBack: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Answer")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(question.staticAnswer ?? "")
                .font(.title3)
                .fontWeight(.medium)

            Spacer()

            recallButtons

            HStack {
                Spacer()
                Button("Show Question") {
                    withAnimation(.easeInOut(duration: 0.6)) {
                        isFlipped = false
                    }
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: QuestionCardView.cardHeight)
        .clipShape(.rect(cornerRadius: 20))
        .glassEffect(
            .regular.tint(colors.accent.opacity(0.15)),
            in: .rect(cornerRadius: 20)
        )
    }

    // MARK: - Recall Buttons

    private var recallButtons: some View {
        HStack(spacing: 8) {
            recallButton("Again", quality: .incorrect, tint: .red)
            recallButton("Hard", quality: .correct, tint: .orange)
            recallButton("Good", quality: .good, tint: .blue)
            recallButton("Easy", quality: .perfect, tint: .green)
        }
    }

    private func recallButton(_ title: String, quality: RecallQuality, tint: Color) -> some View {
        Button {
            onRecall?(quality)
            withAnimation(.easeInOut(duration: 0.6)) {
                isFlipped = false
            }
        } label: {
            Text(title)
                .font(.subheadline.weight(.medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
        .buttonStyle(.plain)
        .clipShape(.rect(cornerRadius: 10))
        .glassEffect(
            .regular.tint(tint.opacity(0.2)),
            in: .rect(cornerRadius: 10)
        )
    }
}

#Preview("Flashcard", traits: .modifier(PreviewAppEnvironment())) {
    ScrollView {
        FlashcardView(question: .sampleFlashcard(order: 0))
            .padding()
    }
}
