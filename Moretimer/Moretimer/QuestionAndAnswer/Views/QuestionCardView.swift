//
//  QuestionCardView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct QuestionCardView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.themeColors) private var colors

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let imageData = question.imageData {
                imageFromData(imageData, contentMode: .fill)
                    .frame(maxWidth: .infinity, maxHeight: 180)
                    .clipShape(.rect(cornerRadius: 12))
            }

            Text(question.questionText)
                .font(.headline)

            if let desc = question.descriptionText {
                Text(desc)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            CategoryBadge(text: question.questionType.displayName, color: colors.accent)

            Divider()

            answerInputView
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .clipShape(.rect(cornerRadius: 20))
        .glassEffect(
            .regular.tint(colors.primary.opacity(0.1)),
            in: .rect(cornerRadius: 20)
        )
    }

    @ViewBuilder
    private var answerInputView: some View {
        switch question.questionType {
        case .multipleChoice:
            MultipleChoiceAnswerView(question: question)
        case .singleChoice:
            SingleChoiceAnswerView(question: question)
        case .openEnded:
            OpenEndedAnswerView(question: question)
        case .scale:
            ScaleAnswerView(question: question)
        case .yesNo:
            YesNoAnswerView(question: question)
        case .fileUpload:
            FileUploadAnswerView(question: question)
        case .imageUpload:
            ImageUploadAnswerView(question: question)
        }
    }
}

#Preview("Multiple Choice", traits: .modifier(PreviewAppEnvironment())) {
    ScrollView {
        QuestionCardView(question: .sample(for: .multipleChoice, order: 0))
            .padding()
    }
}

#Preview("Scale", traits: .modifier(PreviewAppEnvironment())) {
    ScrollView {
        QuestionCardView(question: .sample(for: .scale, order: 0))
            .padding()
    }
}

#Preview("Yes/No", traits: .modifier(PreviewAppEnvironment())) {
    ScrollView {
        QuestionCardView(question: .sample(for: .yesNo, order: 0))
            .padding()
    }
}
