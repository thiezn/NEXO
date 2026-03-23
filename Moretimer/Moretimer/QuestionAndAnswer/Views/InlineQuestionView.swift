//
//  InlineQuestionView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct InlineQuestionView: View {
    @Bindable var question: QuestionEntity
    var onSubmit: (() -> Void)?
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        VStack(alignment: .trailing, spacing: 8) {
            QuestionCardView(question: question)

            Button("Submit") {
                try? modelContext.save()
                onSubmit?()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal)
    }
}
