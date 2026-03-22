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

    var body: some View {
        QuestionCardView(question: question)
            .padding(.horizontal)
    }
}
