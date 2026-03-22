//
//  AnswerInputViews.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData
import PhotosUI
import UniformTypeIdentifiers

// MARK: - Answer Helper

private func ensureAnswer(for question: QuestionEntity, in context: ModelContext) -> AnswerEntity {
    if let existing = question.answer { return existing }
    let answer = AnswerEntity(answerType: question.questionType)
    answer.question = question
    question.answer = answer
    context.insert(answer)
    return answer
}

// MARK: - Choice Row

private struct ChoiceRow: View {
    let icon: String
    let selectedIcon: String
    let text: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: isSelected ? selectedIcon : icon)
                    .foregroundColor(isSelected ? .accentColor : .secondary)
                Text(text)
                    .foregroundStyle(.primary)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .clipShape(.rect(cornerRadius: 12))
            .glassEffect(
                .regular.tint(isSelected ? Color.accentColor.opacity(0.15) : .clear),
                in: .rect(cornerRadius: 12)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Multiple Choice

struct MultipleChoiceAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(question.choices.enumerated()), id: \.offset) { index, choice in
                ChoiceRow(
                    icon: "square",
                    selectedIcon: "checkmark.square.fill",
                    text: choice,
                    isSelected: isSelected(index),
                    action: { toggle(index) }
                )
            }
        }
    }

    private func isSelected(_ index: Int) -> Bool {
        question.answer?.selectedIndices?.contains(index) ?? false
    }

    private func toggle(_ index: Int) {
        let answer = ensureAnswer(for: question, in: modelContext)
        var indices = answer.selectedIndices ?? []
        if indices.contains(index) {
            indices.removeAll { $0 == index }
        } else {
            indices.append(index)
        }
        answer.selectedIndices = indices
        answer.timestamp = Date()
    }
}

// MARK: - Single Choice

struct SingleChoiceAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext

    private var hasOtherOption: Bool {
        question.choices.count <= 3
    }

    private var isOtherSelected: Bool {
        question.answer?.selectedIndex == nil && question.answer?.otherText != nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(question.choices.enumerated()), id: \.offset) { index, choice in
                ChoiceRow(
                    icon: "circle",
                    selectedIcon: AppIcon.singleChoice,
                    text: choice,
                    isSelected: question.answer?.selectedIndex == index && !isOtherSelected,
                    action: { select(index) }
                )
            }

            if hasOtherOption {
                ChoiceRow(
                    icon: "circle",
                    selectedIcon: AppIcon.singleChoice,
                    text: "Other",
                    isSelected: isOtherSelected,
                    action: { selectOther() }
                )

                if isOtherSelected {
                    TextField("Type your answer...", text: otherTextBinding)
                        .textFieldStyle(.plain)
                        .padding(12)
                        .clipShape(.rect(cornerRadius: 12))
                        .glassEffect(.regular, in: .rect(cornerRadius: 12))
                }
            }
        }
    }

    private func select(_ index: Int) {
        let answer = ensureAnswer(for: question, in: modelContext)
        answer.selectedIndex = index
        answer.otherText = nil
        answer.timestamp = Date()
    }

    private func selectOther() {
        let answer = ensureAnswer(for: question, in: modelContext)
        answer.selectedIndex = nil
        answer.otherText = answer.otherText ?? ""
        answer.timestamp = Date()
    }

    private var otherTextBinding: Binding<String> {
        Binding(
            get: { question.answer?.otherText ?? "" },
            set: { newValue in
                let answer = ensureAnswer(for: question, in: modelContext)
                answer.otherText = newValue
                answer.timestamp = Date()
            }
        )
    }
}

// MARK: - Open Ended

struct OpenEndedAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        TextField("Type your answer...", text: textBinding, axis: .vertical)
            .lineLimit(3...8)
            .textFieldStyle(.plain)
            .padding(12)
            .clipShape(.rect(cornerRadius: 12))
            .glassEffect(.regular, in: .rect(cornerRadius: 12))
    }

    private var textBinding: Binding<String> {
        Binding(
            get: { question.answer?.otherText ?? "" },
            set: { newValue in
                let answer = ensureAnswer(for: question, in: modelContext)
                answer.otherText = newValue
                answer.timestamp = Date()
            }
        )
    }
}

// MARK: - Scale

struct ScaleAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        VStack(spacing: 4) {
            Slider(value: scaleBinding, in: 1...10, step: 1) {
                Text("Rating")
            } currentValueLabel: {
                Text("\(Int(question.answer?.scaleValue ?? 5))")
                    .font(.title2.bold())
            } minimumValueLabel: {
                Text("1")
                    .font(.caption)
            } maximumValueLabel: {
                Text("10")
                    .font(.caption)
            } tick: { value in
                SliderTick(value)
            }
        }
    }

    private var scaleBinding: Binding<Double> {
        Binding(
            get: { question.answer?.scaleValue ?? 5 },
            set: { newValue in
                let answer = ensureAnswer(for: question, in: modelContext)
                answer.scaleValue = newValue
                answer.timestamp = Date()
            }
        )
    }
}

// MARK: - Yes/No

struct YesNoAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        HStack(spacing: 12) {
            Button {
                select(true)
            } label: {
                Label("Yes", systemImage: AppIcon.yesNo)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .clipShape(.rect(cornerRadius: 12))
                    .glassEffect(
                        .regular.tint(isSelected(true) ? Color.green.opacity(0.2) : .clear),
                        in: .rect(cornerRadius: 12)
                    )
            }
            .buttonStyle(.plain)

            Button {
                select(false)
            } label: {
                Label("No", systemImage: "hand.thumbsdown")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .clipShape(.rect(cornerRadius: 12))
                    .glassEffect(
                        .regular.tint(isSelected(false) ? Color.red.opacity(0.2) : .clear),
                        in: .rect(cornerRadius: 12)
                    )
            }
            .buttonStyle(.plain)
        }
    }

    private func isSelected(_ value: Bool) -> Bool {
        question.answer?.boolValue == value
    }

    private func select(_ value: Bool) {
        let answer = ensureAnswer(for: question, in: modelContext)
        answer.boolValue = value
        answer.timestamp = Date()
    }
}

// MARK: - File Upload

struct FileUploadAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext
    @State private var showImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                showImporter = true
            } label: {
                Label(
                    question.answer?.fileName ?? "Choose File",
                    systemImage: AppIcon.fileUpload
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.bordered)
            .fileImporter(
                isPresented: $showImporter,
                allowedContentTypes: [.item],
                allowsMultipleSelection: false
            ) { result in
                handleFileResult(result)
            }

            if question.answer?.fileData != nil, let name = question.answer?.fileName {
                HStack {
                    Image(systemName: AppIcon.done)
                        .foregroundStyle(.green)
                    Text(name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func handleFileResult(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result, let url = urls.first else { return }
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }

        guard let data = try? Data(contentsOf: url) else { return }
        let answer = ensureAnswer(for: question, in: modelContext)
        answer.fileData = data
        answer.fileName = url.lastPathComponent
        answer.timestamp = Date()
    }
}

// MARK: - Image Upload

struct ImageUploadAnswerView: View {
    @Bindable var question: QuestionEntity
    @Environment(\.modelContext) private var modelContext
    @State private var showPicker = false
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let data = question.answer?.fileData {
                imageFromData(data, contentMode: .fill)
                    .frame(maxWidth: .infinity, maxHeight: 180)
                    .clipShape(.rect(cornerRadius: 12))
            }

            Button {
                showPicker = true
            } label: {
                Label(
                    question.answer?.fileData != nil ? "Change Image" : "Choose Image",
                    systemImage: AppIcon.imageUpload
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.bordered)
            .photosPicker(isPresented: $showPicker, selection: $pickerItem, matching: .images)
            .onChange(of: pickerItem) { _, newItem in
                Task {
                    await loadImage(from: newItem)
                }
            }
        }
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        guard let item else { return }
        guard let data = try? await item.loadTransferable(type: Data.self) else { return }
        let answer = ensureAnswer(for: question, in: modelContext)
        answer.fileData = data
        answer.timestamp = Date()
    }
}
