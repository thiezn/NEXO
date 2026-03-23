//
//  Question.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import SwiftData

// MARK: - Question Type

enum QuestionType: String, Codable, Sendable, CaseIterable, Identifiable {
    case multipleChoice
    case singleChoice
    case openEnded
    case scale
    case yesNo
    case fileUpload
    case imageUpload

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .multipleChoice: "Multiple Choice"
        case .singleChoice: "Single Choice"
        case .openEnded: "Open Ended"
        case .scale: "Scale"
        case .yesNo: "Yes / No"
        case .fileUpload: "File Upload"
        case .imageUpload: "Image Upload"
        }
    }

    var systemImage: String {
        switch self {
        case .multipleChoice: AppIcon.multipleChoice
        case .singleChoice: AppIcon.singleChoice
        case .openEnded: AppIcon.openEnded
        case .scale: AppIcon.scaleIcon
        case .yesNo: AppIcon.yesNo
        case .fileUpload: AppIcon.fileUpload
        case .imageUpload: AppIcon.imageUpload
        }
    }
}

// MARK: - Question Entity

@Model
final class QuestionEntity {
    var questionText: String
    var descriptionText: String?
    @Attribute(.externalStorage) var imageData: Data?
    var questionTypeRaw: String
    var choices: [String]
    var order: Int
    var createdAt: Date
    var staticAnswer: String?

    @Relationship(deleteRule: .cascade, inverse: \AnswerEntity.question)
    var answer: AnswerEntity?

    var message: MessageEntity?
    var deck: LearningDeck?

    var questionType: QuestionType {
        get { QuestionType(rawValue: questionTypeRaw) ?? .openEnded }
        set { questionTypeRaw = newValue.rawValue }
    }

    var isFlashcard: Bool { staticAnswer != nil }

    init(
        questionText: String,
        questionType: QuestionType,
        descriptionText: String? = nil,
        imageData: Data? = nil,
        choices: [String] = [],
        order: Int = 0,
        staticAnswer: String? = nil
    ) {
        self.questionText = questionText
        self.questionTypeRaw = questionType.rawValue
        self.descriptionText = descriptionText
        self.imageData = imageData
        self.choices = choices
        self.order = order
        self.createdAt = Date()
        self.staticAnswer = staticAnswer
    }
}

// MARK: - Question Entity Codable

extension QuestionEntity: Codable {
    enum CodingKeys: String, CodingKey {
        case questionText = "question_text"
        case descriptionText = "description_text"
        case imageDataBase64 = "image_data_base64"
        case questionType = "question_type"
        case choices
        case order
        case createdAt = "created_at"
        case answer
        case staticAnswer = "static_answer"
    }

    convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let questionText = try container.decode(String.self, forKey: .questionText)
        let typeRaw = try container.decode(String.self, forKey: .questionType)
        let questionType = QuestionType(rawValue: typeRaw) ?? .openEnded
        let descriptionText = try container.decodeIfPresent(String.self, forKey: .descriptionText)
        let choices = try container.decodeIfPresent([String].self, forKey: .choices) ?? []
        let order = try container.decodeIfPresent(Int.self, forKey: .order) ?? 0

        var imageData: Data?
        if let base64 = try container.decodeIfPresent(String.self, forKey: .imageDataBase64) {
            imageData = Data(base64Encoded: base64)
        }

        self.init(
            questionText: questionText,
            questionType: questionType,
            descriptionText: descriptionText,
            imageData: imageData,
            choices: choices,
            order: order
        )

        if let createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) {
            self.createdAt = createdAt
        }

        self.answer = try container.decodeIfPresent(AnswerEntity.self, forKey: .answer)
        self.staticAnswer = try container.decodeIfPresent(String.self, forKey: .staticAnswer)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(questionText, forKey: .questionText)
        try container.encodeIfPresent(descriptionText, forKey: .descriptionText)
        try container.encodeIfPresent(imageData?.base64EncodedString(), forKey: .imageDataBase64)
        try container.encode(questionTypeRaw, forKey: .questionType)
        try container.encode(choices, forKey: .choices)
        try container.encode(order, forKey: .order)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(answer, forKey: .answer)
        try container.encodeIfPresent(staticAnswer, forKey: .staticAnswer)
    }
}

// MARK: - Answer Entity

@Model
final class AnswerEntity {
    var answerTypeRaw: String
    var timestamp: Date
    var question: QuestionEntity?

    // Multiple choice
    var selectedIndices: [Int]?

    // Single choice
    var selectedIndex: Int?

    // Single choice "other" + open ended
    var otherText: String?

    // Scale
    var scaleValue: Double?

    // Yes/No
    var boolValue: Bool?

    // File/Image upload
    @Attribute(.externalStorage) var fileData: Data?
    var fileName: String?

    var answerType: QuestionType {
        get { QuestionType(rawValue: answerTypeRaw) ?? .openEnded }
        set { answerTypeRaw = newValue.rawValue }
    }

    init(answerType: QuestionType) {
        self.answerTypeRaw = answerType.rawValue
        self.timestamp = Date()
    }
}

// MARK: - Answer Data (read-only convenience)

enum AnswerData {
    case multipleChoice(Set<Int>)
    case singleChoice(index: Int?, other: String?)
    case openEnded(String)
    case scale(Double)
    case yesNo(Bool)
    case fileUpload(Data, String)
    case imageUpload(Data)
}

extension AnswerEntity {
    var answerData: AnswerData? {
        switch answerType {
        case .multipleChoice:
            guard let indices = selectedIndices else { return nil }
            return .multipleChoice(Set(indices))
        case .singleChoice:
            return .singleChoice(index: selectedIndex, other: otherText)
        case .openEnded:
            guard let text = otherText else { return nil }
            return .openEnded(text)
        case .scale:
            guard let value = scaleValue else { return nil }
            return .scale(value)
        case .yesNo:
            guard let value = boolValue else { return nil }
            return .yesNo(value)
        case .fileUpload:
            guard let data = fileData, let name = fileName else { return nil }
            return .fileUpload(data, name)
        case .imageUpload:
            guard let data = fileData else { return nil }
            return .imageUpload(data)
        }
    }
}

// MARK: - Answer Entity Codable

extension AnswerEntity: Codable {
    enum CodingKeys: String, CodingKey {
        case answerType = "answer_type"
        case timestamp
        case selectedIndices = "selected_indices"
        case selectedIndex = "selected_index"
        case otherText = "other_text"
        case scaleValue = "scale_value"
        case boolValue = "bool_value"
        case fileDataBase64 = "file_data_base64"
        case fileName = "file_name"
    }

    convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let typeRaw = try container.decode(String.self, forKey: .answerType)
        let answerType = QuestionType(rawValue: typeRaw) ?? .openEnded

        self.init(answerType: answerType)

        if let ts = try container.decodeIfPresent(Date.self, forKey: .timestamp) {
            self.timestamp = ts
        }
        self.selectedIndices = try container.decodeIfPresent([Int].self, forKey: .selectedIndices)
        self.selectedIndex = try container.decodeIfPresent(Int.self, forKey: .selectedIndex)
        self.otherText = try container.decodeIfPresent(String.self, forKey: .otherText)
        self.scaleValue = try container.decodeIfPresent(Double.self, forKey: .scaleValue)
        self.boolValue = try container.decodeIfPresent(Bool.self, forKey: .boolValue)
        self.fileName = try container.decodeIfPresent(String.self, forKey: .fileName)

        if let base64 = try container.decodeIfPresent(String.self, forKey: .fileDataBase64) {
            self.fileData = Data(base64Encoded: base64)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(answerTypeRaw, forKey: .answerType)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encodeIfPresent(selectedIndices, forKey: .selectedIndices)
        try container.encodeIfPresent(selectedIndex, forKey: .selectedIndex)
        try container.encodeIfPresent(otherText, forKey: .otherText)
        try container.encodeIfPresent(scaleValue, forKey: .scaleValue)
        try container.encodeIfPresent(boolValue, forKey: .boolValue)
        try container.encodeIfPresent(fileData?.base64EncodedString(), forKey: .fileDataBase64)
        try container.encodeIfPresent(fileName, forKey: .fileName)
    }
}

// MARK: - Sample Factory

extension QuestionEntity {
    static func sample(for type: QuestionType, order: Int) -> QuestionEntity {
        switch type {
        case .multipleChoice:
            QuestionEntity(
                questionText: "Which fruits do you like?",
                questionType: .multipleChoice,
                descriptionText: "Select all that apply.",
                choices: ["Apple", "Banana", "Cherry", "Mango"],
                order: order
            )
        case .singleChoice:
            QuestionEntity(
                questionText: "What is your favorite color?",
                questionType: .singleChoice,
                descriptionText: "Pick one, or type your own.",
                choices: ["Red", "Blue", "Green"],
                order: order
            )
        case .openEnded:
            QuestionEntity(
                questionText: "Tell us about yourself.",
                questionType: .openEnded,
                descriptionText: "Write a short paragraph.",
                order: order
            )
        case .scale:
            QuestionEntity(
                questionText: "How satisfied are you?",
                questionType: .scale,
                descriptionText: "1 = Not at all, 10 = Very satisfied",
                order: order
            )
        case .yesNo:
            QuestionEntity(
                questionText: "Do you agree with the terms?",
                questionType: .yesNo,
                order: order
            )
        case .fileUpload:
            QuestionEntity(
                questionText: "Upload your resume.",
                questionType: .fileUpload,
                descriptionText: "PDF or DOCX files accepted.",
                order: order
            )
        case .imageUpload:
            QuestionEntity(
                questionText: "Upload a profile photo.",
                questionType: .imageUpload,
                descriptionText: "Choose an image from your library.",
                order: order
            )
        }
    }

    static func sampleFlashcard(order: Int) -> QuestionEntity {
        QuestionEntity(
            questionText: "What is the capital of France?",
            questionType: .openEnded,
            descriptionText: "Geography flashcard",
            order: order,
            staticAnswer: "Paris"
        )
    }
}
