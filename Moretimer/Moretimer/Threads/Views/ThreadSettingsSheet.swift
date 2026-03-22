//
//  ThreadSettingsSheet.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct ThreadSettingsSheet: View {
    @Bindable var thread: ThreadEntity
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext

    private let categories = ["General", "Work", "Personal", "Research", "Creative"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Title") {
                    TextField("Thread title", text: $thread.title)
                }

                Section("Category") {
                    Picker("Category", selection: $thread.category) {
                        ForEach(categories, id: \.self) { category in
                            Text(category).tag(category)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                }

                Section {
                    LabeledContent("Created", value: thread.createdAt, format: .dateTime)
                    LabeledContent("Messages", value: "\(thread.messageCount)")
                }
            }
            .navigationTitle("Thread Settings")
            #if !os(macOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        try? modelContext.save()
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
