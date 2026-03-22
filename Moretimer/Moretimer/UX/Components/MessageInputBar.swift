import SwiftUI

struct MessageInputBar: View {
    @Binding var text: String
    var placeholder: String = "Message..."
    let onSend: (String) -> Void

    private var trimmedText: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        HStack(spacing: 12) {
            TextField(placeholder, text: $text, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.secondary.opacity(0.1), in: .capsule)

            Button {
                let content = trimmedText
                guard !content.isEmpty else { return }
                onSend(content)
                text = ""
            } label: {
                Image(systemName: AppIcon.send)
                    .font(.title2)
            }
            .disabled(trimmedText.isEmpty)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}
