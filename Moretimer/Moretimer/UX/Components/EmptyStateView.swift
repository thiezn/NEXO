import SwiftUI

struct EmptyStateView: View {
    let title: String
    let systemImage: String
    let description: String
    var actionLabel: String?
    var action: (() -> Void)?

    init(
        _ title: String,
        systemImage: String,
        description: String,
        actionLabel: String? = nil,
        action: (() -> Void)? = nil
    ) {
        self.title = title
        self.systemImage = systemImage
        self.description = description
        self.actionLabel = actionLabel
        self.action = action
    }

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: systemImage)
        } description: {
            Text(description)
        } actions: {
            if let actionLabel, let action {
                Button(actionLabel, action: action)
                    .buttonStyle(.borderedProminent)
            }
        }
    }
}
