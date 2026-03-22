import SwiftUI

struct SectionHeader: View {
    let title: String
    var actionLabel: String?
    var action: (() -> Void)?

    init(_ title: String, actionLabel: String? = nil, action: (() -> Void)? = nil) {
        self.title = title
        self.actionLabel = actionLabel
        self.action = action
    }

    var body: some View {
        HStack {
            Text(title)
                .font(.title3.weight(.semibold))

            Spacer()

            if let actionLabel, let action {
                Button(actionLabel, action: action)
                    .font(.subheadline)
            }
        }
        .padding(.horizontal)
    }
}
