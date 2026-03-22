import SwiftUI

// MARK: - Top Level Toolbar

struct TopLevelToolbarContent: ToolbarContent {
    let avatarData: Data?
    let avatarInitials: String
    let onAvatarTap: () -> Void
    var quickActions: [MenuAction] = []
    var listSections: [[MenuAction]] = []
    var destructiveActions: [MenuAction] = []

    private var hasMenuContent: Bool {
        !quickActions.isEmpty || !listSections.isEmpty || !destructiveActions.isEmpty
    }

    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            if hasMenuContent {
                AppMenu(
                    quickActions: quickActions,
                    listSections: listSections,
                    destructiveActions: destructiveActions
                ) {
                    Label("More", systemImage: AppIcon.more)
                }
            }

            Button {
                onAvatarTap()
            } label: {
                AvatarView(imageData: avatarData, initials: avatarInitials, size: 28)
            }
            .buttonStyle(.plain)
        }
    }
}

// MARK: - Detail Toolbar

struct DetailToolbarContent: ToolbarContent {
    let primaryAction: MenuAction?
    var quickActions: [MenuAction] = []
    var listSections: [[MenuAction]] = []
    var destructiveActions: [MenuAction] = []

    private var hasMenuContent: Bool {
        !quickActions.isEmpty || !listSections.isEmpty || !destructiveActions.isEmpty
    }

    var body: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            if let primaryAction {
                if hasMenuContent {
                    // Show primary action directly + overflow menu
                    Button(primaryAction.title, systemImage: primaryAction.icon) {
                        primaryAction.handler()
                    }

                    AppMenu(
                        quickActions: quickActions,
                        listSections: listSections,
                        destructiveActions: destructiveActions
                    ) {
                        Label("More", systemImage: AppIcon.more)
                    }
                } else {
                    // Single action shown directly
                    Button(primaryAction.title, systemImage: primaryAction.icon) {
                        primaryAction.handler()
                    }
                }
            } else if hasMenuContent {
                // No primary, just overflow
                AppMenu(
                    quickActions: quickActions,
                    listSections: listSections,
                    destructiveActions: destructiveActions
                ) {
                    Label("More", systemImage: AppIcon.more)
                }
            }
        }
    }
}
