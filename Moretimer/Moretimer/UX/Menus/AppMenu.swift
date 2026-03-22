import SwiftUI

// MARK: - Menu Action

struct MenuAction: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let handler: @MainActor () -> Void
}

// MARK: - App Menu

struct AppMenu<Label: View>: View {
    var quickActions: [MenuAction] = []
    var listSections: [[MenuAction]] = []
    var destructiveActions: [MenuAction] = []
    @ViewBuilder let label: () -> Label

    var body: some View {
        Menu {
            if !quickActions.isEmpty {
                ControlGroup {
                    ForEach(quickActions) { action in
                        Button(action.title, systemImage: action.icon) {
                            action.handler()
                        }
                    }
                }
                .controlGroupStyle(.palette)
            }

            ForEach(listSections.indices, id: \.self) { idx in
                Section {
                    ForEach(listSections[idx]) { action in
                        Button(action.title, systemImage: action.icon) {
                            action.handler()
                        }
                    }
                }
            }

            if !destructiveActions.isEmpty {
                Section {
                    ForEach(destructiveActions) { action in
                        Button(action.title, systemImage: action.icon, role: .destructive) {
                            action.handler()
                        }
                    }
                }
            }
        } label: {
            label()
        }
    }
}
