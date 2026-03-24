import SwiftUI
import SwiftData

// MARK: - Namespace Environment Key

private struct AppNamespaceKey: EnvironmentKey {
    static let defaultValue: Namespace.ID? = nil
}

extension EnvironmentValues {
    var appNamespace: Namespace.ID? {
        get { self[AppNamespaceKey.self] }
        set { self[AppNamespaceKey.self] = newValue }
    }
}

// MARK: - Navigation Destinations Modifier

struct AppNavigationDestinations: ViewModifier {
    @Namespace private var namespace
    @Environment(\.modelContext) private var modelContext

    func body(content: Content) -> some View {
        content
            .environment(\.appNamespace, namespace)
            .navigationDestination(for: AppDestination.self) { destination in
                switch destination {
                case .book(let id):
                    if let book = modelContext.model(for: id) as? BookEntity {
                        BookReaderView(book: book)
                        #if os(iOS)
                            .navigationTransition(.zoom(sourceID: id, in: namespace))
                        #endif
                    }
                case .thread(let id):
                    if let thread = modelContext.model(for: id) as? ThreadEntity {
                        ThreadDetailView(thread: thread)
                        #if os(iOS)
                            .navigationTransition(.zoom(sourceID: id, in: namespace))
                        #endif
                    }
                }
            }
    }
}

extension View {
    func appNavigationDestinations() -> some View {
        modifier(AppNavigationDestinations())
    }

    @ViewBuilder
    func matchedTransitionSource(id: some Hashable, in namespace: Namespace.ID?) -> some View {
        if let namespace {
            self.matchedTransitionSource(id: id, in: namespace)
        } else {
            self
        }
    }
}
