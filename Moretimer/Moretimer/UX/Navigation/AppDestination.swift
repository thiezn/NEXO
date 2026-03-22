import SwiftData

enum AppDestination: Hashable {
    case book(PersistentIdentifier)
    case thread(PersistentIdentifier)
}
