import SwiftUI
import SwiftData

struct ThreadSearchResultsView: View {
    let threads: [ThreadEntity]

    var body: some View {
        if !threads.isEmpty {
            Section("Threads") {
                ForEach(threads) { thread in
                    NavigationLink(value: AppDestination.thread(thread.persistentModelID)) {
                        ThreadRowView(thread: thread)
                    }
                }
            }
        }
    }
}
