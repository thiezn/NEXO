//
//  ThreadSearchResultsView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct ThreadSearchResultsView: View {
    @Environment(NavigationManager.self) private var navManager
    let threads: [ThreadEntity]

    var body: some View {
        if !threads.isEmpty {
            Section("Threads") {
                ForEach(threads) { thread in
                    Button {
                        navManager.navigateToThread(thread.persistentModelID)
                    } label: {
                        ThreadRowView(thread: thread)
                    }
                    .tint(.primary)
                }
            }
        }
    }
}
