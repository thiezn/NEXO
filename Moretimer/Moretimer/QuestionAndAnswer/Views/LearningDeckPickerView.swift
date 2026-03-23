//
//  LearningDeckPickerView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import SwiftData

struct LearningDeckPickerView: View {
    let thread: ThreadEntity
    let learningService: LearningService
    @Environment(\.dismiss) private var dismiss
    @Environment(\.themeColors) private var colors

    @State private var decks: [LearningDeck] = []
    @State private var expandedDeckID: PersistentIdentifier?
    @State private var selectedCards: Set<PersistentIdentifier> = []

    var body: some View {
        NavigationStack {
            List {
                if decks.isEmpty {
                    ContentUnavailableView(
                        "No Decks",
                        systemImage: AppIcon.deck,
                        description: Text("Create a learning deck first.")
                    )
                } else {
                    ForEach(decks) { deck in
                        Section {
                            deckRow(deck)

                            if expandedDeckID == deck.persistentModelID {
                                ForEach(deck.cards) { card in
                                    cardRow(card)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Choose Cards")
            #if !os(macOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add \(selectedCards.count) Cards") {
                        addSelectedCards()
                    }
                    .disabled(selectedCards.isEmpty)
                }
            }
        }
        .presentationDetents([.large])
        .task {
            decks = learningService.fetchDecks()
        }
    }

    private func deckRow(_ deck: LearningDeck) -> some View {
        Button {
            withAnimation {
                if expandedDeckID == deck.persistentModelID {
                    expandedDeckID = nil
                } else {
                    expandedDeckID = deck.persistentModelID
                }
            }
        } label: {
            HStack {
                Image(systemName: AppIcon.deck)
                VStack(alignment: .leading) {
                    Text(deck.title).font(.headline)
                    Text("\(deck.cardCount) cards")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: expandedDeckID == deck.persistentModelID
                    ? "chevron.down" : "chevron.right")
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.plain)
    }

    private func cardRow(_ card: QuestionEntity) -> some View {
        let isSelected = selectedCards.contains(card.persistentModelID)
        return Button {
            if isSelected {
                selectedCards.remove(card.persistentModelID)
            } else {
                selectedCards.insert(card.persistentModelID)
            }
        } label: {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? colors.accent : .secondary)
                VStack(alignment: .leading) {
                    Text(card.questionText)
                        .font(.subheadline)
                        .lineLimit(2)
                    if let answer = card.staticAnswer {
                        Text(answer)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }

    private func addSelectedCards() {
        let allCards = decks.flatMap(\.cards)
        let cards = allCards.filter { selectedCards.contains($0.persistentModelID) }
        learningService.addCardsToThread(cards, thread: thread)
        dismiss()
    }
}
