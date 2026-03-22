//
//  AppTheme.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI

// MARK: - Theme Colors

struct ThemeColors: Sendable {
    let primary: Color
    let secondary: Color
    let accent: Color
    let background: Color
}

// MARK: - App Theme

enum AppTheme: String, CaseIterable, Identifiable, Sendable {
    case happyPink = "Happy Pink"
    case hackerDark = "Hacker Dark"
    case gamerFlashy = "Gamer Flashy"
    case romanticGreyRed = "Romantic Grey-Red"

    var id: String { rawValue }

    private static let systemBackground: Color = {
        #if canImport(UIKit)
        Color(.systemBackground)
        #else
        Color(.windowBackgroundColor)
        #endif
    }()

    private static let systemGray6: Color = {
        #if canImport(UIKit)
        Color(.systemGray6)
        #else
        Color(.controlBackgroundColor)
        #endif
    }()

    var colors: ThemeColors {
        switch self {
        case .happyPink:
            ThemeColors(
                primary: .pink,
                secondary: .white,
                accent: .mint,
                background: Self.systemBackground
            )
        case .hackerDark:
            ThemeColors(
                primary: .green,
                secondary: .gray,
                accent: .cyan,
                background: .black
            )
        case .gamerFlashy:
            ThemeColors(
                primary: .purple,
                secondary: .yellow,
                accent: .orange,
                background: Self.systemBackground
            )
        case .romanticGreyRed:
            ThemeColors(
                primary: Color(.darkGray),
                secondary: .red,
                accent: .red.opacity(0.7),
                background: Self.systemGray6
            )
        }
    }

    var preferredColorScheme: ColorScheme? {
        switch self {
        case .happyPink: .light
        case .hackerDark: .dark
        case .gamerFlashy, .romanticGreyRed: nil
        }
    }

    var systemImage: String {
        switch self {
        case .happyPink: "heart.fill"
        case .hackerDark: "terminal.fill"
        case .gamerFlashy: "gamecontroller.fill"
        case .romanticGreyRed: "flame.fill"
        }
    }
}
