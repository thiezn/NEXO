import SwiftUI

enum AppTheme: String, CaseIterable, Identifiable, Sendable {
    case happyPink = "Happy Pink"
    case hackerDark = "Hacker Dark"
    case gamerFlashy = "Gamer Flashy"
    case romanticGreyRed = "Romantic Grey-Red"

    var id: String { rawValue }

    // MARK: - Platform Colors

    private static let systemBackground: Color = {
        #if canImport(UIKit)
        Color(.systemBackground)
        #else
        Color(.windowBackgroundColor)
        #endif
    }()

    private static let secondarySystemBackground: Color = {
        #if canImport(UIKit)
        Color(.secondarySystemBackground)
        #else
        Color(.controlBackgroundColor)
        #endif
    }()

    private static let tertiarySystemBackground: Color = {
        #if canImport(UIKit)
        Color(.tertiarySystemBackground)
        #else
        Color(.underPageBackgroundColor)
        #endif
    }()

    private static let systemGray6: Color = {
        #if canImport(UIKit)
        Color(.systemGray6)
        #else
        Color(.controlBackgroundColor)
        #endif
    }()

    private static let systemGray5: Color = {
        #if canImport(UIKit)
        Color(.systemGray5)
        #else
        Color(.gridColor)
        #endif
    }()

    private static let systemGray4: Color = {
        #if canImport(UIKit)
        Color(.systemGray4)
        #else
        Color(.separatorColor)
        #endif
    }()

    private static let systemGray3: Color = {
        #if canImport(UIKit)
        Color(.systemGray3)
        #else
        Color(.tertiaryLabelColor)
        #endif
    }()

    // MARK: - Light Colors

    var lightColors: ThemeColors {
        switch self {
        case .happyPink:
            ThemeColors(
                primary: .pink,
                secondary: Self.systemGray6,
                accent: .mint,
                backgroundPrimary: Self.systemBackground,
                backgroundSecondary: Self.secondarySystemBackground,
                backgroundTertiary: Self.tertiarySystemBackground,
                backgroundQuaternary: Self.systemGray6
            )
        case .hackerDark:
            ThemeColors(
                primary: .green,
                secondary: Self.systemGray3,
                accent: .cyan,
                backgroundPrimary: Color(white: 0.95),
                backgroundSecondary: Color(white: 0.90),
                backgroundTertiary: Color(white: 0.85),
                backgroundQuaternary: Color(white: 0.80)
            )
        case .gamerFlashy:
            ThemeColors(
                primary: .purple,
                secondary: .yellow,
                accent: .orange,
                backgroundPrimary: Self.systemBackground,
                backgroundSecondary: Self.secondarySystemBackground,
                backgroundTertiary: Self.tertiarySystemBackground,
                backgroundQuaternary: Self.systemGray6
            )
        case .romanticGreyRed:
            ThemeColors(
                primary: Color(.darkGray),
                secondary: .red,
                accent: .red.opacity(0.7),
                backgroundPrimary: Self.systemGray6,
                backgroundSecondary: Self.systemGray5,
                backgroundTertiary: Self.systemGray4,
                backgroundQuaternary: Self.systemGray3
            )
        }
    }

    // MARK: - Dark Colors

    var darkColors: ThemeColors {
        switch self {
        case .happyPink:
            ThemeColors(
                primary: .pink,
                secondary: Color(white: 0.22),
                accent: .mint,
                backgroundPrimary: Color(white: 0.08),
                backgroundSecondary: Color(white: 0.12),
                backgroundTertiary: Color(white: 0.16),
                backgroundQuaternary: Color(white: 0.20)
            )
        case .hackerDark:
            ThemeColors(
                primary: .green,
                secondary: .gray,
                accent: .cyan,
                backgroundPrimary: .black,
                backgroundSecondary: Color(white: 0.06),
                backgroundTertiary: Color(white: 0.10),
                backgroundQuaternary: Color(white: 0.14)
            )
        case .gamerFlashy:
            ThemeColors(
                primary: .purple,
                secondary: .yellow,
                accent: .orange,
                backgroundPrimary: Color(white: 0.06),
                backgroundSecondary: Color(white: 0.10),
                backgroundTertiary: Color(white: 0.14),
                backgroundQuaternary: Color(white: 0.18)
            )
        case .romanticGreyRed:
            ThemeColors(
                primary: Color(.lightGray),
                secondary: .red,
                accent: .red.opacity(0.8),
                backgroundPrimary: Color(white: 0.10),
                backgroundSecondary: Color(white: 0.14),
                backgroundTertiary: Color(white: 0.18),
                backgroundQuaternary: Color(white: 0.22)
            )
        }
    }

    // MARK: - Resolution

    func resolvedColors(for colorScheme: ColorScheme) -> ThemeColors {
        colorScheme == .dark ? darkColors : lightColors
    }

    // MARK: - Icon

    var systemImage: String {
        switch self {
        case .happyPink: AppIcon.themeHappyPink
        case .hackerDark: AppIcon.themeHackerDark
        case .gamerFlashy: AppIcon.themeGamerFlashy
        case .romanticGreyRed: AppIcon.themeRomanticGreyRed
        }
    }
}
