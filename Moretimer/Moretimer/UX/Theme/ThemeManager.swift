import SwiftUI

@MainActor @Observable
final class ThemeManager {

    private static let themeKey = "appTheme"
    private static let appearanceKey = "appAppearanceMode"

    var selectedTheme: AppTheme {
        didSet {
            UserDefaults.standard.set(selectedTheme.rawValue, forKey: Self.themeKey)
        }
    }

    var appearanceMode: AppearanceMode {
        didSet {
            UserDefaults.standard.set(appearanceMode.rawValue, forKey: Self.appearanceKey)
        }
    }

    var preferredColorScheme: ColorScheme? { appearanceMode.colorScheme }

    func resolvedColors(for colorScheme: ColorScheme) -> ThemeColors {
        selectedTheme.resolvedColors(for: colorScheme)
    }

    init() {
        let savedTheme = UserDefaults.standard.string(forKey: Self.themeKey) ?? ""
        self.selectedTheme = AppTheme(rawValue: savedTheme) ?? .happyPink

        let savedAppearance = UserDefaults.standard.string(forKey: Self.appearanceKey) ?? ""
        self.appearanceMode = AppearanceMode(rawValue: savedAppearance) ?? .auto
    }
}

// MARK: - Theme Resolving Modifier

struct ThemeResolvingModifier: ViewModifier {
    @Environment(ThemeManager.self) private var themeManager
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        let resolved = themeManager.resolvedColors(for: colorScheme)
        content
            .environment(\.themeColors, resolved)
            .tint(resolved.accent)
    }
}

extension View {
    func resolveThemeColors() -> some View {
        modifier(ThemeResolvingModifier())
    }
}
