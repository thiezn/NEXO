import SwiftUI

struct ThemeColors: Sendable {
    let primary: Color
    let secondary: Color
    let accent: Color
    let backgroundPrimary: Color
    let backgroundSecondary: Color
    let backgroundTertiary: Color
    let backgroundQuaternary: Color
}

// MARK: - Environment Key

private struct ThemeColorsKey: EnvironmentKey {
    static let defaultValue = AppTheme.happyPink.lightColors
}

extension EnvironmentValues {
    var themeColors: ThemeColors {
        get { self[ThemeColorsKey.self] }
        set { self[ThemeColorsKey.self] = newValue }
    }
}
