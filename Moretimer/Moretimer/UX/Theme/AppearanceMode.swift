import SwiftUI

enum AppearanceMode: String, CaseIterable, Identifiable, Sendable {
    case auto = "Auto"
    case light = "Light"
    case dark = "Dark"

    var id: String { rawValue }

    var colorScheme: ColorScheme? {
        switch self {
        case .auto: nil
        case .light: .light
        case .dark: .dark
        }
    }

    var systemImage: String {
        switch self {
        case .auto: AppIcon.appearanceAuto
        case .light: AppIcon.appearanceLight
        case .dark: AppIcon.appearanceDark
        }
    }
}
