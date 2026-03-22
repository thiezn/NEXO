//
//  ThemeManager.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI

@MainActor @Observable
final class ThemeManager {

    private static let themeKey = "appTheme"

    var selectedTheme: AppTheme {
        didSet {
            UserDefaults.standard.set(selectedTheme.rawValue, forKey: Self.themeKey)
        }
    }

    var colors: ThemeColors { selectedTheme.colors }
    var preferredColorScheme: ColorScheme? { selectedTheme.preferredColorScheme }

    init() {
        let saved = UserDefaults.standard.string(forKey: Self.themeKey) ?? ""
        self.selectedTheme = AppTheme(rawValue: saved) ?? .happyPink
    }
}
