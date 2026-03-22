//
//  ErrorManager.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import SwiftUI
import OSLog

@MainActor @Observable
final class ErrorManager {

    private(set) var currentError: AppError?
    private(set) var isShowingError = false

    func show(_ error: AppError) {
        Logger.ui.error("App error: \(error.localizedDescription)")
        currentError = error
        withAnimation(.spring(duration: 0.3)) {
            isShowingError = true
        }
    }

    func dismiss() {
        withAnimation(.spring(duration: 0.3)) {
            isShowingError = false
        }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(350))
            if !isShowingError {
                currentError = nil
            }
        }
    }

    /// Wraps async throwing work with automatic error display.
    func catching(_ work: @Sendable () async throws -> Void) async {
        do {
            try await work()
        } catch let error as AppError {
            show(error)
        } catch {
            show(.unknown(error.localizedDescription))
        }
    }
}
