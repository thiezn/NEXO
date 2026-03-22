//
//  ErrorOverlay.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI

struct ErrorBanner: View {
    let error: AppError
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: AppIcon.error)
                .foregroundStyle(.white)
                .font(.title3)

            Text(error.localizedDescription)
                .foregroundStyle(.white)
                .font(.subheadline.weight(.medium))
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button(action: onDismiss) {
                Image(systemName: AppIcon.close)
                    .foregroundStyle(.white.opacity(0.8))
                    .font(.title3)
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(.red.gradient, in: .rect(cornerRadius: 12))
        .padding(.horizontal)
        .padding(.top, 8)
        .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: - View Modifier

extension View {
    func loadingErrorOverlay() -> some View {
        modifier(ErrorOverlayModifier())
    }
}

struct ErrorOverlayModifier: ViewModifier {
    @Environment(ErrorManager.self) private var errorManager

    func body(content: Content) -> some View {
        content.overlay(alignment: .top) {
            if errorManager.isShowingError, let error = errorManager.currentError {
                ErrorBanner(error: error) {
                    errorManager.dismiss()
                }
            }
        }
        .animation(.spring(duration: 0.3), value: errorManager.isShowingError)
    }
}

#Preview {
    Text("Content")
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay(alignment: .top) {
            ErrorBanner(error: .unknown("Something went wrong")) {}
        }
}
