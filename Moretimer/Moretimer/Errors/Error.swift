//
//  Error.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation

enum AppError: LocalizedError, Sendable, Equatable {

    // Generic
    case unknown(String)
    case network(String)

    // Book-related
    case bookImportFailed(String)
    case bookNotFound

    // Thread-related
    case threadCreationFailed(String)
    case messageSendFailed(String)

    // Auth-related
    case signInFailed(String)
    case signInCancelled
    case credentialRevoked

    // Storage
    case storageFailed(String)

    var errorDescription: String? {
        switch self {
        case .unknown(let msg): "An unexpected error occurred: \(msg)"
        case .network(let msg): "Network error: \(msg)"
        case .bookImportFailed(let msg): "Book import failed: \(msg)"
        case .bookNotFound: "Book not found"
        case .threadCreationFailed(let msg): "Could not create thread: \(msg)"
        case .messageSendFailed(let msg): "Could not send message: \(msg)"
        case .signInFailed(let msg): "Sign in failed: \(msg)"
        case .signInCancelled: "Sign in was cancelled"
        case .credentialRevoked: "Your Apple ID credential has been revoked"
        case .storageFailed(let msg): "Storage error: \(msg)"
        }
    }
}
