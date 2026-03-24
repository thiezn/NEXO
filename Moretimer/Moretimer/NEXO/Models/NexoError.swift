//
//  NexoError.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import Foundation

enum NexoError: LocalizedError, Sendable {
    case connectionFailed(String)
    case connectionClosed
    case handshakeFailed(String)
    case protocolMismatch(client: UInt32, server: UInt32)
    case requestFailed(NexoErrorPayload)
    case timeout
    case encodingFailed(String)
    case decodingFailed(String)

    nonisolated var errorDescription: String? {
        switch self {
        case .connectionFailed(let msg): "Connection failed: \(msg)"
        case .connectionClosed: "Connection closed"
        case .handshakeFailed(let msg): "Handshake failed: \(msg)"
        case .protocolMismatch(let client, let server):
            "Protocol mismatch: client v\(client), server v\(server)"
        case .requestFailed(let payload): "Request failed [\(payload.code)]: \(payload.message)"
        case .timeout: "Request timed out"
        case .encodingFailed(let msg): "Encoding failed: \(msg)"
        case .decodingFailed(let msg): "Decoding failed: \(msg)"
        }
    }
}
