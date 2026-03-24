//
//  NexoConnectionState.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import SwiftUI

enum NexoConnectionState: Equatable, Sendable {
    case disconnected
    case connecting
    case connected
    case reconnecting(attempt: Int)
    case failed(String)

    var isConnected: Bool {
        if case .connected = self { return true }
        return false
    }

    var statusText: String {
        switch self {
        case .disconnected: "Disconnected"
        case .connecting: "Connecting..."
        case .connected: "Connected"
        case .reconnecting(let attempt): "Reconnecting (\(attempt))..."
        case .failed(let msg): "Failed: \(msg)"
        }
    }

    var statusIcon: String {
        switch self {
        case .disconnected: "wifi.slash"
        case .connecting, .reconnecting: "arrow.triangle.2.circlepath"
        case .connected: "wifi"
        case .failed: "exclamationmark.triangle"
        }
    }

    var statusColor: Color {
        switch self {
        case .disconnected: .secondary
        case .connecting, .reconnecting: .orange
        case .connected: .green
        case .failed: .red
        }
    }
}
