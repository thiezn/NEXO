//
//  NexoConstants.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import Foundation

enum NexoConstants {
    nonisolated static let protocolVersion: UInt32 = 3
    nonisolated static let authToken = "Tm90U29TM2N1cmU="
    nonisolated static let authHeader = "X-NEXO-AUTH"
    nonisolated static let defaultHost = "m4.kudu-decibel.ts.net"
    nonisolated static let defaultPort: UInt16 = 6969
    nonisolated static let clientId = "moretimer"
    nonisolated static let deviceIdKey = "nexo.deviceId"
    nonisolated static let hostKey = "nexo.gatewayHost"
    nonisolated static let portKey = "nexo.gatewayPort"
    nonisolated static let requestTimeoutSeconds: UInt64 = 30
    nonisolated static let maxReconnectAttempts = 10
    nonisolated static let maxReconnectDelaySeconds: Double = 60

    nonisolated static let clientVersion: String =
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"

    nonisolated static var storedHost: String {
        UserDefaults.standard.string(forKey: hostKey) ?? defaultHost
    }

    nonisolated static var storedPort: UInt16 {
        let value = UserDefaults.standard.integer(forKey: portKey)
        return value > 0 ? UInt16(value) : defaultPort
    }
}
