//
//  NexoProtocol.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import Foundation

// MARK: - Enums

/// Available request methods in the gateway protocol.
/// Wire format uses kebab-case/dot-notation matching the Rust serde output.
enum NexoMethod: String, Codable, Sendable {
    case connect
    case health
    case status
    case send
    case agent
    case systemPresence = "system-presence"
    case toolsCatalog = "tools.catalog"
}

/// Server-push event kinds.
enum NexoEventKind: String, Codable, Sendable {
    case tick
    case agent
    case presence
    case shutdown
    case heartbeat
    case cron
}

/// Connection role: user (control-plane client) or node (capability host).
enum NexoRole: String, Codable, Sendable {
    case user
    case node
}

/// Authorization scopes for user-role connections.
enum NexoScope: String, Codable, Sendable {
    case userRead = "user.read"
    case userWrite = "user.write"
    case userAdmin = "user.admin"
}

/// Platform the client is running on.
enum NexoPlatform: String, Codable, Sendable {
    case macos
    case ios
    case linux
    case windows

    nonisolated static var current: NexoPlatform {
        #if os(macOS)
        .macos
        #elseif os(iOS)
        .ios
        #else
        .linux
        #endif
    }
}

// MARK: - Frame Envelope

/// Top-level WebSocket frame envelope.
///
/// Wire format uses `"type"` as the discriminator tag:
/// - `{"type":"request", "id":"...", "method":"...", "params":{...}}`
/// - `{"type":"response", "id":"...", "ok":true, "payload":{...}}`
/// - `{"type":"event", "event":"...", "payload":{...}}`
enum NexoFrame: Sendable {
    case request(id: String, method: NexoMethod, params: Data)
    case response(FrameResponse)
    case event(FrameEvent)

    nonisolated static func newId() -> String {
        UUID().uuidString.lowercased()
    }

    nonisolated static func request(method: NexoMethod, params: some Encodable & Sendable) throws -> NexoFrame {
        let data = try JSONEncoder.nexo.encode(params)
        return .request(id: newId(), method: method, params: data)
    }
}

/// Response frame payload.
struct FrameResponse: Sendable {
    let id: String
    let ok: Bool
    let rawPayload: Data?
    let error: NexoErrorPayload?

    nonisolated func payload<T: Decodable>(as type: T.Type) throws -> T {
        guard let rawPayload else {
            throw NexoError.decodingFailed("Response has no payload")
        }
        return try JSONDecoder.nexo.decode(T.self, from: rawPayload)
    }
}

/// Event frame payload.
struct FrameEvent: Sendable {
    let event: NexoEventKind
    let rawPayload: Data
    let seq: UInt64?
    let stateVersion: UInt64?

    nonisolated func payload<T: Decodable>(as type: T.Type) throws -> T {
        try JSONDecoder.nexo.decode(T.self, from: rawPayload)
    }
}

// MARK: - Connect Handshake

/// Client identity included in the connect handshake.
struct NexoClientInfo: Codable, Sendable {
    let id: String
    let version: String
    let platform: NexoPlatform
}

/// Stable device identity for pairing.
struct NexoDeviceInfo: Codable, Sendable {
    let id: String
}

/// Parameters for the `connect` handshake request.
struct NexoConnectParams: Codable, Sendable {
    let minProtocol: UInt32
    let maxProtocol: UInt32
    let client: NexoClientInfo
    let role: NexoRole
    var scopes: [NexoScope] = []
    var capabilities: [String] = []
    var commands: [String] = []
    var locale: String?
    var userAgent: String?
    var device: NexoDeviceInfo?
}

/// Tick/heartbeat policy sent in the hello-ok response.
struct NexoPolicy: Codable, Sendable {
    let tickIntervalMs: UInt64
}

/// Successful connect response payload.
struct NexoHelloOk: Codable, Sendable {
    let payloadType: String
    let protocolVersion: UInt32
    let policy: NexoPolicy

    private enum CodingKeys: String, CodingKey {
        case payloadType = "type"
        case protocolVersion = "protocol"
        case policy
    }
}

// MARK: - Error Payload

/// Wire-format error payload included in error responses.
struct NexoErrorPayload: Codable, Sendable, Equatable {
    let code: String
    let message: String
}

// MARK: - Method Request Params

struct HealthParams: Codable, Sendable {}
struct StatusParams: Codable, Sendable {}

struct SendParams: Codable, Sendable {
    let target: String
    let payload: JSONValue
    let idempotencyKey: String
}

struct AgentParams: Codable, Sendable {
    let prompt: String
    let idempotencyKey: String
    var context: JSONValue?
}

struct SystemPresenceParams: Codable, Sendable {
    let status: String
}

struct ToolsCatalogParams: Codable, Sendable {
    var filter: String?
}

// MARK: - Method Response Payloads

struct HealthResponse: Codable, Sendable {
    let status: String
    let uptimeSecs: UInt64
}

struct StatusResponse: Codable, Sendable {
    let connectedUsers: UInt32
    let connectedNodes: UInt32
    let capabilities: [String]
}

struct SendResponse: Codable, Sendable {
    let delivered: Bool
}

struct AgentResponse: Codable, Sendable {
    let runId: String
    let status: String
    var summary: String?
}

struct ToolEntry: Codable, Sendable {
    let name: String
    let description: String
    let source: String
    let available: Bool
}

struct ToolsCatalogResponse: Codable, Sendable {
    let tools: [ToolEntry]
}

// MARK: - Event Payloads

struct TickPayload: Codable, Sendable {
    let timestamp: String
    let seq: UInt64
}

struct AgentEventPayload: Codable, Sendable {
    let runId: String
    let status: String
    var content: String?
}

struct PresencePayload: Codable, Sendable {
    let clientId: String
    let role: NexoRole
    let status: String
}

struct ShutdownPayload: Codable, Sendable {
    let reason: String
}

struct HeartbeatPayload: Codable, Sendable {}

struct CronPayload: Codable, Sendable {
    let jobId: String
    let name: String
}
