//
//  NexoCoding.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import Foundation

// MARK: - JSON Encoder/Decoder

extension JSONEncoder {
    // JSONEncoder is not Sendable, but this instance is effectively immutable
    // after initialization and only used for encoding (thread-safe in practice).
    nonisolated(unsafe) static let nexo = JSONEncoder()
}

extension JSONDecoder {
    nonisolated(unsafe) static let nexo = JSONDecoder()
}

// MARK: - JSONValue (type-erased Codable JSON)

/// A lightweight type-erased JSON value for dynamic payload fields.
enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    nonisolated init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let v = try? container.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? container.decode(Int64.self) {
            self = .int(v)
        } else if let v = try? container.decode(Double.self) {
            self = .double(v)
        } else if let v = try? container.decode(String.self) {
            self = .string(v)
        } else if let v = try? container.decode([JSONValue].self) {
            self = .array(v)
        } else if let v = try? container.decode([String: JSONValue].self) {
            self = .object(v)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .double(let v): try container.encode(v)
        case .string(let v): try container.encode(v)
        case .array(let v): try container.encode(v)
        case .object(let v): try container.encode(v)
        }
    }
}

// MARK: - NexoFrame Codable

extension NexoFrame: Codable {

    private enum FrameType: String, Codable {
        case request
        case response
        case event
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case id
        case method
        case params
        case ok
        case payload
        case error
        case event
        case seq
        case stateVersion
    }

    nonisolated init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let frameType = try container.decode(FrameType.self, forKey: .type)

        switch frameType {
        case .request:
            let id = try container.decode(String.self, forKey: .id)
            let method = try container.decode(NexoMethod.self, forKey: .method)
            let paramsValue = try container.decode(JSONValue.self, forKey: .params)
            let paramsData = try JSONEncoder.nexo.encode(paramsValue)
            self = .request(id: id, method: method, params: paramsData)

        case .response:
            let id = try container.decode(String.self, forKey: .id)
            let ok = try container.decode(Bool.self, forKey: .ok)
            let error = try container.decodeIfPresent(NexoErrorPayload.self, forKey: .error)
            let rawPayload: Data?
            if container.contains(.payload), !(try container.decodeNil(forKey: .payload)) {
                let payloadValue = try container.decode(JSONValue.self, forKey: .payload)
                rawPayload = try JSONEncoder.nexo.encode(payloadValue)
            } else {
                rawPayload = nil
            }
            self = .response(FrameResponse(id: id, ok: ok, rawPayload: rawPayload, error: error))

        case .event:
            let event = try container.decode(NexoEventKind.self, forKey: .event)
            let payloadValue = try container.decode(JSONValue.self, forKey: .payload)
            let rawPayload = try JSONEncoder.nexo.encode(payloadValue)
            let seq = try container.decodeIfPresent(UInt64.self, forKey: .seq)
            let stateVersion = try container.decodeIfPresent(UInt64.self, forKey: .stateVersion)
            self = .event(FrameEvent(event: event, rawPayload: rawPayload, seq: seq, stateVersion: stateVersion))
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .request(let id, let method, let params):
            try container.encode(FrameType.request, forKey: .type)
            try container.encode(id, forKey: .id)
            try container.encode(method, forKey: .method)
            let paramsValue = try JSONDecoder.nexo.decode(JSONValue.self, from: params)
            try container.encode(paramsValue, forKey: .params)

        case .response(let response):
            try container.encode(FrameType.response, forKey: .type)
            try container.encode(response.id, forKey: .id)
            try container.encode(response.ok, forKey: .ok)
            if let rawPayload = response.rawPayload {
                let payloadValue = try JSONDecoder.nexo.decode(JSONValue.self, from: rawPayload)
                try container.encode(payloadValue, forKey: .payload)
            }
            if let error = response.error {
                try container.encode(error, forKey: .error)
            }

        case .event(let event):
            try container.encode(FrameType.event, forKey: .type)
            try container.encode(event.event, forKey: .event)
            let payloadValue = try JSONDecoder.nexo.decode(JSONValue.self, from: event.rawPayload)
            try container.encode(payloadValue, forKey: .payload)
            if let seq = event.seq {
                try container.encode(seq, forKey: .seq)
            }
            if let stateVersion = event.stateVersion {
                try container.encode(stateVersion, forKey: .stateVersion)
            }
        }
    }
}
