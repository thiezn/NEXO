//
//  NexoService.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import SwiftUI
import OSLog

/// Main observable service for communicating with the NEXO Gateway.
///
/// Manages the WebSocket lifecycle, performs the protocol handshake,
/// dispatches server-push events, and provides typed convenience methods
/// for every gateway request. Integrates with `ErrorManager` to surface
/// connection problems to the user after repeated failures.
@MainActor @Observable
final class NexoService {

    // MARK: - Observable State

    private(set) var connectionState: NexoConnectionState = .disconnected
    private(set) var lastTickTimestamp: String?
    private(set) var serverUptime: UInt64?

    // MARK: - Configuration

    private(set) var host: String
    private(set) var port: UInt16

    // MARK: - Internal

    private var webSocket: NexoWebSocket?
    private var receiveTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var reconnectAttempt = 0
    private let errorManager: ErrorManager

    // MARK: - Event Stream

    private var eventContinuation: AsyncStream<FrameEvent>.Continuation?
    private var _eventStream: AsyncStream<FrameEvent>?

    /// Subscribe to server-push events. Each call creates a fresh stream;
    /// the previous subscriber's stream is finished.
    func subscribe() -> AsyncStream<FrameEvent> {
        eventContinuation?.finish()
        let (stream, continuation) = AsyncStream.makeStream(of: FrameEvent.self)
        eventContinuation = continuation
        _eventStream = stream
        return stream
    }

    // MARK: - Init

    init(
        host: String = NexoConstants.storedHost,
        port: UInt16 = NexoConstants.storedPort,
        errorManager: ErrorManager
    ) {
        self.host = host
        self.port = port
        self.errorManager = errorManager
    }

    /// Update gateway address and reconnect.
    func updateGateway(host: String, port: UInt16) async {
        self.host = host
        self.port = port
        UserDefaults.standard.set(host, forKey: NexoConstants.hostKey)
        UserDefaults.standard.set(Int(port), forKey: NexoConstants.portKey)
        await disconnect()
        await connect()
    }

    // MARK: - Connect

    func connect() async {
        guard connectionState != .connected, connectionState != .connecting else { return }
        connectionState = .connecting

        do {
            let ws = NexoWebSocket(host: host, port: port)
            webSocket = ws
            try await ws.connect()

            let helloOk = try await performHandshake(on: ws)
            connectionState = .connected
            reconnectAttempt = 0
            Logger.nexo.info("Connected to NEXO gateway, protocol v\(helloOk.protocolVersion)")

            startFrameConsumption(ws)
        } catch {
            Logger.nexo.error("Connection failed: \(error.localizedDescription)")
            connectionState = .failed(error.localizedDescription)
            scheduleReconnect()
        }
    }

    // MARK: - Disconnect

    func disconnect() async {
        reconnectTask?.cancel()
        reconnectTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        await webSocket?.disconnect()
        webSocket = nil
        connectionState = .disconnected
        eventContinuation?.finish()
        eventContinuation = nil
    }

    // MARK: - Handshake

    private func performHandshake(on ws: NexoWebSocket) async throws -> NexoHelloOk {
        let params = NexoConnectParams(
            minProtocol: NexoConstants.protocolVersion,
            maxProtocol: NexoConstants.protocolVersion,
            client: NexoClientInfo(
                id: NexoConstants.clientId,
                version: NexoConstants.clientVersion,
                platform: .current
            ),
            role: .user,
            scopes: [.userRead, .userWrite],
            capabilities: [],
            commands: [],
            locale: Locale.current.identifier,
            userAgent: "Moretimer/\(NexoConstants.clientVersion)",
            device: NexoDeviceInfo(id: deviceId)
        )

        let paramsData = try JSONEncoder.nexo.encode(params)
        let response = try await ws.request(.connect, paramsData: paramsData)

        guard response.ok else {
            let msg = response.error?.message ?? "Unknown handshake error"
            throw NexoError.handshakeFailed(msg)
        }

        let hello = try response.payload(as: NexoHelloOk.self)

        guard hello.protocolVersion == NexoConstants.protocolVersion else {
            throw NexoError.protocolMismatch(
                client: NexoConstants.protocolVersion,
                server: hello.protocolVersion
            )
        }

        return hello
    }

    // MARK: - Frame Consumption

    private func startFrameConsumption(_ ws: NexoWebSocket) {
        receiveTask = Task {
            let frameStream = await ws.frames()
            for await frame in frameStream {
                guard !Task.isCancelled else { break }
                handleFrame(frame)
            }
            guard !Task.isCancelled else { return }
            if connectionState == .connected {
                Logger.nexo.warning("Frame stream ended, connection lost")
                connectionState = .disconnected
                scheduleReconnect()
            }
        }
    }

    private func handleFrame(_ frame: NexoFrame) {
        switch frame {
        case .event(let event):
            handleEvent(event)
        case .response:
            Logger.nexo.warning("Received unmatched response in frame stream")
        case .request:
            Logger.nexo.debug("Received server-initiated request (not yet handled)")
        }
    }

    private func handleEvent(_ event: FrameEvent) {
        switch event.event {
        case .tick:
            if let tick = try? event.payload(as: TickPayload.self),
               lastTickTimestamp != tick.timestamp {
                lastTickTimestamp = tick.timestamp
            }
        case .heartbeat:
            Logger.nexo.trace("Heartbeat received")
        case .shutdown:
            Logger.nexo.warning("Server shutdown event")
            connectionState = .disconnected
        case .run, .presence, .cron:
            break
        }
        // Always forward to the public event stream
        eventContinuation?.yield(event)
    }

    // MARK: - Reconnection

    private func scheduleReconnect() {
        guard reconnectAttempt < NexoConstants.maxReconnectAttempts else {
            connectionState = .failed("Unable to reach gateway")
            errorManager.show(.nexoConnectionLost)
            return
        }

        reconnectAttempt += 1
        let delay = min(pow(2.0, Double(reconnectAttempt)), NexoConstants.maxReconnectDelaySeconds)
        connectionState = .reconnecting(attempt: reconnectAttempt)
        Logger.nexo.info("Reconnecting in \(delay)s (attempt \(self.reconnectAttempt)/\(NexoConstants.maxReconnectAttempts))")

        reconnectTask = Task {
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            await connect()
        }
    }

    /// Reset reconnection counter and try immediately.
    func retryConnection() async {
        reconnectAttempt = 0
        reconnectTask?.cancel()
        await connect()
    }

    // MARK: - Device ID

    private var deviceId: String {
        if let stored = UserDefaults.standard.string(forKey: NexoConstants.deviceIdKey) {
            return stored
        }
        let newId = UUID().uuidString
        UserDefaults.standard.set(newId, forKey: NexoConstants.deviceIdKey)
        return newId
    }

    // MARK: - Convenience Request Methods

    func health() async throws -> HealthResponse {
        let response = try await guardedRequest(.health, params: HealthParams())
        return try response.payload(as: HealthResponse.self)
    }

    func status() async throws -> StatusResponse {
        let response = try await guardedRequest(.status, params: StatusParams())
        return try response.payload(as: StatusResponse.self)
    }

    func sendMessage(target: String, payload: JSONValue, idempotencyKey: String) async throws -> SendResponse {
        let params = SendParams(target: target, payload: payload, idempotencyKey: idempotencyKey)
        let response = try await guardedRequest(.send, params: params)
        return try response.payload(as: SendResponse.self)
    }

    func runStart(
        input: String,
        idempotencyKey: String,
        sessionId: String? = nil,
        instructions: JSONValue? = nil,
        modelId: String? = nil,
        thinking: Bool? = nil
    ) async throws -> RunStartResponse {
        let params = RunStartParams(
            input: input,
            idempotencyKey: idempotencyKey,
            sessionId: sessionId,
            instructions: instructions,
            modelId: modelId,
            thinking: thinking
        )
        let response = try await guardedRequest(.runStart, params: params)
        return try response.payload(as: RunStartResponse.self)
    }

    func imageAnalyze(imageData: String, prompt: String, idempotencyKey: String) async throws -> ImageAnalyzeResponse {
        let params = ImageAnalyzeParams(imageData: imageData, prompt: prompt, idempotencyKey: idempotencyKey)
        let response = try await guardedRequest(.imageAnalyze, params: params)
        return try response.payload(as: ImageAnalyzeResponse.self)
    }

    func toolsCatalog(filter: String? = nil) async throws -> ToolsCatalogResponse {
        let params = ToolsCatalogParams(filter: filter)
        let response = try await guardedRequest(.toolsCatalog, params: params)
        return try response.payload(as: ToolsCatalogResponse.self)
    }

    func toolsExecute(tool: String, args: JSONValue, idempotencyKey: String) async throws -> ToolsExecuteResponse {
        let params = ToolsExecuteParams(tool: tool, args: args, idempotencyKey: idempotencyKey)
        let response = try await guardedRequest(.toolsExecute, params: params)
        return try response.payload(as: ToolsExecuteResponse.self)
    }

    // MARK: - Session Methods

    func sessionCreate(name: String? = nil, promptCollectionId: String? = nil) async throws -> SessionCreateResponse {
        let params = SessionCreateParams(name: name, promptCollectionId: promptCollectionId)
        let response = try await guardedRequest(.sessionCreate, params: params)
        return try response.payload(as: SessionCreateResponse.self)
    }

    func sessionList() async throws -> SessionListResponse {
        let response = try await guardedRequest(.sessionList, params: SessionListParams())
        return try response.payload(as: SessionListResponse.self)
    }

    func sessionGet(sessionId: String) async throws -> SessionGetResponse {
        let params = SessionGetParams(sessionId: sessionId)
        let response = try await guardedRequest(.sessionGet, params: params)
        return try response.payload(as: SessionGetResponse.self)
    }

    func sessionClear(sessionId: String) async throws -> SessionClearResponse {
        let params = SessionClearParams(sessionId: sessionId)
        let response = try await guardedRequest(.sessionClear, params: params)
        return try response.payload(as: SessionClearResponse.self)
    }

    // MARK: - Cron Methods

    func cronCreate(name: String, schedule: String, input: String, sessionId: String? = nil) async throws -> CronCreateResponse {
        let params = CronCreateParams(name: name, schedule: schedule, input: input, sessionId: sessionId)
        let response = try await guardedRequest(.cronCreate, params: params)
        return try response.payload(as: CronCreateResponse.self)
    }

    func cronList() async throws -> CronListResponse {
        let response = try await guardedRequest(.cronList, params: CronListParams())
        return try response.payload(as: CronListResponse.self)
    }

    func cronDelete(jobId: String) async throws -> CronDeleteResponse {
        let params = CronDeleteParams(jobId: jobId)
        let response = try await guardedRequest(.cronDelete, params: params)
        return try response.payload(as: CronDeleteResponse.self)
    }

    // MARK: - Prompt Document Methods

    func promptDocumentCreate(id: String, content: String) async throws -> PromptDocumentCreateResponse {
        let params = PromptDocumentCreateParams(id: id, content: content)
        let response = try await guardedRequest(.promptDocumentCreate, params: params)
        return try response.payload(as: PromptDocumentCreateResponse.self)
    }

    func promptDocumentList() async throws -> PromptDocumentListResponse {
        let response = try await guardedRequest(.promptDocumentList, params: PromptDocumentListParams())
        return try response.payload(as: PromptDocumentListResponse.self)
    }

    func promptDocumentDelete(id: String) async throws -> PromptDocumentDeleteResponse {
        let params = PromptDocumentDeleteParams(id: id)
        let response = try await guardedRequest(.promptDocumentDelete, params: params)
        return try response.payload(as: PromptDocumentDeleteResponse.self)
    }

    // MARK: - Prompt Collection Methods

    func promptCollectionCreate(id: String, name: String, description: String? = nil, documents: [String]) async throws -> PromptCollectionCreateResponse {
        let params = PromptCollectionCreateParams(id: id, name: name, description: description, documents: documents)
        let response = try await guardedRequest(.promptCollectionCreate, params: params)
        return try response.payload(as: PromptCollectionCreateResponse.self)
    }

    func promptCollectionList() async throws -> PromptCollectionListResponse {
        let response = try await guardedRequest(.promptCollectionList, params: PromptCollectionListParams())
        return try response.payload(as: PromptCollectionListResponse.self)
    }

    func promptCollectionDelete(id: String) async throws -> PromptCollectionDeleteResponse {
        let params = PromptCollectionDeleteParams(id: id)
        let response = try await guardedRequest(.promptCollectionDelete, params: params)
        return try response.payload(as: PromptCollectionDeleteResponse.self)
    }

    // MARK: - Private Helpers

    private func guardedRequest(_ method: NexoMethod, params: some Encodable) async throws -> FrameResponse {
        guard let webSocket, connectionState.isConnected else {
            throw NexoError.connectionClosed
        }
        // Encode on MainActor before crossing to the WebSocket actor
        let paramsData = try JSONEncoder.nexo.encode(params)
        let response = try await webSocket.request(method, paramsData: paramsData)
        if !response.ok, let error = response.error {
            throw NexoError.requestFailed(error)
        }
        return response
    }
}
