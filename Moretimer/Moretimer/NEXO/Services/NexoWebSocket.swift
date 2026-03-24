//
//  NexoWebSocket.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 23/03/2026.
//

import Foundation
import Network
import OSLog

/// Low-level WebSocket transport for the NEXO gateway.
///
/// Owns the `NWConnection`, manages send/receive, and correlates
/// request IDs to pending response continuations. Runs as an `actor`
/// to protect mutable state (pending requests, connection handle).
actor NexoWebSocket {

    // MARK: - State

    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "nl.mortimer.nexo.ws", qos: .userInitiated)
    private var pendingRequests: [String: (continuation: CheckedContinuation<FrameResponse, any Error>, timeout: Task<Void, Never>)] = [:]
    private var frameContinuation: AsyncStream<NexoFrame>.Continuation?
    private(set) var isConnected = false

    let host: String
    let port: UInt16
    let authToken: String

    // MARK: - Init

    init(
        host: String = NexoConstants.defaultHost,
        port: UInt16 = NexoConstants.defaultPort,
        authToken: String = NexoConstants.authToken
    ) {
        self.host = host
        self.port = port
        self.authToken = authToken
    }

    // MARK: - Connect

    func connect() async throws {
        let wsOptions = NWProtocolWebSocket.Options(.version13)
        wsOptions.autoReplyPing = true
        wsOptions.setAdditionalHeaders([
            (name: NexoConstants.authHeader, value: authToken)
        ])

        let params = NWParameters.tcp
        params.defaultProtocolStack.applicationProtocols.insert(wsOptions, at: 0)

        let conn = NWConnection(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(rawValue: port)!,
            using: params
        )

        // Bridge NWConnection state callbacks to an AsyncStream so we can
        // await the `.ready` state from within the actor.
        let (stateStream, stateCont) = AsyncStream.makeStream(of: NWConnection.State.self)

        conn.stateUpdateHandler = { state in
            stateCont.yield(state)
        }
        conn.start(queue: queue)

        // Await .ready or bail on error
        for await state in stateStream {
            switch state {
            case .ready:
                conn.stateUpdateHandler = nil
                stateCont.finish()
                self.connection = conn
                self.isConnected = true
                Logger.nexo.info("WebSocket connected to \(self.host):\(self.port)")
                startReceiveLoop()
                return
            case .failed(let error):
                conn.stateUpdateHandler = nil
                stateCont.finish()
                conn.cancel()
                throw NexoError.connectionFailed(error.localizedDescription)
            case .waiting(let error):
                conn.stateUpdateHandler = nil
                stateCont.finish()
                conn.cancel()
                throw NexoError.connectionFailed("Waiting: \(error.localizedDescription)")
            case .cancelled:
                stateCont.finish()
                throw NexoError.connectionClosed
            default:
                continue // .setup, .preparing
            }
        }
        stateCont.finish()
        throw NexoError.connectionFailed("State stream ended unexpectedly")
    }

    // MARK: - Disconnect

    func disconnect() {
        connection?.cancel()
        connection = nil
        isConnected = false
        frameContinuation?.finish()
        frameContinuation = nil
        for (_, entry) in pendingRequests {
            entry.timeout.cancel()
            entry.continuation.resume(throwing: NexoError.connectionClosed)
        }
        pendingRequests.removeAll()
        Logger.nexo.info("WebSocket disconnected")
    }

    // MARK: - Send

    func sendFrame(_ frame: NexoFrame) async throws {
        guard let connection else { throw NexoError.connectionClosed }

        let data = try JSONEncoder.nexo.encode(frame)

        let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
        let context = NWConnection.ContentContext(
            identifier: "nexoText",
            metadata: [metadata]
        )

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            connection.send(
                content: data,
                contentContext: context,
                isComplete: true,
                completion: .contentProcessed { error in
                    if let error {
                        continuation.resume(throwing: NexoError.connectionFailed(error.localizedDescription))
                    } else {
                        continuation.resume()
                    }
                }
            )
        }
    }

    // MARK: - Request / Response

    /// Send a request and await the correlated response.
    /// `paramsData` must be pre-encoded JSON (encoding happens on the caller's actor).
    func request(_ method: NexoMethod, paramsData: Data) async throws -> FrameResponse {
        let id = NexoFrame.newId()
        let frame = NexoFrame.request(id: id, method: method, params: paramsData)

        return try await withCheckedThrowingContinuation { continuation in
            let timeoutTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(NexoConstants.requestTimeoutSeconds))
                await self?.expirePending(id: id)
            }

            pendingRequests[id] = (continuation: continuation, timeout: timeoutTask)

            Task { [weak self] in
                do {
                    try await self?.sendFrame(frame)
                } catch {
                    if let c = await self?.removePending(id: id) {
                        c.resume(throwing: error)
                    }
                }
            }
        }
    }

    // MARK: - Frame Stream

    func frames() -> AsyncStream<NexoFrame> {
        // Finish any previous stream
        frameContinuation?.finish()

        let (stream, continuation) = AsyncStream.makeStream(of: NexoFrame.self)
        frameContinuation = continuation
        return stream
    }

    // MARK: - Receive Loop

    private func startReceiveLoop() {
        guard let connection else { return }
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            Task {
                await self.processReceived(data: data, context: context, error: error)
                await self.startReceiveLoop()
            }
        }
    }

    private func processReceived(
        data: Data?,
        context: NWConnection.ContentContext?,
        error: NWError?
    ) {
        if let error {
            Logger.nexo.error("WebSocket receive error: \(error.localizedDescription)")
            isConnected = false
            frameContinuation?.finish()
            failAllPending(NexoError.connectionFailed(error.localizedDescription))
            return
        }

        guard let data else { return }

        // Extract WebSocket metadata to check opcode
        guard let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition)
                as? NWProtocolWebSocket.Metadata else {
            return
        }

        switch metadata.opcode {
        case .text:
            decodeAndDispatch(data)
        case .close:
            Logger.nexo.info("WebSocket server sent close frame")
            isConnected = false
            frameContinuation?.finish()
            failAllPending(NexoError.connectionClosed)
        default:
            Logger.nexo.debug("Ignoring WebSocket opcode: \(String(describing: metadata.opcode))")
        }
    }

    private func decodeAndDispatch(_ data: Data) {
        let frame: NexoFrame
        do {
            frame = try JSONDecoder.nexo.decode(NexoFrame.self, from: data)
        } catch {
            Logger.nexo.error("Failed to decode frame: \(error.localizedDescription)")
            return
        }

        switch frame {
        case .response(let response):
            // Match to a pending request continuation
            if let entry = pendingRequests.removeValue(forKey: response.id) {
                entry.timeout.cancel()
                entry.continuation.resume(returning: response)
            } else {
                Logger.nexo.warning("Received response with no matching request: \(response.id)")
            }
        case .event, .request:
            // Forward to the frame stream for NexoService to consume
            frameContinuation?.yield(frame)
        }
    }

    // MARK: - Pending Request Helpers

    private func expirePending(id: String) {
        if let entry = pendingRequests.removeValue(forKey: id) {
            entry.continuation.resume(throwing: NexoError.timeout)
        }
    }

    private func removePending(id: String) -> CheckedContinuation<FrameResponse, any Error>? {
        guard let entry = pendingRequests.removeValue(forKey: id) else { return nil }
        entry.timeout.cancel()
        return entry.continuation
    }

    private func failAllPending(_ error: any Error) {
        let pending = pendingRequests
        pendingRequests.removeAll()
        for (_, entry) in pending {
            entry.timeout.cancel()
            entry.continuation.resume(throwing: error)
        }
    }
}
