//
//  Logging.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import OSLog

extension Logger {
    static let subsystem = Bundle.main.bundleIdentifier ?? "nl.mortimer.moretimer"

    static let storage = Logger(subsystem: subsystem, category: "storage")
    static let ui = Logger(subsystem: subsystem, category: "ui")
    static let book = Logger(subsystem: subsystem, category: "book")
    static let thread = Logger(subsystem: subsystem, category: "thread")
    static let auth = Logger(subsystem: subsystem, category: "auth")
}

//extension OSSignposter {
//    static let api = OSSignposter(subsystem: Logger.subsystem, category: "api")
//    static let scanner = OSSignposter(subsystem: Logger.subsystem, category: "scanner")
//    static let storage = OSSignposter(subsystem: Logger.subsystem, category: "storage")

    // usage
    // func fetch(_ url: URL) async throws -> Data {
    //    let id = signposter.makeSignpostID()
    //    let state = signposter.beginInterval("fetch", id: id, "\(url.absoluteString, privacy: .public)")
    //    defer { signposter.endInterval("fetch", state) }
    //
    //    // … perform work …
    //     you can emit events related to this signpost using emitEvent(name, id: id, message)
    //    return Data()
    // }
//}

// Prefer using explicit categories: Logger.api/ui/scanner/storage/task.
