//
//  NotificationManager.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import UserNotifications
import OSLog
#if os(macOS)
import AppKit
#else
import UIKit
#endif

enum NotificationManager {

    /// Requests notification authorization and registers for remote notifications.
    static func registerForPushNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .badge, .sound]
        ) { granted, error in
            if let error {
                Logger.auth.error("Push authorization error: \(error)")
                return
            }

            guard granted else {
                Logger.auth.info("Push notifications not granted")
                return
            }

            Logger.auth.info("Push notifications authorized")
            DispatchQueue.main.async {
                #if os(macOS)
                NSApplication.shared.registerForRemoteNotifications()
                #else
                UIApplication.shared.registerForRemoteNotifications()
                #endif
            }
        }
    }

    /// Converts device token to hex string and logs it.
    static func handleDeviceToken(_ deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()
        Logger.auth.info("Device push token: \(token)")
        // TODO: Send token to backend server for push delivery
    }

    /// Logs push registration failure.
    static func handleRegistrationError(_ error: Error) {
        Logger.auth.error("Failed to register for remote notifications: \(error)")
    }
}
