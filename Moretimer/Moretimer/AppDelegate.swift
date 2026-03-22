//
//  AppDelegate.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import UserNotifications
import AuthenticationServices
import OSLog

#if os(macOS)

class AppDelegate: NSObject, NSApplicationDelegate {

    func applicationDidFinishLaunching(_ notification: Notification) {
        NotificationManager.registerForPushNotifications()
        checkAppleSignInCredential()
    }

    func application(
        _ application: NSApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        NotificationManager.handleDeviceToken(deviceToken)
    }

    func application(
        _ application: NSApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        NotificationManager.handleRegistrationError(error)
    }
}

#else

class AppDelegate: NSObject, UIApplicationDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        NotificationManager.registerForPushNotifications()
        checkAppleSignInCredential()
        return true
    }

    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        NotificationManager.handleDeviceToken(deviceToken)
    }

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        NotificationManager.handleRegistrationError(error)
    }
}

#endif

// MARK: - Shared

extension AppDelegate {

    func checkAppleSignInCredential() {
        guard let userID = UserDefaults.standard.string(forKey: UserProfileKeys.userID) else { return }

        ASAuthorizationAppleIDProvider().getCredentialState(forUserID: userID) { state, error in
            if let error {
                Logger.auth.error("Credential state check failed: \(error)")
                return
            }

            if state == .revoked || state == .notFound {
                Logger.auth.warning("Apple ID credential invalid, clearing stored data")
                for key in [UserProfileKeys.userID, UserProfileKeys.fullName, UserProfileKeys.email] {
                    UserDefaults.standard.removeObject(forKey: key)
                }
            }
        }
    }
}
