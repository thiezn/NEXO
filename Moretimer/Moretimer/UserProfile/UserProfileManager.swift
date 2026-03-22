//
//  UserProfileManager.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import Foundation
import AuthenticationServices
import OSLog

nonisolated enum UserProfileKeys {
    static let userID = "appleUserID"
    static let fullName = "appleUserFullName"
    static let email = "appleUserEmail"
    static let avatar = "userAvatarImage"
}

@MainActor @Observable
final class UserProfileManager {

    private(set) var userID: String?
    private(set) var fullName: String?
    private(set) var email: String?
    private(set) var avatarImageData: Data?

    var isSignedIn: Bool { userID != nil }

    var initials: String {
        guard let name = fullName else { return "?" }
        let parts = name.split(separator: " ")
        let firstInitial = parts.first?.first.map(String.init) ?? ""
        let lastInitial = parts.count > 1 ? parts.last?.first.map(String.init) ?? "" : ""
        return firstInitial + lastInitial
    }

    init() {
        self.userID = UserDefaults.standard.string(forKey: UserProfileKeys.userID)
        self.fullName = UserDefaults.standard.string(forKey: UserProfileKeys.fullName)
        self.email = UserDefaults.standard.string(forKey: UserProfileKeys.email)
        self.avatarImageData = UserDefaults.standard.data(forKey: UserProfileKeys.avatar)
    }

    func handleSignIn(_ result: Result<ASAuthorization, Error>) {
        switch result {
        case .success(let authorization):
            guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else {
                Logger.auth.error("Unexpected credential type")
                return
            }

            userID = credential.user
            UserDefaults.standard.set(credential.user, forKey: UserProfileKeys.userID)

            if let name = credential.fullName {
                let formatted = [name.givenName, name.familyName]
                    .compactMap { $0 }
                    .joined(separator: " ")
                if !formatted.isEmpty {
                    fullName = formatted
                    UserDefaults.standard.set(formatted, forKey: UserProfileKeys.fullName)
                }
            }

            if let emailAddress = credential.email {
                email = emailAddress
                UserDefaults.standard.set(emailAddress, forKey: UserProfileKeys.email)
            }

            Logger.auth.info("Signed in with Apple ID: \(credential.user)")

        case .failure(let error):
            Logger.auth.error("Sign in failed: \(error)")
        }
    }

    func signOut() {
        userID = nil
        fullName = nil
        email = nil
        avatarImageData = nil

        for key in [UserProfileKeys.userID, UserProfileKeys.fullName, UserProfileKeys.email, UserProfileKeys.avatar] {
            UserDefaults.standard.removeObject(forKey: key)
        }

        Logger.auth.info("Signed out")
    }

    func checkCredentialState() async {
        guard let userID else { return }

        do {
            let state = try await ASAuthorizationAppleIDProvider().credentialState(forUserID: userID)
            if state == .revoked || state == .notFound {
                Logger.auth.warning("Apple ID credential no longer valid, signing out")
                signOut()
            }
        } catch {
            Logger.auth.error("Failed to check credential state: \(error)")
        }
    }

    func updateAvatar(_ imageData: Data) {
        avatarImageData = imageData
        UserDefaults.standard.set(imageData, forKey: UserProfileKeys.avatar)
    }

    func removeAvatar() {
        avatarImageData = nil
        UserDefaults.standard.removeObject(forKey: UserProfileKeys.avatar)
    }
}
