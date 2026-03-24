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
    static let avatarCrop = "userAvatarCropData"
}

@MainActor @Observable
final class UserProfileManager {

    private(set) var userID: String?
    private(set) var fullName: String?
    private(set) var email: String?
    private(set) var avatarImageData: Data?
    private(set) var avatarCropData: AvatarCropData

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
        if let cropJSON = UserDefaults.standard.data(forKey: UserProfileKeys.avatarCrop),
           let crop = try? JSONDecoder().decode(AvatarCropData.self, from: cropJSON) {
            self.avatarCropData = crop
        } else {
            self.avatarCropData = .default
        }
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
        avatarCropData = .default

        for key in [UserProfileKeys.userID, UserProfileKeys.fullName, UserProfileKeys.email, UserProfileKeys.avatar, UserProfileKeys.avatarCrop] {
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

    func updateFullName(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        fullName = trimmed.isEmpty ? nil : trimmed
        if let fullName {
            UserDefaults.standard.set(fullName, forKey: UserProfileKeys.fullName)
        } else {
            UserDefaults.standard.removeObject(forKey: UserProfileKeys.fullName)
        }
    }

    func updateAvatar(_ imageData: Data, crop: AvatarCropData = .default) {
        avatarImageData = imageData
        avatarCropData = crop
        UserDefaults.standard.set(imageData, forKey: UserProfileKeys.avatar)
        if let cropJSON = try? JSONEncoder().encode(crop) {
            UserDefaults.standard.set(cropJSON, forKey: UserProfileKeys.avatarCrop)
        }
    }

    func updateAvatarCrop(_ crop: AvatarCropData) {
        avatarCropData = crop
        if let cropJSON = try? JSONEncoder().encode(crop) {
            UserDefaults.standard.set(cropJSON, forKey: UserProfileKeys.avatarCrop)
        }
    }

    func removeAvatar() {
        avatarImageData = nil
        avatarCropData = .default
        UserDefaults.standard.removeObject(forKey: UserProfileKeys.avatar)
        UserDefaults.standard.removeObject(forKey: UserProfileKeys.avatarCrop)
    }
}
