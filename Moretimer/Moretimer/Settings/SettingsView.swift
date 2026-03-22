//
//  SettingsView.swift
//  Moretimer
//
//  Created by Mortimer, M (Mathijs) on 22/03/2026.
//

import SwiftUI
import AuthenticationServices
import PhotosUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ThemeManager.self) private var themeManager
    @Environment(UserProfileManager.self) private var userProfile
    @State private var selectedPhoto: PhotosPickerItem?

    var body: some View {
        Form {
            accountSection
            themeSection
            aboutSection
        }
        .navigationTitle("Settings")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
        }
        .onChange(of: selectedPhoto) { _, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self) {
                    userProfile.updateAvatar(data)
                }
            }
        }
    }

    // MARK: - Account

    @ViewBuilder
    private var accountSection: some View {
        Section("Account") {
            if userProfile.isSignedIn {
                HStack(spacing: 16) {
                    avatarView
                    VStack(alignment: .leading, spacing: 4) {
                        Text(userProfile.fullName ?? "User")
                            .font(.headline)
                        if let email = userProfile.email {
                            Text(email)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.vertical, 4)

                Button("Sign Out", role: .destructive) {
                    userProfile.signOut()
                }
            } else {
                SignInWithAppleButton(.signIn) { request in
                    request.requestedScopes = [.fullName, .email]
                } onCompletion: { result in
                    userProfile.handleSignIn(result)
                }
                .signInWithAppleButtonStyle(.whiteOutline)
                .frame(height: 44)
            }
        }
    }

    @ViewBuilder
    private var avatarView: some View {
        PhotosPicker(selection: $selectedPhoto, matching: .images) {
            Group {
                if let data = userProfile.avatarImageData {
                    imageFromData(data, contentMode: .fill)
                        .frame(width: 56, height: 56)
                        .clipShape(.circle)
                } else {
                    Text(userProfile.initials)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(width: 56, height: 56)
                        .background(.blue.gradient, in: .circle)
                }
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Theme

    @ViewBuilder
    private var themeSection: some View {
        @Bindable var tm = themeManager

        Section("Theme") {
            Picker("Theme", selection: $tm.selectedTheme) {
                ForEach(AppTheme.allCases) { theme in
                    Label {
                        Text(theme.rawValue)
                    } icon: {
                        Image(systemName: theme.systemImage)
                            .foregroundStyle(theme.colors.primary)
                    }
                    .tag(theme)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        }
    }

    // MARK: - About

    private var aboutSection: some View {
        Section("About") {
            LabeledContent("Version") {
                Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")
            }
            LabeledContent("Build") {
                Text(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1")
            }
        }
    }
}
