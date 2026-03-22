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
            appearanceSection
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
                let avatarData = userProfile.avatarImageData
                let initials = userProfile.initials
                HStack(spacing: 16) {
                    PhotosPicker(selection: $selectedPhoto, matching: .images) {
                        AvatarView(
                            imageData: avatarData,
                            initials: initials
                        )
                    }
                    .buttonStyle(.plain)

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

    // MARK: - Appearance

    @ViewBuilder
    private var appearanceSection: some View {
        @Bindable var tm = themeManager

        Section("Appearance") {
            Picker("Mode", selection: $tm.appearanceMode) {
                ForEach(AppearanceMode.allCases) { mode in
                    Label(mode.rawValue, systemImage: mode.systemImage)
                        .tag(mode)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        }
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
                            .foregroundStyle(theme.lightColors.primary)
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
