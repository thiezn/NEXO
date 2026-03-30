import SwiftUI
import AuthenticationServices
import PhotosUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ThemeManager.self) private var themeManager
    @Environment(UserProfileManager.self) private var userProfile
    @Environment(NexoService.self) private var nexoService
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var cropSession: AvatarCropSession?
    @State private var gatewayHost: String
    @State private var gatewayPort: String
    @State private var gatewayStatus: StatusResponse?
    @State private var toolsCatalog: [ToolEntry]?

    init() {
        self._gatewayHost = State(initialValue: NexoConstants.storedHost)
        self._gatewayPort = State(initialValue: String(NexoConstants.storedPort))
    }

    var body: some View {
        Form {
            accountSection
            nexoSection
            if nexoService.connectionState.isConnected {
                gatewayStatusSection
                toolsCatalogSection
            }
            appearanceSection
            themeSection
            aboutSection
        }
        .task(id: nexoService.connectionState.isConnected) {
            guard nexoService.connectionState.isConnected else {
                gatewayStatus = nil
                toolsCatalog = nil
                return
            }
            async let s = try? nexoService.status()
            async let t = try? nexoService.toolsCatalog()
            gatewayStatus = await s
            toolsCatalog = await t?.tools ?? []
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
                    cropSession = AvatarCropSession(imageData: data, initialCrop: .default, isNewImage: true)
                }
            }
        }
        .sheet(item: $cropSession) { session in
            AvatarCropView(imageData: session.imageData, initialCrop: session.initialCrop) { crop in
                if session.isNewImage {
                    userProfile.updateAvatar(session.imageData, crop: crop)
                } else {
                    userProfile.updateAvatarCrop(crop)
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
                let cropData = userProfile.avatarCropData
                let initials = userProfile.initials
                HStack(spacing: 16) {
                    Menu {
                        PhotosPicker(selection: $selectedPhoto, matching: .images) {
                            Label("Choose Photo", systemImage: "photo")
                        }
                        if let avatarData {
                            Button {
                                cropSession = AvatarCropSession(imageData: avatarData, initialCrop: cropData, isNewImage: false)
                            } label: {
                                Label("Adjust Crop", systemImage: "crop")
                            }
                            Button(role: .destructive) {
                                userProfile.removeAvatar()
                            } label: {
                                Label("Remove Photo", systemImage: "trash")
                            }
                        }
                    } label: {
                        AvatarView(
                            imageData: avatarData,
                            cropData: cropData,
                            initials: initials
                        )
                        .overlay(alignment: .bottomTrailing) {
                            Image(systemName: "camera.fill")
                                .font(.caption2)
                                .padding(4)
                                .background(.ultraThinMaterial, in: .circle)
                        }
                    }
                    .buttonStyle(.plain)

                    VStack(alignment: .leading, spacing: 4) {
                        TextField("Name", text: Binding(
                            get: { userProfile.fullName ?? "" },
                            set: { userProfile.updateFullName($0) }
                        ))
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

    // MARK: - NEXO Gateway

    private var nexoSection: some View {
        Section("NEXO Gateway") {
            LabeledContent("Status") {
                HStack(spacing: 6) {
                    Image(systemName: nexoService.connectionState.statusIcon)
                        .foregroundStyle(nexoService.connectionState.statusColor)
                    Text(nexoService.connectionState.statusText)
                        .foregroundStyle(.secondary)
                }
            }

            TextField("Host", text: $gatewayHost)
                .textContentType(.URL)
                #if os(iOS)
                .keyboardType(.URL)
                .autocapitalization(.none)
                #endif

            TextField("Port", text: $gatewayPort)
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif

            Button(nexoService.connectionState.isConnected ? "Reconnect" : "Connect") {
                let port = UInt16(gatewayPort) ?? NexoConstants.defaultPort
                Task {
                    await nexoService.updateGateway(host: gatewayHost, port: port)
                }
            }
        }
    }

    // MARK: - Gateway Status

    private var gatewayStatusSection: some View {
        Section("Gateway Status") {
            if let status = gatewayStatus {
                LabeledContent("Connected Nodes") {
                    Text("\(status.connectedNodes)")
                }
                LabeledContent("Connected Users") {
                    Text("\(status.connectedUsers)")
                }
                if !status.capabilities.isEmpty {
                    LabeledContent("Capabilities") {
                        Text(status.capabilities.joined(separator: ", "))
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: - Tools Catalog

    private var toolsCatalogSection: some View {
        Section("Tools (\(toolsCatalog?.count ?? 0))") {
            if let tools = toolsCatalog {
                if tools.isEmpty {
                    Text("No tools registered")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(tools, id: \.name) { tool in
                        LabeledContent {
                            HStack(spacing: 6) {
                                Text(tool.source)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Image(systemName: tool.available ? "checkmark.circle.fill" : "xmark.circle")
                                    .foregroundStyle(tool.available ? .green : .secondary)
                                    .font(.caption)
                            }
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(tool.name)
                                    .font(.body.monospaced())
                                Text(tool.description)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity)
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
