import SwiftUI

struct ImageAnalyzeSheet: View {
    let imageData: Data
    let onSubmit: (Data, String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var prompt = ""
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    @FocusState private var isPromptFocused: Bool

    private let cropSize = CGSize(width: 280, height: 200)

    private var trimmedPrompt: String {
        prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                cropArea
                    .padding(.top, 8)

                TextField("Ask about this image...", text: $prompt, axis: .vertical)
                    .lineLimit(2...5)
                    .textFieldStyle(.plain)
                    .padding(12)
                    .background(.secondary.opacity(0.1), in: .rect(cornerRadius: 12))
                    .padding(.horizontal)
                    .focused($isPromptFocused)

                Spacer()
            }
            .navigationTitle("Analyze Image")
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Send") {
                        guard !trimmedPrompt.isEmpty else { return }
                        let croppedData = renderCroppedImage()
                        onSubmit(croppedData, trimmedPrompt)
                        dismiss()
                    }
                    .disabled(trimmedPrompt.isEmpty)
                }
            }
        }
        .presentationDetents([.medium, .large])
        .onAppear { isPromptFocused = true }
    }

    // MARK: - Crop Area

    private var cropArea: some View {
        ZStack {
            Color.black

            imageFromData(imageData, contentMode: .fit)
                .scaleEffect(scale)
                .offset(offset)
                .gesture(combinedGesture)

            RectCropOverlay(cropSize: cropSize)
        }
        .frame(height: 240)
        .clipShape(.rect(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var combinedGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                scale = max(1.0, lastScale * value.magnification)
            }
            .onEnded { _ in
                scale = max(1.0, scale)
                lastScale = scale
            }
            .simultaneously(with:
                DragGesture()
                    .onChanged { value in
                        offset = CGSize(
                            width: lastOffset.width + value.translation.width,
                            height: lastOffset.height + value.translation.height
                        )
                    }
                    .onEnded { _ in
                        lastOffset = offset
                    }
            )
    }

    // MARK: - Render Cropped Image

    @MainActor
    private func renderCroppedImage() -> Data {
        let renderer = ImageRenderer(content:
            imageFromData(imageData, contentMode: .fit)
                .scaleEffect(scale)
                .offset(offset)
                .frame(width: cropSize.width, height: cropSize.height)
                .clipped()
        )
        renderer.scale = 2.0

        #if canImport(UIKit)
        if let uiImage = renderer.uiImage,
           let data = uiImage.jpegData(compressionQuality: 0.85) {
            return data
        }
        #elseif canImport(AppKit)
        if let nsImage = renderer.nsImage,
           let tiff = nsImage.tiffRepresentation,
           let rep = NSBitmapImageRep(data: tiff),
           let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) {
            return data
        }
        #endif

        return imageData
    }
}

// MARK: - Crop Overlay

private struct RectCropOverlay: View {
    let cropSize: CGSize

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.5)))

                let rect = CGRect(
                    x: (size.width - cropSize.width) / 2,
                    y: (size.height - cropSize.height) / 2,
                    width: cropSize.width,
                    height: cropSize.height
                )
                context.blendMode = .clear
                context.fill(Path(rect), with: .color(.white))
            }
            .allowsHitTesting(false)
        }
    }
}
