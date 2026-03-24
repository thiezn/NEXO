import SwiftUI

struct AvatarCropData: Codable, Equatable {
    var scale: CGFloat
    /// Horizontal offset as a fraction of the crop circle diameter.
    var offsetX: CGFloat
    /// Vertical offset as a fraction of the crop circle diameter.
    var offsetY: CGFloat

    static let `default` = AvatarCropData(scale: 1.0, offsetX: 0, offsetY: 0)
}

struct AvatarCropSession: Identifiable {
    let id = UUID()
    let imageData: Data
    let initialCrop: AvatarCropData
    let isNewImage: Bool
}

struct AvatarCropView: View {
    let imageData: Data
    let initialCrop: AvatarCropData
    let onSave: (AvatarCropData) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var scale: CGFloat
    @State private var lastScale: CGFloat
    @State private var offset: CGSize
    @State private var lastOffset: CGSize

    private let cropSize: CGFloat = 280

    init(imageData: Data, initialCrop: AvatarCropData = .default, onSave: @escaping (AvatarCropData) -> Void) {
        self.imageData = imageData
        self.initialCrop = initialCrop
        self.onSave = onSave
        self._scale = State(initialValue: initialCrop.scale)
        self._lastScale = State(initialValue: initialCrop.scale)
        // Denormalize: stored fractions → pixel offsets for the crop circle
        let cropSize: CGFloat = 280
        self._offset = State(initialValue: CGSize(
            width: initialCrop.offsetX * cropSize,
            height: initialCrop.offsetY * cropSize
        ))
        self._lastOffset = State(initialValue: CGSize(
            width: initialCrop.offsetX * cropSize,
            height: initialCrop.offsetY * cropSize
        ))
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                ZStack {
                    Color.black.ignoresSafeArea()

                    imageFromData(imageData, contentMode: .fit)
                        .frame(width: geo.size.width)
                        .scaleEffect(scale)
                        .offset(offset)
                        .gesture(combinedGesture)

                    CircleCropOverlay(cropSize: cropSize, frameSize: geo.size)
                }
            }
            .navigationTitle("Crop Avatar")
            .toolbarTitleDisplayMode(.inline)
            #if os(iOS)
            .toolbarBackground(.hidden, for: .navigationBar)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .foregroundStyle(.white)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        // Normalize: pixel offsets → fractions of crop circle diameter
                        onSave(AvatarCropData(
                            scale: scale,
                            offsetX: offset.width / cropSize,
                            offsetY: offset.height / cropSize
                        ))
                        dismiss()
                    }
                    .foregroundStyle(.white)
                }
            }
        }
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
}

private struct CircleCropOverlay: View {
    let cropSize: CGFloat
    let frameSize: CGSize

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.5)))

            let rect = CGRect(
                x: (size.width - cropSize) / 2,
                y: (size.height - cropSize) / 2,
                width: cropSize,
                height: cropSize
            )
            context.blendMode = .clear
            context.fill(Path(ellipseIn: rect), with: .color(.white))
        }
        .allowsHitTesting(false)
    }
}
