import SwiftUI

struct AvatarView: View {
    let imageData: Data?
    var cropData: AvatarCropData = .default
    var initials: String = "?"
    var size: CGFloat = 56

    var body: some View {
        if let data = imageData {
            imageFromData(data, contentMode: .fill)
                .scaleEffect(cropData.scale)
                .offset(x: cropData.offsetX * size, y: cropData.offsetY * size)
                .frame(width: size, height: size)
                .clipShape(.circle)
        } else if size <= 32 {
            Image(systemName: AppIcon.profile)
                .font(.system(size: size * 0.85))
                .foregroundStyle(.secondary)
        } else {
            Text(initials)
                .font(.system(size: size * 0.35, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: size, height: size)
                .background(.blue.gradient, in: .circle)
        }
    }
}
