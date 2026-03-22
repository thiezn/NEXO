import SwiftUI

struct AvatarView: View {
    let imageData: Data?
    var initials: String = "?"
    var size: CGFloat = 56

    var body: some View {
        if let data = imageData {
            imageFromData(data, contentMode: .fill)
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
