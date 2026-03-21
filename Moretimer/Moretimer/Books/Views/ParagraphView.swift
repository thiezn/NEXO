import SwiftUI

struct ParagraphView: View {
    let paragraph: ParagraphEntity
    let fontSize: Double
    let fontDesign: Font.Design

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(paragraph.images) { imageRef in
                if let data = imageRef.imageData {
                    imageFromData(data)
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }

            Text(paragraph.fullText)
                .font(.system(size: fontSize, design: fontDesign))
                .lineSpacing(fontSize * 0.4)
        }
    }
}
