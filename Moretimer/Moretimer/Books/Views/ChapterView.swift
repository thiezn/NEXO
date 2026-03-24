import SwiftUI
import SwiftData

struct ChapterView: View {
    let chapter: ChapterEntity
    let fontSize: Double
    let fontDesign: Font.Design

    var body: some View {
        Section {
            ForEach(chapter.displayParagraphs) { paragraph in
                ParagraphView(
                    paragraph: paragraph,
                    fontSize: fontSize,
                    fontDesign: fontDesign
                )
                .id(paragraph.persistentModelID)
            }
        } header: {
            Text(chapter.title)
                .font(.system(.title2, design: fontDesign))
                .fontWeight(.bold)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
