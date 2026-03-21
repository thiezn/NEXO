import SwiftUI

enum ReadingMode: String, CaseIterable, Identifiable {
    case continuous = "Continuous"
    case paged = "Paged"
    var id: String { rawValue }
}

enum FontDesignOption: String, CaseIterable, Identifiable {
    case `default` = "Default"
    case serif = "Serif"
    case rounded = "Rounded"
    case monospaced = "Monospaced"

    var id: String { rawValue }

    var fontDesign: Font.Design {
        switch self {
        case .default: .default
        case .serif: .serif
        case .rounded: .rounded
        case .monospaced: .monospaced
        }
    }
}

struct BookSettingsSheet: View {
    @Binding var readingMode: ReadingMode
    @Binding var fontSize: Double
    @Binding var fontDesignOption: FontDesignOption
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Reading Mode") {
                    Picker("Mode", selection: $readingMode) {
                        ForEach(ReadingMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("Typography") {
                    VStack(alignment: .leading) {
                        Text("Font Size: \(Int(fontSize))pt")
                        Slider(value: $fontSize, in: 12...28, step: 1)
                    }

                    Picker("Font", selection: $fontDesignOption) {
                        ForEach(FontDesignOption.allCases) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                }

                Section("Preview") {
                    Text("The quick brown fox jumps over the lazy dog.")
                        .font(.system(size: fontSize, design: fontDesignOption.fontDesign))
                        .lineSpacing(fontSize * 0.4)
                }
            }
            .navigationTitle("Reading Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
