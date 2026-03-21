import UniformTypeIdentifiers
import CoreTransferable

extension UTType {
    static let bookJSON = UTType(exportedAs: "com.moretimer.book", conformingTo: .json)
}

extension BookOutputJSON: Transferable {
    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .bookJSON)
    }
}
