import UniformTypeIdentifiers
import CoreTransferable

nonisolated extension UTType {
    static let bookJSON = UTType(exportedAs: "com.moretimer.book", conformingTo: .json)
}

extension BookOutputJSON: Transferable {
    nonisolated static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .bookJSON)
    }
}


