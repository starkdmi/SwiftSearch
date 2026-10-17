import CoreGraphics
import CoreText
import Foundation
import ImageIO
import MCP
import SemanticSearch
import Testing
import UniformTypeIdentifiers

@testable import SwiftSearchKit

/// A home and a folder of files in a temporary directory, the lexical profile:
/// no models, nothing downloaded.
struct Fixture: ~Copyable {
    let root: URL
    let home: Home
    let folder: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftsearch-tests-\(UUID().uuidString)", isDirectory: true)
        home = Home(root.appendingPathComponent("home", isDirectory: true))
        folder = root.appendingPathComponent("files", isDirectory: true)
        try home.save(Settings(profile: .lexical, snippetCharacters: 40))
        try write("Sea otters hold hands while they sleep so they do not drift apart.", "otters.txt")
        try write("# Kelp\n\nKelp forests grow up to half a metre a day.", "notes/kelp.md")
        try write("Invoice 42: two otter plush toys.", "invoices/2024/invoice-42.txt")
        try photo("photos/IMG_0001.jpg")
    }

    func write(_ text: String, _ relative: String) throws {
        let url = folder.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// A JPEG with the IPTC title and GPS a photo manager writes.
    func photo(_ relative: String) throws {
        let url = folder.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let context = try #require(
            CGContext(
                data: nil, width: 64, height: 48, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.8, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 48))
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(
            destination, try #require(context.makeImage()),
            [
                kCGImagePropertyIPTCDictionary: [kCGImagePropertyIPTCObjectName: "Harbour lighthouse"],
                kCGImagePropertyExifDictionary: [
                    kCGImagePropertyExifDateTimeOriginal: "2024:06:14 18:23:11",
                    kCGImagePropertyExifOffsetTimeOriginal: "+00:00",
                ],
            ] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
    }

    func path(_ relative: String) -> String { folder.appendingPathComponent(relative).path }

    /// A PDF whose one page is a picture of text: no text layer, as a scan.
    func scannedPDF(_ relative: String, text: String, pageNumber: String? = nil) throws {
        let width = 1_200
        let height = 400
        let bitmap = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        bitmap.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(
                string: text,
                attributes: [
                    kCTFontAttributeName as NSAttributedString.Key:
                        CTFontCreateWithName("Helvetica" as CFString, 64, nil)
                ]))
        bitmap.textPosition = CGPoint(x: 40, y: 170)
        CTLineDraw(line, bitmap)
        let image = try #require(bitmap.makeImage())

        let url = folder.appendingPathComponent(relative)
        var box = CGRect(x: 0, y: 0, width: 600, height: 200)
        let pdf = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        pdf.beginPDFPage(nil)
        pdf.draw(image, in: box)
        if let pageNumber {
            // A scanner's page number, selectable over the picture.
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(
                    string: pageNumber,
                    attributes: [
                        kCTFontAttributeName as NSAttributedString.Key:
                            CTFontCreateWithName("Helvetica" as CFString, 9, nil)
                    ]))
            pdf.textPosition = CGPoint(x: 295, y: 10)
            CTLineDraw(line, pdf)
        }
        pdf.endPDFPage()
        pdf.closePDF()
    }

    /// A PDF with a text layer: lines drawn as text, not as a picture.
    func textPDF(_ relative: String, lines: [String]) throws {
        let url = folder.appendingPathComponent(relative)
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let pdf = try #require(CGContext(url as CFURL, mediaBox: &box, nil))
        pdf.beginPDFPage(nil)
        let font = CTFontCreateWithName("Helvetica" as CFString, 11, nil)
        for (index, text) in lines.enumerated() {
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(
                    string: text, attributes: [kCTFontAttributeName as NSAttributedString.Key: font]))
            pdf.textPosition = CGPoint(x: 40, y: 750 - CGFloat(index) * 14)
            CTLineDraw(line, pdf)
        }
        pdf.endPDFPage()
        pdf.closePDF()
    }

    deinit { try? FileManager.default.removeItem(at: root) }
}

@Suite("Library, lexical profile", .serialized)
struct LibraryTests {

    static func synced(_ fixture: borrowing Fixture, added: Int = 4) async throws -> Library {
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        let summary = try await library.sync()
        #expect(summary.added == added)
        #expect(summary.failed.isEmpty)
        return library
    }

    @Test("Search, list, info and read a folder")
    func library() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)

        let answer = try await library.search("otters sleep")
        let top = try #require(answer.files.first)
        #expect(top.path == fixture.path("otters.txt"))
        #expect(top.type == "text")
        #expect(top.matches.first?.text?.count ?? 0 <= 41)  // 40 and an ellipsis

        // A photo found by its tags, a Markdown section by its heading.
        #expect(try await library.search("lighthouse").files.first?.path
            == fixture.path("photos/IMG_0001.jpg"))
        #expect(try await library.search("kelp forests").files.first?.path
            == fixture.path("notes/kelp.md"))

        let invoices = try await library.list(FileQuery(folder: fixture.path("invoices")))
        #expect(invoices.files.map(\.path) == [fixture.path("invoices/2024/invoice-42.txt")])
        let photos = try await library.list(FileQuery(type: "image"))
        #expect(photos.total == 1)
        let june = try await library.list(
            FileQuery(
                after: ISO8601DateFormatter().date(from: "2024-06-01T00:00:00Z"),
                before: ISO8601DateFormatter().date(from: "2024-07-01T00:00:00Z")))
        #expect(june.files.map(\.title) == ["Harbour lighthouse"])

        let info = try await library.info(fixture.path("photos/IMG_0001.jpg"))
        #expect(info.metadata["title"] == "Harbour lighthouse")

        guard case .text(let text, false) = try await library.read(fixture.path("otters.txt"))
        else { Issue.record("not text"); return }
        #expect(text.hasPrefix("Sea otters"))
        guard case .image(let jpeg, "image/jpeg") = try await library.read(
            fixture.path("photos/IMG_0001.jpg"))
        else { Issue.record("not an image"); return }
        #expect(!jpeg.isEmpty)

        let status = try await library.status()
        #expect(status.files == 4)
        #expect(status.byType["image"] == 1 && status.byType["text"] == 3)
        try await library.close()
    }

    @Test("Nothing outside the folders is read, and a removed folder leaves the index")
    func boundaries() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        // A file that exists but is not in the library.
        let outside = fixture.root.appendingPathComponent("home/settings.json").path
        await #expect(throws: LibraryError.self) { try await library.read(outside) }
        await #expect(throws: LibraryError.self) {
            try await library.read(fixture.path("../home/settings.json"))
        }
        // In the folder but not indexed.
        try fixture.write("new", "later.txt")
        await #expect(throws: LibraryError.self) { try await library.read(fixture.path("later.txt")) }

        #expect(try await library.removeFolder(fixture.folder.path) == 4)
        #expect(try await library.status().files == 0)
        try await library.close()
    }

    @Test("A search says where its time went; the JSON an agent reads does not")
    func searchTimings() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        let answer = try await library.search("otters sleep")
        let timings = try #require(answer.timings)
        #expect(timings.rounds == 1)
        #expect(timings.total >= timings.retrieval + timings.rerank)

        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(answer))
        #expect((json as? [String: Any])?.keys.sorted() == ["files", "notes", "query"])
        try await library.close()
    }

    @Test("Search inside one file")
    func searchWithin() async throws {
        let fixture = try Fixture()
        // Long enough to be cut into many chunks.
        let long = (1...200).map { "Paragraph \($0) is about harbour logistics and nothing else." }
            + ["The otter sanctuary opens at nine."]
            + (201...400).map { "Paragraph \($0) is about harbour logistics and nothing else." }
        try fixture.write(long.joined(separator: "\n\n"), "harbour.txt")
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        _ = try await library.sync()

        // Everywhere, "otter" finds three files; inside one, only that one.
        #expect(try await library.search("otter").files.count == 3)
        let inside = try await library.search(
            "otter", within: fixture.path("harbour.txt"), limit: 5)
        #expect(inside.files.map(\.path) == [fixture.path("harbour.txt")])
        // BM25 finds the one chunk with the word.
        #expect(inside.files[0].matches.count == 1)
        // Places, not files, are what `limit` counts inside a file.
        let places = try await library.search(
            "harbour logistics", within: fixture.path("harbour.txt"), limit: 5)
        #expect(places.files.count == 1 && places.files[0].matches.count == 5)

        await #expect(throws: LibraryError.self) {
            try await library.search("otter", within: "/etc/hosts")
        }
        let tool = await Tools.call(
            "search", ["query": "otter", "path": .string(fixture.path("otters.txt"))],
            library: library)
        guard case .text(let json, _, _)? = tool.content.first else {
            Issue.record("no text"); return
        }
        let answer = try JSONDecoder().decode(SearchAnswer.self, from: Data(json.utf8))
        #expect(answer.files.map(\.path) == [fixture.path("otters.txt")])
        try await library.close()
    }

    @Test("A folder removed while another profile was active is not searched")
    func otherProfileIndex() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        try await library.close()
        // As `remove` under another profile leaves it: the folder gone from the
        // settings, its files still in this profile's index.
        var settings = try fixture.home.loadSettings()
        settings.folders = []
        try fixture.home.save(settings)

        let reopened = try await Library.open(home: fixture.home)
        #expect(try await reopened.search("otters").files.isEmpty)
        #expect(try await reopened.list().total == 0)
        #expect(try await reopened.status().files == 0)
        await #expect(throws: LibraryError.self) {
            try await reopened.read(fixture.path("otters.txt"))
        }
        try await reopened.close()
    }

    @Test("One long document does not crowd the other files out")
    func crowding() async throws {
        let fixture = try Fixture()
        // Hundreds of chunks that all say it, more than a channel's 100
        // candidates, and one file elsewhere that says it once.
        let paragraph = Array(repeating: "The lighthouse keeper logged the lamp.", count: 20)
            .joined(separator: " ")
        try fixture.write(
            (1...600).map { "\($0). \(paragraph)" }.joined(separator: "\n\n"), "log.txt")
        try fixture.write(
            "Letters from the coast. The lighthouse keeper wrote to his sister in May.",
            "letters/sister.txt")
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        _ = try await library.sync()

        let answer = try await library.search("lighthouse keeper", limit: 10)
        #expect(
            Set(answer.files.map(\.path)).isSuperset(of: [
                fixture.path("log.txt"), fixture.path("letters/sister.txt"),
            ]))
        try await library.close()
    }

    @Test("Removing a folder keeps files a remaining folder covers")
    func overlappingFolders() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        _ = try await library.addFolder(fixture.path("notes"))
        #expect(try await library.removeFolder(fixture.folder.path) == 3)
        #expect(try await library.list().files.map(\.path) == [fixture.path("notes/kelp.md")])
        try await library.close()
    }

    @Test("A PDF page too long for one answer is continued with offset")
    func pdfOffset() async throws {
        let fixture = try Fixture()
        try fixture.textPDF(
            "report.pdf",
            lines: (1...40).map { "Line \($0) of the harbour report, about tides and berths." })
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        _ = try await library.sync()

        guard case .text(let first, true) = try await library.read(
            fixture.path("report.pdf"), page: 1, maximumCharacters: 200),
            case .text(let second, true) = try await library.read(
                fixture.path("report.pdf"), page: 1, offset: 200, maximumCharacters: 200),
            case .text(let whole, _) = try await library.read(
                fixture.path("report.pdf"), page: 1, maximumCharacters: 400)
        else {
            Issue.record("not truncated text"); return
        }
        #expect(first != second)
        #expect(first + second == whole)
        try await library.close()
    }

    @Test("A file swapped for a link to one outside the folders is not read")
    func symlinkEscape() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        let secret = fixture.root.appendingPathComponent("secret.txt")
        try "private".write(to: secret, atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(atPath: fixture.path("otters.txt"))
        try FileManager.default.createSymbolicLink(
            atPath: fixture.path("otters.txt"), withDestinationPath: secret.path)
        await #expect(throws: LibraryError.self) { try await library.read(fixture.path("otters.txt")) }
        try await library.close()
    }

    @Test("A folder replaced by a link to another is neither read nor synced")
    func folderRedirect() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        let other = fixture.root.appendingPathComponent("other", isDirectory: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try "private".write(
            to: other.appendingPathComponent("otters.txt"), atomically: true, encoding: .utf8)
        let moved = fixture.root.appendingPathComponent("moved", isDirectory: true)
        try FileManager.default.moveItem(at: fixture.folder, to: moved)
        try FileManager.default.createSymbolicLink(at: fixture.folder, withDestinationURL: other)

        await #expect(throws: LibraryError.self) { try await library.read(fixture.path("otters.txt")) }
        let summary = try await library.sync()
        #expect(summary.failed.count == 1)
        await #expect(throws: LibraryError.self) { try await library.read(fixture.path("otters.txt")) }
        try await library.close()
    }

    @Test("The engine reads only the media read_file would open")
    func engineMediaBoundary() async throws {
        let fixture = try Fixture()
        let folder = Library.resolved(fixture.folder.path)
        let built = await Recipe.build(
            settings: Settings(folders: [folder], profile: .lexical),
            storeURL: fixture.root.appendingPathComponent("boundary.sqlite"),
            boundary: FolderBoundary([folder]))
        let access = try #require(built.configuration.mediaAccess)
        let otters = URL(fileURLWithPath: fixture.path("otters.txt"))
        #expect(access(otters))

        // An indexed file swapped for a link to one outside the folders.
        let secret = fixture.root.appendingPathComponent("secret.txt")
        try "private".write(to: secret, atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: otters)
        try FileManager.default.createSymbolicLink(at: otters, withDestinationURL: secret)
        #expect(!access(otters))
        #expect(!access(secret))
    }

    @Test("Only the multimodal profile swaps its models; idle ones are released")
    func residencyPerProfile() async throws {
        let fixture = try Fixture()
        let folder = Library.resolved(fixture.folder.path)
        for (profile, sequential) in [(Profile.fast, false), (.multimodal, true)] {
            let built = await Recipe.build(
                settings: Settings(folders: [folder], profile: profile),
                storeURL: fixture.root.appendingPathComponent("\(profile).sqlite"),
                boundary: FolderBoundary([folder]))
            #expect(built.configuration.runtime.sequentialModelResidency == sequential)
            #expect(built.configuration.runtime.modelIdleTimeout == .seconds(300))
        }
    }

    @Test("A library's home is private to its owner")
    func privateHome() async throws {
        let fixture = try Fixture()
        let library = try await Library.open(home: fixture.home)
        let permissions = try FileManager.default.attributesOfItem(atPath: fixture.home.url.path)[
            .posixPermissions] as? Int
        #expect(permissions == 0o700)
        try await library.close()
    }

    @Test("A missing folder is reported and the others still sync")
    func missingFolder() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        let drive = fixture.root.appendingPathComponent("drive", isDirectory: true)
        try FileManager.default.createDirectory(at: drive, withIntermediateDirectories: true)
        _ = try await library.addFolder(drive.path)
        try FileManager.default.removeItem(at: drive)
        try fixture.write("Tide tables for the harbour.", "tides.txt")

        let summary = try await library.sync()
        #expect(summary.added == 1)
        #expect(summary.failed.count == 1 && summary.failed[0].hasPrefix(drive.path))
        #expect(summary.error == nil)
        #expect(await !library.isSyncing)
        #expect(try await library.status().files == 5)
        try await library.close()
    }

    @Test("A scanned PDF page reads as the OCR the index holds")
    func scannedPDF() async throws {
        let fixture = try Fixture()
        try fixture.scannedPDF("scan.pdf", text: "Harbour pilot licence")
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        _ = try await library.sync()
        guard case .text(let text, _) = try await library.read(fixture.path("scan.pdf")) else {
            Issue.record("not text"); return
        }
        #expect(text.contains("page 1"))
        #expect(text.localizedCaseInsensitiveContains("pilot licence"))
        try await library.close()
    }

    @Test("A scan with a selectable page number reads as its OCR, not the number")
    func scannedPDFWithPageNumber() async throws {
        let fixture = try Fixture()
        try fixture.scannedPDF("scan.pdf", text: "Harbour pilot licence", pageNumber: "1")
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        _ = try await library.sync()
        let answer = try await library.search("harbour pilot licence")
        #expect(answer.files.first?.path == fixture.path("scan.pdf"))
        guard case .text(let text, _) = try await library.read(fixture.path("scan.pdf")) else {
            Issue.record("not text"); return
        }
        #expect(text.localizedCaseInsensitiveContains("pilot licence"))
        try await library.close()
    }

    @Test("An offset past the end is an empty answer, not a crash")
    func hugeOffset() async throws {
        let fixture = try Fixture()
        try fixture.textPDF("report.pdf", lines: ["Tidal turbines need divers."])
        let library = try await Self.synced(fixture, added: 5)
        for path in ["report.pdf", "otters.txt"] {
            guard
                case .text(let text, let truncated) = try await library.read(
                    fixture.path(path), offset: .max)
            else {
                Issue.record("not text"); continue
            }
            #expect(text.isEmpty)
            #expect(!truncated)
        }
        try await library.close()
    }

    @Test("A query picture must be an indexed picture in the library")
    func queryPictureBoundary() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        // The lexical profile has nothing that sees a picture: refused, not
        // silently searched by the words alone.
        await #expect(throws: LibraryError.self) {
            try await library.search("otters", image: fixture.path("photos/IMG_0001.jpg"))
        }
        try await library.close()

        var settings = try fixture.home.loadSettings()
        settings.profile = .fast
        try fixture.home.save(settings)
        let fast = try await Library.open(home: fixture.home)
        // Outside the folders, and inside but not indexed in this profile:
        // refused before any model sees the file.
        let outside = fixture.root.appendingPathComponent("private.jpg")
        try FileManager.default.copyItem(
            at: URL(fileURLWithPath: fixture.path("photos/IMG_0001.jpg")), to: outside)
        for path in [outside.path, fixture.path("photos/IMG_0001.jpg")] {
            await #expect {
                try await fast.search("", image: path)
            } throws: { error in
                if case LibraryError.notInLibrary = error { return true }
                return false
            }
        }
        try await fast.close()
    }

    @Test("What the engine could not do reaches the answer")
    func diagnosticsNotes() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        var diagnostics = SearchDiagnostics(mode: .hybrid)
        diagnostics.rerankIgnoredQueryImages = true
        diagnostics.skippedSpaces = [.init(space: "vl", reason: "failed to load")]
        diagnostics.unavailableRepresentations = [
            .init(
                key: .init(
                    item: SemanticSearchEngine.itemID(
                        forFile: URL(fileURLWithPath: fixture.path("photos/IMG_0001.jpg"))),
                    representation: "image"),
                reason: "media missing"),
            .init(
                key: .init(
                    item: SemanticSearchEngine.itemID(
                        forFile: fixture.root.appendingPathComponent("elsewhere.jpg")),
                    representation: "image"),
                reason: "media missing"),
        ]
        let notes = await library.notes(from: diagnostics)
        #expect(notes.contains { $0.contains("words alone") })
        #expect(notes.contains { $0.contains("vl") && $0.contains("failed to load") })
        #expect(notes.contains { $0.contains("IMG_0001.jpg") && $0.contains("media missing") })
        // Outside the folders: not named.
        #expect(!notes.contains { $0.contains("elsewhere") })
        try await library.close()
    }

    @Test("Models are not released while a call runs")
    func activity() {
        let activity = Activity()
        activity.begin()
        #expect(activity.inFlight())
        activity.end()
        #expect(!activity.inFlight())
    }

    @Test("A moved file is found at its new path without being indexed again")
    func movedFile() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        try FileManager.default.moveItem(
            atPath: fixture.path("otters.txt"), toPath: fixture.path("notes/sea-otters.txt"))
        let summary = try await library.sync()
        #expect(summary.moved == 1)
        #expect(summary.added == 0 && summary.removed == 0 && summary.updated == 0)

        let answer = try await library.search("otters sleep")
        #expect(answer.files.first?.path == fixture.path("notes/sea-otters.txt"))
        let info = try await library.info(fixture.path("notes/sea-otters.txt"))
        #expect(info.metadata["fileName"] == "sea-otters.txt")
        #expect(info.metadata["contentHash"] == nil)
        guard case .text(let text, _) = try await library.read(fixture.path("notes/sea-otters.txt"))
        else {
            Issue.record("not text"); return
        }
        #expect(text.contains("Sea otters"))
        await #expect(throws: LibraryError.self) {
            try await library.read(fixture.path("otters.txt"))
        }
        try await library.close()
    }

    @Test("Folder names are words, counted from the outermost folder added")
    func folderWords() async throws {
        let fixture = try Fixture()
        try fixture.write("Tide tables for the harbour.", "Trips/Fethiye_2022/tides.txt")
        let library = try await Library.open(home: fixture.home)
        _ = try await library.addFolder(fixture.folder.path)
        // A subfolder added too is walked with its parent, not on its own.
        _ = try await library.addFolder(fixture.path("Trips"))
        _ = try await library.sync()
        _ = try await library.sync()
        let tides = fixture.path("Trips/Fethiye_2022/tides.txt")
        #expect(try await library.info(tides).metadata["folderPath"] == "Trips/Fethiye_2022")
        let answer = try await library.search("Fethiye 2022")
        #expect(answer.files.first?.path == tides)
        #expect(answer.files.first?.matches.first?.kind == "file details")
        try await library.close()
    }

    @Test("The whole disk is not a folder to add")
    func rootRefused() async throws {
        let fixture = try Fixture()
        let library = try await Library.open(home: fixture.home)
        await #expect(throws: LibraryError.self) { try await library.addFolder("/") }
        await #expect(throws: LibraryError.self) { try await library.addFolder("/.") }
        // Nor a link to it.
        let link = fixture.root.appendingPathComponent("disk")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: URL(fileURLWithPath: "/"))
        await #expect(throws: LibraryError.self) { try await library.addFolder(link.path) }
        #expect(await library.settings.folders.isEmpty)
        try await library.close()
    }

    @Test("Rounds of a crowded search are ranked together by the reranker")
    func mergedRounds() {
        func file(_ name: String) -> FileHit {
            FileHit(path: "/f/\(name)", type: "text", title: nil, score: 0, matches: [])
        }
        // Round one: a (0.2), b past the reranker's pool; round two: c (0.9).
        let merged = Library.merged(
            [file("a"), file("b"), file("c")], rerankScores: [0.2, nil, 0.9])
        #expect(merged.map(\.path) == ["/f/c", "/f/a", "/f/b"])
        // No reranker: the engine's order stands.
        let unranked = Library.merged([file("a"), file("b")], rerankScores: [nil, nil])
        #expect(unranked.map(\.path) == ["/f/a", "/f/b"])
    }

    @Test("Settings from an older version still open")
    func olderSettings() throws {
        let old = #"{"folders":["/tmp/a"],"profile":"fast","modelDirectory":"/m"}"#
        let settings = try JSONDecoder().decode(Settings.self, from: Data(old.utf8))
        #expect(settings.folders == ["/tmp/a"])
        #expect(settings.modelDirectory == "/m")
        #expect(settings.shots)
        #expect(settings.shotModel == nil)
        #expect(settings.snippetCharacters == Settings().snippetCharacters)

        let current = Settings(shots: false, shotModel: "/models/omnishotcut")
        let data = try JSONEncoder().encode(current)
        #expect(try JSONDecoder().decode(Settings.self, from: data) == current)
    }

    @Test("One process at a time")
    func lock() async throws {
        let fixture = try Fixture()
        let library = try await Library.open(home: fixture.home)
        await #expect(throws: IndexLock.Held.self) { try await Library.open(home: fixture.home) }
        try await library.close()
    }

    @Test("The tools over MCP: listed, called, and refused")
    func mcp() async throws {
        let fixture = try Fixture()
        let library = try await Self.synced(fixture)
        let server = await MCPServer.makeServer(library: library)
        let (clientTransport, serverTransport) = await InMemoryTransport.createConnectedPair()
        try await server.start(transport: serverTransport)
        let client = Client(name: "test", version: "1")
        _ = try await client.connect(transport: clientTransport)

        let tools = try await client.listTools().tools
        #expect(Set(tools.map(\.name)) == [
            "search", "list_files", "file_info", "read_file", "library_status", "sync_library",
        ])

        let (content, isError) = try await client.callTool(
            name: "search", arguments: ["query": "otter plush", "type": "text"])
        #expect(isError != true)
        guard case .text(let json, _, _)? = content.first else {
            Issue.record("no text"); return
        }
        let answer = try JSONDecoder().decode(SearchAnswer.self, from: Data(json.utf8))
        #expect(answer.files.first?.path == fixture.path("invoices/2024/invoice-42.txt"))

        let (image, _) = try await client.callTool(
            name: "read_file", arguments: ["path": .string(fixture.path("photos/IMG_0001.jpg"))])
        guard case .image(_, "image/jpeg", _, _)? = image.first else {
            Issue.record("no image"); return
        }

        let (_, refused) = try await client.callTool(
            name: "read_file", arguments: ["path": "/etc/hosts"])
        #expect(refused == true)
        let (_, badDate) = try await client.callTool(
            name: "list_files", arguments: ["after": "June"])
        #expect(badDate == true)

        await client.disconnect()
        await server.stop()
        try await library.close()
    }
}
