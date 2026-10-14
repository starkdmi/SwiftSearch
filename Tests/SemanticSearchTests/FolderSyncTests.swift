import Foundation
import Testing

@testable import SemanticSearch

extension SemanticSearchEngine {
    /// The stored row, for tests that check what a sync kept.
    func storedItem(_ id: ItemID) throws -> StoredItem? { try store.item(id) }

    func storedRepresentations(_ id: ItemID) throws -> [StoredRepresentation] {
        try store.representations(forItem: id)
    }
}

/// The names of the files a processor was run on.
private final class Processed: @unchecked Sendable {
    private let lock = NSLock()
    private var names: [String] = []
    func add(_ name: String) { lock.withLock { names.append(name) } }
    var all: [String] { lock.withLock { names } }
}

/// Any file as one item of its text, plus one child; a file named `broken`
/// fails.
private struct WholeFile: ContentProcessor {
    var processed: Processed?
    var identifier: String { "whole-file" }
    var version: String { "1" }
    var capabilities: ProcessorCapabilities { ProcessorCapabilities(producedKinds: [.originalText]) }
    var configurationFingerprint: String { "1" }

    func canProcess(_ content: RawContent) -> Bool {
        guard case .fileURL(let url) = content.payload else { return false }
        return url.pathExtension != "bin"
    }

    func process(
        _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .fileURL(let url) = content.payload else { return [] }
        processed?.add(url.lastPathComponent)
        if url.lastPathComponent.hasPrefix("broken") {
            throw CocoaError(.fileReadCorruptFile)
        }
        let text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        return [
            SearchItem(
                id: content.id, metadata: content.metadata,
                representations: [.text(text, kind: .originalText)]),
            SearchItem(
                id: ItemID("\(content.id.rawValue)#chunk-0"), parentID: content.id,
                metadata: content.metadata,
                representations: [.text("chunk of \(text)", kind: .originalText)]),
        ]
    }
}

/// Marks what it relocates, as a processor whose output depends on the path.
private struct Marking: ContentProcessor {
    var identifier: String { "marking" }
    var version: String { "1" }
    var capabilities: ProcessorCapabilities { ProcessorCapabilities() }
    var configurationFingerprint: String { "1" }
    func canProcess(_ content: RawContent) -> Bool { true }
    func process(
        _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] { [] }
    func relocate(
        _ items: [SearchItem], from old: URL, to content: RawContent
    ) async throws -> [SearchItem] {
        SearchItem.relocated(items, from: old, to: content).map {
            var item = $0
            item.metadata["relocatedBy"] = .string(identifier)
            return item
        }
    }
}

@Suite("Folder sync and listing")
struct FolderSyncTests {

    @Test("Wrappers pass a move on to what they wrap, and paths are compared standardized")
    func wrappersRelocate() async throws {
        let old = URL(fileURLWithPath: "/photos/a.jpg")
        let new = URL(fileURLWithPath: "/photos/trip/b.jpg")
        let content = RawContent(id: "b", payload: .fileURL(new), metadata: ["folders": .string("x")])
        let items = [
            SearchItem(
                id: "b", locator: SourceLocator(source: "file:///photos/./a.jpg"),
                representations: [
                    ContentRepresentation(id: "media", kind: .video, content: .mediaReference(old)),
                    ContentRepresentation(
                        id: "page", kind: .image,
                        content: .mediaReference(URL(string: "file:///photos/a.jpg#page=3")!),
                        locator: SourceLocator(source: "file:///photos/a.jpg#page=3")),
                ])
        ]
        for processor: any ContentProcessor in [
            ContentProcessorChain([Marking()]), CombinedContentProcessor([Marking()]),
        ] {
            let moved = try await processor.relocate(items, from: old, to: content)
            #expect(moved[0].metadata["relocatedBy"] == .string("marking"))
            #expect(moved[0].metadata["folders"] == .string("x"))
            #expect(moved[0].locator.source == new.absoluteString)
            if case .mediaReference(let url) = moved[0].representations[0].content {
                #expect(url == new)
            } else {
                Issue.record("not a media reference")
            }
            // A source that points into the file keeps pointing there.
            let page = moved[0].representations[1]
            #expect(page.locator?.source == new.absoluteString + "#page=3")
            if case .mediaReference(let url) = page.content {
                #expect(url.absoluteString == new.absoluteString + "#page=3")
            } else {
                Issue.record("not a media reference")
            }
        }
    }

    private static func engine(
        storeURL: URL?, processed: Processed? = nil
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text", provider: HashingEmbeddingProvider(dimensions: 64),
                        indexKind: .exact)
                ],
                reranker: nil, processor: WholeFile(processed: processed)))
        try await engine.prepare()
        return engine
    }

    @Test("A moved or renamed file keeps what was indexed from it; a copy or an edit does not")
    func moves() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url
        try Self.write("otters hold hands", "notes/otters.txt", in: root)
        try Self.write("kelp grows fast", "kelp.txt", in: root)
        let processed = Processed()
        let engine = try await Self.engine(storeURL: directory.storeURL, processed: processed)
        _ = try await engine.sync(folder: root)

        let old = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("notes/otters.txt"))
        let oldChunk = ItemID(old.rawValue + "#chunk-0")
        let slots = try await engine.storedRepresentations(oldChunk).map(\.slot)
        #expect(!slots.isEmpty)
        #expect(try await engine.storedItem(old)?.metadata["contentHash"] != nil)

        let archive = root.appendingPathComponent("archive")
        try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: true)
        try FileManager.default.moveItem(
            at: root.appendingPathComponent("notes/otters.txt"),
            to: archive.appendingPathComponent("sea-otters.txt"))
        let moved = try await engine.sync(folder: root)
        let new = SemanticSearchEngine.itemID(forFile: archive.appendingPathComponent("sea-otters.txt"))
        let newChunk = ItemID(new.rawValue + "#chunk-0")
        #expect(moved.moved == [FolderSyncReport.Move(from: old, to: new)])
        #expect(moved.added.isEmpty && moved.removed.isEmpty && moved.unchanged == 1)
        // Not processed, not embedded: the same slots, now under the new id.
        #expect(!processed.all.contains("sea-otters.txt"))
        #expect(moved.indexing.representationsWritten == 0)
        #expect(try await engine.storedRepresentations(newChunk).map(\.slot) == slots)
        #expect(try await engine.storedItem(newChunk)?.parentID == new)
        #expect(try await engine.storedItem(old) == nil)
        #expect(try await engine.storedItem(oldChunk) == nil)
        #expect(
            try await engine.storedItem(newChunk)?.metadata["folders"]?.contains(.string(archive.path))
                == true)
        // Found through its vectors and its postings, under the new path only.
        let hits = try await engine.search("otters hold hands")
        #expect(hits.hits.first?.itemID.rawValue.hasPrefix(new.rawValue) == true)
        #expect(!hits.hits.contains { $0.itemID.rawValue.hasPrefix(old.rawValue) })

        let again = try await engine.sync(folder: root)
        #expect(again.unchanged == 2 && again.moved.isEmpty)

        // A copy beside the original is a new file.
        try FileManager.default.copyItem(
            at: archive.appendingPathComponent("sea-otters.txt"),
            to: root.appendingPathComponent("copy.txt"))
        let copy = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("copy.txt"))
        let copied = try await engine.sync(folder: root)
        #expect(copied.added == [copy] && copied.moved.isEmpty)

        // Moved and edited: other bytes, processed again.
        try FileManager.default.removeItem(at: root.appendingPathComponent("kelp.txt"))
        try Self.write("kelp grows very fast", "kelp-notes.txt", in: root)
        let kelp = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("kelp.txt"))
        let edited = try await engine.sync(folder: root)
        #expect(edited.moved.isEmpty)
        #expect(edited.removed == [kelp])
        #expect(processed.all.contains("kelp-notes.txt"))
        try await engine.close()
    }

    static func write(_ text: String, _ relative: String, in root: URL) throws {
        let url = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    @Test("New, changed, unchanged and deleted files; hidden, unsupported and the index skipped")
    func sync() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url
        try Self.write("otters hold hands", "otters.txt", in: root)
        try Self.write("kelp grows fast", "sea/kelp.md", in: root)
        try Self.write("secret", ".hidden.txt", in: root)
        try Self.write("secret", ".git/config.txt", in: root)
        try Self.write("binary", "blob.bin", in: root)
        // The index itself lives in the folder it indexes.
        let engine = try await Self.engine(storeURL: directory.storeURL)

        let first = try await engine.sync(folder: root)
        let otters = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("otters.txt"))
        let kelp = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("sea/kelp.md"))
        #expect(Set(first.added) == [otters, kelp])
        #expect(first.unsupported.map(\.lastPathComponent) == ["blob.bin"])
        #expect(first.failed.isEmpty && first.removed.isEmpty)
        #expect(otters.rawValue.hasPrefix("file://"))

        let again = try await engine.sync(folder: root)
        #expect(again.unchanged == 2 && again.added.isEmpty && again.updated.isEmpty)

        try Self.write("otters hold hands while they sleep", "otters.txt", in: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("sea/kelp.md"))
        let changed = try await engine.sync(folder: root)
        #expect(changed.updated == [otters])
        #expect(changed.removed == [kelp])
        #expect(try await engine.item(kelp) == nil)
        #expect(try await engine.item(ItemID("\(kelp.rawValue)#chunk-0")) == nil)

        let hits = try await engine.search("sleep", options: SearchOptions(mode: .lexicalOnly))
        #expect(hits.hits.first?.itemID == otters)
        try await engine.close()
    }

    @Test("A folder that is not there throws and removes nothing")
    func missingFolder() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("drive")
        try Self.write("tide tables", "tides.txt", in: root)
        let engine = try await Self.engine(storeURL: directory.storeURL)
        _ = try await engine.sync(folder: root)

        try FileManager.default.removeItem(at: root)
        await #expect(throws: SemanticSearchError.self) {
            try await engine.sync(folder: root)
        }
        #expect(try await engine.listItems().total == 1)
        try await engine.close()
    }

    @Test("A failing file is reported without stopping the rest, and is not marked done")
    func failures() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("files")
        try Self.write("fine", "fine.txt", in: root)
        try Self.write("will fail", "broken.txt", in: root)
        let engine = try await Self.engine(storeURL: nil)

        let first = try await engine.sync(folder: root)
        #expect(first.added.count == 1)
        #expect(first.failed.map(\.file.lastPathComponent) == ["broken.txt"])

        // Not recorded as done: tried again, not skipped as unchanged.
        let second = try await engine.sync(folder: root)
        #expect(second.failed.count == 1)
        #expect(second.unchanged == 1)
        try await engine.close()
    }

    @Test("Listing: filters, folders, order, pages, and children only when asked")
    func listing() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("library")
        try Self.write("a", "a.txt", in: root)
        try Self.write("bb", "trips/2024/b.txt", in: root)
        try Self.write("ccc", "trips/c.md", in: root)
        let engine = try await Self.engine(storeURL: nil)
        _ = try await engine.sync(folder: root)

        let all = try await engine.listItems()
        #expect(all.total == 3)
        #expect(all.items.allSatisfy { $0.representations.isEmpty })
        // By URL: …/library/a.txt, …/library/trips/2024/b.txt, …/library/trips/c.md.
        #expect(
            all.items.map { URL(string: $0.id.rawValue)!.lastPathComponent }
                == ["a.txt", "b.txt", "c.md"])

        let trips = try await engine.listItems(
            matching: .equals(
                key: "folders", value: .string(root.appendingPathComponent("trips").path)))
        #expect(trips.total == 2)
        let deep = try await engine.listItems(
            matching: .equals(
                key: "folders", value: .string(root.appendingPathComponent("trips/2024").path)))
        #expect(deep.items.map(\.id.rawValue).first?.hasSuffix("b.txt") == true)

        let everything = try await engine.listItems(topLevelOnly: false)
        #expect(everything.total == 6)
        let chunks = try await engine.listItems(
            matching: .not(.withinParent(all.items[0].id)), topLevelOnly: false,
            includesRepresentations: true)
        #expect(chunks.total == 4)
        #expect(chunks.items.allSatisfy { !$0.representations.isEmpty })

        let page = try await engine.listItems(order: .id, limit: 2, offset: 2)
        #expect(page.total == 3 && page.items.count == 1)
        try await engine.close()
    }

    @Test("Ordered by a metadata value, missing values last either way")
    func metadataOrder() async throws {
        let engine = try await Self.engine(storeURL: nil)
        func item(_ id: String, _ metadata: Metadata) -> SearchItem {
            SearchItem(
                id: ItemID(id), metadata: metadata, representations: [.text("text \(id)")])
        }
        _ = try await engine.upsert([
            item("old", ["taken": .date(Date(timeIntervalSince1970: 1e9))]),
            item("new", ["taken": .date(Date(timeIntervalSince1970: 1.7e9))]),
            item("none", [:]),
            item("mid", ["taken": .date(Date(timeIntervalSince1970: 1.5e9))]),
        ])
        let newest = try await engine.listItems(order: .metadata("taken", descending: true))
        #expect(newest.items.map(\.id.rawValue) == ["new", "mid", "old", "none"])
        let oldest = try await engine.listItems(order: .metadata("taken"))
        #expect(oldest.items.map(\.id.rawValue) == ["old", "mid", "new", "none"])
        let since = try await engine.listItems(
            matching: .greaterThan(key: "taken", value: .date(Date(timeIntervalSince1970: 1.2e9))),
            order: .metadata("taken"))
        #expect(since.items.map(\.id.rawValue) == ["mid", "new"])
        try await engine.close()
    }

    @Test("A folder that cannot be read is reported, and what came from it stays")
    func unreadableFolder() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("files")
        try Self.write("open", "open.txt", in: root)
        try Self.write("private", "locked/inside.txt", in: root)
        let engine = try await Self.engine(storeURL: nil)
        #expect(try await engine.sync(folder: root).added.count == 2)

        let locked = root.appendingPathComponent("locked")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: locked.path)
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: locked.path)
        }
        let report = try await engine.sync(folder: root)
        #expect(report.removed.isEmpty)
        #expect(report.unreadable.map(\.lastPathComponent) == ["locked"])
        #expect(try await engine.listItems().total == 2)

        // Readable again and emptied: now it is a removal.
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path)
        try FileManager.default.removeItem(at: locked.appendingPathComponent("inside.txt"))
        let emptied = try await engine.sync(folder: root)
        #expect(emptied.removed.count == 1 && emptied.unreadable.isEmpty)
        try await engine.close()
    }

    @Test("Folders depend on the path, not on which folder was synced")
    func foldersAcrossSyncRoots() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("photos")
        try Self.write("tram", "trips/tram.txt", in: root)
        let engine = try await Self.engine(storeURL: nil)
        _ = try await engine.sync(folder: root.appendingPathComponent("trips"))
        let parent = try await engine.sync(folder: root)
        #expect(parent.unchanged == 1)
        let listed = try await engine.listItems(
            matching: .equals(key: "folders", value: .string(root.path)))
        #expect(listed.total == 1)
        try await engine.close()
    }

    @Test("Folder paths from the top down")
    func folders() {
        #expect(
            SemanticSearchEngine.folders(of: URL(fileURLWithPath: "/data/photos/trips/a.jpg"))
                == ["/data", "/data/photos", "/data/photos/trips"])
        #expect(SemanticSearchEngine.folders(of: URL(fileURLWithPath: "/a.jpg")) == [])
    }
}
