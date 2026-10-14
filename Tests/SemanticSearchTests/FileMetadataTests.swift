import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import SemanticSearch
@testable import SemanticSearchApple

/// A file and one child, as a chunker or a transcriber would return them.
private struct ParentAndChild: ContentProcessor {
    var identifier: String { "parent-and-child" }
    var version: String { "1" }
    var capabilities: ProcessorCapabilities { ProcessorCapabilities(producedKinds: [.originalText]) }
    var configurationFingerprint: String { "1" }
    func canProcess(_ content: RawContent) -> Bool { true }
    func process(
        _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        [
            SearchItem(id: content.id, metadata: content.metadata, locator: content.locator),
            SearchItem(
                id: ItemID("\(content.id.rawValue)#chunk-0"), parentID: content.id,
                metadata: content.metadata,
                representations: [.text("what the file says", kind: .originalText)]),
        ]
    }
}

private actor ScriptedPlaceNamer: PlaceNamer {
    nonisolated let identifier = "scripted-place"
    nonisolated let configurationFingerprint = "1"
    private(set) var asked: [(Double, Double)] = []
    func placeName(latitude: Double, longitude: Double) async throws -> String? {
        asked.append((latitude, longitude))
        return "Lisbon, Portugal"
    }
}

/// Refuses the first `failures` lookups, as a server that throttles or a
/// machine that is offline would.
private actor FlakyPlaceNamer: PlaceNamer {
    nonisolated let identifier = "flaky-place"
    nonisolated let configurationFingerprint = "1"
    private var failures: Int
    init(failures: Int) { self.failures = failures }
    func placeName(latitude: Double, longitude: Double) async throws -> String? {
        guard failures == 0 else {
            failures -= 1
            throw URLError(.notConnectedToInternet)
        }
        return "Lisbon, Portugal"
    }
}

@Suite("File metadata")
struct FileMetadataTests {

    static func placeEngine(
        _ processor: FileMetadataProcessor, storeURL: URL? = nil
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: false),
                        kinds: [.originalText, .metadata], indexKind: .exact)
                ],
                reranker: nil, processor: processor))
        try await engine.prepare()
        return engine
    }

    @Test("A place a failed lookup left out is asked again, and only the place is filled in")
    func pendingPlace() async throws {
        let url = try Self.taggedPhoto(named: "DSC_2.jpg")
        let processor = FileMetadataProcessor(
            wrapping: ParentAndChild(), placeNamer: FlakyPlaceNamer(failures: 2))
        let engine = try await Self.placeEngine(processor)
        _ = try await engine.ingest([RawContent(id: "photo", payload: .fileURL(url))])
        let pending = try #require(try await engine.storedItem("photo"))
        #expect(pending.metadata[FileMetadataProcessor.placePendingKey] == .boolean(true))
        #expect(pending.metadata["place"] == nil)
        #expect(pending.metadata["latitude"] == .double(38.7223))

        // Refused again: still pending.
        #expect(try await processor.namePendingPlaces(in: engine) == 0)
        let words = try await engine.storedRepresentations("photo#chunk-0").map(\.slot)
        #expect(try await processor.namePendingPlaces(in: engine) == 1)
        for id: ItemID in ["photo", "photo#chunk-0"] {
            let item = try #require(try await engine.storedItem(id))
            #expect(item.metadata["place"] == .string("Lisbon, Portugal"))
            #expect(item.metadata[FileMetadataProcessor.placePendingKey] == nil)
        }
        // The chunk was not touched; the file's words were, and hold the place.
        #expect(try await engine.storedRepresentations("photo#chunk-0").map(\.slot) == words)
        let text = try await engine.storedRepresentations("photo")
            .first { $0.representationID == "metadata" }?.text
        #expect(text == "DSC 2\nTram 28\ntram, Alfama\nLisbon, Portugal")
        let byPlace = try await engine.search("Lisbon", options: SearchOptions(mode: .lexicalOnly))
        #expect(byPlace.hits.first?.itemID == "photo")
        #expect(try await processor.namePendingPlaces(in: engine) == 0)
        try await engine.close()
    }

    @Test("Folders below the synced one are words; a parent added later changes only them")
    func folderWords() async throws {
        let directory = try TemporaryDirectory()
        let library = directory.url.appendingPathComponent("library")
        let trip = library.appendingPathComponent("Trips/Fethiye_2022")
        try FileManager.default.createDirectory(at: trip, withIntermediateDirectories: true)
        try FileManager.default.moveItem(
            at: try Self.taggedPhoto(named: "DSC_9.jpg"), to: trip.appendingPathComponent("DSC_9.jpg"))
        let namer = ScriptedPlaceNamer()
        let engine = try await Self.placeEngine(
            FileMetadataProcessor(wrapping: ParentAndChild(), placeNamer: namer),
            storeURL: directory.storeURL)
        _ = try await engine.sync(folder: library)
        let id = SemanticSearchEngine.itemID(forFile: trip.appendingPathComponent("DSC_9.jpg"))
        func words() async throws -> String? {
            try await engine.storedRepresentations(id)
                .first { $0.representationID == "metadata" }?.text
        }
        #expect(try await words() == "DSC 9\nTrips / Fethiye 2022\nTram 28\ntram, Alfama\nLisbon, Portugal")
        #expect(
            try await engine.storedItem(ItemID(id.rawValue + "#chunk-0"))?.metadata["folderPath"]
                == .string("Trips/Fethiye_2022"))
        let byFolder = try await engine.search(
            "Fethiye 2022", options: SearchOptions(mode: .lexicalOnly))
        #expect(byFolder.hits.first?.itemID == id)

        // The library's parent synced instead: the same file, one more folder
        // above it — its words follow, nothing is processed again.
        let chunkSlots = try await engine.storedRepresentations(ItemID(id.rawValue + "#chunk-0"))
            .map(\.slot)
        let parent = try await engine.sync(folder: directory.url)
        #expect(parent.unchanged >= 1 && parent.added.isEmpty && parent.updated.isEmpty)
        #expect(
            try await words()
                == "DSC 9\nlibrary / Trips / Fethiye 2022\nTram 28\ntram, Alfama\nLisbon, Portugal")
        #expect(
            try await engine.storedRepresentations(ItemID(id.rawValue + "#chunk-0")).map(\.slot)
                == chunkSlots)
        #expect(await namer.asked.count == 1)
        let again = try await engine.sync(folder: directory.url)
        #expect(again.indexing.itemsWritten == 0)
        try await engine.close()
    }

    @Test("A renamed file's name follows it; its tags and place stay, looked up once")
    func renamedFile() async throws {
        let directory = try TemporaryDirectory()
        let root = directory.url.appendingPathComponent("photos")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.moveItem(
            at: try Self.taggedPhoto(named: "DSC_7.jpg"),
            to: root.appendingPathComponent("DSC_7.jpg"))
        let namer = ScriptedPlaceNamer()
        let engine = try await Self.placeEngine(
            FileMetadataProcessor(wrapping: ParentAndChild(), placeNamer: namer),
            storeURL: directory.storeURL)
        _ = try await engine.sync(folder: root)

        try FileManager.default.moveItem(
            at: root.appendingPathComponent("DSC_7.jpg"),
            to: root.appendingPathComponent("harbour-tram.jpg"))
        let report = try await engine.sync(folder: root)
        let new = SemanticSearchEngine.itemID(forFile: root.appendingPathComponent("harbour-tram.jpg"))
        #expect(report.moved.map(\.to) == [new])
        #expect(await namer.asked.count == 1)

        let file = try #require(try await engine.storedItem(new))
        #expect(file.metadata["fileName"] == .string("harbour-tram.jpg"))
        #expect(file.metadata["place"] == .string("Lisbon, Portugal"))
        #expect(
            try await engine.storedItem(ItemID(new.rawValue + "#chunk-0"))?.metadata["fileName"]
                == .string("harbour-tram.jpg"))
        let text = try await engine.storedRepresentations(new)
            .first { $0.representationID == "metadata" }?.text
        #expect(text == "harbour tram\nTram 28\ntram, Alfama\nLisbon, Portugal")
        let byName = try await engine.search("harbour", options: SearchOptions(mode: .lexicalOnly))
        #expect(byName.hits.first?.itemID == new)
        let byOldName = try await engine.search("DSC", options: SearchOptions(mode: .lexicalOnly))
        #expect(byOldName.hits.isEmpty)
        try await engine.close()
    }

    static func temporaryURL(_ name: String) -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("file-metadata-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(name)
    }

    /// A JPEG with the tags an iPhone and a photo manager write.
    static func taggedPhoto(named name: String = "IMG_0412.jpg") throws -> URL {
        let url = temporaryURL(name)
        let image = try AppleHelpersTests.solidImage(width: 64, height: 48)
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        let properties: [CFString: Any] = [
            kCGImagePropertyOrientation: 6,
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifDateTimeOriginal: "2024:06:14 18:23:11",
                kCGImagePropertyExifOffsetTimeOriginal: "+01:00",
            ],
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFMake: "Apple",
                kCGImagePropertyTIFFModel: "iPhone 15 Pro",
            ],
            kCGImagePropertyGPSDictionary: [
                kCGImagePropertyGPSLatitude: 38.7223,
                kCGImagePropertyGPSLatitudeRef: "N",
                kCGImagePropertyGPSLongitude: 9.1393,
                kCGImagePropertyGPSLongitudeRef: "W",
            ],
            kCGImagePropertyIPTCDictionary: [
                kCGImagePropertyIPTCObjectName: "Tram 28",
                kCGImagePropertyIPTCKeywords: ["tram", "Alfama"],
            ],
        ]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    @Test("A photo's EXIF, GPS and IPTC become filters on every item, and words on the file")
    func photo() async throws {
        let url = try Self.taggedPhoto()
        let namer = ScriptedPlaceNamer()
        let processor = FileMetadataProcessor(wrapping: ParentAndChild(), placeNamer: namer)
        let items = try await processor.process(
            RawContent(id: "photo", payload: .fileURL(url), metadata: ["title": .string("Mine")]))
        #expect(items.count == 2)

        for item in items {
            let metadata = item.metadata
            #expect(metadata["fileName"] == .string("IMG_0412.jpg"))
            #expect(metadata["contentType"]?.contains(.string("public.jpeg")) == true)
            #expect(metadata["contentType"]?.contains(.string("public.image")) == true)
            #expect(metadata["camera"] == .string("Apple iPhone 15 Pro"))
            // Orientation 6: stored landscape, shown portrait.
            #expect(metadata["pixelWidth"] == .integer(48))
            #expect(metadata["pixelHeight"] == .integer(64))
            #expect(metadata["latitude"] == .double(38.7223))
            #expect(metadata["longitude"] == .double(-9.1393))
            #expect(metadata["place"] == .string("Lisbon, Portugal"))
            #expect(metadata["keywords"] == .stringList(["tram", "Alfama"]))
            // 18:23 at +01:00.
            #expect(metadata["contentCreated"] == .date(Date(timeIntervalSince1970: 1_718_385_791)))
            // The caller's title wins over the file's.
            #expect(metadata["title"] == .string("Mine"))
            #expect(metadata["fileSize"] != nil)
        }
        #expect(await namer.asked.count == 1)

        let words = try #require(items[0][representation: "metadata"])
        #expect(words.kind == .metadata)
        #expect(words.content.text == "IMG 0412\nTram 28\ntram, Alfama\nLisbon, Portugal")
        // Dates and cameras are for filters, not words.
        #expect(!(words.content.text ?? "").contains("iPhone"))
        #expect(items[1][representation: "metadata"] == nil)
    }

    @Test("Location off stores no coordinates and asks no one; names can be left out")
    func privacySwitches() async throws {
        let url = try Self.taggedPhoto()
        let namer = ScriptedPlaceNamer()
        let processor = FileMetadataProcessor(
            wrapping: ParentAndChild(), placeNamer: namer,
            configuration: .init(indexesFileName: false, recordsLocation: false))
        let items = try await processor.process(RawContent(id: "photo", payload: .fileURL(url)))
        #expect(items[0].metadata["latitude"] == nil)
        #expect(items[0].metadata["place"] == nil)
        #expect(await namer.asked.isEmpty)
        #expect(items[0][representation: "metadata"]?.content.text == "Tram 28\ntram, Alfama")

        let silent = FileMetadataProcessor(
            wrapping: ParentAndChild(), configuration: .init(indexesWords: false))
        let plain = try await silent.process(RawContent(id: "photo", payload: .fileURL(url)))
        #expect(plain[0][representation: "metadata"] == nil)
        #expect(plain[0].metadata["latitude"] == .double(38.7223))

        #expect(
            Set([processor, silent, FileMetadataProcessor(wrapping: ParentAndChild())]
                .map(\.configurationFingerprint)).count == 3)
    }

    @Test("A movie's QuickTime tags: title, place, date, camera, size and duration")
    func movie() async throws {
        let url = Self.temporaryURL("clip.mov")
        func item(_ identifier: AVMetadataIdentifier, _ value: String) -> AVMetadataItem {
            let item = AVMutableMetadataItem()
            item.identifier = identifier
            item.value = value as NSString
            item.dataType = kCMMetadataBaseDataType_UTF8 as String
            return item
        }
        try await AppleHelpersTests.makeMovie(
            at: url, seconds: 1,
            metadata: [
                item(.quickTimeMetadataTitle, "Harbour at dusk"),
                item(.quickTimeMetadataLocationISO6709, "+38.7223-009.1393+045.000/"),
                item(.quickTimeMetadataCreationDate, "2024-06-14T18:23:11+0100"),
                item(.quickTimeMetadataMake, "Apple"),
                item(.quickTimeMetadataModel, "iPhone 15 Pro"),
            ])
        let items = try await FileMetadataProcessor(wrapping: ParentAndChild())
            .process(RawContent(id: "clip", payload: .fileURL(url)))
        let metadata = items[1].metadata
        #expect(metadata["title"] == .string("Harbour at dusk"))
        #expect(metadata["latitude"] == .double(38.7223))
        #expect(metadata["longitude"] == .double(-9.1393))
        #expect(metadata["contentCreated"] == .date(Date(timeIntervalSince1970: 1_718_385_791)))
        #expect(metadata["camera"] == .string("Apple iPhone 15 Pro"))
        #expect(metadata["pixelWidth"] == .integer(320))
        #expect(metadata["contentType"]?.contains(.string("public.movie")) == true)
        if case .double(let duration)? = metadata["duration"] {
            #expect(abs(duration - 1) < 0.2)
        } else {
            Issue.record("no duration")
        }
        #expect(items[0][representation: "metadata"]?.content.text == "clip\nHarbour at dusk")
    }

    @Test("A song's iTunes tags: title, artist, album")
    func song() async throws {
        let wav = Self.temporaryURL("tone.wav")
        try AppleHelpersTests.makeAudio(at: wav)
        let url = Self.temporaryURL("track01.m4a")
        let session = try #require(
            AVAssetExportSession(
                asset: AVURLAsset(url: wav), presetName: AVAssetExportPresetAppleM4A))
        func item(_ identifier: AVMetadataIdentifier, _ value: String) -> AVMetadataItem {
            let item = AVMutableMetadataItem()
            item.identifier = identifier
            item.value = value as NSString
            return item
        }
        session.metadata = [
            item(.iTunesMetadataSongName, "Come Together"),
            item(.iTunesMetadataArtist, "The Beatles"),
            item(.iTunesMetadataAlbum, "Abbey Road"),
        ]
        try await session.export(to: url, as: .m4a)

        let items = try await FileMetadataProcessor(wrapping: ParentAndChild())
            .process(RawContent(id: "song", payload: .fileURL(url)))
        #expect(items[0].metadata["artist"] == .string("The Beatles"))
        #expect(items[0].metadata["album"] == .string("Abbey Road"))
        #expect(
            items[0][representation: "metadata"]?.content.text
                == "track01\nCome Together\nThe Beatles\nAbbey Road")
    }

    @Test("Markdown front matter: title, author, tags in every YAML list form")
    func frontMatter() {
        let tags = FileFacts.frontMatterTags(
            """
            ---
            title: "Release checklist"
            author: Dana
            tags: [ops, "release"]
            keywords:
              - deploy
              - rollback
            nested:
              title: not this
            ---
            # Body
            title: not this either
            """)
        #expect(tags.title == "Release checklist")
        #expect(tags.authors == ["Dana"])
        #expect(tags.keywords == ["ops", "release", "deploy", "rollback"])
        #expect(FileFacts.frontMatterTags("# No front matter\ntitle: x").title == nil)

        // A quoted value keeps its commas; a plain author is one name.
        let quoted = FileFacts.frontMatterTags(
            """
            ---
            title: 'It''s here'
            author: Doe, Jane
            authors: ["Roe, Richard", 'O''Brien, Pat']
            tags: ["Paris, France", travel, "say \\"hi\\""]
            keywords: "a, b"
            categories: x, y
            ---
            """)
        #expect(quoted.title == "It's here")
        #expect(quoted.authors == ["Doe, Jane", "Roe, Richard", "O'Brien, Pat"])
        #expect(quoted.keywords == ["Paris, France", "travel", "say \"hi\"", "a, b", "x", "y"])
        #expect(FileFacts.flowItems(" a ,, 'b' ") == ["a", "b"])
    }

    @Test("Parsers: file names, ISO 6709, EXIF offsets, cameras")
    func parsers() throws {
        #expect(FileFacts.nameWords("invoice-march_2024.final.pdf") == "invoice march 2024 final")
        #expect(FileFacts.nameWords("__.txt") == nil)
        let point = try #require(FileFacts.iso6709("-33.8688+151.2093/"))
        #expect(point.0 == -33.8688 && point.1 == 151.2093)
        #expect(FileFacts.iso6709("+4012-07400/") == nil)  // degrees and minutes
        // A Pixel's `©xyz`: two-digit longitude degrees.
        let android = try #require(FileFacts.iso6709("+36.6672+29.1057/"))
        #expect(android.0 == 36.6672 && android.1 == 29.1057)
        if #available(macOS 26.0, iOS 26.0, *) {
            // At sea MapKit's name is inside the city with its context.
            #expect(
                MapKitPlaceNamer.joined([
                    "Mediterranean Sea", "Mediterranean Sea, Türkiye", "Türkiye",
                ]) == "Mediterranean Sea, Türkiye")
            #expect(
                MapKitPlaceNamer.joined(["Eiffel Tower", "Paris", "France"])
                    == "Eiffel Tower, Paris, France")
            #expect(MapKitPlaceNamer.joined([nil, " ", nil]) == nil)
        }
        #expect(FileFacts.timeZone(offset: "-05:30")?.secondsFromGMT() == -19_800)
        #expect(FileFacts.timeZone(offset: "Z") == nil)
        #expect(FileFacts.camera(make: "Canon", model: "Canon EOS R5") == "Canon EOS R5")
        #expect(FileFacts.camera(make: "  ", model: nil) == nil)
    }

    @Test("A file is found by its tags, and filtered by its type at the chunk level")
    func searchable() async throws {
        let url = try Self.taggedPhoto(named: "DSC_1.jpg")
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: HashingEmbeddingProvider(dimensions: 64, acceptsVisual: false),
                        kinds: [.originalText, .metadata], indexKind: .exact)
                ],
                reranker: nil,
                processor: FileMetadataProcessor(
                    wrapping: ParentAndChild(), placeNamer: ScriptedPlaceNamer())))
        try await engine.prepare()
        _ = try await engine.ingest([RawContent(id: "photo", payload: .fileURL(url))])

        let byPlace = try await engine.search(
            "Lisbon", options: SearchOptions(mode: .lexicalOnly))
        #expect(byPlace.hits.first?.itemID == "photo")
        #expect(byPlace.hits.first?.kind == .metadata)

        let pictures = try await engine.search(
            "what the file says",
            options: SearchOptions(
                mode: .lexicalOnly,
                filter: .and([
                    .equals(key: "contentType", value: .string("public.image")),
                    .greaterThan(
                        key: "contentCreated", value: .date(Date(timeIntervalSince1970: 1.7e9))),
                ])))
        #expect(pictures.hits.first?.itemID == "photo#chunk-0")
        let movies = try await engine.search(
            "what the file says",
            options: SearchOptions(
                mode: .lexicalOnly,
                filter: .equals(key: "contentType", value: .string("public.movie"))))
        #expect(movies.hits.isEmpty)
        try await engine.close()
    }
}
