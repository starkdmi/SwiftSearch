import Foundation
import SemanticSearch
import SemanticSearchApple
import XCTest

/// ``FileMetadataProcessor`` on real files, and MapKit's reverse geocoder
/// live.
final class FileMetadataIntegrationTests: XCTestCase {

    /// Reads nothing, so only the file's own details come back.
    private struct Nothing: ContentProcessor {
        var identifier: String { "nothing" }
        var version: String { "1" }
        var capabilities: ProcessorCapabilities { ProcessorCapabilities() }
        var configurationFingerprint: String { "1" }
        func canProcess(_ content: RawContent) -> Bool { true }
        func process(
            _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
        ) async throws -> [SearchItem] { [] }
    }

    /// What the fixture library's files say about themselves. Printed; the
    /// type is asserted, the rest depends on how each file was made.
    func testLibraryFiles() async throws {
        let processor = FileMetadataProcessor(
            wrapping: Nothing(),
            configuration: .init(indexesFileName: false))
        let names = [
            "arctic-terns.txt", "printing-press.md", "kelp-forests.txt",
            "hydrothermal-vents-ja.txt", "tidal-energy.pdf", "bramble-recipe.png",
            "oxford-timetable.png",
        ]
        for name in names {
            let url = try Integration.library(name)
            let items = try await processor.process(RawContent(id: ItemID(name), payload: .fileURL(url)))
            let item = try XCTUnwrap(items.first)
            let facts = item.metadata.sorted { $0.key < $1.key }
                .filter { !["fileName", "fileCreated", "fileModified"].contains($0.key) }
                .map { "\($0.key)=\($0.value)" }
            print("[metadata] \(name): " + facts.joined(separator: " "))
            if let words = item[representation: "metadata"]?.content.text {
                print("[metadata]   words: \(words.replacingOccurrences(of: "\n", with: " | "))")
            }
            XCTAssertNotNil(item.metadata["contentType"], name)
        }
    }

    func testMapKitNamesPlaces() async throws {
        try XCTSkipUnless(
            ["1", "true", "yes", "on"].contains(
                ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_RUN_GEOCODING_TESTS"]?
                    .lowercased() ?? ""),
            "set SEMANTIC_SEARCH_RUN_GEOCODING_TESTS=1 to ask Apple's servers")
        guard #available(macOS 26.0, iOS 26.0, *) else {
            throw XCTSkip("MKReverseGeocodingRequest needs macOS 26")
        }
        let namer = MapKitPlaceNamer(configuration: .init(locale: Locale(identifier: "en_US")))
        let tower = try await namer.placeName(latitude: 48.8584, longitude: 2.2945)
        print("[metadata] 48.8584, 2.2945 → \(tower ?? "nil")")
        XCTAssertEqual(tower, "Eiffel Tower, Paris, France")
        // Within the cache's 100 m: answered without asking again.
        let nearby = try await namer.placeName(latitude: 48.85842, longitude: 2.29448)
        XCTAssertEqual(nearby, tower)
        let sea = try await namer.placeName(latitude: 0, longitude: -30)
        print("[metadata] 0, -30 → \(sea ?? "nil")")
    }
}
