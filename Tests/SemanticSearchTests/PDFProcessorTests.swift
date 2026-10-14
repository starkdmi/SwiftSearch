import Foundation
import SemanticSearch
import SemanticSearchApple
import Testing

/// `tidal-energy.pdf`: page 1 printed from HTML, so its text is in the file;
/// page 2 a picture of a typed log, with no text layer at all.
@Suite("PDF processor")
struct PDFProcessorTests {

    static func fixture() throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: "tidal-energy", withExtension: "pdf", subdirectory: "Fixtures"))
    }

    @Test("A page with a text layer is read from it, a scan through Vision")
    func routesEachPage() async throws {
        let items = try await PDFProcessor().process(
            RawContent(id: "tidal", payload: .fileURL(try Self.fixture())), onProgress: nil)

        let parent = try #require(items.first)
        #expect(parent.id == "tidal")
        #expect(parent.metadata["title"] == .string("Tidal Stream Energy"))

        let first = try #require(items.first { $0.id == "tidal#page-1" })
        #expect(first.parentID == "tidal")
        #expect(first.locator.pageIndex == 0)
        let layer = try #require(first.representations.first { $0.kind == .originalText })
        #expect(layer.content.text?.contains("Pentland Firth") == true)
        #expect(!first.representations.contains { $0.kind == .image })

        let second = try #require(items.first { $0.id == "tidal#page-2" })
        #expect(second.locator.pageIndex == 1)
        // The rendering is kept: the page is a picture, and a multimodal
        // space can embed it.
        #expect(second.representations.contains { $0.kind == .image })
        let read = try #require(second.representations.first { $0.kind == .ocrText })
        #expect(read.content.text?.lowercased().contains("barnacles") == true)
    }

    @Test("A thin or unmapped text layer routes the page to recognition")
    func routing() {
        let processor = PDFProcessor()
        #expect(processor.route(forLayer: "12") == .recognized)
        #expect(processor.route(forLayer: String(repeating: "tide ", count: 20)) == .textLayer)
        let broken = String(repeating: "\u{FFFD}", count: 10) + String(repeating: "a", count: 60)
        #expect(processor.route(forLayer: broken) == .recognized)
    }

    @Test("PDF pages are searchable end to end")
    func endToEnd() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(acceptsVisual: true),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                processor: PDFProcessor()))
        try await engine.prepare()
        try await engine.ingest([
            RawContent(id: "tidal", payload: .fileURL(try Self.fixture()))
        ])

        let layer = try await engine.search(
            "MeyGen", options: SearchOptions(mode: .lexicalOnly))
        #expect(layer.hits.first?.itemID == "tidal#page-1")
        let scan = try await engine.search(
            "barnacles", options: SearchOptions(mode: .lexicalOnly))
        #expect(scan.hits.first?.itemID == "tidal#page-2")
        #expect(scan.hits.first?.locator.pageIndex == 1)
    }
}
