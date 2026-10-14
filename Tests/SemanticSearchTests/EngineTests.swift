import Foundation
import Testing

@testable import SemanticSearch

/// Scratch directory that cleans itself up.
struct TemporaryDirectory: ~Copyable {
    let url: URL

    init() throws {
        url = FileManager.default.temporaryDirectory
            .appendingPathComponent("semanticsearch-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    var storeURL: URL { url.appendingPathComponent("index.sqlite") }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }
}

extension SemanticSearchEngine {
    /// An engine with a deterministic embedder and a term-overlap reranker.
    static func testEngine(
        storeURL: URL? = nil,
        indexKind: VectorIndexKind = .exact,
        reranker: (any RerankingProvider)? = ScriptedRerankingProvider.termOverlap(),
        kinds: Set<RepresentationKind> = [],
        dimensions: Int = 64,
        matryoshkaDimensions: [Int] = [],
        acceptsVisual: Bool = true,
        spaceID: EmbeddingSpaceID = "test"
    ) async throws -> SemanticSearchEngine {
        let provider = HashingEmbeddingProvider(
            dimensions: dimensions,
            matryoshkaDimensions: matryoshkaDimensions,
            acceptsVisual: acceptsVisual)
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: spaceID, provider: provider, kinds: kinds, indexKind: indexKind,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                reranker: reranker))
        try await engine.prepare()
        return engine
    }
}

func textItem(
    _ id: String,
    _ text: String,
    parent: String? = nil,
    kind: RepresentationKind = .originalText,
    language: String? = "en",
    metadata: Metadata = [:],
    locator: SourceLocator = .unspecified
) -> SearchItem {
    SearchItem(
        id: ItemID(id),
        parentID: parent.map { ItemID($0) },
        metadata: metadata,
        locator: locator,
        representations: [
            ContentRepresentation(
                id: "text", kind: kind, content: .text(text), language: language,
                locator: locator)
        ])
}

@Suite("Engine")
struct EngineTests {

    // MARK: - Round trip

    @Test("Upsert, search, and the hit reports the representation that matched")
    func basicRoundTrip() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("a", "the quick brown fox jumps over the lazy dog"),
            textItem("b", "a treatise on marine biology and coral reefs"),
        ])

        let results = try await engine.search("brown fox")
        #expect(results.hits.first?.itemID == "a")
        #expect(results.hits.first?.representationID == "text")
        #expect(results.hits.first?.kind == .originalText)
        #expect(results.hits.first?.text?.contains("fox") == true)
    }

    @Test("An unchanged upsert is skipped by fingerprint")
    func unchangedUpsertIsSkipped() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let item = textItem("a", "content that will not change")

        let first = try await engine.upsert([item])
        #expect(first.itemsWritten == 1)
        #expect(first.vectorsWritten == 1)

        let second = try await engine.upsert([item])
        #expect(second.itemsUnchanged == 1)
        #expect(second.itemsWritten == 0)
        // The point of the fingerprint: no re-embedding.
        #expect(second.vectorsWritten == 0)
    }

    @Test("Changing metadata alone does not re-embed")
    func metadataChangeDoesNotReembed() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "stable text")])

        var updated = textItem("a", "stable text")
        updated.metadata = ["author": .string("someone new")]
        let report = try await engine.upsert([updated])

        #expect(report.vectorsWritten == 0)
        // But the new metadata has to be visible to filters immediately.
        let filtered = try await engine.search(
            "stable",
            options: SearchOptions(
                filter: .equals(key: "author", value: .string("someone new"))))
        #expect(filtered.hits.count == 1)
    }

    @Test("Changing content re-embeds")
    func contentChangeReembeds() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "original")])
        let report = try await engine.upsert([textItem("a", "completely different")])
        #expect(report.itemsWritten == 1)
        #expect(report.vectorsWritten == 1)
    }

    @Test("Removal takes the item out of every channel")
    func removal() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("a", "alpha content here"), textItem("b", "beta content here"),
        ])
        try await engine.remove("a")

        let results = try await engine.search("alpha content")
        #expect(!results.hits.contains { $0.itemID == "a" })
        #expect(try await engine.itemCount() == 1)
    }

    @Test("Removing a parent removes its children")
    func cascadingRemoval() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("doc", "a document"),
            textItem("doc#1", "first passage", parent: "doc"),
            textItem("doc#2", "second passage", parent: "doc"),
        ])
        try await engine.remove("doc", includingDescendants: true)
        #expect(try await engine.itemCount() == 0)
    }

    // MARK: - Modes

    @Test("Lexical-only search loads no model")
    func lexicalOnlyNeedsNoModel() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("a", "hydrothermal vents on the ocean floor"),
            textItem("b", "the history of the printing press"),
        ])
        await engine.releaseModels()

        let results = try await engine.search(
            "hydrothermal vents", options: SearchOptions(mode: .lexicalOnly))
        #expect(results.hits.first?.itemID == "a")
        #expect(results.diagnostics.channels.allSatisfy { $0.channel.family == .lexical })
    }

    @Test("A short query still runs the full pipeline")
    func shortQueriesAreNotFastPathed() async throws {
        // The extracted implementation skipped dense retrieval and reranking
        // for short queries, so "Tokyo" and "the Tokyo office" took different
        // code paths. Hybrid is now the default for every query length.
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "Tokyo is the capital of Japan")])

        let results = try await engine.search("Tokyo")
        #expect(results.diagnostics.channels.contains { $0.channel.family != .lexical })
        #expect(results.hits.first?.evidence.rerankRank != nil)
    }

    @Test("Retrieval-only mode skips the reranker")
    func retrievalOnlySkipsReranker() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content")])
        let results = try await engine.search(
            "content", options: SearchOptions(mode: .retrievalOnly))
        #expect(results.diagnostics.rerankedCandidates == 0)
        #expect(results.hits.first?.evidence.rerankScore == nil)
    }

    @Test("Dense-only mode skips lexical")
    func denseOnlySkipsLexical() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content about something")])
        let results = try await engine.search(
            "content", options: SearchOptions(mode: .denseOnly))
        #expect(!results.diagnostics.channels.contains { $0.channel.family == .lexical })
    }

    // MARK: - Filtering

    @Test("Metadata filters apply to every channel")
    func metadataFiltering() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("a", "quarterly revenue report", metadata: ["year": .integer(2025)]),
            textItem("b", "quarterly revenue report", metadata: ["year": .integer(2026)]),
        ])

        let results = try await engine.search(
            "quarterly revenue",
            options: SearchOptions(filter: .equals(key: "year", value: .integer(2026))))
        #expect(results.hits.count == 1)
        #expect(results.hits[0].itemID == "b")
    }

    @Test("Range, existence and list filters work")
    func filterVariants() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem(
                "a", "first document", metadata: ["score": .double(10), "tags": .stringList(["x", "y"])]),
            textItem("b", "second document", metadata: ["score": .double(90)]),
        ])

        let high = try await engine.search(
            "document", options: SearchOptions(filter: .greaterThan(key: "score", value: .double(50))))
        #expect(high.hits.map(\.itemID) == ["b"])

        let tagged = try await engine.search(
            "document", options: SearchOptions(filter: .equals(key: "tags", value: .string("y"))))
        #expect(tagged.hits.map(\.itemID) == ["a"])

        let present = try await engine.search(
            "document", options: SearchOptions(filter: .exists(key: "tags")))
        #expect(present.hits.map(\.itemID) == ["a"])

        let between = try await engine.search(
            "document",
            options: SearchOptions(
                filter: .between(key: "score", lower: .double(5), upper: .double(50))))
        #expect(between.hits.map(\.itemID) == ["a"])
    }

    @Test("Composed filters combine")
    func composedFilters() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("a", "report", metadata: ["year": .integer(2026), "kind": .string("q1")]),
            textItem("b", "report", metadata: ["year": .integer(2026), "kind": .string("q2")]),
            textItem("c", "report", metadata: ["year": .integer(2025), "kind": .string("q1")]),
        ])

        let results = try await engine.search(
            "report",
            options: SearchOptions(
                filter: .and([
                    .equals(key: "year", value: .integer(2026)),
                    .equals(key: "kind", value: .string("q1")),
                ])))
        #expect(results.hits.map(\.itemID) == ["a"])
    }

    @Test("A selective filter still fills the candidate pool")
    func selectiveFilterDoesNotStarveChannels() async throws {
        // Filtering after retrieval leaves a 1%-selective filter with one
        // survivor out of a hundred candidates. The filter runs inside the
        // search instead.
        let engine = try await SemanticSearchEngine.testEngine()
        var items: [SearchItem] = []
        for index in 0..<500 {
            items.append(
                textItem(
                    "item-\(index)", "document number \(index) about shared subject matter",
                    metadata: ["bucket": .integer(Int64(index % 100))]))
        }
        try await engine.upsert(items)

        let results = try await engine.search(
            "shared subject matter",
            options: SearchOptions(
                limit: 5, filter: .equals(key: "bucket", value: .integer(7))))
        #expect(results.hits.count == 5)
        #expect(
            results.hits.allSatisfy {
                if case .integer(7)? = $0.metadata["bucket"] { return true }
                return false
            })
    }

    @Test("A filter matching nothing returns nothing, not everything")
    func emptyFilterResult() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content")])
        let results = try await engine.search(
            "content",
            options: SearchOptions(filter: .equals(key: "missing", value: .string("nope"))))
        #expect(results.hits.isEmpty)
    }

    // MARK: - Hierarchy

    @Test("searchInside scopes to a subtree")
    func searchInsideScopes() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("book-1", "Book One"),
            textItem("book-1#c1", "the migration of arctic terns", parent: "book-1"),
            textItem("book-2", "Book Two"),
            textItem("book-2#c1", "the migration of arctic terns", parent: "book-2"),
        ])

        let results = try await engine.searchInside("book-2", "arctic terns")
        #expect(!results.hits.isEmpty)
        #expect(results.hits.allSatisfy { $0.itemID.rawValue.hasPrefix("book-2") })
    }

    @Test("searchInside reaches grandchildren")
    func searchInsideIsTransitive() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("root", "root"),
            textItem("root#a", "middle", parent: "root"),
            textItem("root#a#1", "deeply nested passage about kelp forests", parent: "root#a"),
        ])

        let results = try await engine.searchInside("root", "kelp forests")
        #expect(results.hits.contains { $0.itemID == "root#a#1" })
    }

    @Test("Hits carry bounded parent context")
    func parentContextIsAttached() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let longSummary = String(repeating: "context ", count: 500)
        try await engine.upsert([
            SearchItem(
                id: "doc",
                metadata: ["title": .string("The Document")],
                representations: [
                    ContentRepresentation(
                        id: "summary", kind: .summary, content: .text(longSummary))
                ]),
            textItem("doc#1", "a passage about tidal patterns", parent: "doc"),
        ])

        let results = try await engine.search(
            "tidal patterns", options: SearchOptions(parentContextCharacters: 100))
        let hit = try #require(results.hits.first { $0.itemID == "doc#1" })
        #expect(hit.parent?.itemID == "doc")
        #expect((hit.parent?.text?.count ?? 0) <= 100)
    }

    @Test("Parent grouping ranks by the strongest hit, not the hit count")
    func groupingUsesStrongestHit() async throws {
        // A long document with many weak matches must not outrank a short one
        // that actually answers the query.
        let engine = try await SemanticSearchEngine.testEngine()
        var items: [SearchItem] = [textItem("long", "Long Document"), textItem("short", "Short")]
        for index in 0..<8 {
            items.append(
                textItem("long#\(index)", "tangentially about oceans", parent: "long"))
        }
        items.append(
            textItem("short#0", "hydrothermal vents on the ocean floor", parent: "short"))
        try await engine.upsert(items)

        let results = try await engine.search(
            "hydrothermal vents ocean floor", options: SearchOptions(groupByParent: true))
        #expect(results.groups.first?.parentID == "short")
    }

    // MARK: - Locators

    @Test("Locators survive the round trip exactly")
    func locatorRoundTrip() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let locator = SourceLocator(
            source: "file:///corpus/report.pdf",
            pageIndex: 4,
            timeRange: SourceLocator.TimeRange(start: 12.5, end: 30),
            characterRange: SourceLocator.CharacterRange(location: 120, length: 360),
            boundingBoxes: [SourceLocator.BoundingBox(x: 0.1, y: 0.2, width: 0.5, height: 0.05)],
            focusTime: 15)
        try await engine.upsert([
            textItem("page", "the annual figures were unusually strong", locator: locator)
        ])

        let hit = try #require(try await engine.search("annual figures").hits.first)
        #expect(hit.locator.source == locator.source)
        #expect(hit.locator.pageIndex == 4)
        #expect(hit.locator.characterRange?.location == 120)
        #expect(hit.locator.characterRange?.length == 360)
        #expect(hit.locator.timeRange?.start == 12.5)
        #expect(hit.locator.focusTime == 15)
        #expect(hit.locator.boundingBoxes.count == 1)
    }

    @Test("Chunk locators point at the right span of the source")
    func chunkLocatorsAreNarrowed() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(), indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                processor: TextChunkingProcessor(
                    configuration: .init(
                        chunker: ChunkerConfiguration(
                            targetTokens: 20, maximumTokens: 24, overlapTokens: 0,
                            minimumTokens: 1)))))
        try await engine.prepare()

        let source = (1...40)
            .map { "Paragraph \($0) is about subject \($0) and nothing else whatsoever." }
            .joined(separator: "\n\n")
        try await engine.ingest([
            RawContent(
                id: "doc", payload: .text(source),
                locator: SourceLocator(source: "file:///doc.txt"))
        ])

        let hit = try #require(try await engine.search("subject 37").hits.first)
        let range = try #require(hit.locator.characterRange)
        let selected = (source as NSString).substring(
            with: NSRange(location: range.location, length: range.length))
        #expect(selected.contains("subject"))
        #expect(hit.locator.source == "file:///doc.txt")
    }

    // MARK: - Evidence and diagnostics

    @Test("Hits carry per-channel ranking evidence")
    func evidenceIsReported() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "distinctive phrase about penguins")])
        let hit = try #require(try await engine.search("penguins").hits.first)

        #expect(hit.evidence.fusedScore > 0)
        #expect(!hit.evidence.channels.isEmpty)
        #expect(hit.evidence.channels.allSatisfy { $0.rank >= 1 })
        #expect(hit.evidence.rerankScore != nil)
        // The reranker orders the pool, so the final score is its rank's.
        #expect(hit.evidence.rerankRank == 1)
        #expect(hit.evidence.finalScore > 0)
    }

    @Test("Diagnostics report channels, timings and the query terms")
    func diagnosticsAreReported() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "diagnostics content here")])
        let results = try await engine.search("diagnostics content")

        #expect(!results.diagnostics.queryTerms.isEmpty)
        #expect(results.diagnostics.channels.count >= 2)
        #expect(results.diagnostics.fusedCandidates > 0)
        #expect(results.diagnostics.totalDuration > .zero)
        #expect(results.diagnostics.channels.allSatisfy { $0.appliedWeight >= 0 })
    }

    @Test("A space with no vectors is reported, not silently dropped")
    func emptySpaceIsDiagnosed() async throws {
        // Degraded search must be requested rather than substituted.
        let engine = try await SemanticSearchEngine.testEngine(kinds: [.image])
        try await engine.upsert([textItem("a", "text that no space will hold")])

        let results = try await engine.search("text")
        #expect(results.diagnostics.skippedSpaces.contains { $0.space == "test" })
    }

    @Test("Results are ordered deterministically")
    func orderingIsStable() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let items = (0..<60).map {
            textItem("item-\($0)", "shared subject matter, variant \($0)")
        }
        try await engine.upsert(items)

        let first = try await engine.search("shared subject matter").hits.map(\.itemID)
        for _ in 0..<5 {
            #expect(try await engine.search("shared subject matter").hits.map(\.itemID) == first)
        }
    }

    // MARK: - Reranking behaviour

    @Test("The reranker sees at most the configured pool")
    func rerankPoolIsBounded() async throws {
        let reranker = ScriptedRerankingProvider.termOverlap()
        let engine = try await SemanticSearchEngine.testEngine(reranker: reranker)
        try await engine.upsert(
            (0..<120).map { textItem("item-\($0)", "common subject matter \($0)") })

        _ = try await engine.search(
            "common subject matter", options: SearchOptions(rerankPoolSize: 20))
        let calls = await reranker.calls
        #expect(calls.last?.candidates.count == 20)
    }

    @Test("Late fusion, when asked for, demotes rather than deletes")
    func lateFusionKeepsRetrievalSignal() async throws {
        // A reranker that hates the true answer should not be able to remove
        // it from the results when retrieval was confident.
        let hostile = ScriptedRerankingProvider { _, candidate in
            RerankScore(score: candidate.text.contains("penguins") ? 0 : 1)
        }
        let engine = try await SemanticSearchEngine.testEngine(reranker: hostile)
        try await engine.upsert([
            textItem("target", "a detailed study of emperor penguins"),
            textItem("other-1", "unrelated material about soil chemistry"),
            textItem("other-2", "unrelated material about bridge design"),
        ])

        let results = try await engine.search(
            "emperor penguins", options: SearchOptions(rerankerOrderingOnly: false))
        #expect(results.hits.contains { $0.itemID == "target" })
    }

    @Test("By default the reranker alone orders its pool")
    func rerankerOrderingOnly() async throws {
        let hostile = ScriptedRerankingProvider { _, candidate in
            RerankScore(score: candidate.text.contains("penguins") ? 0 : 1)
        }
        let engine = try await SemanticSearchEngine.testEngine(reranker: hostile)
        try await engine.upsert([
            textItem("target", "a detailed study of emperor penguins"),
            textItem("other", "unrelated material about soil chemistry"),
        ])

        let results = try await engine.search("emperor penguins")
        #expect(results.hits.first?.itemID == "other")
    }

    @Test("Reranker scores are never described as calibrated")
    func rerankerScoresAreNotCalibrated() async throws {
        // Semantic "no match" needs an explicitly calibrated policy; the
        // bundled providers do not claim one.
        let reranker = ScriptedRerankingProvider.termOverlap()
        #expect(await reranker.capabilities.producesCalibratedScores == false)
    }
}
