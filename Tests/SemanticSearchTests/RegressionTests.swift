import CoreGraphics
import Foundation
import Testing

@testable import SemanticSearch
@testable import SemanticSearchApple

// One test per fixed defect. Each failed, or would have, against the code
// before its fix.

@Suite("Regressions: persistence")
struct PersistenceRegressionTests {

    @Test("A vector deleted after a snapshot does not survive reopen in the graph")
    func deletedVectorLeavesTheGraphOnReopen() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: storeURL, indexKind: .hnsw)
            try await engine.upsert(
                (0..<50).map { textItem("item-\($0)", "document number \($0)") })
            try await engine.rebuildVectorIndex(space: "test")  // publishes a snapshot
            try await engine.remove("item-49")  // the highest slot
            // Context-only: stored, never embedded, never lexically indexed.
            try await engine.upsert(
                SearchItem(
                    id: "unrelated",
                    representations: [
                        ContentRepresentation(
                            id: "note", kind: .originalText,
                            content: .text("completely unrelated context"), weight: 0)
                    ]))
            try await engine.close()
        }

        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: storeURL, indexKind: .hnsw, reranker: nil)
        let provider = HashingEmbeddingProvider()
        try await provider.prepare()
        let query = try await provider.embed(
            EmbeddingInput(text: "document number 49", role: .query(instruction: nil))
        ).vector

        let neighbours = try await reopened.approximateNeighbours(
            space: "test", query: query, limit: 10)
        let live = try await reopened.store.vectorSlots(space: "test")
        #expect(neighbours.allSatisfy { live.contains($0.slot) })

        let results = try await reopened.search(
            "document number 49", options: SearchOptions(mode: .denseOnly, limit: 60))
        #expect(!results.hits.contains { $0.itemID == "unrelated" })
        #expect(!results.hits.contains { $0.itemID == "item-49" })
        try await reopened.close()
    }

    @Test("Slots are never reused, even for the most recently deleted row")
    func slotsAreNeverReused() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL
        var used: Set<Int> = []

        do {
            let engine = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
            try await engine.upsert([textItem("a", "alpha"), textItem("b", "beta")])
            used.formUnion(try await engine.store.slotKinds().keys)
            try await engine.remove("b")
            try await engine.close()
        }

        // Across a reopen too: the high-water mark is persisted.
        let reopened = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
        try await reopened.upsert(textItem("c", "gamma"))
        let slot = try #require(
            try await reopened.store.representations(forItem: "c").first?.slot)
        #expect(!used.contains(slot))
        try await reopened.close()
    }

    @Test("Removing an item while its vectors are being computed leaves nothing behind")
    func removeDuringEmbedding() async throws {
        let provider = BlockingEmbeddingProvider()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: provider)]))

        let upsert = Task { try await engine.upsert(textItem("a", "alpha")) }
        await provider.waitUntilEmbedding()
        try await engine.remove("a")
        await provider.release(embedding: ())

        let report = try await upsert.value
        #expect(report.vectorsWritten == 0)
        #expect(try await engine.vectorCount(space: "s") == 0)
        #expect(try await engine.search("alpha").hits.isEmpty)
    }

    @Test("Postings written by a different analyzer are rebuilt on open")
    func analyzerChangeRebuildsPostings() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine(
                configuration: EngineConfiguration(storeURL: storeURL))
            #expect(await !engine.lexicalIndexRebuiltOnOpen)
            try await engine.upsert(textItem("a", "alpha beta"))
            try await engine.close()
        }

        // Same analyzer: nothing to do.
        do {
            let engine = try await SemanticSearchEngine(
                configuration: EngineConfiguration(storeURL: storeURL))
            #expect(await !engine.lexicalIndexRebuiltOnOpen)
            try await engine.close()
        }

        let reversing = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL, analyzer: ReversingAnalyzer()))
        #expect(await reversing.lexicalIndexRebuiltOnOpen)
        // The query is reversed by the new analyzer; it only matches if the
        // stored postings were reversed too.
        let results = try await reversing.search(
            "alpha", options: SearchOptions(mode: .lexicalOnly))
        #expect(results.hits.map(\.itemID) == ["a"])
        try await reversing.close()
    }
}

/// A root by one producer and, when the text says "both", a child by
/// another.
private struct TwoProducers: ContentProcessor {
    var identifier: String { "two-producers" }
    var version: String { "1" }
    var capabilities: ProcessorCapabilities { ProcessorCapabilities() }
    var configurationFingerprint: String { "1" }
    func canProcess(_ content: RawContent) -> Bool { true }
    func process(
        _ content: RawContent, onProgress: (@Sendable (ProcessingProgress) -> Void)?
    ) async throws -> [SearchItem] {
        guard case .text(let text) = content.payload else { return [] }
        var items = [
            SearchItem(
                id: content.id,
                representations: [
                    ContentRepresentation(
                        id: "text", kind: .originalText, content: .text("root words"),
                        provenance: Provenance(producer: "first"))
                ])
        ]
        if text == "both" {
            items.append(
                SearchItem(
                    id: ItemID("\(content.id.rawValue)#child"), parentID: content.id,
                    representations: [
                        ContentRepresentation(
                            id: "text", kind: .originalText, content: .text("child words"),
                            provenance: Provenance(producer: "second"))
                    ]))
        }
        return items
    }
}

@Suite("Regressions: ingest")
struct IngestRegressionTests {

    @Test("Re-ingesting shorter content removes the chunks it no longer produces")
    func reingestRemovesStaleChildren() async throws {
        let chunker = ChunkerConfiguration(
            targetTokens: 16, maximumTokens: 16, overlapTokens: 0, minimumTokens: 1)
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                processor: TextChunkingProcessor(configuration: .init(chunker: chunker))))

        let long = (0..<12).map { "Paragraph \($0) mentions zeppelin\($0) specifically." }
            .joined(separator: "\n\n")
        try await engine.ingest([RawContent(id: "doc", payload: .text(long))])
        let before = try await engine.itemCount()
        #expect(before > 2)

        let report = try await engine.ingest(
            [RawContent(id: "doc", payload: .text("A short note."))])
        #expect(report.itemsRemoved == before - 1)
        #expect(try await engine.itemCount() == 1)
        let stale = try await engine.search(
            "zeppelin11", options: SearchOptions(mode: .lexicalOnly))
        #expect(stale.hits.isEmpty)
        #expect(try await engine.search("short note").hits.first?.itemID == "doc")
    }

    @Test("An emptied file loses its old text; a new empty file is indexed")
    func emptiedContent() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(processor: TextChunkingProcessor()))
        try await engine.ingest([RawContent(id: "note", payload: .text("zeppelin minutes"))])
        #expect(
            try await engine.search("zeppelin", options: SearchOptions(mode: .lexicalOnly))
                .hits.map(\.itemID) == ["note"])

        try await engine.ingest([RawContent(id: "note", payload: .text("  \n"))])
        #expect(
            try await engine.search("zeppelin", options: SearchOptions(mode: .lexicalOnly))
                .hits.isEmpty)
        #expect(try await engine.item("note") != nil)

        try await engine.ingest([RawContent(id: "blank", payload: .text(""))])
        #expect(try await engine.item("blank") != nil)
        let again = try await engine.ingest([RawContent(id: "blank", payload: .text(""))])
        #expect(again.itemsUnchanged == 1)
    }

    @Test("Children of a producer that no longer runs are removed; the caller's stay")
    func replacedProducerChildren() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(processor: TwoProducers()))
        // "both": a root by one producer, a child by another — a transcriber
        // beside a frame extractor.
        try await engine.ingest([RawContent(id: "clip", payload: .text("both"))])
        try await engine.upsert(
            SearchItem(
                id: "clip#note", parentID: "clip",
                representations: [.text("the caller's note")]))
        #expect(try await engine.item("clip#child") != nil)

        // The second producer taken out: nothing in this run names it.
        let report = try await engine.ingest([RawContent(id: "clip", payload: .text("root"))])
        #expect(report.itemsRemoved == 1)
        #expect(try await engine.item("clip#child") == nil)
        #expect(try await engine.item("clip#note") != nil)
    }

    @Test("Re-ingesting keeps children the caller attached")
    func reingestKeepsCallerChildren() async throws {
        let chunker = ChunkerConfiguration(
            targetTokens: 16, maximumTokens: 16, overlapTokens: 0, minimumTokens: 1)
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                processor: TextChunkingProcessor(configuration: .init(chunker: chunker))))

        let long = (0..<12).map { "Paragraph \($0) mentions zeppelin\($0) specifically." }
            .joined(separator: "\n\n")
        try await engine.ingest([RawContent(id: "doc", payload: .text(long))])
        // The caller's own evidence under the same parent — the way frames sit
        // under a video.
        try await engine.upsert(
            SearchItem(
                id: "doc#annotation", parentID: "doc",
                representations: [.text("A reviewer's note about airships.")]))

        let report = try await engine.ingest(
            [RawContent(id: "doc", payload: .text("A short note."))])
        #expect(report.itemsRemoved > 0)
        #expect(try await engine.item("doc#annotation") != nil)
        #expect(try await engine.itemCount() == 2)
    }
}

@Suite("Regressions: model residency")
struct ResidencyRegressionTests {

    private func engine(sequential: Bool, ledger: ResidencyLedger) async throws
        -> SemanticSearchEngine
    {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "s", provider: LedgerEmbeddingProvider(ledger: ledger))
                ],
                reranker: LedgerRerankingProvider(ledger: ledger),
                runtime: RuntimePolicy(
                    maximumConcurrentInferenceJobs: 2,
                    sequentialModelResidency: sequential)))
        try await engine.upsert([textItem("a", "alpha beta"), textItem("b", "gamma")])
        return engine
    }

    @Test("Sequential residency never has two models loaded at once")
    func sequentialResidencyHoldsOneModel() async throws {
        let ledger = ResidencyLedger()
        let engine = try await engine(sequential: true, ledger: ledger)
        try await engine.prepare()
        for query in ["alpha", "gamma", "beta"] { _ = try await engine.search(query) }
        try await engine.upsert(textItem("c", "delta"))
        _ = try await engine.search("delta")

        #expect(await ledger.peak == 1)
        #expect(await ledger.loads > 4)  // it really did swap
    }

    @Test("A representation the provider returns no vector for is reported")
    func emptyVectorReported() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: VoidEmbeddingProvider())]))
        let report = try await engine.upsert([
            textItem("kept", "alpha words"), textItem("lost", "void words"),
        ])
        #expect(report.skipped.map(\.key.item) == ["lost"])
        #expect(report.skipped.first?.reason.contains("empty vector") == true)
    }

    @Test("With several permits, a release waits for every job in flight")
    func releaseTakesEveryPermit() async throws {
        let provider = SlowEmbeddingProvider()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: provider)],
                reranker: nil,
                runtime: RuntimePolicy(
                    maximumConcurrentInferenceJobs: 2, sequentialModelResidency: false)))
        try await engine.upsert([textItem("a", "alpha beta")])
        let search = Task { try await engine.search("alpha") }
        try await Task.sleep(for: .milliseconds(100))
        await engine.releaseModels()
        _ = try await search.value
        #expect(await provider.releasedMidEmbed == false)
        #expect(await provider.releases == 1)
    }

    @Test("Without sequential residency both models stay loaded")
    func concurrentResidencyKeepsBoth() async throws {
        let ledger = ResidencyLedger()
        let engine = try await engine(sequential: false, ledger: ledger)
        _ = try await engine.search("alpha")
        _ = try await engine.search("gamma")
        #expect(await ledger.peak == 2)
        #expect(await ledger.loads == 2)
    }

    @Test("Idle models are released after the configured timeout")
    func idleTimeoutReleasesModels() async throws {
        let embedder = HashingEmbeddingProvider()
        let reranker = ScriptedRerankingProvider.termOverlap()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: embedder)],
                reranker: reranker,
                runtime: RuntimePolicy(modelIdleTimeout: .milliseconds(100))))
        try await engine.upsert(textItem("a", "alpha"))
        _ = try await engine.search("alpha")
        #expect(await embedder.isPrepared)

        // Polled with a generous deadline rather than slept: the scale tests
        // saturate the cooperative pool when the whole suite runs in parallel,
        // and a 100 ms timer — this loop's own sleeps included — has been
        // observed to wake more than a minute late. The loop exits as soon as
        // the release lands, so the deadline costs nothing when it passes.
        let deadline = ContinuousClock.now + .seconds(600)
        while ContinuousClock.now < deadline {
            if await !embedder.isPrepared, await !reranker.isPrepared { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(await !embedder.isPrepared)
        #expect(await !reranker.isPrepared)

        // And the next search reloads what it needs.
        #expect(try await engine.search("alpha").hits.first?.itemID == "a")
    }

    @Test("A provider that learns its dimensions on prepare gets a usable space")
    func lateDimensions() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: LateDimensionsProvider())]))
        try await engine.upsert([textItem("a", "alpha"), textItem("b", "beta")])
        let first = try await engine.search("alpha", options: SearchOptions(mode: .denseOnly))
        let second = try await engine.search("beta", options: SearchOptions(mode: .denseOnly))
        #expect(first.hits.first?.itemID == "a")
        #expect(second.hits.first?.itemID == "b")
    }
}

@Suite("Regressions: query handling")
struct QueryRegressionTests {

    @Test("A text-and-image query against a text-only space is answered and reported")
    func mixedQueryAgainstTextOnlySpace() async throws {
        let engine = try await SemanticSearchEngine.testEngine(acceptsVisual: false)
        try await engine.upsert([textItem("a", "harbour at dawn"), textItem("b", "tax forms")])

        let results = try await engine.search(
            SearchQuery(
                text: "harbour at dawn",
                images: [.encoded(Data([0x89, 0x50]), contentType: "image/png")]),
            options: SearchOptions(mode: .denseOnly))
        #expect(results.hits.first?.itemID == "a")
        #expect(results.diagnostics.partiallyAnsweredSpaces.map(\.space) == ["test"])
        #expect(results.diagnostics.channels.contains { $0.channel.space == "test" })
    }

    @Test("A query is embedded once per instruction, not once per family")
    func queryEmbeddedOncePerInstruction() async throws {
        let provider = QueryLedgerEmbeddingProvider()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [EmbeddingSpaceConfiguration(id: "s", provider: provider)]))
        try await engine.upsert([textItem("a", "alpha beta"), textItem("b", "gamma")])

        let plain = try await engine.search("alpha", options: SearchOptions(mode: .denseOnly))
        #expect(plain.diagnostics.channels.count > 1)
        #expect(plain.hits.first?.itemID == "a")
        #expect(await provider.queries == [nil])

        _ = try await engine.search(
            "alpha",
            options: SearchOptions(mode: .denseOnly, queryInstructions: [.caption: "a picture"]))
        let second = await provider.queries.dropFirst()
        #expect(Set(second) == [nil, "a picture"])
        #expect(second.count == 2)
    }
}

@Suite("Regressions: filters")
struct FilterRegressionTests {

    static let metadata: [ItemID: Metadata] = [
        "done": ["status": .string("done"), "n": .integer(5), "tags": .stringList(["a", "b"])],
        "open": ["status": .string("open"), "n": .double(5), "tags": .stringList(["a"])],
        "none": [:],
        "quoted": ["say \"hi\"": .string("yes"), "tag \"list\"": .stringList(["x"])],
    ]

    private func engine() async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        try await engine.upsert([
            textItem("done", "shared words", metadata: Self.metadata["done"]!),
            textItem("open", "shared words", metadata: Self.metadata["open"]!),
            textItem("none", "shared words", metadata: Self.metadata["none"]!),
            textItem("quoted", "shared words", metadata: Self.metadata["quoted"]!),
        ])
        return engine
    }

    private func ids(_ engine: SemanticSearchEngine, _ filter: SearchFilter) async throws
        -> Set<ItemID>
    {
        let results = try await engine.search(
            "shared words", options: SearchOptions(mode: .lexicalOnly, filter: filter))
        return Set(results.hits.map(\.itemID))
    }

    @Test("notEquals includes items that lack the key")
    func notEqualsIsTheComplementOfEquals() async throws {
        let engine = try await engine()
        #expect(
            try await ids(engine, .notEquals(key: "status", value: .string("done")))
                == ["open", "none", "quoted"])
        #expect(
            try await ids(engine, .not(.equals(key: "status", value: .string("done"))))
                == ["open", "none", "quoted"])
    }

    @Test("A metadata key containing a double quote can be filtered on")
    func keysWithQuotes() async throws {
        let engine = try await engine()
        #expect(
            try await ids(engine, .equals(key: "say \"hi\"", value: .string("yes")))
                == ["quoted"])
        #expect(try await ids(engine, .exists(key: "say \"hi\"")) == ["quoted"])
        #expect(
            try await ids(engine, .notEquals(key: "say \"hi\"", value: .string("yes")))
                == ["done", "open", "none"])
        #expect(
            try await ids(engine, .equals(key: "tag \"list\"", value: .stringList(["x"])))
                == ["quoted"])
        #expect(try await ids(engine, .equals(key: "tag \"list\"", value: .string("x"))) == ["quoted"])
    }

    @Test("withinParent: the parent and its subtree, negated, several parents, a missing one")
    func withinParentSubtrees() async throws {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        func item(_ id: ItemID, _ parent: ItemID?) -> SearchItem {
            SearchItem(
                id: id, parentID: parent,
                representations: [.text("shared words", kind: .originalText)])
        }
        try await engine.upsert([
            item("a", nil), item("a#1", "a"), item("a#1#x", "a#1"), item("b", nil),
            item("b#1", "b"), item("ghost#1", "ghost"),
        ])
        let cases: [(SearchFilter, Set<ItemID>)] = [
            (.withinParent("a"), ["a", "a#1", "a#1#x"]),
            (.withinParent("a#1"), ["a#1", "a#1#x"]),
            (.not(.withinParent("a")), ["b", "b#1", "ghost#1"]),
            // A parent that was never stored still has its children.
            (.withinParent("ghost"), ["ghost#1"]),
            (.or([.withinParent("a#1"), .withinParent("b")]), ["a#1", "a#1#x", "b", "b#1"]),
            (.and([.withinParent("a"), .not(.withinParent("a#1"))]), ["a"]),
            (.withinParent("missing"), []),
        ]
        for (filter, expected) in cases {
            #expect(try await ids(engine, filter) == expected, "\(filter)")
            let listed = try await engine.listItems(
                matching: filter, topLevelOnly: false, limit: 100)
            #expect(Set(listed.items.map(\.id)) == expected, "\(filter)")
            #expect(listed.total == expected.count, "\(filter)")
        }
    }

    @Test("The SQL filter agrees with the in-memory evaluator")
    func sqlAgreesWithEvaluator() async throws {
        let engine = try await engine()
        let metadata = Self.metadata
        let filters: [SearchFilter] = [
            .notEquals(key: "status", value: .string("done")),
            .not(.exists(key: "status")),
            .not(.anyOf(key: "status", values: [.string("done"), .string("open")])),
            .or([.equals(key: "say \"hi\"", value: .string("yes")), .not(.exists(key: "status"))]),
            .not(.lessThan(key: "status", value: .string("m"))),
            .lessThan(key: "status", value: .string("m")),
            .equals(key: "n", value: .integer(5)),
            .equals(key: "n", value: .double(5)),
            .between(key: "n", lower: .integer(4), upper: .double(6)),
            .greaterThan(key: "status", value: .integer(0)),
            .equals(key: "tags", value: .stringList(["a"])),
            .equals(key: "tags", value: .string("b")),
            .anyOf(key: "tags", values: [.string("b"), .stringList(["a"])]),
            .notEquals(key: "tags", value: .stringList(["a", "b"])),
            .equals(key: "status", value: .stringList([])),
        ]
        for filter in filters {
            let expected = Set(
                metadata.filter {
                    filter.matches(
                        itemID: $0.key, kind: .originalText, metadata: $0.value, ancestors: [])
                }.keys)
            #expect(try await ids(engine, filter) == expected, "\(filter)")
        }
    }
}

@Suite("Regressions: inference gate")
struct InferenceGateTests {

    @Test("Interactive waiters are served before earlier background waiters")
    func interactiveJumpsTheQueue() async throws {
        let gate = AsyncSemaphore(value: 1)
        let order = OrderLog()
        try await gate.wait()

        let background = Task {
            try await gate.wait(priority: .background)
            await order.append("background")
            gate.signal()
        }
        try await Task.sleep(for: .milliseconds(50))
        let interactive = Task {
            try await gate.wait(priority: .interactive)
            await order.append("interactive")
            gate.signal()
        }
        try await Task.sleep(for: .milliseconds(50))

        gate.signal()
        try await background.value
        try await interactive.value
        #expect(await order.entries == ["interactive", "background"])
    }

    @Test("A cancelled waiter leaves the queue without taking a permit")
    func cancelledWaiterThrows() async throws {
        let gate = AsyncSemaphore(value: 1)
        try await gate.wait()

        let waiter = Task { try await gate.wait(priority: .background) }
        try await Task.sleep(for: .milliseconds(50))
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }

        // The permit returns to the pool rather than to the cancelled waiter.
        gate.signal()
        try await gate.wait()
        gate.signal()
    }
}

// MARK: - Fixtures

/// Tokenises on whitespace and reverses every token. Distinguishable from the
/// default analyzer by construction, which is all the analyzer test needs.
struct ReversingAnalyzer: LexicalAnalyzer {
    var analyzerVersion: String { "reversing.v1" }

    func analyzeDocument(_ text: String, language: String?) -> AnalyzedDocument {
        let terms = analyzeQuery(text, language: language)
        return AnalyzedDocument(terms: terms, tokenCount: terms.count)
    }

    func analyzeQuery(_ text: String, language: String?) -> [String] {
        text.lowercased().split(whereSeparator: \.isWhitespace).map { String($0.reversed()) }
    }
}

actor OrderLog {
    private(set) var entries: [String] = []
    func append(_ entry: String) { entries.append(entry) }
}

/// Counts how many models are loaded at once.
actor ResidencyLedger {
    private var resident: Set<String> = []
    private(set) var peak = 0
    private(set) var loads = 0

    func loaded(_ name: String) {
        if resident.insert(name).inserted { loads += 1 }
        peak = max(peak, resident.count)
    }

    func released(_ name: String) { resident.remove(name) }
}

actor LedgerEmbeddingProvider: EmbeddingProvider {
    private let ledger: ResidencyLedger
    private let inner = HashingEmbeddingProvider()
    private var prepared = false

    init(ledger: ResidencyLedger) { self.ledger = ledger }

    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "ledger-embedder", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: 64)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        await ledger.loaded("embedder")
        prepared = true
    }

    func release() async {
        prepared = false
        await ledger.released("embedder")
    }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard prepared else { throw SemanticSearchError.providerNotPrepared("ledger-embedder") }
        return try await inner.embed(inputs)
    }
}

/// Records the instruction of every query it embeds.
actor QueryLedgerEmbeddingProvider: EmbeddingProvider {
    private let inner = HashingEmbeddingProvider()
    private var prepared = false
    private(set) var queries: [String?] = []

    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "query-ledger", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: 64)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        prepared = true
    }

    func release() async { prepared = false }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard prepared else { throw SemanticSearchError.providerNotPrepared("query-ledger") }
        for input in inputs {
            if case .query(let instruction) = input.role { queries.append(instruction) }
        }
        return try await inner.embed(inputs)
    }
}

/// Returns no vector for any input whose text mentions "void".
actor VoidEmbeddingProvider: EmbeddingProvider {
    private let inner = HashingEmbeddingProvider()
    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "void-embedder", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: 64)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { true }
    func prepare() async throws {}
    func release() async {}
    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        try await inner.prepare()
        let outputs = try await inner.embed(inputs)
        return zip(inputs, outputs).map { input, output in
            guard input.text.contains("void") else { return output }
            var empty = output
            empty.vector = []
            return empty
        }
    }
}

/// Embeds slowly, and notes a release that lands while it is embedding.
actor SlowEmbeddingProvider: EmbeddingProvider {
    private let inner = HashingEmbeddingProvider()
    private var prepared = false
    private var embedding = 0
    private(set) var releasedMidEmbed = false
    private(set) var releases = 0

    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "slow-embedder", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: 64)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        prepared = true
    }

    func release() async {
        if embedding > 0 { releasedMidEmbed = true }
        releases += 1
        prepared = false
    }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        guard prepared else { throw SemanticSearchError.providerNotPrepared("slow-embedder") }
        embedding += 1
        defer { embedding -= 1 }
        try await Task.sleep(for: .milliseconds(300))
        return try await inner.embed(inputs)
    }
}

actor LedgerRerankingProvider: RerankingProvider {
    private let ledger: ResidencyLedger
    private let inner = ScriptedRerankingProvider.termOverlap()
    private var prepared = false

    init(ledger: ResidencyLedger) { self.ledger = ledger }

    var identity: RerankerIdentity {
        RerankerIdentity(modelIdentity: "ledger-reranker", promptVersion: "1")
    }
    var capabilities: RerankingCapabilities { RerankingCapabilities() }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        await ledger.loaded("reranker")
        prepared = true
    }

    func release() async {
        prepared = false
        await ledger.released("reranker")
    }

    func score(query: RerankQuery, candidates: [RerankCandidate]) async throws -> [RerankScore] {
        guard prepared else { throw SemanticSearchError.providerNotPrepared("ledger-reranker") }
        return try await inner.score(query: query, candidates: candidates)
    }
}

/// Reports zero dimensions until prepared, like `MLXEmbeddingProvider(model:)`.
actor LateDimensionsProvider: EmbeddingProvider {
    private let inner = HashingEmbeddingProvider(dimensions: 64)
    private var dimensions = 0
    private var prepared = false

    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "late", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: dimensions)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        dimensions = 64
        prepared = true
    }

    func release() async { prepared = false }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        try await inner.embed(inputs)
    }
}

/// Suspends inside `embed` until released, so a test can act mid-embedding.
actor BlockingEmbeddingProvider: EmbeddingProvider {
    private let inner = HashingEmbeddingProvider(dimensions: 64)
    private var prepared = false
    private var started: CheckedContinuation<Void, Never>?
    private var hasStarted = false
    private var gate: CheckedContinuation<Void, Never>?

    var descriptor: EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: "blocking", tokenizerVersion: "t", preprocessingVersion: "p",
            pooling: "sum", normalized: true, dimensions: 64)
    }
    var capabilities: EmbeddingCapabilities {
        EmbeddingCapabilities(acceptsText: true, maximumBatchSize: 8, maximumInputTokens: 512)
    }
    var isPrepared: Bool { prepared }

    func prepare() async throws {
        try await inner.prepare()
        prepared = true
    }

    func release() async { prepared = false }

    func waitUntilEmbedding() async {
        if hasStarted { return }
        await withCheckedContinuation { started = $0 }
    }

    func release(embedding: Void) {
        gate?.resume()
        gate = nil
    }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
        hasStarted = true
        started?.resume()
        started = nil
        await withCheckedContinuation { gate = $0 }
        return try await inner.embed(inputs)
    }
}

@Suite("Regressions: chunking")
struct ChunkingRegressionTests {

    @Test(
        "Overlap counts against the ceiling",
        arguments: [
            ChunkerConfiguration.default,
            ChunkerConfiguration(
                targetTokens: 96, maximumTokens: 96, overlapTokens: 16, minimumTokens: 24),
            ChunkerConfiguration(
                targetTokens: 64, maximumTokens: 128, overlapTokens: 32, minimumTokens: 8),
        ])
    func chunksNeverExceedTheCeiling(configuration: ChunkerConfiguration) {
        // Paragraphs of mixed length, so chunks close at every kind of
        // boundary: under the target, at it, and at a paragraph that alone
        // nearly fills the ceiling.
        let text = (0..<40).map { paragraph in
            let sentences = [3, 18, 40, 7, 25][paragraph % 5]
            return (0..<sentences).map {
                "Sentence \($0) of paragraph \(paragraph) talks about harbour logistics."
            }.joined(separator: " ")
        }.joined(separator: "\n\n")

        let chunks = TextChunker(configuration: configuration).chunk(text)
        #expect(chunks.count > 1)
        for chunk in chunks {
            #expect(
                chunk.estimatedTokens <= configuration.maximumTokens,
                "chunk \(chunk.ordinal) is \(chunk.estimatedTokens) tokens")
        }
        // Overlap still happens where it fits.
        #expect(chunks.dropFirst().contains { $0.overlapPrefixLength > 0 })
    }
}

@Suite("Regressions: media helpers")
struct MediaHelperRegressionTests {

    private func line(_ text: String, y: Double) -> RecognizedText {
        RecognizedText(
            text: text,
            boundingBox: SourceLocator.BoundingBox(x: 0.1, y: y, width: 0.8, height: 0.05),
            confidence: 1)
    }

    @Test("OCR boxes are assigned to chunks by offset, not by substring")
    func ocrBoxesFollowOffsets() {
        let lines = [
            line("Quarterly harbour report", y: 0.1),
            line("OK", y: 0.2),
            line("Tonnage rose sharply. OK", y: 0.3),
        ]
        let text = lines.map(\.text).joined(separator: "\n")
        let ns = text as NSString
        let first = ns.range(of: "Quarterly harbour report\nOK")
        let second = ns.range(of: "Tonnage rose sharply. OK")
        let processor = VisionOCRProcessor()

        let chunkA = TextChunk(
            ordinal: 0, text: ns.substring(with: first),
            range: .init(location: first.location, length: first.length), estimatedTokens: 6)
        let chunkB = TextChunk(
            ordinal: 1, text: ns.substring(with: second),
            range: .init(location: second.location, length: second.length), estimatedTokens: 6)

        let boxesA = processor.locator(for: chunkA, lines: lines, base: .unspecified)
            .boundingBoxes.map(\.y)
        let boxesB = processor.locator(for: chunkB, lines: lines, base: .unspecified)
            .boundingBoxes.map(\.y)
        // "OK" is a substring of the third line, so substring matching put
        // its box in the second chunk too.
        #expect(boxesA == [0.1, 0.2])
        #expect(boxesB == [0.3])
    }

    @Test("OCR language is never guessed from the first requested language")
    func ocrLanguage() {
        let lines = [line("Отчёт о грузообороте порта за третий квартал", y: 0.1)]
        let text = lines[0].text

        let single = VisionOCRProcessor(configuration: .init(recognitionLanguages: ["ru-RU"]))
        #expect(single.pageLanguage(of: text, lines: lines, hint: nil) == "ru-RU")

        let several = VisionOCRProcessor(
            configuration: .init(recognitionLanguages: ["en-US", "ru-RU"]))
        let detected = several.pageLanguage(of: text, lines: lines, hint: nil)
        #expect(detected == "ru-RU")

        let automatic = VisionOCRProcessor()
        #expect(automatic.pageLanguage(of: text, lines: lines, hint: nil) == "ru")
        // Too little to detect: the caller's hint stands.
        #expect(automatic.pageLanguage(of: "42", lines: [], hint: "tr") == "tr")
    }

    @Test("Frame time ranges tile the asset when the interval is widened")
    func frameRangesTileTheAsset() {
        let extractor = VideoFrameExtractor(
            configuration: .init(interval: 5, maximumFrames: 4))
        // 100 s at 5 s would be 21 frames; the cap widens the interval to 25 s.
        let times = extractor.sampleTimes(duration: 100)
        #expect(times == [0, 25, 50, 75])

        let frames = times.enumerated().map {
            ExtractedFrame(image: Self.pixel, time: $0.element, index: $0.offset)
        }
        let ranges = VideoFrameExtractor.timeRanges(for: frames, duration: 100)
        #expect(ranges.map(\.start) == [0, 25, 50, 75])
        #expect(ranges.map(\.end) == [25, 50, 75, 100])
    }

    @Test("Capabilities accept each modality only when the provider says so")
    func capabilitiesArePerModality() {
        let imagesOnly = EmbeddingCapabilities(acceptsText: true, acceptsImages: true)
        #expect(imagesOnly.accepts(.image))
        #expect(!imagesOnly.accepts(.video))
        #expect(!imagesOnly.accepts(.audio))
        #expect(imagesOnly.accepts(.transcript))
        #expect(!EmbeddingCapabilities(acceptsText: true).accepts(.audio))
        #expect(!RepresentationKind.audio.isLexicallySearchable)
        #expect(!RepresentationKind.audio.isVisual)
    }

    private static let pixel: CGImage = {
        let context = CGContext(
            data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }()
}

@Suite("Regressions: upsert")
struct UpsertRegressionTests {

    @Test("A corrected locator on unchanged content is stored without re-embedding")
    func locatorChangesLand() async throws {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        func item(start: Double) -> SearchItem {
            SearchItem(
                id: "clip",
                representations: [
                    ContentRepresentation(
                        id: "transcript", kind: .transcript,
                        content: .text("the tide turns at noon"),
                        provenance: Provenance(producer: "asr", confidence: Float(start)),
                        locator: SourceLocator(
                            timeRange: SourceLocator.TimeRange(start: start, end: start + 2)))
                ])
        }

        try await engine.upsert(item(start: 10))
        let report = try await engine.upsert(item(start: 12))
        #expect(report.itemsUnchanged == 1)
        #expect(report.vectorsWritten == 0)

        let hit = try #require(try await engine.search("tide").hits.first)
        #expect(hit.locator.timeRange?.start == 12)
        #expect(hit.provenance.confidence == 12)
    }
}

@Suite("Regressions: graph lifecycle")
struct GraphLifecycleRegressionTests {

    private func engine(storeURL: URL, minimumVectorsForGraph: Int) async throws
        -> SemanticSearchEngine
    {
        try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "s", provider: HashingEmbeddingProvider(dimensions: 32),
                        indexKind: .hnsw,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: minimumVectorsForGraph))
                ],
                runtime: RuntimePolicy(embeddingBatchSize: 32)))
    }

    @Test("A space that opens empty moves to its graph once it passes the threshold")
    func emptySpaceIsPromotedToGraph() async throws {
        let directory = try TemporaryDirectory()
        let engine = try await engine(storeURL: directory.storeURL, minimumVectorsForGraph: 40)
        #expect(await engine.usesExactSearch(space: "s") == true)

        try await engine.upsert((0..<30).map { textItem("a\($0)", "first batch \($0)") })
        #expect(await engine.usesExactSearch(space: "s") == true)

        try await engine.upsert((0..<30).map { textItem("b\($0)", "second batch \($0)") })
        #expect(await engine.usesExactSearch(space: "s") == false)

        let results = try await engine.search(
            "second batch 7", options: SearchOptions(mode: .denseOnly, limit: 60))
        #expect(results.hits.count == 60)
        #expect(results.hits.first?.itemID == "b7")
        try await engine.close()
    }

    @Test("Flush publishes a snapshot after thousands of vectors, whatever the batch size")
    func snapshotThresholdCountsVectors() async throws {
        let directory = try TemporaryDirectory()
        do {
            let engine = try await engine(
                storeURL: directory.storeURL, minimumVectorsForGraph: 100)
            // 2,100 vectors is 66 batches of 32: far under a threshold counted
            // in generations, over one counted in vectors.
            try await engine.upsert((0..<2_100).map { textItem("i\($0)", "entry \($0)") })
            try await engine.flush()
            let stored = try #require(try await engine.store.loadSpaces().first)
            #expect(stored.snapshotFile != nil)
            #expect(stored.snapshotGeneration == stored.generation)
            try await engine.close()
        }

        let reopened = try await engine(storeURL: directory.storeURL, minimumVectorsForGraph: 100)
        #expect(await reopened.usesExactSearch(space: "s") == false)
        let results = try await reopened.search(
            "entry 1234", options: SearchOptions(mode: .denseOnly))
        #expect(results.hits.first?.itemID == "i1234")
        try await reopened.close()
    }
}

@Suite("Regressions: reranker modality")
struct RerankModalityRegressionTests {

    private static let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01])

    private func engine(reranker: ScriptedRerankingProvider) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine.testEngine(reranker: reranker)
        try await engine.upsert([
            textItem("red", "a red bicycle"),
            textItem("blue", "a blue bicycle"),
            SearchItem(
                id: "photo",
                representations: [
                    ContentRepresentation(
                        id: "media", kind: .image,
                        content: .imageData(Self.png, contentType: "public.png"))
                ]),
        ])
        return engine
    }

    @Test("A text-only reranker does not rerank an image-only query")
    func imageOnlyQuerySkipsTextReranker() async throws {
        let reranker = ScriptedRerankingProvider.termOverlap()
        let engine = try await engine(reranker: reranker)

        let results = try await engine.search(
            SearchQuery(images: [.encoded(Self.png, contentType: "public.png")]))
        #expect(results.diagnostics.rerankSkippedReason != nil)
        #expect(results.diagnostics.rerankedCandidates == 0)
        #expect(await reranker.calls.isEmpty)
        #expect(results.hits.allSatisfy { $0.evidence.rerankScore == nil })
    }

    @Test("A candidate with nothing a text reranker can read is not scored as empty text")
    func imageCandidateIsNotScoredAsEmptyText() async throws {
        let reranker = ScriptedRerankingProvider.termOverlap()
        let engine = try await engine(reranker: reranker)

        let results = try await engine.search("bicycle")
        let sent = await reranker.calls.flatMap(\.candidates)
        #expect(!sent.isEmpty)
        #expect(sent.allSatisfy { !$0.text.isEmpty })

        let photo = try #require(results.hits.first { $0.itemID == "photo" })
        #expect(photo.evidence.rerankScore == nil)
        #expect(results.diagnostics.rerankUnscoredCandidates == 1)
    }

    @Test("A mixed query in front of a text-only reranker is reported as partly judged")
    func mixedQueryReportsIgnoredImages() async throws {
        let reranker = ScriptedRerankingProvider.termOverlap()
        let engine = try await engine(reranker: reranker)

        let results = try await engine.search(
            SearchQuery(text: "bicycle", images: [.encoded(Self.png, contentType: "public.png")]))
        #expect(results.diagnostics.rerankIgnoredQueryImages)
        #expect(results.diagnostics.rerankSkippedReason == nil)
        #expect(results.diagnostics.rerankedCandidates > 0)
    }

    @Test("A limit above the rerank pool is filled from retrieval order", arguments: [false, true])
    func limitAbovePool(orderingOnly: Bool) async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert((0..<30).map { textItem("i\($0)", "bicycle number \($0)") })

        let results = try await engine.search(
            "bicycle",
            options: SearchOptions(
                rerankPoolSize: 10, limit: 25, rerankerOrderingOnly: orderingOnly))
        #expect(results.hits.count == 25)
        #expect(results.hits.prefix(10).allSatisfy { $0.evidence.rerankScore != nil })
        #expect(results.hits.dropFirst(10).allSatisfy { $0.evidence.rerankScore == nil })
        let scores = results.hits.map(\.evidence.finalScore)
        #expect(zip(scores, scores.dropFirst()).allSatisfy { $0 >= $1 })
    }

    @Test("Hybrid search without a reranker says so")
    func missingRerankerIsReported() async throws {
        let engine = try await SemanticSearchEngine.testEngine(reranker: nil)
        try await engine.upsert(textItem("red", "a red bicycle"))
        let results = try await engine.search("bicycle")
        #expect(results.diagnostics.rerankSkippedReason != nil)
        let retrievalOnly = try await engine.search(
            "bicycle", options: SearchOptions(mode: .retrievalOnly))
        #expect(retrievalOnly.diagnostics.rerankSkippedReason == nil)
    }
}

@Suite("Regressions: kind routing")
struct KindRoutingRegressionTests {

    @Test("A configured kind the provider cannot take opens no channel and needs no vectors")
    func unacceptedConfiguredKindIsIgnored() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "s", provider: HashingEmbeddingProvider(),
                        kinds: Set(RepresentationKind.allCases))
                ]))
        try await engine.upsert([
            textItem("a", "harbour tides"),
            SearchItem(
                id: "recording",
                representations: [
                    ContentRepresentation(
                        id: "media", kind: .audio,
                        content: .mediaReference(URL(fileURLWithPath: "/tmp/r.m4a")))
                ]),
        ])

        let results = try await engine.search(
            "harbour", options: SearchOptions(mode: .denseOnly))
        #expect(!results.diagnostics.channels.contains { $0.channel.family == .audio })
        #expect(results.hits.first?.itemID == "a")

        // Nothing is left for `reindex` to chase forever.
        let report = try await engine.reindex()
        #expect(report.skipped.isEmpty)
        #expect(report.vectorsWritten == 0)
    }

    @Test("A space configured for kinds its provider rejects embeds nothing else instead")
    func noAcceptedKindsEmbedsNothing() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "s", provider: HashingEmbeddingProvider(acceptsVisual: false),
                        kinds: [.image])
                ]))
        try await engine.upsert(textItem("a", "harbour tides"))
        #expect(try await engine.vectorCount(space: "s") == 0)
    }
}

@Suite("Regressions: tombstones")
struct TombstoneRegressionTests {

    @Test("A space with no published snapshot accumulates no tombstones")
    func noSnapshotNoTombstones() async throws {
        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(storeURL: directory.storeURL)
        try await engine.upsert((0..<20).map { textItem("i\($0)", "entry \($0)") })
        for index in 0..<20 { try await engine.remove(ItemID("i\(index)")) }
        #expect(try await engine.store.tombstones(space: "test", newerThan: 0).isEmpty)
        try await engine.close()
    }

    @Test("Tombstones are still written, and honoured, once a snapshot exists")
    func snapshotKeepsTombstones() async throws {
        let directory = try TemporaryDirectory()
        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: directory.storeURL, indexKind: .hnsw)
            try await engine.upsert((0..<20).map { textItem("i\($0)", "entry \($0)") })
            try await engine.rebuildVectorIndex(space: "test")
            try await engine.remove("i3")
            #expect(try await engine.store.tombstones(space: "test", newerThan: 0).count == 1)
            try await engine.close()
        }
        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: directory.storeURL, indexKind: .hnsw)
        let results = try await reopened.search(
            "entry 3", options: SearchOptions(mode: .denseOnly, limit: 30))
        #expect(!results.hits.contains { $0.itemID == "i3" })
        try await reopened.close()
    }
}

@Suite("Regressions: evaluation")
struct EvaluationRegressionTests {

    @Test("ANN recall refuses a space that has no graph to measure")
    func annRecallRefusesExactSpace() async throws {
        let engine = try await SemanticSearchEngine.testEngine(indexKind: .exact)
        try await engine.upsert((0..<10).map { textItem("i\($0)", "entry \($0)") })
        let runner = EvaluationRunner(
            corpus: EvaluationCorpus(name: "empty", documents: [], queries: []))
        await #expect(throws: SemanticSearchError.self) {
            _ = try await runner.annRecall(
                against: engine, space: "test", queryVectors: [[Float](repeating: 0.1, count: 64)])
        }
    }
}

@Suite("Regressions: query language")
struct QueryLanguageRegressionTests {

    private func englishEngine() async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(configuration: EngineConfiguration())
        try await engine.upsert([
            SearchItem(
                id: "bar",
                representations: [
                    ContentRepresentation(
                        id: "t", kind: .transcript,
                        content: .text("We're making Nina's espresso martini with chocolate."),
                        language: "en")
                ]),
            SearchItem(
                id: "office",
                representations: [
                    ContentRepresentation(
                        id: "t", kind: .originalText,
                        content: .text("Offices in Tokyo and Moscow."), language: "en")
                ]),
        ])
        return engine
    }

    @Test(
        "A short query misdetected as another language still matches an English index",
        arguments: [("espresso martini", "bar"), ("Nina Carter", "bar"), ("Tokyo office", "office")])
    func shortQueryUsesIndexedLanguage(query: String, expected: String) async throws {
        let engine = try await englishEngine()
        let results = try await engine.search(query, options: SearchOptions(mode: .lexicalOnly))
        #expect(results.hits.first?.itemID == ItemID(expected), "terms: \(results.diagnostics.queryTerms)")
        #expect(results.diagnostics.detectedQueryLanguage == "en")
    }

    @Test("A confidently detected language is kept even when the index lacks it")
    func confidentDetectionIsKept() async throws {
        let engine = try await englishEngine()
        let results = try await engine.search(
            "где находится офис", options: SearchOptions(mode: .lexicalOnly))
        #expect(results.diagnostics.detectedQueryLanguage == "ru")
    }
}

@Suite("Regressions: media access")
struct MediaAccessRegressionTests {

    static func engine(at store: URL, reranker: ScriptedRerankingProvider? = nil, allows: Bool)
        async throws -> SemanticSearchEngine
    {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: store,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test",
                        provider: HashingEmbeddingProvider(dimensions: 32, acceptsVisual: true),
                        indexKind: .exact)
                ],
                reranker: reranker,
                mediaAccess: { _ in allows }))
        try await engine.prepare()
        return engine
    }

    @Test("Media the caller's boundary refuses is neither embedded nor shown to the reranker")
    func refusedMediaIsNotRead() async throws {
        let directory = try TemporaryDirectory()
        let store = directory.url.appendingPathComponent("index.sqlite")
        let picture = directory.url.appendingPathComponent("harbour.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: picture)
        let item = SearchItem(
            id: "harbour",
            representations: [
                ContentRepresentation(id: "media", kind: .image, content: .mediaReference(picture))
            ])

        let refusing = try await Self.engine(at: store, allows: false)
        let report = try await refusing.upsert([item])
        #expect(report.skipped.contains { $0.reason.contains("may not be read") })
        try await refusing.close()

        // Indexed while it was allowed; the file then became one the boundary
        // refuses — a link to a picture outside the library's folders.
        let allowing = try await Self.engine(at: store, allows: true)
        _ = try await allowing.reindex()
        try await allowing.close()
        let reranker = ScriptedRerankingProvider(
            capabilities: RerankingCapabilities(acceptsImages: true, maximumImagesPerPair: 2)
        ) { _, candidate in RerankScore(score: candidate.images.isEmpty ? 0 : 1) }
        let engine = try await Self.engine(at: store, reranker: reranker, allows: false)
        let results = try await engine.search("harbour")
        let sent = await reranker.calls.flatMap(\.candidates)
        #expect(sent.allSatisfy { $0.images.isEmpty })
        #expect(
            results.diagnostics.unavailableRepresentations.contains {
                $0.reason.contains("may not be read")
            })
        try await engine.close()
    }
}
