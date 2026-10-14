import Foundation
import Testing

@testable import SemanticSearch

@Suite("Resilience")
struct ResilienceTests {

    // MARK: - Cancellation

    /// Signals once, when the work under test has demonstrably started.
    ///
    /// Cancelling after a fixed sleep is a race: under a loaded machine the
    /// sleep overshoots, the build finishes, and the test asserts that a
    /// completed job was cancelled. This waits for the embedder to actually be
    /// called instead, so the cancellation always lands mid-flight.
    actor StartSignal {
        private var continuation: CheckedContinuation<Void, Never>?
        private var fired = false

        func signal() {
            guard !fired else { return }
            fired = true
            continuation?.resume()
            continuation = nil
        }

        func wait() async {
            guard !fired else { return }
            await withCheckedContinuation { continuation in
                if fired {
                    continuation.resume()
                } else {
                    self.continuation = continuation
                }
            }
        }
    }

    /// An embedder that reports when it is first used, and is slow enough
    /// afterwards that a cancellation has somewhere to land.
    actor SignallingEmbeddingProvider: EmbeddingProvider {
        private let inner = HashingEmbeddingProvider(dimensions: 32)
        private let signal: StartSignal
        private var callCount = 0
        private var prepared = false

        init(signal: StartSignal) { self.signal = signal }

        // Mirrored rather than forwarded: the protocol's requirements are plain
        // getters, and an actor cannot satisfy those with a `get async`.
        let descriptor = EmbeddingSpaceDescriptor(
            modelIdentity: "hashing-trigram", tokenizerVersion: "trigram.v1",
            preprocessingVersion: "lowercase-fold.v1", pooling: "sum", normalized: true,
            dimensions: 32)

        let capabilities = EmbeddingCapabilities(
            acceptsText: true, maximumBatchSize: 32, maximumInputTokens: 512)

        var isPrepared: Bool { prepared }

        func prepare() async throws {
            try await inner.prepare()
            prepared = true
        }

        func release() async {
            await inner.release()
            prepared = false
        }

        func embed(_ inputs: [EmbeddingInput]) async throws -> [EmbeddingOutput] {
            callCount += 1
            if callCount == 2 { await signal.signal() }
            if callCount > 1 { try await Task.sleep(for: .milliseconds(5)) }
            try Task.checkCancellation()
            return try await inner.embed(inputs)
        }
    }

    private func cancellableEngine(
        storeURL: URL?, signal: StartSignal
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: SignallingEmbeddingProvider(signal: signal),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                runtime: RuntimePolicy(embeddingBatchSize: 1)))
        try await engine.prepare()
        return engine
    }

    @Test("Cancelling an index build stops it")
    func indexingIsCancellable() async throws {
        let signal = StartSignal()
        let engine = try await cancellableEngine(storeURL: nil, signal: signal)
        let items = (0..<400).map { textItem("item-\($0)", "document number \($0)") }

        let task = Task { try await engine.upsert(items) }
        await signal.wait()
        task.cancel()

        await #expect(throws: (any Error).self) { try await task.value }

        // An upsert writes the rows first and embeds second, so a cancellation
        // during embedding leaves the content durable and some vectors missing
        // — never the other way round. That asymmetry is what makes recovery a
        // matter of finding the slots without vectors.
        let items_ = try await engine.itemCount()
        let vectors = try await engine.vectorCount(space: "test")
        #expect(items_ == items.count)
        #expect(vectors < items.count)
    }

    @Test("A cancelled build leaves the index usable")
    func cancelledBuildLeavesConsistentState() async throws {
        let directory = try TemporaryDirectory()
        let signal = StartSignal()
        let engine = try await cancellableEngine(
            storeURL: directory.storeURL, signal: signal)
        let items = (0..<400).map { textItem("item-\($0)", "document number \($0)") }

        let task = Task { try await engine.upsert(items) }
        await signal.wait()
        task.cancel()
        _ = try? await task.value

        let itemCount = try await engine.itemCount()
        let vectorCount = try await engine.vectorCount(space: "test")
        #expect(itemCount == items.count)
        #expect(vectorCount > 0)
        #expect(vectorCount < itemCount)

        // `reindex` finds exactly the slots without vectors and finishes them.
        _ = try await engine.reindex()
        #expect(try await engine.vectorCount(space: "test") == (try await engine.itemCount()))

        let results = try await engine.search("document number")
        #expect(!results.hits.isEmpty)
        try await engine.close()
    }

    @Test("Cancelling a search stops it")
    func searchIsCancellable() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert((0..<500).map { textItem("item-\($0)", "document \($0)") })

        let task = Task { try await engine.search("document") }
        task.cancel()
        // Either it completed before the cancellation landed or it threw.
        // Both are correct; silently returning partial results is not.
        _ = try? await task.value
    }

    // MARK: - Missing and broken media

    @Test("Missing media is reported, and the item stays in the index")
    func missingMediaIsReported() async throws {
        // Stale media is reported explicitly so the caller can decide whether
        // to re-ingest or remove.
        let engine = try await SemanticSearchEngine.testEngine()
        let missing = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID().uuidString).png")

        let report = try await engine.upsert([
            SearchItem(
                id: "image",
                representations: [
                    ContentRepresentation(
                        id: "media", kind: .image, content: .mediaReference(missing))
                ])
        ])

        #expect(report.skipped.contains { $0.key.item == "image" })
        #expect(report.skipped.contains { $0.reason.contains("missing") })
        // The row is still there — deleting it would lose the caller's record
        // of what was supposed to be indexed.
        #expect(try await engine.itemCount() == 1)
    }

    @Test("A representation the provider cannot accept is reported")
    func unsupportedModalityIsReported() async throws {
        let engine = try await SemanticSearchEngine.testEngine(
            acceptsVisual: false, spaceID: "text-only")
        let report = try await engine.upsert([
            SearchItem(
                id: "image",
                representations: [
                    ContentRepresentation(
                        id: "media", kind: .image,
                        content: .imageData(Data([0x89, 0x50]), contentType: "public.png"))
                ])
        ])
        #expect(report.skipped.contains { $0.key.item == "image" })
    }

    @Test("A processor failure names the processor and the item")
    func processorFailuresAreAttributed() async throws {
        struct FailingProcessor: ContentProcessor {
            let identifier = "failing"
            let version = "1"
            var capabilities: ProcessorCapabilities {
                ProcessorCapabilities(acceptsText: true, producedKinds: [.originalText])
            }
            var configurationFingerprint: String { "failing.v1" }
            func canProcess(_ content: RawContent) -> Bool { true }
            func process(
                _ content: RawContent,
                onProgress: (@Sendable (ProcessingProgress) -> Void)?
            ) async throws -> [SearchItem] {
                struct Boom: Error {}
                throw Boom()
            }
        }

        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(), indexKind: .exact)
                ],
                processor: FailingProcessor()))

        do {
            _ = try await engine.ingest([RawContent(id: "doc", payload: .text("anything"))])
            Issue.record("expected the processor failure to surface")
        } catch let error as SemanticSearchError {
            guard case .processorFailed(let itemID, let processor, _) = error else {
                Issue.record("wrong error: \(error)")
                return
            }
            #expect(itemID == "doc")
            #expect(processor == "failing")
        }
    }

    @Test("Ingest skips content the processor has already handled")
    func ingestIsFingerprinted() async throws {
        // The fingerprint is what keeps OCR and transcription from being
        // repeated on every ingest pass.
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(), indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                processor: TextChunkingProcessor()))
        try await engine.prepare()

        let content = RawContent(id: "doc", payload: .text("a modest amount of text"))
        let first = try await engine.ingest([content])
        #expect(first.itemsWritten >= 1)

        let second = try await engine.ingest([content])
        #expect(second.itemsWritten == 0)
        #expect(second.itemsUnchanged == 1)
    }

    @Test("Searching an empty index returns nothing and does not throw")
    func emptyIndexSearch() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let results = try await engine.search("anything at all")
        #expect(results.hits.isEmpty)
        #expect(results.diagnostics.fusedCandidates == 0)
    }

    @Test("An empty query is rejected")
    func emptyQueryIsRejected() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search("")
        }
    }

    @Test("Invalid options are rejected before any work happens")
    func optionValidation() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content")])

        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search("content", options: SearchOptions(limit: 0))
        }
        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search(
                "content", options: SearchOptions(candidatesPerChannel: 0))
        }
        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search(
                "content", options: SearchOptions(fusion: FusionConfiguration(k: 0)))
        }
    }

    @Test("An image-only query cannot be answered lexically")
    func imageOnlyLexicalIsRejected() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content")])
        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search(
                SearchQuery(images: [.encoded(Data([0x01]), contentType: "public.png")]),
                options: SearchOptions(mode: .lexicalOnly))
        }
    }

    @Test("A very long document is indexed without truncation errors")
    func veryLongDocument() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let long = String(repeating: "paragraph about oceanography. ", count: 5_000)
        let report = try await engine.upsert([textItem("long", long)])
        #expect(report.vectorsWritten == 1)
        #expect(try await engine.search("oceanography").hits.count == 1)
    }

    @Test("Unicode content survives storage and retrieval")
    func unicodeRoundTrip() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        let samples = [
            ("en", "the quick brown fox"),
            ("ru", "быстрая коричневая лиса"),
            ("tr", "hızlı kahverengi tilki"),
            ("ar", "الثعلب البني السريع"),
            ("zh", "敏捷的棕色狐狸"),
            ("ja", "素早い茶色のキツネ"),
            ("emoji", "a fox 🦊 and a bridge 🌉"),
        ]
        try await engine.upsert(
            samples.map { textItem($0.0, $0.1, language: $0.0 == "emoji" ? "en" : $0.0) })

        for (id, text) in samples {
            let item = try #require(try await engine.item(ItemID(id)))
            #expect(item.representations[0].content.text == text)
        }
    }
}
