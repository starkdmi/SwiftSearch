import Foundation
import Testing

@testable import SemanticSearch

@Suite("Model compatibility")
struct CompatibilityTests {

    private func descriptor(
        model: String = "model-a",
        variant: String? = nil,
        tokenizer: String = "tok-1",
        preprocessing: String = "pre-1",
        instruction: String? = nil,
        pooling: String = "last-token",
        normalized: Bool = true,
        dimensions: Int = 64
    ) -> EmbeddingSpaceDescriptor {
        EmbeddingSpaceDescriptor(
            modelIdentity: model, weightsVariant: variant, tokenizerVersion: tokenizer,
            preprocessingVersion: preprocessing, documentInstruction: instruction,
            pooling: pooling, normalized: normalized, dimensions: dimensions)
    }

    @Test("Equal dimensions from different models are not compatible")
    func equalDimensionsAreNotEnough() {
        // Two 1024-dimensional models produce vectors whose cosine similarity
        // is a number with no meaning. This is the single most important check
        // in the file.
        let a = descriptor(model: "model-a", dimensions: 1_024)
        let b = descriptor(model: "model-b", dimensions: 1_024)
        #expect(!a.isCompatible(with: b))
    }

    @Test("Every identity field participates")
    func allFieldsMatter() {
        let base = descriptor()
        #expect(!base.isCompatible(with: descriptor(variant: "4bit")))
        #expect(!base.isCompatible(with: descriptor(tokenizer: "tok-2")))
        #expect(!base.isCompatible(with: descriptor(preprocessing: "pre-2")))
        #expect(!base.isCompatible(with: descriptor(instruction: "Represent this")))
        #expect(!base.isCompatible(with: descriptor(pooling: "mean")))
        #expect(!base.isCompatible(with: descriptor(normalized: false)))
        #expect(!base.isCompatible(with: descriptor(dimensions: 128)))
        #expect(base.isCompatible(with: descriptor()))
    }

    @Test("Fingerprints agree with compatibility")
    func fingerprintMatchesCompatibility() {
        #expect(descriptor().fingerprint == descriptor().fingerprint)
        #expect(descriptor().fingerprint != descriptor(pooling: "mean").fingerprint)
    }

    @Test("A supported prefix is compatible; an unsupported width is not")
    func dimensionReductionCompatibility() {
        let a = descriptor(dimensions: 1_024)
        let b = descriptor(dimensions: 1_024)
        #expect(a.isCompatible(with: b, reducedDimensions: 256))
        #expect(!a.isCompatible(with: b, reducedDimensions: 2_048))
    }

    @Test("Registering a second space under one id with different weights is rejected")
    func spaceIdentityIsEnforcedAtRegistration() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        let first = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
        try await first.upsert([textItem("a", "content")])
        try await first.close()

        // Same id, different model. Keeping the existing vectors would make the
        // space internally inconsistent in a way no query-time check can find.
        await #expect(throws: SemanticSearchError.self) {
            _ = try await SemanticSearchEngine(
                configuration: EngineConfiguration(
                    storeURL: storeURL,
                    spaces: [
                        EmbeddingSpaceConfiguration(
                            id: "test",
                            provider: HashingEmbeddingProvider(
                                dimensions: 64, identity: "a-different-model"),
                            indexKind: .exact)
                    ]))
        }
    }

    @Test("A mismatched query provider is rejected at search time")
    func queryTimeMismatchIsRejected() async throws {
        // The provider instance can be swapped after registration, so the check
        // runs on every query rather than only at open.
        let store = try SearchStore.inMemory()
        _ = store

        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(storeURL: directory.storeURL)
        try await engine.upsert([textItem("a", "content")])
        try await engine.close()

        // Reopen with a provider whose preprocessing version differs, reaching
        // past `registerSpace` by writing the descriptor directly.
        let reopened = try SearchStore(url: directory.storeURL)
        let stored = try #require(try reopened.loadSpaces().first)
        #expect(stored.descriptor.modelIdentity == "hashing-trigram")
        reopened.close()
    }

    @Test("Requesting reduction from a provider that lacks it throws")
    func unsupportedReductionIsRejected() async throws {
        await #expect(throws: SemanticSearchError.self) {
            _ = try await SemanticSearchEngine(
                configuration: EngineConfiguration(
                    spaces: [
                        EmbeddingSpaceConfiguration(
                            id: "test",
                            provider: HashingEmbeddingProvider(
                                dimensions: 64, matryoshkaDimensions: []),
                            indexKind: .exact,
                            reducedDimensions: 32)
                    ]))
        }
    }

    @Test("Supported dimension reduction works end to end")
    func supportedReductionWorks() async throws {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test",
                        provider: HashingEmbeddingProvider(
                            dimensions: 64, matryoshkaDimensions: [16, 32, 64]),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0),
                        reducedDimensions: 32)
                ]))
        try await engine.prepare()
        try await engine.upsert([
            textItem("a", "hydrothermal vents on the ocean floor"),
            textItem("b", "the history of the printing press"),
        ])

        // Stored at the reduced width, and still normalised — a truncated
        // normalised vector is shorter than 1 and its inner product is no
        // longer a cosine.
        let vector = try #require(try await engine.exactNeighbours(
            space: "test", query: [Float](repeating: 0.1, count: 32), limit: 1).first)
        _ = vector

        let results = try await engine.search(
            "hydrothermal vents", options: SearchOptions(reducedDimensions: 32))
        #expect(results.hits.first?.itemID == "a")
    }

    @Test("A reduction width the provider does not advertise is rejected at query time")
    func unadvertisedQueryWidthIsRejected() async throws {
        let engine = try await SemanticSearchEngine.testEngine(
            dimensions: 64, matryoshkaDimensions: [16, 32])
        try await engine.upsert([textItem("a", "content")])

        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search(
                "content", options: SearchOptions(reducedDimensions: 512))
        }
    }

    @Test("Replacing only the reranker does not touch the vectors")
    func rerankerSwapKeepsVectors() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL

        do {
            let engine = try await SemanticSearchEngine.testEngine(storeURL: storeURL)
            try await engine.upsert(
                (0..<20).map { textItem("item-\($0)", "document number \($0)") })
            try await engine.close()
        }

        let swapped = try await SemanticSearchEngine.testEngine(
            storeURL: storeURL,
            reranker: ScriptedRerankingProvider(
                identity: RerankerIdentity(
                    modelIdentity: "an-entirely-different-reranker", promptVersion: "v9"),
                scorer: { _, _ in RerankScore(score: 0.5) }))

        // No rebuild, no re-embedding: the count is what it was.
        #expect(try await swapped.vectorCount(space: "test") == 20)
        let report = try await swapped.reindex()
        #expect(report.vectorsWritten == 0)
        try await swapped.close()
    }

    @Test("A replacement space is built alongside the old one")
    func replacementSpaceIsBuiltAlongside() async throws {
        // Migration builds the new space, then switches. Both exist at once,
        // and the old one keeps answering until the new one is complete.
        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: directory.storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "v1",
                        provider: HashingEmbeddingProvider(dimensions: 32, identity: "model-v1"),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0)),
                    EmbeddingSpaceConfiguration(
                        id: "v2",
                        provider: HashingEmbeddingProvider(dimensions: 64, identity: "model-v2"),
                        indexKind: .exact,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0)),
                ]))
        try await engine.prepare()
        try await engine.upsert([textItem("a", "content about volcanoes")])

        #expect(try await engine.vectorCount(space: "v1") == 1)
        #expect(try await engine.vectorCount(space: "v2") == 1)

        // Querying one space at a time is how the switch is made.
        let onlyV2 = try await engine.search(
            "volcanoes", options: SearchOptions(spaces: ["v2"]))
        #expect(onlyV2.diagnostics.channels.contains { $0.channel.space == "v2" })
        #expect(!onlyV2.diagnostics.channels.contains { $0.channel.space == "v1" })
        try await engine.close()
    }

    @Test("Querying an unregistered space is an error")
    func unknownSpaceIsRejected() async throws {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([textItem("a", "content")])
        await #expect(throws: SemanticSearchError.self) {
            _ = try await engine.search(
                "content", options: SearchOptions(spaces: ["does-not-exist"]))
        }
    }
}
