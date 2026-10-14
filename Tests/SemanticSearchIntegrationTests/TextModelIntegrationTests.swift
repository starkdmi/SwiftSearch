import Foundation
import SemanticSearch
import SemanticSearchMLX
import XCTest

/// The text reference profile: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B.
final class TextModelIntegrationTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            Integration.runsModelTests,
            "set SEMANTIC_SEARCH_RUN_INTEGRATION_TESTS=1 to run model-backed tests")
    }

    private func makeProvider() -> MLXEmbeddingProvider {
        ModelProfile.Text.embeddingProvider(model: Integration.textEmbeddingModel)
    }

    private func makeEngine(
        storeURL: URL?, indexKind: VectorIndexKind = .hnsw, withReranker: Bool = true
    ) async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text",
                        provider: makeProvider(),
                        kinds: [
                            .originalText, .ocrText, .transcript, .caption, .summary,
                            .translation,
                        ],
                        indexKind: indexKind,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                reranker: withReranker
                    ? ModelProfile.Text.rerankingProvider(model: Integration.textRerankerModel)
                    : nil))
        try await engine.prepare()
        return engine
    }

    // MARK: - Embedding behaviour

    /// A vector must not depend on what else shared its batch.
    ///
    /// This is the test that catches the padding bug: Qwen3's PAD is its EOS,
    /// so a mask built from token identity masks the real EOS and last-token
    /// pooling reads a padding position instead. The vectors still look
    /// plausible; only this comparison shows they are wrong.
    func testBatchedAndSingleEmbeddingsAgree() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let texts = [
            "short",
            "a considerably longer passage about the migration of arctic terns across two "
                + "hemispheres, with enough words to pad every other row in the batch",
            "medium length text about hydrothermal vents",
        ]

        var singles: [[Float]] = []
        for text in texts {
            singles.append(try await provider.embed(EmbeddingInput(text: text)).vector)
        }
        let batched = try await provider.embed(texts.map { EmbeddingInput(text: $0) })

        XCTAssertEqual(batched.count, texts.count)
        for (index, text) in texts.enumerated() {
            let similarity = VectorMath.dot(singles[index], batched[index].vector)
            XCTAssertGreaterThan(
                similarity, 0.999,
                "batched and single embeddings diverged for \(text.prefix(24))…: \(similarity)")
        }
    }

    /// Vectors are unit length, so an inner product is a cosine.
    func testEmbeddingsAreNormalized() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let output = try await provider.embed(EmbeddingInput(text: "normalisation check"))
        XCTAssertEqual(VectorMath.l2Norm(output.vector), 1, accuracy: 1e-3)
        XCTAssertEqual(output.vector.count, ModelProfile.Text.dimensions)
    }

    /// The query prompt has to differ from the document prompt, or the
    /// asymmetric retrieval the model was trained for is not happening.
    func testQueryAndDocumentPromptsDiffer() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let text = "arctic tern migration"
        let asDocument = try await provider.embed(
            EmbeddingInput(text: text, role: .document)).vector
        let asQuery = try await provider.embed(
            EmbeddingInput(text: text, role: .query(instruction: nil))).vector

        XCTAssertLessThan(
            VectorMath.dot(asDocument, asQuery), 0.9999,
            "the query instruction is not reaching the model")
    }

    /// Matryoshka prefixes are supported and renormalised.
    func testSupportedDimensionReduction() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let capabilities = await provider.capabilities
        XCTAssertTrue(capabilities.supportsDimensionReduction)

        let full = try await provider.embed(EmbeddingInput(text: "dimension reduction")).vector
        let reduced = try await provider.embed(
            EmbeddingInput(text: "dimension reduction", requestedDimensions: 256)).vector

        XCTAssertEqual(reduced.count, 256)
        XCTAssertEqual(VectorMath.l2Norm(reduced), 1, accuracy: 1e-3)

        // The prefix must be the prefix, up to renormalisation — otherwise it
        // is a different space wearing the same name.
        let renormalizedPrefix = VectorMath.normalizedPrefix(full, dimensions: 256)
        XCTAssertGreaterThan(VectorMath.dot(reduced, renormalizedPrefix), 0.999)
    }

    /// An unadvertised width is rejected rather than silently truncated.
    func testUnsupportedDimensionIsIgnoredRatherThanTruncated() async throws {
        let provider = MLXEmbeddingProvider(
            configuration: .init(
                model: Integration.textEmbeddingModel,
                dimensions: ModelProfile.Text.dimensions,
                matryoshkaDimensions: []))
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let output = try await provider.embed(
            EmbeddingInput(text: "no matryoshka here", requestedDimensions: 128))
        XCTAssertEqual(
            output.vector.count, ModelProfile.Text.dimensions,
            "a provider without Matryoshka support must not truncate")
    }

    /// Long input is truncated and the truncation is reported, not swallowed.
    func testTruncationIsReported() async throws {
        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        let long = String(repeating: "a sentence about oceanography. ", count: 400)
        let output = try await provider.embed(EmbeddingInput(text: long))
        XCTAssertTrue(output.wasTruncated, "a 2000-token input must report truncation")
        XCTAssertFalse(output.vector.isEmpty)
    }

    // MARK: - Reranking

    func testRerankerScoresRelevanceHigher() async throws {
        let reranker = ModelProfile.Text.rerankingProvider(model: Integration.textRerankerModel)
        try await reranker.prepare()
        defer { Task { await reranker.release() } }

        let query = RerankQuery(
            text: "How far do arctic terns migrate?",
            instruction: EngineConfiguration.defaultRerankInstruction)
        let scores = try await reranker.score(
            query: query,
            candidates: [
                RerankCandidate(
                    text:
                        "Arctic terns fly from the Arctic to the Antarctic and back each year, "
                        + "covering about ninety thousand kilometres."),
                RerankCandidate(
                    text:
                        "The printing press used movable metal type and an oil based ink to "
                        + "reproduce text at scale."),
            ])

        XCTAssertEqual(scores.count, 2)
        XCTAssertGreaterThan(scores[0].score, scores[1].score)
    }

    func testRerankerScoresAreNotAdvertisedAsCalibrated() async throws {
        let reranker = ModelProfile.Text.rerankingProvider(model: Integration.textRerankerModel)
        let capabilities = await reranker.capabilities
        XCTAssertFalse(
            capabilities.producesCalibratedScores,
            "no threshold on these scores means anything without a calibration set")
    }

    /// Batched and single reranking agree, for the same padding reason as the
    /// embedder.
    func testBatchedAndSingleRerankingAgree() async throws {
        let reranker = MLXRerankingProvider(
            configuration: .init(model: Integration.textRerankerModel, batchSize: 4))
        try await reranker.prepare()
        defer { Task { await reranker.release() } }

        let query = RerankQuery(
            text: "hydrothermal vents", instruction: EngineConfiguration.defaultRerankInstruction)
        let candidates = [
            RerankCandidate(text: "Short."),
            RerankCandidate(
                text:
                    "Hydrothermal vents host ecosystems that depend on chemosynthesis rather "
                    + "than sunlight, with tube worms and crabs clustering around the plumes "
                    + "in complete darkness."),
            RerankCandidate(text: "Kelp forests shelter sea otters."),
        ]

        let batched = try await reranker.score(query: query, candidates: candidates)
        var singles: [Float] = []
        for candidate in candidates {
            singles.append(
                try await reranker.score(query: query, candidates: [candidate])[0].score)
        }

        for index in candidates.indices {
            XCTAssertEqual(
                batched[index].score, singles[index], accuracy: 0.02,
                "batched reranking diverged at \(index): the batch's padding is being scored")
        }
    }

    /// The fast path (shared prefix, float32 score head at the last
    /// position) against mlx-swift-lm's Qwen3 scoring the same pairs.
    func testFastRerankingMatchesStock() async throws {
        let corpus = try Integration.multilingualCorpus()
        let candidates = corpus.documents.prefix(10).map {
            RerankCandidate(text: $0.text, kind: $0.kind)
        }
        let queries = corpus.queries.prefix(5).map {
            RerankQuery(text: $0.text, instruction: EngineConfiguration.defaultRerankInstruction)
        }
        func run(_ scoring: MLXRerankingProvider.Scoring) async throws -> ([[Float]], Double) {
            let reranker = ModelProfile.Text.rerankingProvider(
                model: Integration.textRerankerModel, scoring: scoring)
            try await reranker.prepare()
            _ = try await reranker.score(query: queries[0], candidates: Array(candidates.prefix(2)))
            var scores: [[Float]] = []
            let started = ContinuousClock.now
            for query in queries {
                scores.append(try await reranker.score(query: query, candidates: candidates).map(\.score))
            }
            let seconds = Integration.milliseconds(ContinuousClock.now - started) / 1000
            await reranker.release()
            return (scores, seconds)
        }
        let stock = try await run(.stock)
        let fast = try await run(.fast)
        print(String(format: "[rerank.fast.text] %d queries × %d candidates: stock %.2f s, fast %.2f s",
                     queries.count, candidates.count, stock.1, fast.1))
        for q in queries.indices {
            for i in candidates.indices {
                XCTAssertEqual(stock.0[q][i], fast.0[q][i], accuracy: 0.03, "query \(q) #\(i)")
            }
        }
    }

    // MARK: - End to end

    func testMultilingualCorpusRetrieval() async throws {
        let corpus = try Integration.multilingualCorpus()
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"))

        let indexStarted = ContinuousClock.now
        let runner = EvaluationRunner(corpus: corpus)
        let indexReport = try await runner.index(into: engine)
        let indexElapsed = ContinuousClock.now - indexStarted

        let coldStarted = ContinuousClock.now
        _ = try await engine.search("arctic tern migration")
        let cold = ContinuousClock.now - coldStarted

        let (report, identity) = try await runner.run(against: engine, label: "text-profile")

        Integration.measure(
            "text-profile",
            [
                "documents": "\(indexReport.itemsWritten)",
                "vectors": "\(indexReport.vectorsWritten)",
                "index_ms": String(format: "%.0f", Integration.milliseconds(indexElapsed)),
                "cold_ms": String(format: "%.0f", Integration.milliseconds(cold)),
                "warm_median_ms": String(format: "%.0f", report.medianLatencyMilliseconds),
                "warm_p95_ms": String(format: "%.0f", report.p95LatencyMilliseconds),
                "index_bytes": "\(await engine.indexSizeBytes())",
                "peak_mb": String(format: "%.0f", Integration.peakFootprintMegabytes()),
                "ndcg10": String(format: "%.4f", report.overall.ndcg10),
                "mrr": String(format: "%.4f", report.overall.mrr),
            ])

        for slice in report.slices {
            print(
                "[measure] slice name=\(slice.name) queries=\(slice.queryCount) "
                    + String(format: "ndcg10=%.4f mrr=%.4f", slice.ndcg10, slice.mrr))
        }

        XCTAssertGreaterThan(report.overall.ndcg10, 0.5)

        // Cross-language is the slice a real multilingual space is bought for:
        // an English query reaching the Japanese and Arabic documents on the
        // same topic, which no lexical channel can do.
        if let cross = report.slices.first(where: { $0.name == "cross-language" }) {
            XCTAssertGreaterThan(
                cross.ndcg10, 0.5,
                "cross-language retrieval is the reason for a multilingual space")
        }

        if let baselineURL = Integration.baselineURL,
            FileManager.default.fileExists(atPath: baselineURL.path)
        {
            let baseline = try EvaluationBaseline.load(from: baselineURL)
            if identity.isComparable(with: baseline.identity) {
                let gate = report.gate(against: baseline.report)
                XCTAssertTrue(gate.passed, gate.failures.joined(separator: "\n"))
            } else {
                print("[measure] baseline is not comparable; gate skipped")
            }
        }

        try await engine.close()
    }

    /// ANN recall against exact search, gated at 99%, on real vectors.
    func testAnnRecallAgainstExactSearch() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let engine = try await makeEngine(
            storeURL: directory.appendingPathComponent("index.sqlite"), withReranker: false)

        // Enough documents for the graph to be doing real work.
        var items: [SearchItem] = []
        for index in 0..<3_000 {
            items.append(
                SearchItem(
                    id: ItemID("doc-\(index)"),
                    representations: [
                        .text(
                            "Document \(index) discusses topic \(index % 137) and its effect "
                                + "on subject \(index % 53) in some detail.")
                    ]))
        }
        try await engine.upsert(items)
        try await engine.rebuildVectorIndex(space: "text")

        let provider = makeProvider()
        try await provider.prepare()
        defer { Task { await provider.release() } }

        var vectors: [[Float]] = []
        for index in 0..<25 {
            vectors.append(
                try await provider.embed(
                    EmbeddingInput(
                        text: "what is known about topic \(index * 5)?",
                        role: .query(instruction: nil))
                ).vector)
        }

        let recall = try await EvaluationRunner(
            corpus: EvaluationCorpus(name: "recall", documents: [], queries: [])
        ).annRecall(against: engine, space: "text", queryVectors: vectors, depth: 100)

        Integration.measure("ann-recall", ["recall_at_100": String(format: "%.4f", recall)])
        XCTAssertGreaterThanOrEqual(recall, 0.99, "the ANN gate is recall@100 ≥ 0.99")

        try await engine.close()
    }

    /// A query provider that does not match the index is rejected rather than
    /// answered with meaningless similarities.
    func testModelMismatchIsRejected() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let storeURL = directory.appendingPathComponent("index.sqlite")

        let engine = try await makeEngine(storeURL: storeURL, withReranker: false)
        try await engine.upsert([SearchItem(id: "a", representations: [.text("content")])])
        try await engine.close()

        do {
            _ = try await SemanticSearchEngine(
                configuration: EngineConfiguration(
                    storeURL: storeURL,
                    spaces: [
                        EmbeddingSpaceConfiguration(
                            id: "text",
                            provider: MLXEmbeddingProvider(
                                configuration: .init(
                                    model: Integration.textEmbeddingModel,
                                    documentInstruction: "An instruction the index was not "
                                        + "built with",
                                    dimensions: ModelProfile.Text.dimensions)),
                            indexKind: .exact)
                    ]))
            XCTFail("a different document instruction must not be accepted for this space")
        } catch let error as SemanticSearchError {
            guard case .incompatibleEmbeddingSpace = error else {
                return XCTFail("wrong error: \(error)")
            }
        }
    }

    /// Reopening with only the reranker changed must not rebuild anything.
    func testRerankerReplacementDoesNotRebuildEmbeddings() async throws {
        let directory = try Integration.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let storeURL = directory.appendingPathComponent("index.sqlite")

        do {
            let engine = try await makeEngine(storeURL: storeURL)
            try await engine.upsert(
                (0..<20).map {
                    SearchItem(
                        id: ItemID("doc-\($0)"),
                        representations: [.text("document number \($0)")])
                })
            try await engine.close()
        }

        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                storeURL: storeURL,
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "text", provider: makeProvider(),
                        kinds: [
                            .originalText, .ocrText, .transcript, .caption, .summary,
                            .translation,
                        ],
                        indexKind: .hnsw,
                        hnsw: HNSWConfiguration(minimumVectorsForGraph: 0))
                ],
                reranker: MLXRerankingProvider(
                    configuration: .init(
                        model: Integration.textRerankerModel, maximumPairTokens: 512))))

        let reopenedCount = try await engine.vectorCount(space: "text")
        XCTAssertEqual(reopenedCount, 20)
        let report = try await engine.reindex()
        XCTAssertEqual(report.vectorsWritten, 0)
        try await engine.close()
    }
}
