import Foundation
import Testing

@testable import SemanticSearch

@Suite("Evaluation")
struct EvaluationTests {

    private func query(
        _ id: String, relevance: [String: Int], slice: String = "default"
    ) -> EvaluationQuery {
        EvaluationQuery(id: id, text: id, slice: slice, relevance: relevance)
    }

    // MARK: - Metrics

    @Test("A perfect ranking scores 1")
    func perfectNdcg() {
        let q = query("q", relevance: ["a": 3, "b": 2, "c": 1])
        let score = RetrievalMetrics.ndcg(ranked: ["a", "b", "c"], query: q, k: 10)
        #expect(abs(score - 1) < 1e-9)
    }

    @Test("A reversed ranking scores less than a perfect one")
    func reversedNdcg() {
        let q = query("q", relevance: ["a": 3, "b": 2, "c": 1])
        let perfect = RetrievalMetrics.ndcg(ranked: ["a", "b", "c"], query: q, k: 10)
        let reversed = RetrievalMetrics.ndcg(ranked: ["c", "b", "a"], query: q, k: 10)
        #expect(reversed < perfect)
        #expect(reversed > 0)
    }

    @Test("nDCG is graded, not binary")
    func gradedRelevance() {
        // One highly relevant answer must not score the same as several
        // marginal ones.
        let q = query("q", relevance: ["a": 3, "b": 1, "c": 1])
        let best = RetrievalMetrics.ndcg(ranked: ["a", "b", "c"], query: q, k: 10)
        let worse = RetrievalMetrics.ndcg(ranked: ["b", "c", "a"], query: q, k: 10)
        #expect(best > worse)
    }

    @Test("Results past k do not count")
    func cutoffIsRespected() {
        let q = query("q", relevance: ["target": 3])
        let inside = RetrievalMetrics.ndcg(ranked: ["target", "x", "y"], query: q, k: 3)
        let outside = RetrievalMetrics.ndcg(ranked: ["x", "y", "target"], query: q, k: 2)
        #expect(inside > 0)
        #expect(outside == 0)
    }

    @Test("A query with no relevant documents contributes nothing")
    func unlabelledQueriesAreSkipped() {
        // Scoring it 0 or 1 would move an average for a reason that has nothing
        // to do with the system.
        let q = query("q", relevance: [:])
        #expect(RetrievalMetrics.ndcg(ranked: ["a"], query: q, k: 10).isNaN)
        #expect(RetrievalMetrics.mean([1.0, .nan, 0.5]) == 0.75)
    }

    @Test("Reciprocal rank finds the first relevant result")
    func reciprocalRank() {
        let q = query("q", relevance: ["target": 1])
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["target", "a"], query: q) == 1)
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["a", "target"], query: q) == 0.5)
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["a", "b"], query: q) == 0)
    }

    @Test("ANN recall compares two candidate lists")
    func annRecallBySlot() {
        #expect(RetrievalMetrics.recall(approximate: [1, 2, 3], exact: [1, 2, 3], k: 3) == 1)
        #expect(RetrievalMetrics.recall(approximate: [1, 2, 9], exact: [1, 2, 3], k: 3) > 0.66)
        #expect(RetrievalMetrics.recall(approximate: [7, 8, 9], exact: [1, 2, 3], k: 3) == 0)
    }

    @Test("Tie-aware ANN recall credits an equally-close neighbour")
    func annRecallHandlesTies() {
        // Slot 9 is not in the exact top three, but it sits at exactly the same
        // distance as the third. Every implementation picks a different subset
        // of a tied block and every one of them is right.
        let exact = [
            VectorMatch(slot: 1, score: 0.9), VectorMatch(slot: 2, score: 0.8),
            VectorMatch(slot: 3, score: 0.7),
        ]
        let approximate = [
            VectorMatch(slot: 1, score: 0.9), VectorMatch(slot: 2, score: 0.8),
            VectorMatch(slot: 9, score: 0.7),
        ]
        #expect(RetrievalMetrics.annRecall(approximate: approximate, exact: exact, k: 3) == 1)
    }

    @Test("Tie-aware ANN recall still catches a genuine miss")
    func annRecallCatchesRealMisses() {
        let exact = [
            VectorMatch(slot: 1, score: 0.9), VectorMatch(slot: 2, score: 0.8),
            VectorMatch(slot: 3, score: 0.7),
        ]
        let approximate = [
            VectorMatch(slot: 1, score: 0.9), VectorMatch(slot: 2, score: 0.8),
            VectorMatch(slot: 9, score: 0.2),
        ]
        let recall = RetrievalMetrics.annRecall(approximate: approximate, exact: exact, k: 3)
        #expect(abs(recall - 2.0 / 3.0) < 1e-9)
    }

    // MARK: - Reports and gates

    @Test("A report is built per slice as well as overall")
    func reportSlices() {
        let results: [(query: EvaluationQuery, ranked: [ItemID], duration: Duration)] = [
            (query("q1", relevance: ["a": 3], slice: "en"), ["a"], .milliseconds(10)),
            (query("q2", relevance: ["b": 3], slice: "en"), ["x", "b"], .milliseconds(20)),
            (query("q3", relevance: ["c": 3], slice: "ru"), ["c"], .milliseconds(30)),
        ]
        let report = EvaluationReport.make(label: "run", results: results)

        #expect(report.overall.queryCount == 3)
        #expect(report.slices.map(\.name) == ["en", "ru"])
        #expect(report.slices.first { $0.name == "ru" }?.ndcg10 == 1)
        #expect(report.medianLatencyMilliseconds > 0)
    }

    @Test("The gate passes when the run retains the baseline")
    func gatePasses() {
        let baseline = EvaluationReport.make(
            label: "baseline",
            results: [(query("q", relevance: ["a": 3]), ["a"], .milliseconds(1))])
        let current = EvaluationReport.make(
            label: "current",
            results: [(query("q", relevance: ["a": 3]), ["a"], .milliseconds(1))])
        #expect(current.gate(against: baseline).passed)
    }

    @Test("The gate fails a regression below 98%")
    func gateFailsRegression() {
        let baseline = EvaluationReport.make(
            label: "baseline",
            results: [(query("q", relevance: ["a": 3, "b": 1]), ["a", "b"], .milliseconds(1))])
        let regressed = EvaluationReport.make(
            label: "current",
            results: [(query("q", relevance: ["a": 3, "b": 1]), ["b", "a"], .milliseconds(1))])

        let gate = regressed.gate(against: baseline)
        #expect(!gate.passed)
        #expect(gate.failures.contains { $0.contains("ndcg@10") })
    }

    @Test("The gate fails per slice even when the overall average holds")
    func gateIsPerSlice() {
        // An average that holds while one language collapses is not a pass.
        let baseline = EvaluationReport.make(
            label: "baseline",
            results: [
                (query("q1", relevance: ["a": 3, "b": 1], slice: "en"), ["a", "b"], .milliseconds(1)),
                (query("q2", relevance: ["c": 3, "d": 1], slice: "ar"), ["c", "d"], .milliseconds(1)),
            ])
        let current = EvaluationReport.make(
            label: "current",
            results: [
                (query("q1", relevance: ["a": 3, "b": 1], slice: "en"), ["a", "b"], .milliseconds(1)),
                (query("q2", relevance: ["c": 3, "d": 1], slice: "ar"), ["d", "c"], .milliseconds(1)),
            ])

        let gate = current.gate(against: baseline)
        #expect(!gate.passed)
        #expect(gate.failures.contains { $0.contains("ar/") })
    }

    @Test("A slice dropped from the run is a failure, not a pass")
    func droppedSliceFails() {
        // Removing the Arabic queries is not a way to keep the Arabic numbers.
        let baseline = EvaluationReport.make(
            label: "baseline",
            results: [
                (query("q1", relevance: ["a": 3], slice: "en"), ["a"], .milliseconds(1)),
                (query("q2", relevance: ["c": 3], slice: "ar"), ["c"], .milliseconds(1)),
            ])
        let current = EvaluationReport.make(
            label: "current",
            results: [(query("q1", relevance: ["a": 3], slice: "en"), ["a"], .milliseconds(1))])

        let gate = current.gate(against: baseline)
        #expect(!gate.passed)
        #expect(gate.failures.contains { $0.contains("missing") })
    }

    // MARK: - Identity

    @Test("Identities that differ in a measured dimension are not comparable")
    func identityComparability() {
        // Benchmark comparisons are only meaningful on identical corpora and
        // candidate pools. This is what makes a mismatch visible.
        func identity(candidates: Int = 100, models: [String] = ["m"]) -> EvaluationIdentity {
            EvaluationIdentity(
                corpusName: "c", documentCount: 10, queryCount: 5, embeddingModels: models,
                rerankerModel: "r", analyzerVersion: "v1", bm25: .default, fusionK: 60,
                candidatesPerChannel: candidates, rerankPoolSize: 50, limit: 10,
                mode: "hybrid", indexKind: "hnsw")
        }

        #expect(identity().isComparable(with: identity()))
        #expect(!identity().isComparable(with: identity(candidates: 50)))
        #expect(!identity().isComparable(with: identity(models: ["other"])))
    }

    @Test("Index kind is excluded from comparability")
    func indexKindDoesNotBlockComparison() {
        // Exact versus HNSW is precisely the comparison the ANN gate makes.
        let exact = EvaluationIdentity(
            corpusName: "c", documentCount: 10, queryCount: 5, embeddingModels: ["m"],
            rerankerModel: nil, analyzerVersion: "v1", bm25: .default, fusionK: 60,
            candidatesPerChannel: 100, rerankPoolSize: 50, limit: 10, mode: "hybrid",
            indexKind: "exact")
        var graph = exact
        graph.indexKind = "hnsw"
        #expect(exact.isComparable(with: graph))
    }

    @Test("A baseline round-trips through JSON")
    func baselineRoundTrip() throws {
        let directory = try TemporaryDirectory()
        let file = directory.url.appendingPathComponent("baseline.json")

        let baseline = EvaluationBaseline(
            identity: EvaluationIdentity(
                corpusName: "c", documentCount: 3, queryCount: 2, embeddingModels: ["m"],
                rerankerModel: "r", analyzerVersion: "v1", bm25: .default, fusionK: 60,
                candidatesPerChannel: 100, rerankPoolSize: 50, limit: 10, mode: "hybrid",
                indexKind: "exact",
                spaces: [
                    .init(
                        id: "s", fingerprint: "f", dimensions: 256, maximumInputTokens: 512,
                        queryInstructions: ["text": "fact checking"])
                ],
                ignoresIdenticalIDs: true),
            report: EvaluationReport.make(
                label: "baseline",
                results: [(query("q", relevance: ["a": 3]), ["a"], .milliseconds(5))]))

        try baseline.write(to: file)
        let loaded = try EvaluationBaseline.load(from: file)
        #expect(loaded.identity == baseline.identity)
        #expect(abs(loaded.report.overall.ndcg10 - baseline.report.overall.ndcg10) < 1e-9)
    }

    @Test("A baseline recorded before the identity held its spaces still loads and gates")
    func legacyBaselineLoads() throws {
        let directory = try TemporaryDirectory()
        let file = directory.url.appendingPathComponent("legacy.json")
        // As `--output` wrote it before spaces, the identical-id rule and
        // degraded queries were recorded.
        let legacyJSON = #"""
            {
              "identity" : {
                "analyzerVersion" : "linguistic.v2",
                "bm25" : { "b" : 0.75, "k1" : 1.2 },
                "candidatesPerChannel" : 100,
                "corpusName" : "c",
                "documentCount" : 10,
                "embeddingModels" : [ "m" ],
                "fusionK" : 60,
                "indexKind" : "exact",
                "limit" : 10,
                "mode" : "denseOnly",
                "queryCount" : 5,
                "rerankPoolSize" : 50,
                "rerankerModel" : "r"
              },
              "recordedAt" : "2026-10-10T19:04:04Z",
              "report" : {
                "label" : "baseline",
                "medianLatencyMilliseconds" : 1,
                "overall" : {
                  "mrr" : 0.5, "name" : "overall", "ndcg10" : 0.6, "queryCount" : 5,
                  "recall10" : 0.9
                },
                "p95LatencyMilliseconds" : 2,
                "slices" : [ ]
              }
            }
            """#
        try Data(legacyJSON.utf8).write(to: file)

        let legacy = try EvaluationBaseline.load(from: file)
        #expect(legacy.identity.spaces == nil)
        #expect(legacy.identity.ignoresIdenticalIDs == nil)
        #expect(legacy.report.degradedQueries == nil)

        // What it did not record is not compared, as it was not before.
        var current = legacy.identity
        current.spaces = [
            .init(id: "s", fingerprint: "f", dimensions: 128, maximumInputTokens: 2_048)
        ]
        current.ignoresIdenticalIDs = true
        #expect(current.measurementDifferences(from: legacy.identity).isEmpty)
        #expect(current.isComparable(with: legacy.identity))
    }

    @Test("Another width, token cap, instruction or identical-id rule is another measurement")
    func measurementDifferencesAreFound() {
        func identity(
            dimensions: Int = 768, tokens: Int = 512, instruction: String? = "fact checking",
            ignoring: Bool = true, fingerprint: String = "f"
        ) -> EvaluationIdentity {
            let instructions = instruction.map { ["caption": $0, "text": $0] } ?? [:]
            return EvaluationIdentity(
                corpusName: "c", documentCount: 10, queryCount: 5, embeddingModels: ["m"],
                rerankerModel: nil, analyzerVersion: "v1", bm25: .default, fusionK: 60,
                candidatesPerChannel: 100, rerankPoolSize: 50, limit: 10, mode: "denseOnly",
                indexKind: "exact",
                spaces: [
                    .init(
                        id: "s", fingerprint: fingerprint, dimensions: dimensions,
                        maximumInputTokens: tokens, queryInstructions: instructions)
                ],
                ignoresIdenticalIDs: ignoring)
        }
        let base = identity()
        #expect(identity().measurementDifferences(from: base).isEmpty)
        #expect(identity().isComparable(with: base))

        let changed = [
            identity(dimensions: 128), identity(tokens: 2_048),
            identity(instruction: "search result"), identity(instruction: nil),
            identity(ignoring: false),
        ]
        for run in changed {
            #expect(run.measurementDifferences(from: base).count == 1)
            #expect(!run.isComparable(with: base))
        }
        // One instruction for every family is one change, naming them.
        #expect(
            identity(instruction: "search result").measurementDifferences(from: base) == [
                "space 's' query instruction (caption, text): \"fact checking\" → "
                    + "\"search result\""
            ])

        // Another conversion of the model is a change a gate run is made to
        // judge: reported, not refused.
        let converted = identity(fingerprint: "g")
        #expect(converted.measurementDifferences(from: base).isEmpty)
        #expect(!converted.isComparable(with: base))
    }

    @Test("The identity records a reopened space as stored, not as configured")
    func identityReadsTheStore() async throws {
        let directory = try TemporaryDirectory()
        let storeURL = directory.storeURL
        func engine(reducedDimensions: Int?, indexKind: VectorIndexKind) async throws
            -> SemanticSearchEngine
        {
            try await SemanticSearchEngine(
                configuration: EngineConfiguration(
                    storeURL: storeURL,
                    spaces: [
                        EmbeddingSpaceConfiguration(
                            id: "test",
                            provider: HashingEmbeddingProvider(
                                dimensions: 64, matryoshkaDimensions: [32]),
                            indexKind: indexKind,
                            hnsw: HNSWConfiguration(minimumVectorsForGraph: 0),
                            reducedDimensions: reducedDimensions)
                    ]))
        }
        let runner = EvaluationRunner(
            corpus: EvaluationCorpus(
                name: "c", documents: [.init(id: "a", text: "alpha")], queries: []))

        let created = try await engine(reducedDimensions: 32, indexKind: .hnsw)
        try await runner.index(into: created)
        let first = await runner.identity(of: created)
        try await created.close()

        // Asked for the full width and exact search, the reopened store keeps
        // the width and the graph it was created with — and says so.
        let reopened = try await engine(reducedDimensions: nil, indexKind: .exact)
        let second = await runner.identity(of: reopened)
        try await reopened.close()

        let fingerprint = await HashingEmbeddingProvider(dimensions: 64).descriptor.fingerprint
        let stored = EvaluationIdentity.Space(
            id: "test", fingerprint: fingerprint, dimensions: 32, maximumInputTokens: 512)
        #expect(first.spaces == [stored])
        #expect(second.spaces == [stored])
        #expect(first.indexKind == "hnsw")
        #expect(second.indexKind == "hnsw")
        #expect(second.ignoresIdenticalIDs == false)

        // An HNSW space below its graph threshold is searched exactly.
        let small = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(), indexKind: .hnsw)
                ]))
        try await runner.index(into: small)
        #expect(await runner.identity(of: small).indexKind == "exact")
        try await small.close()
    }

    @Test("The identity records the instruction of each family the space routes")
    func identityRecordsInstructions() async throws {
        // In the engine's order: the query's, the space's, the engine's. The
        // visual family is not routed to a text-only space, so its
        // instruction is never read.
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(
                        id: "test", provider: HashingEmbeddingProvider(),
                        kinds: [.originalText, .caption, .transcript],
                        queryInstructions: [.caption: "space", .visual: "never read"])
                ],
                queryInstructions: [.text: "engine", .caption: "engine"]))
        let runner = EvaluationRunner(
            corpus: EvaluationCorpus(name: "c", documents: [], queries: []),
            options: .init(search: SearchOptions(queryInstructions: [.transcript: "query"])))

        let identity = await runner.identity(of: engine)
        #expect(
            identity.spaces?.first?.queryInstructions
                == ["caption": "space", "text": "engine", "transcript": "query"])
        try await engine.close()
    }

    // MARK: - End to end

    @Test("The runner indexes a corpus and produces a report")
    func runnerEndToEnd() async throws {
        let corpus = EvaluationCorpus(
            name: "tiny",
            documents: [
                .init(id: "a", text: "hydrothermal vents on the ocean floor"),
                .init(id: "b", text: "the history of the printing press"),
                .init(id: "c", text: "arctic tern migration patterns"),
            ],
            queries: [
                EvaluationQuery(
                    id: "q1", text: "hydrothermal vents", slice: "en", relevance: ["a": 3]),
                EvaluationQuery(
                    id: "q2", text: "printing press history", slice: "en", relevance: ["b": 3]),
            ])

        let engine = try await SemanticSearchEngine.testEngine()
        let runner = EvaluationRunner(
            corpus: corpus, options: .init(search: .default, warmupQueries: 0))
        try await runner.index(into: engine)

        let (report, identity) = try await runner.run(against: engine, label: "test")
        #expect(report.overall.queryCount == 2)
        #expect(report.overall.ndcg10 > 0)
        #expect(identity.corpusName == "tiny")
        #expect(identity.embeddingModels == ["hashing-trigram"])
        #expect(identity.bm25 == .default)
        #expect(identity.fusionK == 60)
    }

    @Test("A corpus that ignores identical ids scores past the query's own copy")
    func identicalIDsAreDropped() async throws {
        // ArguAna's shape: the query is also a document, word for word, and
        // what it should find is another document. Scored as is, the copy
        // ranks first and the answer can never be top-1.
        func corpus(ignoring: Bool?) -> EvaluationCorpus {
            EvaluationCorpus(
                name: "identical",
                documents: [
                    .init(id: "claim", text: "tidal energy is predictable"),
                    .init(id: "counter", text: "tidal energy is predictable but costly"),
                    .init(id: "other", text: "arctic tern migration patterns"),
                ],
                queries: [
                    EvaluationQuery(
                        id: "claim", text: "tidal energy is predictable", slice: "en",
                        relevance: ["counter": 1])
                ],
                ignoresIdenticalIDs: ignoring)
        }

        var reciprocalRanks: [Bool?: Double] = [:]
        for ignoring in [nil, true] as [Bool?] {
            let engine = try await SemanticSearchEngine.testEngine()
            let runner = EvaluationRunner(
                corpus: corpus(ignoring: ignoring),
                options: .init(search: SearchOptions(limit: 1), warmupQueries: 0))
            try await runner.index(into: engine)
            let (report, _) = try await runner.run(against: engine, label: "test")
            reciprocalRanks[ignoring] = report.overall.mrr
            try await engine.close()
        }
        // At a limit of one the copy is all there is to score.
        #expect(reciprocalRanks[nil] == 0)
        // One hit more is fetched, so dropping the copy still leaves `limit`:
        // without it the one hit is the copy, dropped, and the answer is never
        // seen. At a limit of two the answer would rank first either way.
        #expect(reciprocalRanks[true] == 1)
    }

    @Test("A degraded query is counted once per reason, not once per channel family")
    func degradedQueriesAreCounted() async throws {
        // The provider returns no vector for text that mentions "void", and
        // the engine skips the space once for every text family it routes.
        let corpus = EvaluationCorpus(
            name: "degraded",
            documents: [
                .init(id: "a", text: "alpha words"),
                .init(id: "b", text: "beta words"),
            ],
            queries: [
                EvaluationQuery(id: "q1", text: "void alpha", relevance: ["a": 1]),
                EvaluationQuery(id: "q2", text: "void beta", relevance: ["b": 1]),
                EvaluationQuery(id: "q3", text: "alpha", relevance: ["a": 1]),
            ])
        let engine = try await SemanticSearchEngine(
            configuration: EngineConfiguration(
                spaces: [
                    EmbeddingSpaceConfiguration(id: "s", provider: VoidEmbeddingProvider())
                ]))
        let runner = EvaluationRunner(corpus: corpus, options: .init(warmupQueries: 0))
        try await runner.index(into: engine)

        let (report, _) = try await runner.run(against: engine, label: "test")
        #expect(
            report.degradedQueries == ["s skipped: provider returned an empty query vector": 2])
        try await engine.close()
    }

    @Test("The runner measures ANN recall against exact search")
    func runnerMeasuresAnnRecall() async throws {
        let corpus = EvaluationCorpus(
            name: "recall",
            documents: (0..<3_000).map {
                .init(id: "doc-\($0)", text: "document \($0) about subject \($0 % 311)")
            },
            queries: [])

        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(
            storeURL: directory.storeURL, indexKind: .hnsw)
        let runner = EvaluationRunner(corpus: corpus)
        try await runner.index(into: engine)
        try await engine.rebuildVectorIndex(space: "test")

        let provider = HashingEmbeddingProvider(dimensions: 64)
        try await provider.prepare()
        var queryVectors: [[Float]] = []
        for index in 0..<20 {
            let output = try await provider.embed(
                EmbeddingInput(text: "query about subject \(index)"))
            queryVectors.append(output.vector)
        }

        let recall = try await runner.annRecall(
            against: engine, space: "test", queryVectors: queryVectors, depth: 100)
        #expect(recall >= 0.99, "ANN recall@100 was \(recall), the gate is 0.99")
        try await engine.close()
    }
}
