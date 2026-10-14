import Foundation
import Testing

@testable import SemanticSearch

/// The portable multilingual fixture, exercised without a model.
///
/// With the hashing embedder these establish that the corpus loads, indexes and
/// retrieves in every language through the lexical channel. The same fixture is
/// run against the real models in the integration suite, where the
/// cross-language slice becomes meaningful.
@Suite("Fixtures")
struct FixtureTests {

    static func corpusURL() throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: "multilingual-corpus", withExtension: "json",
                subdirectory: "Fixtures"))
    }

    static func corpus() throws -> EvaluationCorpus {
        try EvaluationCorpus.load(from: try corpusURL())
    }

    @Test("The corpus loads and is well formed")
    func corpusLoads() throws {
        let corpus = try Self.corpus()
        #expect(corpus.name == "multilingual-v1")
        #expect(corpus.documents.count >= 19)
        #expect(corpus.queries.count >= 22)

        // Every query has to name at least one document that exists, or the
        // metrics measure nothing.
        let ids = Set(corpus.documents.map(\.id))
        for query in corpus.queries {
            #expect(!query.relevantItemIDs.isEmpty, "\(query.id) has no relevant documents")
            #expect(query.relevantItemIDs.isSubset(of: ids), "\(query.id) names a missing document")
        }
    }

    @Test("The corpus covers the six required languages")
    func languageCoverage() throws {
        let corpus = try Self.corpus()
        let languages = Set(corpus.documents.compactMap(\.language))
        for required in ["en", "ru", "tr", "ar", "zh", "ja"] {
            #expect(languages.contains(required), "no \(required) documents")
        }
    }

    @Test("The corpus has a cross-language and a single-character slice")
    func requiredSlices() throws {
        let corpus = try Self.corpus()
        let slices = Set(corpus.queries.map(\.slice))
        #expect(slices.contains("cross-language"))
        #expect(slices.contains("zh-single-char"))
    }

    @Test("Same-language queries retrieve their own documents")
    func inLanguageRetrieval() async throws {
        let corpus = try Self.corpus()
        let engine = try await SemanticSearchEngine.testEngine()
        let runner = EvaluationRunner(
            corpus: corpus,
            options: .init(search: SearchOptions(mode: .lexicalOnly), warmupQueries: 0))
        try await runner.index(into: engine)

        let (report, identity) = try await runner.run(against: engine, label: "lexical")

        // Lexical only, with a deterministic embedder: the in-language slices
        // should be near perfect, because the words are right there.
        for slice in report.slices where slice.name != "cross-language" {
            #expect(slice.mrr > 0.5, "\(slice.name) MRR was \(slice.mrr)")
        }
        #expect(identity.corpusName == "multilingual-v1")
        #expect(identity.mode == "lexicalOnly")
    }

    @Test("A single Chinese character retrieves through the lexical channel")
    func singleCharacterSlice() async throws {
        let corpus = try Self.corpus()
        let engine = try await SemanticSearchEngine.testEngine()
        _ = try await engine.upsert(corpus.items())

        let query = try #require(corpus.queries.first { $0.slice == "zh-single-char" })
        let results = try await engine.search(
            SearchQuery(text: query.text, languageHint: "zh"),
            options: SearchOptions(mode: .lexicalOnly, filter: corpus.filter(for: query)))

        let top = try #require(results.hits.first)
        #expect(query.relevantItemIDs.contains(top.itemID.rawValue))
    }

    @Test("Cross-language retrieval needs a real embedding space")
    func crossLanguageIsHonest() async throws {
        // Recorded rather than asserted away. The lexical channel cannot answer
        // a cross-language query and the hashing embedder cannot either, so
        // this slice is expected to be poor here and is the reason the
        // integration suite exists.
        let corpus = try Self.corpus()
        let engine = try await SemanticSearchEngine.testEngine()
        let runner = EvaluationRunner(
            corpus: corpus, options: .init(search: .default, warmupQueries: 0))
        try await runner.index(into: engine)

        let (report, _) = try await runner.run(against: engine, label: "hashing")
        let cross = try #require(report.slices.first { $0.name == "cross-language" })
        #expect(cross.queryCount == 3)
        print("[fixture] cross-language nDCG@10 without a real model: \(cross.ndcg10)")
    }

    @Test("A baseline recorded from this corpus gates a later run")
    func baselineGatesARun() async throws {
        let corpus = try Self.corpus()
        let directory = try TemporaryDirectory()
        let baselineURL = directory.url.appendingPathComponent("baseline.json")

        let engine = try await SemanticSearchEngine.testEngine()
        let runner = EvaluationRunner(
            corpus: corpus,
            options: .init(search: SearchOptions(mode: .lexicalOnly), warmupQueries: 0))
        try await runner.index(into: engine)

        let (report, identity) = try await runner.run(against: engine, label: "baseline")
        try EvaluationBaseline(identity: identity, report: report).write(to: baselineURL)

        // The same configuration must reproduce itself, or the extraction
        // parity gate can never be trusted.
        let (again, identityAgain) = try await runner.run(against: engine, label: "rerun")
        let loaded = try EvaluationBaseline.load(from: baselineURL)
        #expect(identityAgain.isComparable(with: loaded.identity))
        #expect(again.gate(against: loaded.report).passed)
    }
}
