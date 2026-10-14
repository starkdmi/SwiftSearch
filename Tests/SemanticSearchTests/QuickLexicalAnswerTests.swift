import Foundation
import SemanticSearch
import Testing

@Suite("Quick lexical answer")
struct QuickLexicalAnswerTests {

    static func engine() async throws -> SemanticSearchEngine {
        let engine = try await SemanticSearchEngine.testEngine()
        try await engine.upsert([
            textItem("bramble", "the Bramble is gin, lemon, sugar and blackberry liqueur"),
            textItem("tides", "tidal turbines turn with the flood and the ebb"),
        ])
        return engine
    }

    static let quick = SearchOptions(quickLexicalAnswer: .default)

    @Test("A short query BM25 can answer is answered by BM25 alone")
    func shortQueryWithHits() async throws {
        let results = try await Self.engine().search("bramble", options: Self.quick)
        #expect(results.diagnostics.answeredLexically)
        #expect(results.hits.first?.itemID == "bramble")
        #expect(results.diagnostics.rerankedCandidates == 0)
        #expect(results.diagnostics.rerankSkippedReason == nil)
        #expect(results.diagnostics.channels.map(\.channel) == [.lexical])
    }

    @Test("Searching further is the same search without the option")
    func searchFurther() async throws {
        let engine = try await Self.engine()
        let further = try await engine.search("bramble")
        #expect(!further.diagnostics.answeredLexically)
        #expect(further.diagnostics.rerankedCandidates > 0)
        #expect(further.diagnostics.channels.count > 1)
    }

    @Test(
        "Long queries and queries BM25 finds nothing for take the whole pipeline",
        arguments: [
            "how are tidal turbines driven by the flood and the ebb",  // five words and more
            "kelp otters",  // no hit
        ])
    func fullPipeline(query: String) async throws {
        let results = try await Self.engine().search(query, options: Self.quick)
        #expect(!results.diagnostics.answeredLexically)
    }

    @Test("Function words do not count as terms")
    func termsAfterAnalysis() async throws {
        // Four words, two terms once the determiner and preposition go.
        let results = try await Self.engine().search("the flood of tides", options: Self.quick)
        #expect(results.diagnostics.answeredLexically)
    }

    @Test("Limits are validated")
    func validation() async throws {
        let engine = try await Self.engine()
        await #expect(throws: SemanticSearchError.self) {
            try await engine.search(
                "bramble",
                options: SearchOptions(quickLexicalAnswer: .init(minimumHits: 0)))
        }
    }
}
