import Foundation
import Testing

@testable import SemanticSearch

@Suite("BM25")
struct BM25Tests {

    /// Reference implementation, written out longhand.
    ///
    /// Checking the scorer against a second expression of the same formula is
    /// the only way to catch a transcription error in it; an assertion that the
    /// scorer agrees with itself would pass whatever it computed.
    private func referenceScore(
        termFrequency tf: Double, documentLength: Double, averageLength: Double,
        documentFrequency df: Double, documentCount n: Double, k1: Double, b: Double
    ) -> Double {
        let idf = log(1 + ((n - df + 0.5) / (df + 0.5)))
        return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (documentLength / averageLength))))
    }

    @Test("Scores match the textbook formula")
    func matchesReferenceFormula() {
        let parameters = BM25Parameters(k1: 1.2, b: 0.75)
        let statistics = BM25Statistics(documentCount: 10, totalDocumentLength: 100)
        let postings = [
            BM25Posting(slot: 1, termFrequency: 3, documentLength: 12),
            BM25Posting(slot: 2, termFrequency: 1, documentLength: 8),
        ]

        let scores = BM25.score(
            queryTerms: ["alpha"], parameters: parameters, statistics: statistics,
            allowedSlots: nil, postings: { _ in postings })

        let expectedOne = referenceScore(
            termFrequency: 3, documentLength: 12, averageLength: 10, documentFrequency: 2,
            documentCount: 10, k1: 1.2, b: 0.75)
        let expectedTwo = referenceScore(
            termFrequency: 1, documentLength: 8, averageLength: 10, documentFrequency: 2,
            documentCount: 10, k1: 1.2, b: 0.75)

        #expect(abs(Double(scores[1] ?? 0) - expectedOne) < 1e-4)
        #expect(abs(Double(scores[2] ?? 0) - expectedTwo) < 1e-4)
    }

    @Test("Defaults are k1 = 1.2 and b = 0.75")
    func defaultParameters() {
        // The extracted implementation's values. Changing them silently would
        // make every extraction-parity comparison meaningless.
        #expect(BM25Parameters.default.k1 == 1.2)
        #expect(BM25Parameters.default.b == 0.75)
    }

    @Test("A repeated query term counts twice")
    func queryTermFrequencyMultiplies() {
        let statistics = BM25Statistics(documentCount: 10, totalDocumentLength: 100)
        let postings = [BM25Posting(slot: 1, termFrequency: 2, documentLength: 10)]

        let single = BM25.score(
            queryTerms: ["alpha"], parameters: .default, statistics: statistics,
            allowedSlots: nil, postings: { _ in postings })
        let doubled = BM25.score(
            queryTerms: ["alpha", "alpha"], parameters: .default, statistics: statistics,
            allowedSlots: nil, postings: { _ in postings })

        #expect(abs((doubled[1] ?? 0) - 2 * (single[1] ?? 0)) < 1e-5)
    }

    @Test("A term in every document scores zero")
    func universalTermHasNoIdf() {
        // idf = log(1 + (N − df + 0.5)/(df + 0.5)) is log(1 + ~0) when df == N.
        let statistics = BM25Statistics(documentCount: 3, totalDocumentLength: 30)
        let postings = (1...3).map {
            BM25Posting(slot: $0, termFrequency: 1, documentLength: 10)
        }
        let scores = BM25.score(
            queryTerms: ["the"], parameters: .default, statistics: statistics,
            allowedSlots: nil, postings: { _ in postings })
        for score in scores.values { #expect(score < 0.3) }
    }

    @Test("Length normalisation penalises longer documents")
    func lengthNormalization() {
        let statistics = BM25Statistics(documentCount: 10, totalDocumentLength: 100)
        let scores = BM25.score(
            queryTerms: ["alpha"], parameters: .default, statistics: statistics,
            allowedSlots: nil,
            postings: { _ in
                [
                    BM25Posting(slot: 1, termFrequency: 2, documentLength: 5),
                    BM25Posting(slot: 2, termFrequency: 2, documentLength: 50),
                ]
            })
        #expect((scores[1] ?? 0) > (scores[2] ?? 0))
    }

    @Test("b = 0 ignores document length")
    func bZeroDisablesLengthNormalization() {
        let statistics = BM25Statistics(documentCount: 10, totalDocumentLength: 100)
        let scores = BM25.score(
            queryTerms: ["alpha"], parameters: BM25Parameters(k1: 1.2, b: 0),
            statistics: statistics, allowedSlots: nil,
            postings: { _ in
                [
                    BM25Posting(slot: 1, termFrequency: 2, documentLength: 5),
                    BM25Posting(slot: 2, termFrequency: 2, documentLength: 500),
                ]
            })
        #expect(abs((scores[1] ?? 0) - (scores[2] ?? 0)) < 1e-6)
    }

    @Test("The allow-list is applied inside the loop")
    func allowedSlotsFilterInline() {
        let statistics = BM25Statistics(documentCount: 10, totalDocumentLength: 100)
        let scores = BM25.score(
            queryTerms: ["alpha"], parameters: .default, statistics: statistics,
            allowedSlots: [2],
            postings: { _ in
                [
                    BM25Posting(slot: 1, termFrequency: 2, documentLength: 10),
                    BM25Posting(slot: 2, termFrequency: 2, documentLength: 10),
                ]
            })
        #expect(scores[1] == nil)
        #expect(scores[2] != nil)
    }

    @Test("An empty corpus scores nothing")
    func emptyCorpus() {
        let scores = BM25.score(
            queryTerms: ["alpha"], parameters: .default, statistics: .empty,
            allowedSlots: nil, postings: { _ in [] })
        #expect(scores.isEmpty)
    }

    @Test("Top-k breaks ties on the lower slot")
    func topKTiesAreDeterministic() {
        let top = BM25.top([5: 1.0, 2: 1.0, 9: 1.0], limit: 2)
        #expect(top.map(\.slot) == [2, 5])
    }

    @Test("Top-k is stable across repeated runs")
    func topKIsStable() {
        let scores: [Int: Float] = Dictionary(
            uniqueKeysWithValues: (1...200).map { ($0, Float($0 % 7)) })
        let first = BM25.top(scores, limit: 10).map(\.slot)
        for _ in 0..<20 {
            #expect(BM25.top(scores, limit: 10).map(\.slot) == first)
        }
    }

    @Test("Scoring is order-independent across repeated runs")
    func scoringIsDeterministic() {
        // Float addition is not associative, so an unordered traversal of the
        // query terms makes exact ties depend on hash seeding.
        let statistics = BM25Statistics(documentCount: 100, totalDocumentLength: 1_000)
        let terms = ["alpha", "beta", "gamma", "delta", "epsilon"]
        let postings: (String) -> [BM25Posting] = { term in
            (1...20).map {
                BM25Posting(
                    slot: $0, termFrequency: (term.count + $0) % 5 + 1, documentLength: 10 + $0)
            }
        }
        let first = BM25.score(
            queryTerms: terms, parameters: .default, statistics: statistics,
            allowedSlots: nil, postings: postings)
        for _ in 0..<10 {
            let again = BM25.score(
                queryTerms: terms, parameters: .default, statistics: statistics,
                allowedSlots: nil, postings: postings)
            #expect(first == again)
        }
    }

    @Test("Invalid parameters are rejected")
    func parameterValidation() {
        #expect(throws: SemanticSearchError.self) {
            _ = try BM25Parameters(k1: 0, b: 0.75).validated()
        }
        #expect(throws: SemanticSearchError.self) {
            _ = try BM25Parameters(k1: 1.2, b: 1.5).validated()
        }
        #expect(throws: Never.self) { _ = try BM25Parameters.default.validated() }
    }

    // MARK: - In-memory index

    @Test("The in-memory index scores end to end")
    func inMemoryIndex() {
        let analyzer = LinguisticLexicalAnalyzer()
        var index = InMemoryBM25Index(
            parameters: .default, analyzerVersion: analyzer.analyzerVersion)
        index.add(
            slot: 1,
            analyzed: analyzer.analyzeDocument("the quick brown fox", language: "en"))
        index.add(
            slot: 2,
            analyzed: analyzer.analyzeDocument("a slow green turtle", language: "en"))

        let scores = index.score(queryTerms: analyzer.analyzeQuery("fox", language: "en"))
        #expect(scores[1] != nil)
        #expect(scores[2] == nil)
    }

    @Test("Removing a document removes its postings")
    func inMemoryRemoval() {
        let analyzer = LinguisticLexicalAnalyzer()
        var index = InMemoryBM25Index(analyzerVersion: analyzer.analyzerVersion)
        index.add(slot: 1, analyzed: analyzer.analyzeDocument("fox", language: "en"))
        index.add(slot: 2, analyzed: analyzer.analyzeDocument("fox", language: "en"))
        index.remove(slot: 1)

        #expect(index.statistics.documentCount == 1)
        let scores = index.score(queryTerms: ["fox"])
        #expect(scores[1] == nil)
        #expect(scores[2] != nil)
    }
}
