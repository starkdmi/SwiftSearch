import Foundation

extension SearchStore {

    /// Corpus statistics for BM25.
    ///
    /// One aggregate over `lexical_documents`, which has a row per lexically
    /// indexed representation and no blobs. Cheap enough to run per query, and
    /// running it per query is what keeps it correct across concurrent writes
    /// without a cache to invalidate.
    func lexicalStatistics() throws -> BM25Statistics {
        try withStatement("SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM lexical_documents")
        { statement in
            guard try statement.step() else { return .empty }
            return BM25Statistics(
                documentCount: statement.int(0), totalDocumentLength: statement.int(1))
        }
    }

    /// Postings for one term, with each document's length attached.
    func postings(term: String) throws -> [BM25Posting] {
        try withStatement(
            """
            SELECT t.slot, t.term_frequency, d.token_count
            FROM lexical_terms t
            JOIN lexical_documents d ON d.slot = t.slot
            WHERE t.term = ?
            """
        ) { statement in
            statement.bind(1, term)
            var result: [BM25Posting] = []
            try statement.forEachRow { row in
                result.append(
                    BM25Posting(
                        slot: row.int(0), termFrequency: row.int(1), documentLength: row.int(2)))
            }
            return result
        }
    }

    /// Score a query against the lexical index.
    func lexicalScores(
        queryTerms: [String],
        parameters: BM25Parameters,
        allowedSlots: Set<Int>?
    ) throws -> [Int: Float] {
        let statistics = try lexicalStatistics()
        return try BM25.score(
            queryTerms: queryTerms,
            parameters: parameters,
            statistics: statistics,
            allowedSlots: allowedSlots,
            postings: { try postings(term: $0) })
    }
}
