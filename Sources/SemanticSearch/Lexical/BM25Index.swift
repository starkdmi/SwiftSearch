import Foundation

/// BM25 parameters. The defaults are the values the extracted implementation
/// shipped with, kept so that extraction is verifiably behaviour-preserving.
public struct BM25Parameters: Sendable, Hashable, Codable {
    /// Term-frequency saturation. Higher values keep rewarding repeats for
    /// longer.
    public var k1: Float
    /// Length normalisation, 0...1. At 0, document length is ignored entirely.
    public var b: Float

    public init(k1: Float = 1.2, b: Float = 0.75) {
        self.k1 = k1
        self.b = b
    }

    public static let `default` = BM25Parameters()

    func validated() throws -> BM25Parameters {
        guard k1 > 0 else { throw SemanticSearchError.configuration("BM25 k1 must be > 0") }
        guard b >= 0, b <= 1 else {
            throw SemanticSearchError.configuration("BM25 b must be in [0, 1]")
        }
        return self
    }

    var fingerprint: String {
        StableHash.of { hasher in
            hasher.append("bm25.v1")
            hasher.append(Double(k1))
            hasher.append(Double(b))
        }
    }
}

/// One document's contribution for one term.
public struct BM25Posting: Sendable, Hashable {
    public var slot: Int
    public var termFrequency: Int
    /// The document's primary-token count. Carried on the posting because the
    /// scorer needs it per posting and fetching it separately would be a
    /// lookup per document rather than a join.
    public var documentLength: Int

    public init(slot: Int, termFrequency: Int, documentLength: Int) {
        self.slot = slot
        self.termFrequency = termFrequency
        self.documentLength = max(1, documentLength)
    }
}

/// Corpus-wide statistics BM25 needs.
public struct BM25Statistics: Sendable, Hashable {
    public var documentCount: Int
    public var totalDocumentLength: Int

    public init(documentCount: Int, totalDocumentLength: Int) {
        self.documentCount = documentCount
        self.totalDocumentLength = totalDocumentLength
    }

    public var averageDocumentLength: Float {
        guard documentCount > 0 else { return 1 }
        return Float(max(1, totalDocumentLength)) / Float(documentCount)
    }

    public static let empty = BM25Statistics(documentCount: 0, totalDocumentLength: 0)
}

/// Okapi BM25.
///
/// Separated from storage so the same arithmetic serves the SQLite-backed index
/// the engine uses and the in-memory one the tests use. The formula and its
/// parameterisation are unchanged from the extracted implementation:
/// `idf = log(1 + (N − df + 0.5) / (df + 0.5))`, the standard saturation term,
/// and query term frequency as a multiplier so a word repeated in the query
/// counts twice.
public enum BM25 {

    /// Score every document holding at least one query term.
    ///
    /// `postings` is called once per *distinct* query term. `allowedSlots`
    /// filters inside the loop rather than after it, which is what keeps a
    /// selective filter from emptying the channel.
    public static func score(
        queryTerms: [String],
        parameters: BM25Parameters,
        statistics: BM25Statistics,
        allowedSlots: Set<Int>?,
        postings: (String) throws -> [BM25Posting]
    ) rethrows -> [Int: Float] {
        guard !queryTerms.isEmpty, statistics.documentCount > 0 else { return [:] }

        var queryTermFrequency: [String: Float] = [:]
        queryTermFrequency.reserveCapacity(queryTerms.count)
        for term in queryTerms { queryTermFrequency[term, default: 0] += 1 }

        let documentCount = Float(statistics.documentCount)
        let averageLength = max(statistics.averageDocumentLength, 0.000_001)
        let k1 = parameters.k1
        let b = parameters.b

        var scores: [Int: Float] = [:]

        // Sorted so that two runs over one corpus accumulate the same floating
        // point sum in the same order. Float addition is not associative, and
        // an unordered dictionary traversal makes exact score ties depend on
        // hash seeding.
        for term in queryTermFrequency.keys.sorted() {
            let queryFrequency = queryTermFrequency[term]!
            let termPostings = try postings(term)
            let documentFrequency = Float(termPostings.count)
            guard documentFrequency > 0 else { continue }

            let idf = log(1 + ((documentCount - documentFrequency + 0.5) / (documentFrequency + 0.5)))
            guard idf > 0 else { continue }

            for posting in termPostings {
                if let allowedSlots, !allowedSlots.contains(posting.slot) { continue }
                let tf = Float(posting.termFrequency)
                let denominator =
                    tf + k1 * (1 - b + b * (Float(posting.documentLength) / averageLength))
                guard denominator > 0 else { continue }
                scores[posting.slot, default: 0] +=
                    idf * ((tf * (k1 + 1)) / denominator) * queryFrequency
            }
        }

        return scores
    }

    /// Top `limit` slots by score, ties broken on the lower slot.
    public static func top(_ scores: [Int: Float], limit: Int) -> [(slot: Int, score: Float)] {
        guard limit > 0, !scores.isEmpty else { return [] }
        var collector = TopKCollector(limit: limit)
        for slot in scores.keys.sorted() {
            collector.offer(row: slot, score: scores[slot]!)
        }
        return collector.sortedDescending().map { (slot: $0.row, score: $0.score) }
    }
}

/// An in-memory BM25 index.
///
/// Used by callers with a corpus small enough to hold and by the tests, which
/// need a posting source with no database behind it. The engine uses the
/// store-backed path instead: at 100k units the resident posting lists run to
/// tens of megabytes and have to be rebuilt on every open, and a SQLite index
/// on `term` answers the same query without either cost.
public struct InMemoryBM25Index: Sendable {
    public private(set) var parameters: BM25Parameters
    public private(set) var analyzerVersion: String

    private var postings: [String: [BM25Posting]] = [:]
    private var lengths: [Int: Int] = [:]

    public init(parameters: BM25Parameters = .default, analyzerVersion: String = "") {
        self.parameters = parameters
        self.analyzerVersion = analyzerVersion
    }

    public var statistics: BM25Statistics {
        BM25Statistics(
            documentCount: lengths.count, totalDocumentLength: lengths.values.reduce(0, +))
    }

    public mutating func add(slot: Int, analyzed: AnalyzedDocument) {
        remove(slot: slot)
        let length = max(1, analyzed.tokenCount)
        lengths[slot] = length

        var frequencies: [String: Int] = [:]
        frequencies.reserveCapacity(analyzed.terms.count)
        for term in analyzed.terms { frequencies[term, default: 0] += 1 }
        for (term, frequency) in frequencies {
            postings[term, default: []].append(
                BM25Posting(slot: slot, termFrequency: frequency, documentLength: length))
        }
    }

    public mutating func remove(slot: Int) {
        guard lengths.removeValue(forKey: slot) != nil else { return }
        for term in postings.keys {
            postings[term]?.removeAll { $0.slot == slot }
            if postings[term]?.isEmpty == true { postings.removeValue(forKey: term) }
        }
    }

    public func score(queryTerms: [String], allowedSlots: Set<Int>? = nil) -> [Int: Float] {
        BM25.score(
            queryTerms: queryTerms,
            parameters: parameters,
            statistics: statistics,
            allowedSlots: allowedSlots,
            postings: { postings[$0] ?? [] })
    }
}
