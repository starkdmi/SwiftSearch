import Foundation

/// A reranker whose scores come from a closure rather than a model.
///
/// Lets a test state exactly what the reranker believes and then assert what
/// the engine does with that belief — that late fusion demotes rather than
/// deletes, that ties break deterministically, that truncation is reported.
/// None of which a real reranker can be made to demonstrate on demand.
public actor ScriptedRerankingProvider: RerankingProvider {

    public typealias Scorer = @Sendable (RerankQuery, RerankCandidate) -> RerankScore

    private let scorer: Scorer
    private let identityValue: RerankerIdentity
    private let capabilitiesValue: RerankingCapabilities
    private var prepared = false
    /// Every call, in order. Lets a test assert on the pool the engine actually
    /// sent — its size, its order, and whether parent context was attached.
    public private(set) var calls: [(query: RerankQuery, candidates: [RerankCandidate])] = []

    public init(
        identity: RerankerIdentity = RerankerIdentity(
            modelIdentity: "scripted", promptVersion: "test.v1"),
        capabilities: RerankingCapabilities = RerankingCapabilities(),
        scorer: @escaping Scorer
    ) {
        self.identityValue = identity
        self.capabilitiesValue = capabilities
        self.scorer = scorer
    }

    /// Scores by how many of the query's terms appear in the candidate's text.
    /// A plausible-looking default for tests that need *a* reranker rather than
    /// a particular one.
    public static func termOverlap() -> ScriptedRerankingProvider {
        ScriptedRerankingProvider { query, candidate in
            let terms = Set(query.text.lowercased().split(separator: " ").map(String.init))
            guard !terms.isEmpty else { return RerankScore(score: 0) }
            let text = candidate.text.lowercased()
            let matched = terms.filter { text.contains($0) }.count
            return RerankScore(score: Float(matched) / Float(terms.count))
        }
    }

    public var identity: RerankerIdentity { identityValue }
    public var capabilities: RerankingCapabilities { capabilitiesValue }
    public var isPrepared: Bool { prepared }

    public func prepare() async throws { prepared = true }
    public func release() async { prepared = false }

    public func score(query: RerankQuery, candidates: [RerankCandidate]) async throws
        -> [RerankScore]
    {
        guard prepared else {
            throw SemanticSearchError.providerNotPrepared(identityValue.modelIdentity)
        }
        try Task.checkCancellation()
        calls.append((query: query, candidates: candidates))
        return candidates.map { scorer(query, $0) }
    }

    public func reset() {
        calls.removeAll()
    }
}
