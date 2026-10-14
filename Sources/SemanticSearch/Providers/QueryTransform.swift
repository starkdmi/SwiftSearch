import Foundation

/// One query the engine will actually run.
public struct QueryVariant: Sendable {
    public var query: SearchQuery
    /// Distinguishes this variant's channels from the original's, so fusion can
    /// split the family weight between them rather than letting expansion
    /// multiply a family's say in the ranking.
    public var variant: String?
    /// Why this variant exists, for diagnostics.
    public var reason: String?

    public init(query: SearchQuery, variant: String? = nil, reason: String? = nil) {
        self.query = query
        self.variant = variant
        self.reason = reason
    }
}

/// Rewrites or expands a query before retrieval. Replaceable, and off by
/// default.
///
/// Translation and expansion are extension points rather than behaviour. The
/// plan's default is that multilingual retrieval runs on original-language
/// content and original-language queries: a translated query is a second guess
/// at what the user meant, and pivoting every query through one language loses
/// to a genuinely multilingual embedding space more often than it wins.
///
/// When a transform is installed, the original query is always retained as the
/// first variant and the number of extra variants is bounded by the engine.
public protocol QueryTransform: Sendable {
    var identifier: String { get }
    /// Upper bound on variants this transform will add, excluding the original.
    var maximumAdditionalVariants: Int { get }

    func transform(_ query: SearchQuery) async throws -> [QueryVariant]
}

/// The default: the query, unchanged.
public struct IdentityQueryTransform: QueryTransform {
    public init() {}
    public var identifier: String { "identity" }
    public var maximumAdditionalVariants: Int { 0 }

    public func transform(_ query: SearchQuery) async throws -> [QueryVariant] {
        [QueryVariant(query: query)]
    }
}
