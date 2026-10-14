import Foundation

/// One neighbour returned by a vector index.
public struct VectorMatch: Sendable, Hashable {
    /// Row identity in the store's representation table.
    public var slot: Int
    /// Inner product with the query. Higher is better.
    public var score: Float

    public init(slot: Int, score: Float) {
        self.slot = slot
        self.score = score
    }
}

/// A searchable set of vectors for one embedding space. Replaceable.
///
/// Implementations are shared across tasks and must be internally synchronised;
/// the engine calls them from its own isolation but a rebuild runs concurrently
/// with queries by design.
public protocol VectorIndex: AnyObject, Sendable {
    var dimensions: Int { get }
    var count: Int { get }
    /// True when every returned neighbour is guaranteed to be a true nearest
    /// neighbour. False for HNSW, which is why recall is a release gate.
    var isExact: Bool { get }

    func insert(slot: Int, vector: [Float]) throws
    func remove(slot: Int) throws
    func contains(slot: Int) -> Bool

    /// Nearest neighbours by inner product.
    ///
    /// `isAllowed` applies the caller's filter *inside* the search rather than
    /// after it. Filtering afterwards returns fewer than `limit` results
    /// whenever the filter is selective, and returns none at all when it is
    /// very selective — the failure mode where a metadata filter silently
    /// empties the dense channel while lexical search still answers.
    func search(query: [Float], limit: Int, isAllowed: (@Sendable (Int) -> Bool)?) throws
        -> [VectorMatch]

    func removeAll() throws
}

/// How a space's vectors are searched.
public enum VectorIndexKind: String, Sendable, Codable, CaseIterable {
    /// Brute force over full-precision vectors. Exact, and fast enough below a
    /// few thousand vectors or behind a selective filter.
    case exact
    /// USearch HNSW over half-precision vectors, with the top candidates
    /// rescored against the full-precision copies the store holds.
    case hnsw
}

/// Tuning for the HNSW index.
///
/// The two knobs here are the two the USearch Swift wrapper actually exposes.
/// Its `make` hard-codes `expansion_add` and `expansion_search` to the library
/// defaults and offers no setter, so a third field for search expansion would
/// be a control that silently does nothing — recall is bought through
/// connectivity and the rescore pool instead.
///
/// The defaults are chosen to clear ANN recall@100 at 99% of exact search, with
/// margin: measured at 1.00 on random vectors and 0.999 on the clustered ones a
/// real embedder produces.
public struct HNSWConfiguration: Sendable, Hashable, Codable {
    /// Graph connectivity (`M`). Higher improves recall and costs memory.
    public var connectivity: UInt32
    /// Multiplier on `limit` for how many ANN candidates to pull before
    /// rescoring against full precision.
    ///
    /// This is the cheaper of the two recall levers: the extra candidates are
    /// graph hops, not vector comparisons, and the rescoring that follows uses
    /// the full-precision copies SQLite already holds. At 4 the gate is missed
    /// on clustered vectors; at 8 it is cleared with room.
    public var rescoreMultiplier: Int
    /// Below this many vectors the space uses exact search regardless. Building
    /// a graph over 500 vectors costs more than scanning them.
    public var minimumVectorsForGraph: Int

    public init(
        connectivity: UInt32 = 24,
        rescoreMultiplier: Int = 8,
        minimumVectorsForGraph: Int = 2_000
    ) {
        self.connectivity = connectivity
        self.rescoreMultiplier = max(1, rescoreMultiplier)
        self.minimumVectorsForGraph = max(0, minimumVectorsForGraph)
    }

    public static let `default` = HNSWConfiguration()
}
