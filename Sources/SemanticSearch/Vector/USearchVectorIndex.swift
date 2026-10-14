import Foundation
import USearch

/// HNSW index over half-precision vectors, with full-precision rescoring.
///
/// Two-stage by design. The graph is traversed over `f16` copies, which halves
/// its resident size and costs a little recall; the candidates it returns are
/// then rescored against the full-precision vectors SQLite holds, which
/// restores the ordering quantisation perturbed. Pulling
/// ``HNSWConfiguration/rescoreMultiplier`` times `limit` candidates before
/// rescoring is what makes the second stage able to fix the first.
///
/// Slots are used directly as USearch keys. Slots are non-negative and stable
/// for the life of a representation, so no key map is needed and a reopened
/// snapshot addresses the same rows.
public final class USearchVectorIndex: VectorIndex, @unchecked Sendable {

    private let lock = NSLock()
    private var index: USearchIndex
    private var liveSlots: Set<Int>
    private let configuration: HNSWConfiguration
    /// Supplies the full-precision vector for a slot. Returning `nil` leaves
    /// the candidate at its approximate score rather than dropping it: a vector
    /// the store cannot produce is a consistency problem for the writer to
    /// resolve, not a reason to lose a hit mid-query.
    private let fullPrecisionVector: @Sendable (Int) -> [Float]?

    public let dimensions: Int

    public init(
        dimensions: Int,
        configuration: HNSWConfiguration = .default,
        fullPrecisionVector: @escaping @Sendable (Int) -> [Float]?
    ) throws {
        self.dimensions = dimensions
        self.configuration = configuration
        self.fullPrecisionVector = fullPrecisionVector
        self.liveSlots = []
        self.index = try USearchIndex.make(
            metric: .ip,
            dimensions: UInt32(dimensions),
            connectivity: configuration.connectivity,
            quantization: .f16)
    }

    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return liveSlots.count
    }

    public var isExact: Bool { false }

    public func insert(slot: Int, vector: [Float]) throws {
        guard slot >= 0 else {
            throw SemanticSearchError.configuration("vector index slots must be non-negative")
        }
        guard vector.count >= dimensions else {
            throw SemanticSearchError.configuration(
                "vector has \(vector.count) dimensions, index expects \(dimensions)")
        }
        lock.lock()
        defer { lock.unlock() }

        if liveSlots.contains(slot) {
            // USearch has no in-place update; a re-added key would become a
            // second vector under the same key in a multi-index and an error
            // here.
            _ = try index.remove(key: USearchKey(slot))
        }
        try reserveIfNeeded(additional: 1)
        try vector.withUnsafeBufferPointer { buffer in
            try index.addSingle(key: USearchKey(slot), vector: buffer.baseAddress!)
        }
        liveSlots.insert(slot)
    }

    public func remove(slot: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard slot >= 0 else { return }
        liveSlots.remove(slot)
        // Removed from the graph whether or not `liveSlots` knew about it.
        // After ``load(from:slots:)`` the graph can hold keys the store has
        // since deleted — that is exactly what replaying a tombstone is for —
        // and `liveSlots`, which comes from the store, does not list them.
        // Skipping the graph here would leave the deleted vector searchable.
        _ = try index.remove(key: USearchKey(slot))
    }

    public func contains(slot: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return liveSlots.contains(slot)
    }

    public func search(query: [Float], limit: Int, isAllowed: (@Sendable (Int) -> Bool)?) throws
        -> [VectorMatch]
    {
        guard limit > 0, !query.isEmpty else { return [] }
        // USearch takes a pointer and reads `dimensions` floats from it,
        // whatever the array holds: a Matryoshka query narrower than the
        // graph read past its end. Padded with zeros, its inner product with
        // a stored vector is the prefix product ``ExactVectorIndex`` computes,
        // so both indexes score a reduced query alike.
        let query =
            query.count >= dimensions
            ? query : query + [Float](repeating: 0, count: dimensions - query.count)

        lock.lock()
        let liveCount = liveSlots.count
        let approximate: [(slot: Int, distance: Float)]
        do {
            approximate = try searchLocked(
                query: query,
                limit: min(limit * configuration.rescoreMultiplier, max(limit, liveCount)),
                isAllowed: isAllowed)
        } catch {
            lock.unlock()
            throw error
        }
        lock.unlock()

        // Rescoring runs outside the lock: it reads SQLite, and holding the
        // index lock across a database read would serialise every concurrent
        // query behind the slowest one.
        var collector = TopKCollector(limit: limit)
        var scores: [Int: Float] = [:]
        scores.reserveCapacity(approximate.count)
        for (offset, candidate) in approximate.enumerated() {
            let score: Float
            if let exact = fullPrecisionVector(candidate.slot) {
                score = VectorMath.dot(query, exact, dimensions: dimensions)
            } else {
                // USearch reports `ip` as a distance, 1 - similarity.
                score = 1 - candidate.distance
            }
            scores[candidate.slot] = score
            // Ties break on the slot, matching ``ExactVectorIndex``. Breaking
            // them on the ANN's own ordering instead would make the exact and
            // approximate indexes disagree on inputs where both are right,
            // which is indistinguishable from a recall failure in a diff.
            collector.offer(row: offset, score: score, tieBreaker: candidate.slot)
        }

        return collector.sortedDescending().map {
            let slot = approximate[$0.row].slot
            return VectorMatch(slot: slot, score: scores[slot] ?? $0.score)
        }
    }

    private func searchLocked(
        query: [Float], limit: Int, isAllowed: (@Sendable (Int) -> Bool)?
    ) throws -> [(slot: Int, distance: Float)] {
        guard !liveSlots.isEmpty else { return [] }
        let wanted = UInt32(max(1, min(limit, liveSlots.count)))

        var keys = [USearchKey](repeating: 0, count: Int(wanted))
        var distances = [Float](repeating: 0, count: Int(wanted))

        let found: UInt32 = try query.withUnsafeBufferPointer { queryBuffer in
            try keys.withUnsafeMutableBufferPointer { keyBuffer in
                try distances.withUnsafeMutableBufferPointer { distanceBuffer in
                    if let isAllowed {
                        return try index.filteredSearchSingle(
                            vector: queryBuffer.baseAddress!,
                            count: wanted,
                            filter: { key in isAllowed(Int(key)) },
                            keys: keyBuffer.baseAddress!,
                            distances: distanceBuffer.baseAddress!)
                    }
                    return try index.searchSingle(
                        vector: queryBuffer.baseAddress!,
                        count: wanted,
                        keys: keyBuffer.baseAddress!,
                        distances: distanceBuffer.baseAddress!)
                }
            }
        }

        return (0..<Int(found)).map { (slot: Int(keys[$0]), distance: distances[$0]) }
    }

    public func removeAll() throws {
        lock.lock()
        defer { lock.unlock() }
        try index.clear()
        liveSlots.removeAll(keepingCapacity: true)
    }

    // MARK: - Persistence

    /// Write the graph to disk.
    ///
    /// Callers publish snapshots atomically by saving to a temporary path and
    /// renaming; see `VectorIndexStore`.
    public func save(to url: URL) throws {
        lock.lock()
        defer { lock.unlock() }
        try index.save(path: url.path)
    }

    /// Replace the in-memory graph with one read from disk.
    ///
    /// `slots` has to come from the store rather than from the file: USearch
    /// exposes no key enumeration, and the index needs to know which keys are
    /// live to answer `contains` and to size reservations.
    public func load(from url: URL, slots: Set<Int>) throws {
        lock.lock()
        defer { lock.unlock() }
        try index.load(path: url.path)
        liveSlots = slots
    }

    private func reserveIfNeeded(additional: Int) throws {
        // USearch grows by reallocation; reserving in powers of two keeps a
        // 100k-vector build to seventeen reallocations instead of 100k.
        let needed = liveSlots.count + additional
        guard needed > reservedCapacity else { return }
        var capacity = max(1_024, reservedCapacity)
        while capacity < needed { capacity *= 2 }
        try index.reserve(UInt32(capacity))
        reservedCapacity = capacity
    }

    private var reservedCapacity = 0
}
