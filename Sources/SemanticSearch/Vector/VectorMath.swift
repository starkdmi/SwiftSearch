import Accelerate
import Foundation

/// Vector primitives, Accelerate-backed.
///
/// Every embedding the engine stores is L2-normalised, so the inner product
/// *is* the cosine similarity and there is no division on the hot path.
public enum VectorMath {

    /// Inner product over the first `dimensions` components.
    ///
    /// A shorter `dimensions` is the Matryoshka case: comparing prefixes of two
    /// vectors from a provider that supports it. The prefix is not itself
    /// normalised, which is why ``normalizedPrefix(_:dimensions:)`` exists and
    /// why the engine calls it before storing or querying at a reduced width.
    public static func dot(_ lhs: [Float], _ rhs: [Float], dimensions: Int? = nil) -> Float {
        let limit = min(dimensions ?? Int.max, lhs.count, rhs.count)
        guard limit > 0 else { return 0 }
        var result: Float = 0
        lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                vDSP_dotpr(lhsBuffer.baseAddress!, 1, rhsBuffer.baseAddress!, 1, &result,
                    vDSP_Length(limit))
            }
        }
        return result
    }

    public static func l2Norm(_ vector: [Float]) -> Float {
        guard !vector.isEmpty else { return 0 }
        var result: Float = 0
        vector.withUnsafeBufferPointer { buffer in
            vDSP_svesq(buffer.baseAddress!, 1, &result, vDSP_Length(buffer.count))
        }
        return sqrt(result)
    }

    /// L2-normalise in place semantics. A zero vector is returned unchanged
    /// rather than producing NaNs.
    public static func normalized(_ vector: [Float]) -> [Float] {
        let norm = l2Norm(vector)
        guard norm > 0, norm.isFinite else { return vector }
        var output = vector
        var divisor = norm
        output.withUnsafeMutableBufferPointer { buffer in
            vDSP_vsdiv(buffer.baseAddress!, 1, &divisor, buffer.baseAddress!, 1,
                vDSP_Length(buffer.count))
        }
        return output
    }

    /// The first `dimensions` components, renormalised.
    ///
    /// Truncating a normalised vector leaves a vector of length < 1, and the
    /// inner product of two such vectors is not a cosine. Renormalising restores
    /// the identity the whole ranking pipeline assumes.
    public static func normalizedPrefix(_ vector: [Float], dimensions: Int?) -> [Float] {
        guard let dimensions, dimensions > 0, dimensions < vector.count else {
            return normalized(vector)
        }
        return normalized(Array(vector.prefix(dimensions)))
    }

    /// Exact top-k by inner product over a contiguous matrix of row vectors.
    ///
    /// `storage` is `rows * dimensions` floats in row-major order. One
    /// `vDSP_mmul` over the whole matrix beats a per-row `dotpr` loop by a wide
    /// margin once rows run into the thousands.
    static func topMatches(
        query: [Float],
        storage: [Float],
        rows: Int,
        dimensions: Int,
        comparedDimensions: Int,
        limit: Int,
        isAllowed: ((Int) -> Bool)?,
        tieBreaker: ((Int) -> Int)? = nil
    ) -> [(row: Int, score: Float)] {
        guard rows > 0, dimensions > 0, limit > 0, query.count >= comparedDimensions else {
            return []
        }

        var scores = [Float](repeating: 0, count: rows)

        if comparedDimensions == dimensions {
            storage.withUnsafeBufferPointer { matrix in
                query.withUnsafeBufferPointer { vector in
                    scores.withUnsafeMutableBufferPointer { output in
                        vDSP_mmul(
                            matrix.baseAddress!, 1,
                            vector.baseAddress!, 1,
                            output.baseAddress!, 1,
                            vDSP_Length(rows), 1, vDSP_Length(dimensions))
                    }
                }
            }
        } else {
            // A prefix comparison is a strided dot product per row; `mmul`
            // cannot express the gap between rows.
            storage.withUnsafeBufferPointer { matrix in
                query.withUnsafeBufferPointer { vector in
                    for row in 0..<rows {
                        var value: Float = 0
                        vDSP_dotpr(
                            matrix.baseAddress! + row * dimensions, 1,
                            vector.baseAddress!, 1,
                            &value, vDSP_Length(comparedDimensions))
                        scores[row] = value
                    }
                }
            }
        }

        var heap = TopKCollector(limit: limit)
        for row in 0..<rows {
            if let isAllowed, !isAllowed(row) { continue }
            // Ties break on the tie-breaker, not the row: rows are assigned in
            // insertion order and a deletion moves one, so two indexes holding
            // the same vectors would otherwise resolve a tie differently.
            heap.offer(row: row, score: scores[row], tieBreaker: tieBreaker?(row) ?? row)
        }
        return heap.sortedDescending().map { (row: $0.row, score: $0.score) }
    }
}

/// Bounded top-k with deterministic ties.
///
/// Ties break on the lower tie-breaker value, which is what makes two runs over
/// the same corpus produce the same ranking — a requirement an unordered
/// traversal would break.
struct TopKCollector {
    typealias Entry = (row: Int, score: Float, tieBreaker: Int)

    private var entries: [Entry] = []
    private let limit: Int

    init(limit: Int) {
        self.limit = max(0, limit)
        entries.reserveCapacity(self.limit)
    }

    static func isBetter(_ lhs: Entry, than rhs: Entry) -> Bool {
        if lhs.score == rhs.score { return lhs.tieBreaker < rhs.tieBreaker }
        return lhs.score > rhs.score
    }

    mutating func offer(row: Int, score: Float, tieBreaker: Int? = nil) {
        guard limit > 0 else { return }
        let candidate: Entry = (row: row, score: score, tieBreaker: tieBreaker ?? row)
        if entries.count < limit {
            insert(candidate)
            return
        }
        guard let worst = entries.last, Self.isBetter(candidate, than: worst) else { return }
        insert(candidate)
        entries.removeLast()
    }

    private mutating func insert(_ candidate: Entry) {
        // Linear insertion into a list bounded by `limit` (100 by default).
        // A heap wins asymptotically and loses at this size.
        var index = 0
        while index < entries.count, !Self.isBetter(candidate, than: entries[index]) {
            index += 1
        }
        entries.insert(candidate, at: index)
    }

    func sortedDescending() -> [Entry] { entries }
}
