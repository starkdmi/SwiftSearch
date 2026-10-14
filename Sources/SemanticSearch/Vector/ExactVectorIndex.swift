import Foundation

/// Brute-force inner-product search over a contiguous matrix.
///
/// The reference implementation. It is what recall@k for the HNSW index is
/// measured against, what small or heavily filtered spaces use outright, and
/// what the engine falls back to when a graph snapshot fails to load.
public final class ExactVectorIndex: VectorIndex, @unchecked Sendable {

    private let lock = NSLock()
    public let dimensions: Int

    /// `rows * dimensions` floats, row-major.
    private var storage: [Float]
    /// Row → slot. Rows are dense and reused: a removal swaps the last row into
    /// the hole so the matrix stays contiguous for `vDSP_mmul`.
    private var slots: [Int]
    private var rowForSlot: [Int: Int]

    public init(dimensions: Int) {
        self.dimensions = max(0, dimensions)
        self.storage = []
        self.slots = []
        self.rowForSlot = [:]
    }

    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return slots.count
    }

    public var isExact: Bool { true }

    public func insert(slot: Int, vector: [Float]) throws {
        guard vector.count >= dimensions else {
            throw SemanticSearchError.configuration(
                "vector has \(vector.count) dimensions, index expects \(dimensions)")
        }
        lock.lock()
        defer { lock.unlock() }

        let row: Int
        if let existing = rowForSlot[slot] {
            row = existing
        } else {
            row = slots.count
            slots.append(slot)
            rowForSlot[slot] = row
            storage.append(contentsOf: repeatElement(0, count: dimensions))
        }
        let base = row * dimensions
        for index in 0..<dimensions { storage[base + index] = vector[index] }
    }

    public func remove(slot: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let row = rowForSlot.removeValue(forKey: slot) else { return }

        let lastRow = slots.count - 1
        if row != lastRow {
            let movedSlot = slots[lastRow]
            slots[row] = movedSlot
            rowForSlot[movedSlot] = row
            let destination = row * dimensions
            let source = lastRow * dimensions
            for index in 0..<dimensions { storage[destination + index] = storage[source + index] }
        }
        slots.removeLast()
        storage.removeLast(dimensions)
    }

    public func contains(slot: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return rowForSlot[slot] != nil
    }

    public func search(query: [Float], limit: Int, isAllowed: (@Sendable (Int) -> Bool)?) throws
        -> [VectorMatch]
    {
        lock.lock()
        defer { lock.unlock() }
        guard !slots.isEmpty, limit > 0 else { return [] }

        let comparedDimensions = min(dimensions, query.count)
        let rowFilter: ((Int) -> Bool)? = isAllowed.map { predicate in
            { [slots] row in predicate(slots[row]) }
        }

        let matches = VectorMath.topMatches(
            query: query,
            storage: storage,
            rows: slots.count,
            dimensions: dimensions,
            comparedDimensions: comparedDimensions,
            limit: limit,
            isAllowed: rowFilter,
            // Rows move when a slot is removed, so a tie resolved on the row
            // would depend on the index's deletion history.
            tieBreaker: { [slots] row in slots[row] })

        return matches.map { VectorMatch(slot: slots[$0.row], score: $0.score) }
    }

    public func removeAll() throws {
        lock.lock()
        defer { lock.unlock() }
        storage.removeAll(keepingCapacity: true)
        slots.removeAll(keepingCapacity: true)
        rowForSlot.removeAll(keepingCapacity: true)
    }

    /// Full-precision vector for one slot, for rescoring and for evaluation.
    public func vector(forSlot slot: Int) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        guard let row = rowForSlot[slot] else { return nil }
        let base = row * dimensions
        return Array(storage[base..<(base + dimensions)])
    }
}
