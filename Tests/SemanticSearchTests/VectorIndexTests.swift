import Foundation
import Testing

@testable import SemanticSearch

@Suite("Vector indexes")
struct VectorIndexTests {

    /// Deterministic unit vectors, from a seeded generator so a failure is
    /// reproducible rather than "sometimes".
    private func vectors(count: Int, dimensions: Int, seed: UInt64 = 42) -> [[Float]] {
        var state = seed
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let mantissa: UInt64 = state >> 11
            let scale: Double = Double(UInt64(1) << 53)
            let unit: Double = Double(mantissa) / scale
            return Float(unit * 2 - 1)
        }
        return (0..<count).map { _ in
            VectorMath.normalized((0..<dimensions).map { _ in next() })
        }
    }

    // MARK: - Exact index

    @Test("Exact search returns true nearest neighbours")
    func exactIsExact() throws {
        let data = vectors(count: 200, dimensions: 32)
        let index = ExactVectorIndex(dimensions: 32)
        for (slot, vector) in data.enumerated() { try index.insert(slot: slot, vector: vector) }

        let query = data[7]
        let matches = try index.search(query: query, limit: 5, isAllowed: nil)
        #expect(matches.first?.slot == 7)
        #expect(abs((matches.first?.score ?? 0) - 1) < 1e-4)

        // Brute force, computed independently.
        var scored: [(slot: Int, score: Float)] = []
        for (slot, vector) in data.enumerated() {
            scored.append((slot: slot, score: VectorMath.dot(query, vector)))
        }
        scored.sort { lhs, rhs in
            lhs.score == rhs.score ? lhs.slot < rhs.slot : lhs.score > rhs.score
        }
        let expected = scored.prefix(5).map(\.slot)
        #expect(matches.map(\.slot) == Array(expected))
    }

    @Test("Removal compacts without disturbing the rest")
    func removalKeepsOthers() throws {
        let data = vectors(count: 50, dimensions: 16)
        let index = ExactVectorIndex(dimensions: 16)
        for (slot, vector) in data.enumerated() { try index.insert(slot: slot, vector: vector) }

        try index.remove(slot: 10)
        #expect(index.count == 49)
        #expect(!index.contains(slot: 10))
        #expect(index.contains(slot: 49))

        // The swap-with-last compaction must not corrupt the moved row.
        let roundTripped = index.vector(forSlot: 49)
        #expect(roundTripped != nil)
        #expect(abs(VectorMath.dot(roundTripped!, data[49]) - 1) < 1e-5)
    }

    @Test("Re-inserting a slot replaces its vector")
    func reinsertionReplaces() throws {
        let index = ExactVectorIndex(dimensions: 4)
        try index.insert(slot: 1, vector: [1, 0, 0, 0])
        try index.insert(slot: 1, vector: [0, 1, 0, 0])
        #expect(index.count == 1)
        #expect(index.vector(forSlot: 1) == [0, 1, 0, 0])
    }

    @Test("The filter runs inside the search")
    func filterIsApplied() throws {
        let data = vectors(count: 100, dimensions: 16)
        let index = ExactVectorIndex(dimensions: 16)
        for (slot, vector) in data.enumerated() { try index.insert(slot: slot, vector: vector) }

        // A selective filter must still return `limit` results, which is only
        // possible if it is applied during the scan rather than after it.
        let allowed = Set(stride(from: 0, to: 100, by: 10))
        let matches = try index.search(
            query: data[0], limit: 5, isAllowed: { allowed.contains($0) })
        #expect(matches.count == 5)
        #expect(matches.allSatisfy { allowed.contains($0.slot) })
    }

    @Test("Ties break on the lower slot")
    func exactTiesAreDeterministic() throws {
        let index = ExactVectorIndex(dimensions: 4)
        for slot in [9, 3, 7] { try index.insert(slot: slot, vector: [1, 0, 0, 0]) }
        let matches = try index.search(query: [1, 0, 0, 0], limit: 3, isAllowed: nil)
        #expect(matches.map(\.slot) == [3, 7, 9])
    }

    @Test("A prefix comparison renormalises")
    func matryoshkaPrefix() {
        let vector = VectorMath.normalized([3, 4, 12, 0])
        let prefix = VectorMath.normalizedPrefix(vector, dimensions: 2)
        #expect(prefix.count == 2)
        #expect(abs(VectorMath.l2Norm(prefix) - 1) < 1e-5)
    }

    @Test("Searching an empty index returns nothing")
    func emptyIndex() throws {
        let index = ExactVectorIndex(dimensions: 8)
        #expect(try index.search(query: [Float](repeating: 1, count: 8), limit: 5,
            isAllowed: nil).isEmpty)
    }

    @Test("A vector of the wrong width is rejected")
    func dimensionMismatchIsRejected() {
        let index = ExactVectorIndex(dimensions: 8)
        #expect(throws: SemanticSearchError.self) {
            try index.insert(slot: 1, vector: [1, 2, 3])
        }
    }

    // MARK: - HNSW

    @Test("HNSW recall against exact search clears the 99% gate")
    func hnswRecallGate() throws {
        // The release gate, measured directly rather than inferred from
        // ranking quality: a reranker can rescue a poor candidate list often
        // enough to hide a graph problem in nDCG.
        let dimensions = 64
        let data = vectors(count: 4_000, dimensions: dimensions, seed: 7)

        let exact = ExactVectorIndex(dimensions: dimensions)
        let graph = try USearchVectorIndex(
            dimensions: dimensions,
            configuration: HNSWConfiguration(
                connectivity: 24, rescoreMultiplier: 8),
            fullPrecisionVector: { slot in
                slot < data.count ? data[slot] : nil
            })

        for (slot, vector) in data.enumerated() {
            try exact.insert(slot: slot, vector: vector)
            try graph.insert(slot: slot, vector: vector)
        }

        let queries = vectors(count: 40, dimensions: dimensions, seed: 99)
        var recalls: [Double] = []
        for query in queries {
            let approximate = try graph.search(query: query, limit: 100, isAllowed: nil)
            let truth = try exact.search(query: query, limit: 100, isAllowed: nil)
            recalls.append(
                RetrievalMetrics.annRecall(approximate: approximate, exact: truth, k: 100))
        }

        let mean = RetrievalMetrics.mean(recalls)
        #expect(mean >= 0.99, "recall@100 was \(mean), gate is 0.99")
    }

    @Test("A narrower query scores as the exact index's prefix, not past the buffer")
    func hnswPrefixQuery() throws {
        let dimensions = 64
        let data = vectors(count: 500, dimensions: dimensions, seed: 5)
        let exact = ExactVectorIndex(dimensions: dimensions)
        let graph = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        for (slot, vector) in data.enumerated() {
            try exact.insert(slot: slot, vector: vector)
            try graph.insert(slot: slot, vector: vector)
        }
        let query = VectorMath.normalizedPrefix(data[9], dimensions: 32)
        #expect(query.count == 32)
        let approximate = try graph.search(query: query, limit: 5, isAllowed: nil)
        let truth = try exact.search(query: query, limit: 5, isAllowed: nil)
        #expect(approximate.first?.slot == 9)
        #expect(approximate.first?.slot == truth.first?.slot)
        if let a = approximate.first?.score, let b = truth.first?.score {
            #expect(abs(a - b) < 1e-5)
        }
    }

    @Test("Rescoring restores full-precision scores")
    func rescoringUsesFullPrecision() throws {
        let dimensions = 32
        let data = vectors(count: 500, dimensions: dimensions, seed: 11)
        let graph = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        for (slot, vector) in data.enumerated() { try graph.insert(slot: slot, vector: vector) }

        let query = data[3]
        let matches = try graph.search(query: query, limit: 5, isAllowed: nil)
        #expect(matches.first?.slot == 3)
        // Half-precision alone would not land within 1e-4 of 1.0.
        #expect(abs((matches.first?.score ?? 0) - 1) < 1e-4)
    }

    @Test("HNSW honours a filter")
    func hnswFiltering() throws {
        let dimensions = 32
        let data = vectors(count: 1_000, dimensions: dimensions, seed: 13)
        let graph = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        for (slot, vector) in data.enumerated() { try graph.insert(slot: slot, vector: vector) }

        let allowed = Set(stride(from: 0, to: 1_000, by: 5))
        let matches = try graph.search(
            query: data[0], limit: 10, isAllowed: { allowed.contains($0) })
        #expect(!matches.isEmpty)
        #expect(matches.allSatisfy { allowed.contains($0.slot) })
    }

    @Test("Removal takes a vector out of the results")
    func hnswRemoval() throws {
        let dimensions = 16
        let data = vectors(count: 200, dimensions: dimensions, seed: 17)
        let graph = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        for (slot, vector) in data.enumerated() { try graph.insert(slot: slot, vector: vector) }

        try graph.remove(slot: 5)
        #expect(!graph.contains(slot: 5))
        let matches = try graph.search(query: data[5], limit: 10, isAllowed: nil)
        #expect(!matches.contains { $0.slot == 5 })
    }

    @Test("A saved graph reloads and answers the same queries")
    func snapshotRoundTrip() throws {
        let dimensions = 32
        let data = vectors(count: 600, dimensions: dimensions, seed: 23)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("semanticsearch-snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let original = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        for (slot, vector) in data.enumerated() { try original.insert(slot: slot, vector: vector) }

        let file = directory.appendingPathComponent("space.usearch")
        try original.save(to: file)

        let restored = try USearchVectorIndex(
            dimensions: dimensions,
            fullPrecisionVector: { slot in slot < data.count ? data[slot] : nil })
        try restored.load(from: file, slots: Set(0..<data.count))

        #expect(restored.count == data.count)
        let before = try original.search(query: data[10], limit: 10, isAllowed: nil)
        let after = try restored.search(query: data[10], limit: 10, isAllowed: nil)
        #expect(before.map(\.slot) == after.map(\.slot))
    }
}
