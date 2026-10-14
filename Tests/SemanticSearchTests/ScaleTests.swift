import Foundation
import Testing

@testable import SemanticSearch

/// Scale tests at 1k, 10k and 100k units with multiple representations.
///
/// The 100k case is opt-in: it writes roughly a gigabyte and takes minutes, so
/// it runs only when `SEMANTIC_SEARCH_RUN_SCALE_TESTS=1` is set. The 1k and 10k
/// cases run every time, because the shape of a scaling problem is usually
/// visible by 10k and the difference between "fine" and "quadratic" is not
/// something to discover in CI at 100k.
@Suite("Scale")
struct ScaleTests {

    static var runsLargeScale: Bool {
        ["1", "true", "yes"].contains(
            ProcessInfo.processInfo.environment["SEMANTIC_SEARCH_RUN_SCALE_TESTS"]?.lowercased()
                ?? "")
    }

    /// Items with three representations each — original text, a caption and a
    /// summary — so the store holds three times the unit count.
    static func corpus(units: Int) -> [SearchItem] {
        (0..<units).map { index in
            SearchItem(
                id: ItemID("unit-\(index)"),
                parentID: ItemID("group-\(index / 100)"),
                metadata: [
                    "bucket": .integer(Int64(index % 50)),
                    "band": .string(index % 2 == 0 ? "even" : "odd"),
                ],
                locator: SourceLocator(
                    source: "corpus://unit-\(index)", pageIndex: index / 20,
                    characterRange: SourceLocator.CharacterRange(location: 0, length: 64)),
                representations: [
                    ContentRepresentation(
                        id: "text", kind: .originalText,
                        content: .text(
                            "Unit \(index) discusses subject \(index % 997) in the context "
                                + "of topic \(index % 31) with supporting detail."),
                        language: "en"),
                    ContentRepresentation(
                        id: "caption", kind: .caption,
                        content: .text("A diagram illustrating subject \(index % 997)."),
                        language: "en"),
                    ContentRepresentation(
                        id: "summary", kind: .summary,
                        content: .text("Summary of unit \(index), topic \(index % 31)."),
                        language: "en"),
                ],
                ordinal: index)
        }
    }

    private func run(units: Int, indexKind: VectorIndexKind) async throws {
        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(
            storeURL: directory.storeURL, indexKind: indexKind, dimensions: 64)

        let indexStarted = ContinuousClock.now
        let report = try await engine.upsert(Self.corpus(units: units))
        let indexDuration = ContinuousClock.now - indexStarted

        #expect(report.itemsWritten == units)
        #expect(report.representationsWritten == units * 3)
        #expect(try await engine.representationCount() == units * 3)
        #expect(try await engine.vectorCount(space: "test") == units * 3)

        if indexKind == .hnsw { try await engine.rebuildVectorIndex(space: "test") }
        try await engine.flush()

        // A handful of queries, timed. The assertion is loose on purpose — this
        // is a guard against a quadratic regression, not a latency budget,
        // which belongs on real hardware with a real model.
        var durations: [Duration] = []
        for index in stride(from: 0, to: units, by: max(1, units / 20)) {
            let started = ContinuousClock.now
            let results = try await engine.search("subject \(index % 997)")
            durations.append(ContinuousClock.now - started)
            #expect(!results.hits.isEmpty)
        }

        let filtered = try await engine.search(
            "subject 100",
            options: SearchOptions(filter: .equals(key: "band", value: .string("even"))))
        #expect(filtered.hits.allSatisfy { $0.itemID.rawValue.hasPrefix("unit-") })

        let slowest = durations.max() ?? .zero
        #expect(slowest < .seconds(10), "slowest query took \(slowest) at \(units) units")

        print(
            """
            [scale] units=\(units) representations=\(units * 3) index=\(indexKind.rawValue) \
            build=\(indexDuration) slowest_query=\(slowest) \
            size=\(await engine.indexSizeBytes()) bytes
            """)

        try await engine.close()
    }

    @Test("1k units with three representations each")
    func oneThousand() async throws {
        try await run(units: 1_000, indexKind: .exact)
    }

    @Test("10k units, exact index")
    func tenThousandExact() async throws {
        try await run(units: 10_000, indexKind: .exact)
    }

    @Test("10k units, HNSW index")
    func tenThousandGraph() async throws {
        try await run(units: 10_000, indexKind: .hnsw)
    }

    @Test(
        "100k units with multiple representations",
        .enabled(if: ScaleTests.runsLargeScale))
    func oneHundredThousand() async throws {
        try await run(units: 100_000, indexKind: .hnsw)
    }

    @Test("Reopening a large index is fast")
    func reopenIsFast() async throws {
        let directory = try TemporaryDirectory()
        do {
            let engine = try await SemanticSearchEngine.testEngine(
                storeURL: directory.storeURL, indexKind: .hnsw)
            try await engine.upsert(Self.corpus(units: 5_000))
            try await engine.rebuildVectorIndex(space: "test")
            try await engine.close()
        }

        // The published snapshot means the graph is read rather than rebuilt.
        let started = ContinuousClock.now
        let reopened = try await SemanticSearchEngine.testEngine(
            storeURL: directory.storeURL, indexKind: .hnsw)
        let elapsed = ContinuousClock.now - started

        #expect(try await reopened.vectorCount(space: "test") == 15_000)
        #expect(elapsed < .seconds(30), "reopen took \(elapsed)")
        print("[scale] reopen_5k_units=\(elapsed)")
        try await reopened.close()
    }

    @Test("Incremental upserts into a populated index stay cheap")
    func incrementalUpsert() async throws {
        let directory = try TemporaryDirectory()
        let engine = try await SemanticSearchEngine.testEngine(
            storeURL: directory.storeURL, indexKind: .hnsw)
        try await engine.upsert(Self.corpus(units: 5_000))

        let started = ContinuousClock.now
        try await engine.upsert(
            [textItem("late-arrival", "an entirely new document about photogrammetry")])
        let elapsed = ContinuousClock.now - started

        #expect(elapsed < .seconds(5), "single incremental upsert took \(elapsed)")

        // Checked lexically. The hashing embedder is deterministic, not
        // semantic, so its idea of "nearest" among 15,000 vectors is noise —
        // asserting on the fused ranking here would be asserting on that noise
        // rather than on whether the new item reached the index.
        let results = try await engine.search(
            "photogrammetry", options: SearchOptions(mode: .lexicalOnly))
        #expect(results.hits.first?.itemID == "late-arrival")
        try await engine.close()
    }
}
